# CDRscope Tier 2: Unified TCR Quantization Pipeline

Tier 2 solves the fundamental problem of AIRR-seq analysis: **different samples
have wildly different CDR3 sequences and counts, making direct comparison
impossible**. The solution is a "TCR transcriptome" — a fixed-dimensional
reference space analogous to the ~30k genes in RNA-seq.

## Core Idea

```
                        TCR Transcriptome
                     ┌──────────────────────┐
  Sample A CDR3s ───► │  m=10,000 prototypes │ ───► [v1, v2, ..., v10000]
  Sample B CDR3s ───► │  (disease-agnostic)  │ ───► [v1, v2, ..., v10000]
  Sample C CDR3s ───► │  fixed reference     │ ───► [v1, v2, ..., v10000]
                     └──────────────────────┘
                          All samples now comparable
```

1. **Reference pool** — collect all unique CDR3s from multiple datasets (RA,
   CMV, MS, SLE, VDJdb) → ~846k unique sequences
2. **ESM-2 embedding** — `facebook/esm2_t12_35M_UR50D`, 480-dim per sequence
3. **K-means quantization** — find m=10,000 prototype centroids (m << n)
4. **Sample projection** — any sample's CDR3s → nearest centroid → count vector
5. **L2 normalization** — removes sequencing depth bias
6. **Linear SVM classification** — optimal for sparse high-dim data

## Saturation Hypothesis

As the reference pool grows, the optimal m (at fixed 80% variance explained)
grows but eventually **saturates** — the TCR sequence space is finite and
structured.

| Pool size | Fraction | m needed (80% VE) |
|-----------|----------|--------------------|
| 52,892 | 0.0625 | ~3,200 |
| 105,783 | 0.125 | ~4,500 |
| 211,567 | 0.25 | ~6,200 |
| 423,134 | 0.5 | ~8,800 |
| 846,267 | 1.0 | ~10,000 |

Saturation index: **0.45** (pool grew 20x, m grew only 3.8x)

## Pipeline Architecture

```
run_tier2.py (unified entry point)
  │
  ├── 1. build_reference_pool()     ← tcr_reference_quantization.py
  ├── 2. saturation_analysis()      ← tcr_reference_quantization.py
  ├── 3. train_reference_panel()     ← tcr_reference_quantization.py
  ├── 4. project_sample()           ← tcr_reference_quantization.py
  ├── 5. classify (Linear SVM)     ← run_tier2.py
  ├── 6. supervised_visualization  ← supervised_visualization.py
  ├── 7. interpretability          ← interpretability.py
  ├── 8. find_markers              ← find_markers.py
  └── cross_disease_benchmark      ← cross_disease_benchmark.py
```

## Usage

```bash
# Full pipeline
python run_tier2.py --all

# Step by step
python run_tier2.py --build-pool --datasets "RA CMV MS SLE VDJdb"
python run_tier2.py --saturation
python run_tier2.py --train-panel --m 10000
python run_tier2.py --project RA
python run_tier2.py --classify RA
python run_tier2.py --visualize
python run_tier2.py --interpret
```

## Module Overview

| Module | File | Description |
|--------|------|-------------|
| Reference quantization | `tcr_reference_quantization.py` | Pool construction, ESM-2 embedding, k-means, saturation, projection |
| Unified pipeline | `unified_pipeline.py` | Seurat-style analysis: L2 norm → PCA → UMAP → KMeans → SVM |
| Supervised visualization | `supervised_visualization.py` | SVM 1D, SVM+PC 2D, PLS-DA, LDA, supervised UMAP |
| Interpretability | `interpretability.py` | SVM weight ranking, V/J gene enrichment, motif analysis, physicochemical profiling, convergence |
| FindMarkers | `find_markers.py` | Wilcoxon rank-sum, logFC, AUC per prototype, volcano plot, heatmap |
| Cross-disease benchmark | `cross_disease_benchmark.py` | Multi-disease feature extraction and RF classification |
| Entry point | `run_tier2.py` | Unified CLI for all modules |

## Key Results (RA-TRB, m=10,000)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9964 |
| Specificity | 100% |
| Sensitivity | 98.2% |
| F1 | 0.989 |
| MCC | 0.978 |

### Interpretability highlights

- **TRBV10-3**: exclusively enriched in patient prototypes (360× in patients vs 0 in controls)
- **CDR3 motifs**: GYEQ, SSIA, SSIV — patient-specific
- **V gene diversity**: 1.2 in patients vs 2.6 in controls (reduced diversity = antigen-driven expansion)
- **Convergence**: patient prototypes show significant V/J gene convergence

## Tier 1 → Tier 2 Integration

```
Tier 1 (per-sample data)              Tier 2 (pool-only data)
┌────────────────────────┐           ┌────────────────────────┐
│ 65-feature extraction  │           │  Reference panel       │
│  RF/SVM classification │           │  (m=10,000 prototypes) │
│  ↓                     │           │  ↓                     │
│  Model checkpoint      │──────────►│  Project CDR3 pool →  │
│  (trained on labeled   │           │  m-dim vector          │
│   per-sample data)     │           │  ↓                     │
│                        │           │  Apply Tier 1 model →  │
│                        │           │  Disease score         │
└────────────────────────┘           └────────────────────────┘
```

## Reference Coordinate System (RCS) — the final piece (NEW, v2.3)

### Problem

Tier 2's original route is **supervised**: it depends on prior labels
(healthy vs disease). Real-world screening needs an **unsupervised** mode —
samples arrive unlabeled (or all nominally healthy), and we still need to answer:

1. Are there individual differences, and where do they lie?
2. If labels exist (e.g. RA cohorts), can they validate the unsupervised result?

### Core principle

A **fixed, invariant coordinate space** built **once** from CordBlood TRA data
(1.32M unique CDR3s → m=10,000 prototypes). Every new sample projects to a
**unique, stable position** through frozen transformers — the reference space
never changes after construction:

```
CDR3 sequences → ESM-2 embedding → nearest-centroid assignment (m=10,000)
              → L2 normalize → frozen PCA (50 PCs) → frozen UMAP (2D)
```

- Same sample → same coordinates (**invariance**)
- Same donor, different timepoints → nearby coordinates (**stability**)
- Reference origin = CordBlood mean vector

### Deviation: magnitude + direction

Each sample is characterized by two quantities relative to the reference origin:

| Quantity | Type | Question answered |
|----------|------|-------------------|
| **Deviation magnitude** | scalar distance from origin | "How abnormal?" |
| **Deviation direction** | vector of per-PC / per-prototype offsets | "Where abnormal?" |

Two samples can deviate equally far but in different directions — different
underlying biology. Direction vectors can themselves be clustered to discover
deviation subtypes.

### 7-view deviation visualization (`deviation_direction_viz.py`)

1. **PCA scatter** colored by magnitude (+ label overlay for post-hoc validation)
2. **Polar scatter** — radius = magnitude, angle = direction
3. **Radar chart** — deviation profile across top PC axes
4. **Top prototypes** — which CDR3 prototypes drive the deviation
5. **Direction heatmap** — samples × top-15 PCs
6. **Direction clusters** — K-means on direction → candidate subtypes
7. **Arrow field** — vectors from the reference origin

### Longitudinal stability (`longitudinal_validation_v2.py`)

46 samples from 25 donors (SLE GSE254176, Zenodo MDA1/HD1-3, RA controls).
Same-donor pairs are significantly closer than cross-donor pairs:

| Metric | Inter/Intra ratio | p-value |
|--------|-------------------|---------|
| Cosine distance | 1.39× | 3.9e-15 |
| Euclidean distance | 1.21× | 3.9e-15 |
| JS divergence | 1.08× | 7.4e-7 |
| RCS space | 1.16× | 1.5e-3 |

→ A donor's samples stay tightly grouped in the fixed space over time, while
remaining separable from other donors — exactly the property a reference
coordinate system needs.

### Unified entry point (`cdrscope_unified.py`)

- **Route A (supervised)** — runs when labels exist: Linear SVM 5-fold CV,
  AUC 0.9593, differential prototypes, ROC analysis
- **Route B (unsupervised)** — always runs: deviation magnitude scoring
  (AUC 0.73), anomaly ensemble OCSVM+LOF+JS (AUC up to 0.95), diversity
  analysis, composite High/Moderate/Normal stratification
  (RA cohort: High=54, Moderate=82, Normal=409)

### Key lesson

RA disease signal in TRA is **distributed** across many prototypes (GWAS-like),
not concentrated in a few markers. Unsupervised clustering does **not** recover
it (ARI −0.0076). Deviation-from-reference scoring is the right unsupervised
tool for screening; clustering is for subtype discovery; supervised SVM remains
the most powerful when labels are available.

### Chain-specific panels

CordBlood TRA is the first chain-specific reference panel. The same
construction applies to the remaining six chains (TRB, TRG, TRD, IGH, IGL, IGK).

## Dependencies

- Python 3.9+
- `torch`, `transformers` (ESM-2)
- `scikit-learn`, `numpy`, `pandas`
- `umap-learn`, `matplotlib`
- `scipy`
