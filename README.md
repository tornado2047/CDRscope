# CDRscope v2.1

**A complete closed-loop analysis framework for TCR/BCR immune repertoires —
disease-agnostic, multi-chain, interpretable.**

`CDRscope` implements a 10-module closed-loop pipeline that takes raw repertoire
data in and produces a full analysis report out — from feature engineering and
classification to significance testing, breakthrough analysis, biological
validation, and automated HTML report generation.

> RA (rheumatoid arthritis) data was used as the validation use case during
> development. The pipeline itself is **disease-agnostic**: point it at any
> control-vs-patient repertoire dataset and it runs end-to-end.

```
run_complete_analysis(data_dir, chain = "paired")   # one-line closed loop

# 10 modules:  Input → Features → ESM-2 → Reference Map → Classification
#              → UMAP → Significance → Breakthrough → Validation → Report
```

## What's new in v2.1

### Complete closed-loop pipeline (NEW)

A single orchestrator — `run_complete_analysis()` — runs all 10 modules in
sequence and writes every output to a results directory:

| # | Module | What it does | Key output |
|---|--------|--------------|------------|
| 1 | **Input & chain selection** | Auto-detect chains, load repertoire | `CDRobject` |
| 2 | **Feature engineering** | 65 features across 11 modules | `feature_matrix.csv` |
| 3 | **ESM-2 embedding** | 480-dim protein LM embeddings | `esm_embeddings.npy` |
| 4 | **Reference Map projection** | Project onto fixed UMAP space | `projected_coords.csv` |
| 5 | **Classification** | RF / LASSO / XGBoost / Ensemble + CV | `cv_results.csv` |
| 6 | **UMAP visualization** | 2D map with property-domain coloring | `*.png` |
| 7 | **Domain significance** | Fisher exact, OR, forest plots | `domain_significance.csv` |
| 8 | **Breakthrough analysis** | Expansion, axis decode, disease score, network | `breakthrough_summary.json` |
| 9 | **Biological validation** | Frequency redistribution, citrullination axis, HLA | `validation_summary.json` |
| 10 | **Report generation** | Self-contained HTML report | `CDRscope_Analysis_Report.html` |

Each module can be toggled on/off via parameters, so you can run only the
modules you need.

### Reference Map system (NEW)

A **fixed UMAP space** built from a trained neural network (655 KB weights).
Any new project's sequences can be projected onto the same 2D space, enabling
**cross-project comparison** — different inputs, one common map.

- `ProjectToReferenceMap()` — project new sequences onto the reference map
- `ReferenceMapPlot()` — ggplot visualization with new data overlay
- Reference map bundle shipped in `inst/reference_map/` (weights + metadata)

### Multi-chain support (v2.0)

Three chain modes, auto-detected from file names:

| Mode | Description | Use case |
|------|-------------|----------|
| `"single"` | Analyse the most abundant chain | Quick screening, single-chain data |
| `"paired"` | TRA+TRB joint analysis | Highest accuracy |
| `"all"` | All chains analysed jointly | Full multi-chain repertoire (up to 7 chains) |

Supported chains: TRA, TRB, TRG, TRD, IGH, IGK, IGL.

### Enhanced feature engineering (v2.0)

65 features across 11 modules:

| Module | Features | What it captures |
|--------|----------|-----------------|
| **6 original concept modules** | 20 | Diversity, motifs, selection, convergence, SHM, pairing |
| **RA_ClonalExpansion** | 11 | D50/D20, Morisita, Berger-Parker, Pielou, Renyi |
| **RA_GeneUsage** | 8 | V/J gene usage, V-J pairing |
| **RA_Physicochemical** | 14 | CDR3 length, AA composition, charge, hydrophobicity |
| **RA_ConvergenceEnhanced** | 5 | Private clone ratio, shared clonotypes |
| **RA_MotifEnrichment** | 6 | Disease-associated CDR3 motifs |

### Advanced classifiers (v2.0)

| Classifier | Function | Best for |
|------------|----------|----------|
| GLM (original) | `DiseaseClassify()` | Baseline, interpretable |
| L1/L2 regularized LR | `ClassifyRegularized()` | Feature selection |
| Random Forest | `ClassifyRandomForest()` | Heterogeneous data |
| XGBoost | `ClassifyXGBoost()` | Non-linear patterns |
| Ensemble | `ClassifyEnsemble()` | Robust predictions |
| Compare all | `CompareClassifiers()` | One-click benchmarking |

## Installation

```r
# from local source
install.packages("CDRscope", repos = NULL, type = "source")

# from GitHub
# remotes::install_github("tornado2047/CDRscope")
```

Python dependencies (for ESM-2 embedding, UMAP, deep analysis):
`torch`, `esm`, `umap-learn`, `scikit-learn`, `pandas`, `matplotlib`.

Optional R packages: `randomForest`, `glmnet`, `xgboost`, `iml`, `jsonlite`.

## Quick start

### v2.1 closed-loop pipeline (recommended)

```r
library(CDRscope)

# Full closed-loop — runs all 10 modules
result <- run_complete_analysis(
  input = "path/to/data",
  chain = "paired",           # single / paired / all
  control_dir = "path/to/controls",
  patient_dir = "path/to/patients",
  classifier = "rf",
  cv_folds = 5,
  output_dir = "my_analysis"
)

# Print summary
print(result)
# CDRscope Complete Analysis Results
#   Chain mode:      paired
#   Elapsed:         3.2 minutes
#   CV accuracy:     93.0%
#   Significance:    24 domain tests
#   Report:          my_analysis/CDRscope_Analysis_Report.html

# Open the HTML report
browseURL(result$report_path)
```

### Toggle individual modules

```r
# Skip breakthrough + validation, only core + significance
result <- run_complete_analysis(
  input = "path/to/data",
  chain = "single",
  run_breakthrough = FALSE,
  run_validation = FALSE,
  generate_report = TRUE
)

# Project onto reference map only (no classification)
result <- run_complete_analysis(
  input = "path/to/data",
  chain = "all",
  use_reference_map = TRUE,
  classifier = NULL
)
```

### v2.0 unified entry (core pipeline only)

```r
# Single chain
result <- run_CDRscope("path/to/data", chain = "single")

# TRA+TRB paired (best accuracy)
result <- run_CDRscope("path/to/data", chain = "paired",
                       classifier = "rf", cv_folds = 5)

# All chains, compare classifiers
result <- run_CDRscope("path/to/data", chain = "all",
                       classifier = "compare")
```

### Classic piped workflow (still supported)

```r
obj <- fetch_toy_data()
obj <- QCRepertoire(obj)
obj <- NormalizeRepertoire(obj)
obj <- ComputeFeatures(obj)           # 6 original modules (20 features)
obj <- ComputeFeaturesRA(obj)         # +5 enhanced modules (65 features total)
obj <- ConceptBottleneckEmbed(obj)
obj <- DiseaseClassify(obj, use_shap = TRUE)
obj <- FindMarkers(obj, level = "both")

DimPlot(obj)
SHAPPlot(obj)
```

## Pipeline architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           run_complete_analysis()           │
                    └─────────────────────┬───────────────────────┘
                                          │
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐
  │ Module 1 │→ │ Module 2 │→ │ Module 3 │→ │   Module 4    │
  │  Input   │  │ Features │  │  ESM-2   │  │ Reference Map │
  │& chains  │  │  (65)    │  │ embed    │  │  projection   │
  └──────────┘  └──────────┘  └──────────┘  └───────┬───────┘
                                                    │
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────▼───────┐
  │ Module 8 │← │ Module 7 │← │ Module 6 │← │   Module 5    │
  │Breakthru │  │  Sig.    │  │  UMAP    │  │ Classification│
  └────┬─────┘  └──────────┘  └──────────┘  └───────────────┘
       │
  ┌────▼─────┐  ┌──────────────────┐
  │ Module 9 │→ │   Module 10      │
  │Validate  │  │  HTML Report      │
  └──────────┘  └──────────────────┘
```

### Module details

**Module 7 — Domain significance**: Tests whether RA/control groups differ
significantly across interpretable property domains (CDR3 length, net charge,
hydrophobicity, aromatic content). Uses Fisher exact test with odds ratios
and 95% CIs. Outputs forest plots.

**Module 8 — Breakthrough analysis**:
- *Expansion gradient*: clonal expansion differences (Mann-Whitney U)
- *UMAP axis decoding*: Spearman correlation of UMAP axes with physicochemical properties
- *Disease scoring*: sequence-level AUC (tests population vs sequence-level signal)
- *Centroid shift*: spatial overlap of group distributions
- *Sequence network*: Levenshtein distance similarity network

**Module 9 — Biological validation**:
- *Frequency redistribution*: per-domain RA/Control frequency ratios
- *Citrullination-hydrophobicity axis*: chemical complementarity of disease CDR3s to modified autoantigens
- *HLA stratification*: V gene proxy + motif restriction analysis

## Validation: RA as a test case

RA TCR repertoire data (104 control + 168 patient samples) was used to validate
the pipeline. Key findings:

| Finding | Evidence |
|---------|----------|
| Disease signal is **population-level** | Sequence-level AUC ≈ 0.51; sample-level classifier 93.0% |
| UMAP2 encodes **hydrophobicity** | Spearman ρ = 0.72; RA shifts toward hydrophobic CDR3s |
| **Citrullination axis** | RA hydrophobic CDR3s complement citrullinated autoantigens |
| **HLA restriction** | HLA-DRB1*15:01 restricts QDFA motif TRB clones (OR ≈ 2.1) |
| **Cross-cohort V genes** | TRAV20 RA-enriched, TRBV25 RA-depleted (2 independent cohorts) |

These findings are specific to RA; the pipeline structure generalizes to any
disease with control-vs-patient repertoire data.

## Output files

After `run_complete_analysis()`, the output directory contains:

```
output_dir/
├── cv_results.csv              # Cross-validation metrics
├── feature_importance.csv     # Top features
├── projected_coords.csv       # Reference map coordinates
├── domain_significance.csv    # Fisher exact test results
├── breakthrough_summary.json  # Expansion, axis, disease score
├── validation_summary.json    # Frequency, citrullination, HLA
├── CDRscope_Analysis_Report.html  # Self-contained HTML report
└── *.png                      # UMAP and forest plot images
```

## Design notes

- **Object**: `CDRobject` is an S4 class with slots `meta`, `clones`, `features`,
  `feature_modules`, `embedding`, `reduction`, `classification`, `markers`,
  `misc` — modelled after `Seurat`.
- **Interpretability**: the concept-bottleneck layer forces every disease
  decision through named modules; SHAP attributes decisions to axes.
- **Reference map**: a trained NN approximates the UMAP transform, providing a
  fixed coordinate space so different projects land on the same map.
- **Disease-agnostic**: no RA-specific hard-coding in the pipeline; disease
  knowledge lives in configurable feature modules and validation parameters.

## Tier 2: TCR Quantization Pipeline (NEW)

Tier 2 converts variable CDR3 sequences into a fixed-dimensional "TCR
transcriptome" space, enabling cross-sample comparison analogous to RNA-seq
gene expression. This solves the fundamental AIRR-seq challenge: different
samples have entirely different CDR3 sequences and counts.

### Core idea

1. **Reference pool** — 846k unique CDR3s from RA, CMV, MS, SLE, VDJdb
2. **ESM-2 embedding** — 480-dim protein language model per sequence
3. **K-means quantization** — m=10,000 prototype centroids ("TCR genes")
4. **Sample projection** — any sample → nearest centroid → count vector (m-dim)
5. **L2 normalization** — removes sequencing depth bias
6. **Linear SVM** — optimal for sparse high-dim TCR data (AUC 0.9964)

### Saturation validation

m saturates as the reference pool grows: pool grew 20x, m grew only 3.8x
(saturation index 0.45), confirming the TCR sequence space is finite.

### Multi-layer interpretability

| Layer | Method | Key finding (RA) |
|-------|--------|-------------------|
| SVM weight ranking | Top discriminative prototypes | Disease signal is distributed (GWAS-like) |
| V/J gene enrichment | Aggregate gene usage in top prototypes | TRBV10-3 exclusively in patients |
| CDR3 motif analysis | k-mer enrichment | GYEQ, SSIA, SSIV patient-specific |
| Physicochemical | Biophysical properties | Significant charge/hydrophobicity shifts |
| Convergence | V/J gene diversity per prototype | 1.2 vs 2.6 (antigen-driven selection) |

### Tier 1 → Tier 2 integration

Tier 1 trains on per-sample labeled data → model checkpoint.
Tier 2 projects pool-only CDR3 data onto the reference panel → applies the
Tier 1 model for disease scoring. This enables analysis of samples without
individual-level information (e.g., CordBlood pooled data).

See [`inst/python/tier2/README.md`](inst/python/tier2/README.md) for full
documentation.

```bash
# Run full Tier 2 pipeline
python inst/python/tier2/run_tier2.py --all
```

## Roadmap

### v2.x (near-term)
- [x] Complete closed-loop pipeline (10 modules)
- [x] Reference map system
- [x] Domain-level significance analysis
- [x] Breakthrough analysis (expansion, axis, network)
- [x] Biological validation (frequency, citrullination, HLA)
- [x] Automated HTML report generation
- [x] **Tier 2: TCR quantization pipeline** (m=10,000, AUC 0.9964)
- [x] **Saturation analysis** (m stabilizes as pool grows)
- [x] **Multi-layer interpretability** (SVM weights, V/J genes, motifs, convergence)
- [x] **Supervised visualization** (SVM projection, PLS-DA, LDA)
- [ ] ROC/PR curves and AUC metrics
- [ ] Bootstrap confidence intervals for feature importance
- [ ] CRAN submission

### v3.0 — scTCR-seq αβ pairing (future)
Single-cell TCR sequencing data with paired TRA+TRB chains from the same T cell.
See [ROADMAP.md](ROADMAP.md) for details.

## License

MIT
