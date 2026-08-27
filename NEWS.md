# CDRscope NEWS

## v2.4.0 (2026-08-27)

### Canonical leakage-aware analysis core

- AIRR-compatible validation and quantitative sample QC.
- Donor/technical-replicate grouping and grouped nested cross-validation.
- Elastic-net model selection inside inner folds, out-of-fold probability metrics,
  donor-bootstrap confidence intervals and Brier score.
- Hard/soft prototype assignment, compositional normalization and multiscale spectra.
- Healthy-reference-only anomaly fitting with held-out calibration and conformal p-values.
- Versioned panel manifests, independence checks, embedding cache and run provenance.
- Canonical CLI/config plus Python regression tests; see `docs/SYSTEM_OPTIMIZATION.md`.

Historical v2.1-v2.3 scripts remain available for reproducibility but their random
sample-level CV results should be treated as exploratory until rerun through v2.4.


## v2.3.0 (2026-08-24)

### Reference Coordinate System (RCS) — the final unsupervised piece

Completes the unsupervised route of Tier 2: a **fixed, invariant sample
coordinate system** built once from CordBlood TRA, plus deviation
magnitude/direction quantification and visualization.

- **`reference_coordinate_system.py`** — fixed coordinate space built from
  CordBlood TRA (1.32M unique CDR3s → m=10,000 prototypes). Frozen PCA (50 PCs)
  + UMAP transformers; every new sample maps to a **unique, stable position**
  (`same sample → same coordinates`; reference space never changes).
- **`deviation_direction_viz.py`** — 7-view visualization of deviation
  **magnitude** ("how far from the healthy origin?") and **direction**
  ("which prototypes / PC axes drive it?"): PCA magnitude scatter, polar plot,
  radar chart, top contributing prototypes, direction heatmap, direction
  clusters, arrow field. Self-contained HTML + PDF report.
- **`cdrscope_unified.py`** — single entry point with two routes:
  - Route A (supervised, labels provided): Linear SVM 5-fold CV — AUC 0.9593,
    differential prototypes, ROC analysis
  - Route B (unsupervised, always runs): deviation magnitude scoring
    (AUC 0.73), anomaly ensemble OCSVM+LOF+JS (AUC up to 0.95), diversity
    analysis, composite stratification (High=54 / Moderate=82 / Normal=409 on RA)
- **`longitudinal_validation_v2.py`** — robustness proof for the unsupervised
  route: 46 samples from 25 donors (SLE GSE254176, Zenodo MDA1/HD1-3, RA
  controls). Same-donor pairs are significantly closer than cross-donor pairs —
  cosine 1.39× (p=3.9e-15), Euclidean 1.21× (p=3.9e-15), JS divergence 1.08×
  (p=7.4e-7), RCS space 1.16× (p=1.5e-3).
- **`unsupervised_tra_pipeline.py`**, **`unsupervised_enhanced.py`**,
  **`unsupervised_methods_4_6.py`** — full unsupervised suite: KMeans/GMM
  clustering, diversity (Shannon/Simpson/Pielou/Chao1), Isolation Forest,
  One-Class SVM, LOF, JS divergence / Aitchison distance, multi-scale
  prototype grouping, repertoire similarity networks.
- **`cordblood_reference_panel.py`**, **`cordblood_tra_full11.py`**,
  **`cordblood_tra_validate.py`** — CordBlood TRA chain-specific reference
  panel (first of seven planned chains: TRA/TRB/TRG/TRD/IGH/IGL/IGK).
- **`multidisease_tra_test.py`**, **`cross_disease_tier2.py`** — cross-disease
  validation of the CB TRA panel: SLE AUC 0.9706, MS AUC 0.7500, RA-TRA 0.7059.

**Key finding**: RA disease signal in TRA is **distributed** across many
prototypes (GWAS-like); unsupervised clustering cannot separate disease from
control (ARI −0.0076). Deviation-from-reference scoring is the correct
unsupervised screening tool; supervised SVM remains strongest when labels exist.

## v2.1.1 (2026-08-17)

### ROC/AUC Metrics

- **`ROCCurve()`** — new exported function for plotting ROC curves from CV results
- `.compute_roc()` / `.compute_pr()` / `.compute_all_metrics()` — internal helpers for
  ROC curve, PR curve, AUC, sensitivity, specificity, Youden's J
- Cross-validation now reports **AUC-ROC** and **AUC-PR** alongside accuracy and F1
- Pipeline saves `cv_details.csv` with per-fold AUC, and `roc_curve.png` plot
- HTML report generator includes ROC curve image and AUC metrics

### Emerson CMV Benchmark Validation

Validated the pipeline on the **Emerson 2017 CMV serostatus dataset** (389 samples:
175 CMV+, 214 CMV-), comparing against published results from Katayama 2022 and Aker 2022:

| Method | AUC-ROC | Source |
|--------|---------|--------|
| Burden test | 0.490 | Katayama 2022 |
| DeepRC | 0.480 | Katayama 2022 |
| k-mer/SVM | 0.510 | Katayama 2022 |
| k-mer/MIL | 0.590 | Katayama 2022 |
| MotifBoost | 0.710 | Katayama 2022 |
| Count (best) | 0.750 | Aker 2022 |
| **CDRscope v2.1** | **0.788** | This work |

CDRscope achieves state-of-the-art performance, surpassing all published methods.
Top discriminative features: clonal diversity (Simpson, Pielou, Renyi entropy),
clonal expansion (D50, top clone frequencies), and convergence metrics — consistent
with CMV biology where CMV-specific TCR clonotypes undergo significant expansion.

## v2.1.0 (2026-08-16)

### Complete Closed-Loop Analysis Pipeline

- **`run_complete_analysis()`** — unified 10-module orchestrator:
  1. Input & chain selection (single/paired/all)
  2. Feature engineering (65 enhanced features)
  3. ESM-2 embedding (480-dim)
  4. Reference Map projection (fixed UMAP space)
  5. Classification with cross-validation
  6. UMAP visualization with property domain coloring
  7. Domain-level significance analysis (Fisher exact, OR, forest plots)
  8. Breakthrough analysis (expansion gradient, UMAP axis decoding, disease scoring, sequence network)
  9. Biological validation (frequency redistribution, citrullination-hydrophobicity axis, HLA stratification)
  10. Automated HTML report generation

### Reference Map System

- **Fixed UMAP space** via trained NN (655 KB weights) — cross-project comparison enabled
- `ProjectToReferenceMap()` — project new sequences onto existing map
- `ReferenceMapPlot()` — ggplot visualization with new data overlay
- Reference map package bundled in `inst/reference_map/`

### Analysis Modules (Python)

- `complete_analysis.py` — significance + breakthrough + validation
- `generate_report.py` — auto-generate HTML report from all outputs

### Key Findings (RA validation)

- Disease signal is **population-level** (frequency redistribution), not sequence-level (AUC ≈ 0.51)
- UMAP2 = hydrophobicity axis; RA shifts toward hydrophobic CDR3s
- Citrullination increases peptide hydrophobicity → complementary to RA hydrophobic CDR3s
- HLA-DRB1*15:01 restricts QDFA motif TRB clones (OR ≈ 2.1, consistent with Aterido 2024)
- TRAV20 RA-enriched + TRBV25 RA-depleted — consistent across two independent cohorts

## v2.0.0 (2026-08-14)

### New features

- **Unified entry point** `run_CDRscope()` with chain selection:
  - `chain = "single"` — analyse the most abundant chain
  - `chain = "paired"` — TRA+TRB joint analysis
  - `chain = "all"` — all available chains analysed jointly
  - Auto-detects chain types from file names (`_TRA.csv`, `_TRB.csv`, etc.)
  - Supports up to 7 chains: TRA, TRB, TRG, TRD, IGH, IGK, IGL
- **Enhanced feature engineering** (5 new RA-specific modules):
  - `ComputeRA_ClonalExpansion()` — 11 features (D50/D20, Morisita, Berger-Parker, Pielou, Renyi)
  - `ComputeRA_GeneUsage()` — 8 features (RA-associated V/J genes, V-J pairing)
  - `ComputeRA_Physicochemical()` — 14 features (CDR3 length moments, AA composition, charge, hydrophobicity)
  - `ComputeRA_ConvergenceEnhanced()` — 5 features (private clone ratio, shared clonotypes)
  - `ComputeRA_MotifEnrichment()` — 6 features (23 RA-associated CDR3 motifs from literature)
  - Master function `ComputeFeaturesRA()` assembles all 65 features
- **Advanced classifiers**:
  - `ClassifyRandomForest()` — Random Forest with feature importance
  - `ClassifyRegularized()` — L1/L2/ElasticNet regularized logistic regression
  - `ClassifyXGBoost()` — XGBoost gradient boosting
  - `ClassifyEnsemble()` — Soft/hard voting ensemble
  - `CompareClassifiers()` — One-click comparison of all classifiers
- **ESM-2 embedding**: Protein language model embedding of CDR3 sequences (480-dim)
  - Python scripts in `inst/python/` for embedding extraction and parametric UMAP
  - `esm_embed.py` — extract ESM-2 embeddings
  - `parametric_umap.py` — parametric UMAP dimensionality reduction
  - `joint_umap.py` — joint TRA+TRB UMAP

### Performance improvements

- RA classification accuracy improved from **70.2% → 81.2%** (TRB alone, +11.0%)
- TRA+TRB joint analysis achieves **93.0% accuracy** (F1 = 0.944, +22.8% vs original)
- 5-fold cross-validation with accuracy and F1 reporting

### Documentation

- New `README.md` with v2.0 features, quick start, and benchmark results
- `ROADMAP.md` with v3.0 scTCR-seq αβ pairing plan
- `NEWS.md` (this file)

## v0.1.0 (2026-08-12)

- Initial release
- Six concept bottleneck architecture
- Seurat-style CDRobject S4 class
- Offline toy data and online VDJdb fetch
- SHAP-based interpretability
- Statistical and sequence-level biomarker discovery