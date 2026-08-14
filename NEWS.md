# CDRscope NEWS

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