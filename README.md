# CDRscope v2.0

**Interpretable TCR/BCR repertoire analysis with enhanced features, multi-chain support, and deep-learning embeddings.**

`CDRscope` implements the five-layer interpretable algorithm architecture for
TCR/BCR immune repertoires. **v2.0** adds:
- **Multi-chain support** — single chain, TRA+TRB paired, or all chains joint
- **Enhanced feature engineering** — 11 modules, 65 features with RA-specific motifs
- **Advanced classifiers** — Random Forest, LASSO, XGBoost, Ensemble
- **ESM-2 embedding** — protein language model embedding + parametric UMAP

```
run_CDRscope(data_dir, chain = "paired")   # one-line unified entry

# Or the classic piped workflow:
ReadRepertoire → QCRepertoire → ComputeFeatures → ComputeFeaturesRA
              → ConceptBottleneckEmbed → DiseaseClassify → FindMarkers
```

## What's new in v2.0

### Chain selection (NEW)

The unified entry point `run_CDRscope()` accepts a `chain` parameter with three modes:

| Mode | Description | Use case |
|------|-------------|----------|
| `"single"` | Analyse the most abundant chain | Quick screening, TRB-only |
| `"paired"` | TRA+TRB joint analysis | Highest accuracy (93.0% on RA) |
| `"all"` | All chains analysed jointly | Full multi-chain repertoire |

The input data can be **single-chain** (TRA/TRB/TRG/TRD/IGH/IGK/IGL), **paired** (TRA+TRB), or **multi-chain** (up to 7 chains). CDRscope auto-detects available chains from file names.

### Enhanced feature engineering (NEW)

v2.0 adds 5 RA-specific feature modules (45 new features) on top of the original 6 concept modules:

| Module | Features | What it captures |
|--------|----------|-----------------|
| **RA_ClonalExpansion** | 11 | D50/D20, Morisita, Berger-Parker, Pielou, Renyi entropy, HEC stats |
| **RA_GeneUsage** | 8 | RA-associated V genes (TRBV25-1, etc.), J genes, V-J pairing |
| **RA_Physicochemical** | 14 | CDR3 length moments, N-terminal AA composition, charge, hydrophobicity |
| **RA_ConvergenceEnhanced** | 5 | Private clone ratio, shared clonotypes, between-group convergence |
| **RA_MotifEnrichment** | 6 | 23 RA-associated CDR3 motifs from literature (Aterido 2024, JCI, ARD) |

**Performance improvement:** Enhanced features (65 total) improve RA classification accuracy from **70.2% → 81.2%** (+11.0%) on real data.

### Advanced classifiers (NEW)

| Classifier | Function | Best for |
|------------|----------|----------|
| GLM (original) | `DiseaseClassify()` | Baseline, interpretable |
| L1-regularized LR | `ClassifyRegularized()` | Feature selection |
| Random Forest | `ClassifyRandomForest()` | Heterogeneous data, feature importance |
| XGBoost | `ClassifyXGBoost()` | Non-linear patterns |
| Ensemble | `ClassifyEnsemble()` | Robust predictions |
| Compare all | `CompareClassifiers()` | One-click benchmarking |

### ESM-2 embedding (NEW)

Protein language model (ESM-2, 480-dim) embedding of CDR3 sequences, with parametric UMAP for 2D visualization. Python scripts in `inst/python/`.

### Benchmark results (272 RA samples, 104 Control + 168 RA Patient)

| Pipeline | Features | Accuracy | F1 |
|----------|:---:|:---:|:---:|
| Original CDRscope (TRB) | 20 | 70.2% | — |
| Enhanced RF (TRB) | 65 | 81.2% | 0.831 |
| Enhanced RF (TRA) | 32 | 76.9% | 0.826 |
| **Enhanced RF (TRA+TRB)** | **71** | **93.0%** | **0.944** |

## Installation

```r
# from local source
install.packages("CDRscope", repos = NULL, type = "source")

# from GitHub
# remotes::install_github("tornado2047/CDRscope")
```

Optional packages for advanced features: `randomForest`, `glmnet`, `xgboost` (classifiers), `iml` (SHAP), `umap`.

## Quick start

### v2.0 unified entry (recommended)

```r
library(CDRscope)

# Single chain
result <- run_CDRscope("path/to/RA_data", chain = "single")

# TRA+TRB paired (best accuracy)
result <- run_CDRscope("path/to/RA_data", chain = "paired",
                        classifier = "rf", cv_folds = 5)

# All chains, compare classifiers
result <- run_CDRscope("path/to/RA_data", chain = "all",
                        classifier = "compare")

# View results
print(result$cv_results)
head(result$feature_importance)
```

### Classic piped workflow (still supported)

```r
obj <- fetch_toy_data()
obj <- QCRepertoire(obj)
obj <- NormalizeRepertoire(obj)
obj <- ComputeFeatures(obj)           # 6 original modules (20 features)
obj <- ComputeFeaturesRA(obj)         # +5 RA modules (65 features total)
obj <- ConceptBottleneckEmbed(obj)
obj <- DiseaseClassify(obj, use_shap = TRUE)
obj <- FindMarkers(obj, level = "both")

DimPlot(obj)
SHAPPlot(obj)
```

## Design notes

- **Object**: `CDRobject` is an S4 class with slots `meta`, `clones`, `features`, `feature_modules`, `embedding`, `reduction`, `classification`, `markers`, `misc` — modelled after `Seurat`.
- **Interpretability**: the concept-bottleneck layer forces every disease decision through named modules; SHAP attributes decisions to axes.
- **Markers**: statistical level (feature differences + FDR) and sequence level (motif enrichment + public disease clonotypes).
- **Multi-chain**: the `chain` column in the clones table enables automatic chain detection and filtering.

## Roadmap

### v2.x (near-term)
- [ ] ROC/PR curves and AUC metrics
- [ ] Bootstrap confidence intervals for feature importance
- [ ] UMAP visualization with chain labeling
- [ ] CRAN submission

### v3.0 — scTCR-seq αβ pairing (future)
Single-cell TCR sequencing data with paired TRA+TRB chains from the same T cell.
- **αβ pairing analysis**: joint TRA CDR3 + TRB CDR3 embedding per cell
- **Antigen recognition modules**: map complete TCRs (α+β) to 2D space
- **scTCR-seq support**: `pair_id` column in clones table
- **Clonotype-level analysis**: paired-chain convergence, selection, and motif co-occurrence

See [ROADMAP.md](ROADMAP.md) for details.

## License

MIT