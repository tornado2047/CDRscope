# CDRscope Roadmap

## v2.1 (current, 2026-08)

### Completed in v2.1

- [x] **Complete closed-loop pipeline** — `run_complete_analysis()` orchestrates 10 modules end-to-end
- [x] **Reference map system** — fixed UMAP space via trained NN, cross-project comparison
- [x] **Domain-level significance analysis** — Fisher exact, OR, forest plots
- [x] **Breakthrough analysis** — expansion gradient, UMAP axis decoding, disease scoring, sequence network
- [x] **Biological validation** — frequency redistribution, citrullination-hydrophobicity axis, HLA stratification
- [x] **Automated HTML report generation** — self-contained report from all analysis outputs

### Completed in v2.2 (Tier 2)

- [x] **TCR reference quantization** — ESM-2 + k-means → m=10,000 prototype "TCR transcriptome"
- [x] **Saturation analysis** — m stabilizes as pool grows (saturation index 0.45)
- [x] **Unified pipeline** — L2 norm + Linear SVM, AUC 0.9964 on RA-TRB
- [x] **Supervised visualization** — SVM 1D projection, SVM+PC 2D, PLS-DA, LDA, supervised UMAP
- [x] **Multi-layer interpretability** — SVM weight ranking, V/J gene enrichment, CDR3 motif analysis, physicochemical profiling, convergence
- [x] **FindMarkers** — Wilcoxon rank-sum, logFC, AUC per prototype, volcano plot, heatmap
- [x] **Cross-disease benchmark** — RA, CMV, SLE, VDJdb multi-disease testing

## v2.x (near-term, 2026 Q3-Q4)

- [ ] **ROC/PR curves** — AUC metrics for classifier evaluation
- [ ] **Bootstrap confidence intervals** — for feature importance
- [ ] **UMAP visualization with chain labeling** — color by chain type in 2D maps
- [ ] **CDR3 length distribution plots** — per-chain per-group comparison
- [ ] **CRAN submission** — after API stabilization
- [ ] **Vignettes** — multi-chain workflows, reference map tutorial, closed-loop quick start
- [ ] **Disease-agnostic validation** — test pipeline on additional disease datasets (MS, T1D, COVID-19)

## v3.0 — scTCR-seq αβ Pairing Analysis (2026 Q4 - 2027)

### Motivation

In single-cell TCR sequencing (scTCR-seq), each T cell has a **paired TRA + TRB** CDR3,
which together form the complete antigen recognition module. Current bulk TCR-seq
analysis treats TRA and TRB independently (or merged at the sample level), losing
the pairing information that determines antigen specificity.

### Proposed features

#### 1. Paired-chain data support
- New `pair_id` column in clones table linking TRA and TRB from the same cell
- `ReadRepertoire()` detects `pair_id` and enables paired mode
- `fetch_toy_data_sc()` for generating toy scTCR-seq data

#### 2. αβ-paired embedding
- Joint TRA+TRB CDR3 embedding: `concat(TRA_embed, TRB_embed)` → 960-dim
- Paired parametric UMAP: each point = one T cell (α+β)
- Antigen recognition module visualization: each T cell's complete TCR in 2D

#### 3. Paired-chain features
- `ComputeRA_PairingEnhanced()` — αβ pairing statistics
  - TRA-TRB V gene co-occurrence
  - CDR3 length correlation (α vs β)
  - Charge complementarity (α vs β CDR3)
  - Motif co-occurrence across chains
  - Pairing diversity (Shannon entropy of pairings)

#### 4. Clonotype-level analysis
- Clone tracking across samples with paired chains
- Convergence analysis at the clonotype (α+β) level
- Selection imprint on paired repertoires

#### 5. Single-cell visualization
- UMAP colored by paired chain properties
- V-J gene co-occurrence heatmaps
- αβ CDR3 length scatter plots
- Joint motif logo for paired chains

### Data requirements

```
scTCR-seq clone table:
  sample_id | pair_id | cdr3_aa | v_gene | j_gene | chain | count
  ----------|---------|---------|--------|--------|-------|------
  S01       | T1      | CASS... | TRBV25 | TRBJ2  | TRB   | 5
  S01       | T1      | CAVR... | TRAV12 | TRAJ23 | TRA   | 5
  S01       | T2      | CSAR... | TRBV7  | TRBJ1  | TRB   | 3
  S01       | T2      | CALM... | TRAV3  | TRAJ31 | TRA   | 3
```

### Dependencies
- Existing CDRscope v2.1 infrastructure (CDRobject, feature modules, classifiers, reference map)
- ESM-2 embedding pipeline (extended to paired mode)
- scTCR-seq data availability from public datasets (10x Genomics, SMART-seq)
