# CDRscope Roadmap

## v2.x (near-term, 2026 Q3-Q4)

- [ ] **ROC/PR curves** — AUC metrics for classifier evaluation
- [ ] **Bootstrap confidence intervals** — for feature importance
- [ ] **UMAP visualization with chain labeling** — color by chain type in 2D maps
- [ ] **CDR3 length distribution plots** — per-chain per-group comparison
- [ ] **CRAN submission** — after API stabilization
- [ ] **Improved documentation** — vignettes for multi-chain workflows

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
- Existing CDRscope v2.0 infrastructure (CDRobject, feature modules, classifiers)
- ESM-2 embedding pipeline (extended to paired mode)
- scTCR-seq data availability from public datasets (10x Genomics, SMART-seq)