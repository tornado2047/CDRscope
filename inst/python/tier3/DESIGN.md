# CDRscope v3.0 Technical Design — Multi-Layer Adaptive Spectrum

> From single-layer frequency spectrum to multi-layer adaptive spectrum:
> T/B separation, four-layer representation, temporal dynamics, selection imprints.

## Architecture Overview

```
                        CDRscope v3.0 Multi-Layer Spectrum
                       ┌─────────────────────────────────────────┐
                       │                                         │
  CDR3 sequences ─────►│  L1: Prototype Spectrum (10,000-d)      │
  V/J genes      ─────►│  L2: Motif Spectrum (500-d)             │
  CDR3 length    ─────►│  L3: Structure Profile (50-d)           │
  clone counts   ─────►│  L4: Macro Indices (10-d)               │
  N-insertions   ─────►│  S1: N-Insertion Axis (10-d)            │
  AA properties  ─────►│  S2: Physicochemical Axis (10-d)       │
  V-J usage      ─────►│  S3: V-J Bias Axis (20-d)              │
  germline dist  ─────►│  S4: Germline Distance Axis (5-d)      │
  HLA type (opt) ─────►│  S5: MHC Covariate (categorical)        │
                       │                                         │
  Time series    ─────►│  T1: Δ-Frequency Spectrum (10,000-d)    │
                       │  T2: Δ-Diversity Vector (5-d)           │
                       │  T3: Velocity in RCS (2-d)              │
                       │                                         │
  SHM lineage    ─────►│  B1: Lineage Tree Topology (15-d)  [BCR]│
  Mutation spec  ─────►│  B2: Mutation Spectrum (20-d)     [BCR]│
                       │                                         │
                       │  Total: ~10,640-d (TCR) / ~10,655-d (B)│
                       └─────────────────────────────────────────┘
```

## Module Map

### Direction 5: Selection Imprint Axes (`selection_imprints.py`)
Independent axes, orthogonal to prototype spectrum.

| Axis | Dim | Source | Cross-individual alignment |
|------|-----|--------|-----------------------------|
| S1: N-insertion | 10 | CDR3 vs germline alignment | Percentile curves |
| S2: Physicochemical | 10 | AA properties of CDR3 | Fixed property bins |
| S3: V-J bias | 20 | V/J gene usage frequency | PCA of V-J matrix |
| S4: Germline distance | 5 | Edit distance to germline | Fixed distance bins |
| S5: MHC covariate | cat | HLA typing (optional) | Stratification variable |

### Direction 3: Four-Layer Representation
| Layer | Module | Dim | Status |
|-------|--------|-----|--------|
| L1: Prototype spectrum | `tcr_reference_quantization.py` (existing) | 10,000 | ✅ Done |
| L2: Motif spectrum | `motif_layer.py` (new) | 500 | ❌ Build |
| L3: Structure profile | `structure_layer.py` (new) | 50 | ❌ Build |
| L4: Macro indices | `macro_indices.py` (new) | 10 | ❌ Build |

### Direction 4: Temporal Dynamics (`temporal_dynamics.py`)
| Component | Dim | Signal type |
|-----------|-----|------------|
| T1: Δ-frequency spectrum | 10,000 | Fast (clone expansion/contraction) |
| T2: Δ-diversity vector | 5 | Slow (diversity drift) |
| T3: Velocity in RCS | 2 | Position change rate |
| Sampling QC | — | Min interval, noise floor |

### Direction 1: T/B Separation (`track_separator.py`)
- TCR track: L1 (frequency) + S-axes + T-components
- BCR track: B1 (lineage) + B2 (mutation) + S-axes
- Chain-type-specific CDR3 length/charge parameters
- MHC dependency axis only for TCR

### Direction 2: BCR Lineage Trees (`bcr_lineage.py`)
| Component | Dim | Content |
|-----------|-----|---------|
| B1: Tree topology | 15 | branching ratio, trunk length, depth, trunk-to-tip |
| B2: Mutation spectrum | 20 | position mutation rate, R/S ratio, hotspot enrichment |

## Implementation Order

| Phase | Module | Priority | Dependencies |
|-------|--------|----------|--------------|
| 1 | `selection_imprints.py` | P1 | None (standalone) |
| 1 | `macro_indices.py` | P1 | None (standalone) |
| 2 | `motif_layer.py` | P2 | Reference pool for motif dictionary |
| 2 | `temporal_dynamics.py` | P2 | L1 spectrum + RCS |
| 3 | `track_separator.py` | P2 | Phase 1+2 modules |
| 4 | `bcr_lineage.py` | P0 (concept) | Phase 3 |
| 5 | `spectrum_assembler.py` | — | All above |
