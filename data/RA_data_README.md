# RA-CDRscope-data

**Rheumatoid Arthritis (RA) T-cell receptor repertoire raw data** for the
CDRscope analysis pipeline. Stored as a private repository — this is
pseudonymised clinical immune-repertoire data and must NOT be made public
without ethics approval and participant consent.

## Data overview

| Group | Samples | Files | Raw size | Archive |
|-------|---------|-------|----------|---------|
| Control | 105 | 840 CSV | 387 MB | `RA_Control_Files.zip` (80 MB) |
| Patient (RA) | ~168 | 1342 CSV | 699 MB | `RA_Patient_Files.zip` + `.z01` (144 MB, split) |

Each sample produces 8 CSV files: four TCR chains (TRA / TRB / TRG / TRD) ×
two versions (raw `__` and rearranged `_r__`).

## File naming

```
IXTCB#####__TRB.csv      raw repertoire, TRB chain
IXTCB#####_r__TRB.csv    rearranged/filtered repertoire, TRB chain
```

`IXTCB#####` is a pseudonymised sample identifier. The mapping to real
identities is held offline and is NOT included here.

## CSV schema (AIRR-aligned)

```
junction_aa, junction, v_call, d_call, j_call, duplicate_count
```

- `junction_aa` — CDR3 amino-acid sequence
- `junction` — CDR3 nucleotide sequence
- `v_call` / `d_call` / `j_call` — V/D/J gene allele assignments
- `duplicate_count` — clone frequency (UMI/read count)

## Restoring the archives

The Patient archive is a **split zip** (GitHub's 100 MB single-file limit).

```bash
# Control (single zip)
unzip RA_Control_Files.zip

# Patient (split zip: RA_Patient_Files.zip + RA_Patient_Files.z01)
zip -s 0 RA_Patient_Files.zip --out RA_Patient_Files_combined.zip
unzip RA_Patient_Files_combined.zip
```

After extraction you get two folders:
```
RA_Control_Files/   (840 CSV)
RA_Patient_Files/   (1342 CSV)
```

## Usage with CDRscope

```r
library(CDRscope)
# point ReadRepertoire at the extracted CSV folder
# (build a sample metadata table mapping IXTCB##### -> group)
```

See https://github.com/tornado2047/CDRscope for the analysis package.

## Privacy & ethics

- Data is **pseudonymised** (no direct identifiers).
- TCR repertoires are re-identifiable biological traits — treat as sensitive.
- Patient/Control status is sensitive clinical information.
- **Keep this repository PRIVATE.** Do not transfer to public repositories
  without IRB / ethics approval and participant consent for public sharing.

## Provenance

- Source: AIDeN files (Nutstore sync)
- Format: AIRR-aligned TCR CSV (MiXCR-style output)
- Chains: TRA / TRB / TRG / TRD (seven-chain framework, four TCR chains here)
- Committed: 2026-08-13
