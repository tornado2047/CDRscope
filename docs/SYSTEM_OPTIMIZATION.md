# CDRscope v2.4 system-level optimization

This directory introduces one **canonical, leakage-aware pipeline** alongside the
legacy v2.3 research scripts. It does not silently rewrite historical analyses;
instead, new benchmarks should use these modules and clearly label old results as
exploratory.

## What is implemented

- AIRR-like schema normalization, sequence validation and per-sample QC.
- Conservative donor inference that groups `_r`, `_r2`, `-rep1`, and technical replicates.
- Hard or top-k soft prototype assignment.
- L2, relative, Hellinger, CLR and TF-IDF spectrum transforms.
- Multi-resolution prototype aggregation.
- Donor-level **nested** `StratifiedGroupKFold`; scaling and hyperparameter tuning occur
  inside the inner loop.
- Elastic-net probabilistic classifier, out-of-fold predictions, Brier score,
  AUC-ROC/AUC-PR and donor-bootstrap confidence intervals.
- Healthy-reference-only OCSVM/LOF ensemble with a held-out healthy calibration set,
  reference percentiles and finite-sample conformal p-values.
- Versioned panel manifests with SHA-256 hashes and explicit independence status.
- Content-addressed ESM embedding cache.
- Reproducible run manifests containing config, input hashes, Git commit and environment.

## Canonical usage

```bash
python inst/python/canonical_pipeline.py \
  --matrix spectra_counts.npy \
  --metadata metadata.csv \
  --panel-manifest panel_manifest.json \
  --training-sample-ids panel_training_samples.txt \
  --reference-matrix independent_healthy_counts.npy \
  --output results/canonical_run \
  --normalization hellinger \
  --outer-folds 5 --inner-folds 4 --repeats 3 --bootstrap 1000
```

Required metadata columns are `sample_id,label`; `donor_id` is strongly recommended.
When it is absent, technical-replicate suffixes are grouped conservatively. A benchmark
should not be considered confirmatory unless panel independence reports `independent`.

## Evaluation contract

1. The reference panel must be frozen before target-cohort evaluation.
2. Evaluation samples and their sequences must not train that panel.
3. All samples/time points/replicates from one donor stay in one outer fold.
4. Feature filtering, scaling, model selection and threshold selection use training data only.
5. Anomaly models use an independent healthy reference; patient labels are evaluation only.
6. Report out-of-fold predictions and uncertainty, not training accuracy.
7. External leave-one-cohort-out or locked-model blind tests remain required for clinical claims.

## Panel manifest example

```json
{
  "panel_id": "cordblood-tra",
  "version": "1.0.0",
  "chain": "TRA",
  "embedding_model": "esm2_t12_35M_UR50D",
  "embedding_layer": 12,
  "pooling": "mean_residue",
  "n_prototypes": 10000,
  "training_sources": ["independent cord-blood cohort"],
  "training_sample_hash": "...",
  "training_sequence_hash": "...",
  "centroid_sha256": "...",
  "applicability": {"minimum_unique_cdr3": 300, "assay": "AIRR-seq"}
}
```

## Still data-dependent

The code enables but cannot replace the following experiments: chain-specific depth
rarefaction, external cohort validation, platform/batch audits, model comparisons,
antigen-retrieval ablations and wet-lab validation. These require approved source data
and must be run before the corresponding scientific claims are upgraded.
