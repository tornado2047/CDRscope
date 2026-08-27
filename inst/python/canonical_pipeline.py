#!/usr/bin/env python3
"""Canonical system-level CDRscope evaluation entry point.

This command intentionally consumes a frozen spectrum/count matrix. Reference
panel construction is a separate, versioned operation so evaluation samples
cannot silently enter the panel.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cdrscope_core.io_qc import infer_donor_groups
from cdrscope_core.spectra import transform_counts
from cdrscope_core.validation import NestedGroupEvaluator
from cdrscope_core.anomaly import HealthyReferenceDetector
from cdrscope_core.panel import PanelManifest, verify_panel_independence
from cdrscope_core.provenance import RunTracker


def main():
    p = argparse.ArgumentParser(description="Leakage-aware CDRscope benchmark")
    p.add_argument("--matrix", required=True, help=".npy sample x prototype count matrix")
    p.add_argument("--metadata", required=True, help="CSV with sample_id,label and donor_id")
    p.add_argument("--panel-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--normalization", default="hellinger", choices=["l2", "relative", "hellinger", "clr", "tfidf"])
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--reference-matrix", help="Independent healthy-only .npy matrix for anomaly scoring")
    p.add_argument("--training-sample-ids", help="Optional text file proving panel/evaluation independence")
    args = p.parse_args()

    X_raw = np.load(args.matrix)
    meta = pd.read_csv(args.metadata)
    required = {"sample_id", "label"}
    if not required.issubset(meta):
        raise ValueError(f"metadata requires columns {sorted(required)}")
    if len(meta) != len(X_raw):
        raise ValueError("metadata row count differs from matrix")
    groups = infer_donor_groups(meta.sample_id, meta.donor_id if "donor_id" in meta else None)
    y = pd.to_numeric(meta.label, errors="raise").astype(int).to_numpy()
    X = transform_counts(X_raw, args.normalization)

    manifest = PanelManifest.load(args.panel_manifest)
    training_ids = None
    if args.training_sample_ids:
        training_ids = Path(args.training_sample_ids).read_text().splitlines()
    independence = verify_panel_independence(manifest, meta.sample_id, training_ids)
    if independence["status"] != "independent":
        print("WARNING: panel independence is unverified; provide --training-sample-ids", file=sys.stderr)

    config = vars(args).copy()
    tracker = RunTracker(args.output, config, repo_dir=HERE.parents[2])
    tracker.start([args.matrix, args.metadata, args.panel_manifest])
    evaluator = NestedGroupEvaluator(
        outer_splits=args.outer_folds, inner_splits=args.inner_folds,
        repeats=args.repeats, n_bootstrap=args.bootstrap,
    )
    result = evaluator.evaluate(X, y, groups, meta.sample_id)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": meta.sample_id, "donor_group": groups, "label": y,
        "oof_probability": result.predictions,
    }).to_csv(out / "out_of_fold_predictions.csv", index=False)
    payload = result.to_dict()
    payload["panel_independence"] = independence

    if args.reference_matrix:
        X_ref = transform_counts(np.load(args.reference_matrix), args.normalization)
        detector = HealthyReferenceDetector().fit(X_ref)
        anomaly = detector.score_samples(X)
        ci = detector.bootstrap_scores(X)
        pd.DataFrame({"sample_id": meta.sample_id, **anomaly,
                      "score_ci_low": ci[:, 0], "score_ci_high": ci[:, 1]}).to_csv(
                          out / "anomaly_scores.csv", index=False)
        payload["anomaly_reference_n"] = detector.n_reference_
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    tracker.finish({"metrics_file": "metrics.json", "metrics": result.metrics})
    print(json.dumps(result.metrics, indent=2))

if __name__ == "__main__":
    main()
