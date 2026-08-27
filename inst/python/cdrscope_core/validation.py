"""Leakage-aware grouped nested cross-validation and calibrated metrics."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Sequence[float]]
    predictions: np.ndarray
    fold_ids: np.ndarray
    best_params: list
    leakage_checks: Dict[str, object]

    def to_dict(self):
        out = asdict(self)
        out["predictions"] = self.predictions.tolist()
        out["fold_ids"] = self.fold_ids.tolist()
        return out


def _metrics(y, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y, score)),
        "auc_pr": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def _group_bootstrap_ci(y, score, groups, metric, n_boot, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        if len(np.unique(y[idx])) != 2:
            continue
        if metric == "auc_roc":
            values.append(roc_auc_score(y[idx], score[idx]))
        elif metric == "auc_pr":
            values.append(average_precision_score(y[idx], score[idx]))
        elif metric == "brier":
            values.append(brier_score_loss(y[idx], score[idx]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else [float("nan")] * 2


class NestedGroupEvaluator:
    """Nested CV where every donor/replicate remains in exactly one fold.

    Feature scaling and C/l1_ratio selection are fit inside the inner loop.
    The default elastic-net logistic model emits probabilities without a
    post-hoc calibration leak and is suitable when p >> n.
    """
    def __init__(self, outer_splits=5, inner_splits=4, repeats=1, random_state=42,
                 n_bootstrap=500, n_jobs=1, c_grid=(0.01, 0.1, 1.0, 10.0),
                 l1_ratios=(0.0, 0.5, 1.0)):
        self.outer_splits = int(outer_splits)
        self.inner_splits = int(inner_splits)
        self.repeats = int(repeats)
        self.random_state = int(random_state)
        self.n_bootstrap = int(n_bootstrap)
        self.n_jobs = int(n_jobs)
        self.c_grid = tuple(c_grid)
        self.l1_ratios = tuple(l1_ratios)

    def _estimator(self):
        return Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=5000,
                class_weight="balanced", random_state=self.random_state,
            )),
        ])

    def evaluate(self, X, y, groups, sample_ids: Optional[Sequence[str]] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups).astype(str)
        if X.ndim != 2 or len(y) != len(X) or len(groups) != len(X):
            raise ValueError("X, y and groups are not aligned")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("binary labels encoded as 0/1 are required")
        # A donor carrying conflicting labels indicates metadata corruption.
        conflicts = [g for g in np.unique(groups) if len(np.unique(y[groups == g])) > 1]
        if conflicts:
            raise ValueError(f"Groups with conflicting labels: {conflicts[:5]}")

        score_sum = np.zeros(len(y), dtype=float)
        seen = np.zeros(len(y), dtype=int)
        fold_ids = np.full((self.repeats, len(y)), -1, dtype=int)
        best_params = []
        overlaps = []
        for rep in range(self.repeats):
            seed = self.random_state + rep
            outer = StratifiedGroupKFold(self.outer_splits, shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(outer.split(X, y, groups)):
                overlap = set(groups[train]).intersection(groups[test])
                overlaps.append(len(overlap))
                if overlap:
                    raise RuntimeError("donor leakage detected in outer split")
                n_inner = min(self.inner_splits, len(np.unique(groups[train])))
                inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed + 1000 + fold)
                search = GridSearchCV(
                    self._estimator(),
                    {"model__C": self.c_grid, "model__l1_ratio": self.l1_ratios},
                    scoring="roc_auc", cv=inner, n_jobs=self.n_jobs, refit=True,
                    error_score="raise",
                )
                search.fit(X[train], y[train], groups=groups[train])
                score_sum[test] += search.predict_proba(X[test])[:, 1]
                seen[test] += 1
                fold_ids[rep, test] = fold
                best_params.append({"repeat": rep, "fold": fold, **search.best_params_})
        if np.any(seen != self.repeats):
            raise RuntimeError("outer CV did not predict each sample exactly once per repeat")
        scores = score_sum / seen
        metrics = _metrics(y, scores)
        cis = {m: _group_bootstrap_ci(y, scores, groups, m, self.n_bootstrap,
                                      self.random_state + 9000)
               for m in ("auc_roc", "auc_pr", "brier")}
        checks = {
            "group_overlap_max": int(max(overlaps, default=0)),
            "n_groups": int(len(np.unique(groups))),
            "repeats": self.repeats,
            "all_samples_out_of_fold": bool(np.all(seen == self.repeats)),
        }
        return EvaluationResult(metrics, cis, scores, fold_ids, best_params, checks)
