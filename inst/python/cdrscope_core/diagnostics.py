"""Depth calibration, negative controls, FDR and cohort-level robustness diagnostics."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.clip(q, 0, 1)
    return out


def rarefaction_stability(counts, depths=(100, 300, 1000, 3000, 10000), repeats=50,
                          normalization="l2", random_state=42):
    """Estimate per-depth spectrum drift against each sample's full profile."""
    from .spectra import transform_counts
    X = np.asarray(counts, dtype=float)
    if X.ndim != 2 or np.any(X < 0):
        raise ValueError("counts must be a non-negative 2D matrix")
    rng = np.random.default_rng(random_state)
    full = transform_counts(X, normalization)
    result = []
    for depth in depths:
        similarities = []
        eligible = np.flatnonzero(X.sum(axis=1) >= depth)
        for i in eligible:
            probs = X[i] / X[i].sum()
            for _ in range(repeats):
                sampled = rng.multinomial(int(depth), probs)[None, :]
                z = transform_counts(sampled, normalization)[0]
                denom = np.linalg.norm(z) * np.linalg.norm(full[i])
                similarities.append(float(np.dot(z, full[i]) / denom) if denom else 0.0)
        result.append({
            "depth": int(depth), "n_eligible": int(len(eligible)),
            "median_cosine_similarity": float(np.median(similarities)) if similarities else float("nan"),
            "p05_cosine_similarity": float(np.quantile(similarities, 0.05)) if similarities else float("nan"),
        })
    return result


def label_permutation_test(evaluator, X, y, groups, n_permutations=100, random_state=42):
    """Group-level label permutation negative control using the canonical evaluator."""
    y = np.asarray(y); groups = np.asarray(groups)
    observed = evaluator.evaluate(X, y, groups).metrics["auc_roc"]
    rng = np.random.default_rng(random_state)
    unique = np.unique(groups)
    group_label = {g: y[np.flatnonzero(groups == g)[0]] for g in unique}
    null = []
    labels = np.array([group_label[g] for g in unique])
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        mapping = dict(zip(unique, shuffled))
        yp = np.array([mapping[g] for g in groups])
        null.append(evaluator.evaluate(X, yp, groups).metrics["auc_roc"])
    p = (1 + np.sum(np.asarray(null) >= observed)) / (n_permutations + 1)
    return {"observed_auc": float(observed), "null_auc": null, "permutation_p": float(p)}


def leave_one_cohort_out(X, y, cohorts, C=1.0, random_state=42):
    """Locked linear baseline across cohorts; no target cohort enters fitting."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int); cohorts = np.asarray(cohorts)
    rows = []
    for cohort in np.unique(cohorts):
        train, test = cohorts != cohort, cohorts == cohort
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            rows.append({"cohort": str(cohort), "auc_roc": float("nan"), "n_test": int(test.sum())})
            continue
        model = LogisticRegression(C=C, penalty="l2", class_weight="balanced",
                                   max_iter=5000, random_state=random_state)
        model.fit(X[train], y[train])
        score = model.predict_proba(X[test])[:, 1]
        rows.append({"cohort": str(cohort), "auc_roc": float(roc_auc_score(y[test], score)),
                     "n_test": int(test.sum())})
    return rows
