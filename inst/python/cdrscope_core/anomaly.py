"""Healthy-only anomaly detection with held-out calibration and conformal p-values."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


@dataclass
class HealthyReferenceDetector:
    n_components: int = 30
    nu: float = 0.1
    n_neighbors: int = 20
    random_state: int = 42
    calibration_fraction: float = 0.25

    def fit(self, X_reference: np.ndarray):
        X = np.asarray(X_reference, dtype=float)
        if X.ndim != 2 or len(X) < 12:
            raise ValueError("at least 12 independent healthy reference samples are required")
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(X))
        n_cal = max(3, int(round(len(X) * self.calibration_fraction)))
        cal, train = order[:n_cal], order[n_cal:]
        n_pc = min(self.n_components, len(train) - 1, X.shape[1])
        self.pca_ = PCA(n_components=n_pc, random_state=self.random_state).fit(X[train])
        z_train = self.pca_.transform(X[train])
        z_cal = self.pca_.transform(X[cal])
        self.center_ = np.median(z_train, axis=0)
        self.scale_ = np.median(np.abs(z_train - self.center_), axis=0) * 1.4826 + 1e-8
        self.ocsvm_ = OneClassSVM(kernel="rbf", gamma="scale", nu=self.nu).fit(z_train)
        neighbors = min(self.n_neighbors, max(2, len(train) - 1))
        self.lof_ = LocalOutlierFactor(n_neighbors=neighbors, novelty=True).fit(z_train)
        self.reference_profile_ = np.mean(X[train], axis=0)
        train_components = self._components(z_train)
        self.component_center_ = np.median(train_components, axis=0)
        self.component_scale_ = (
            np.median(np.abs(train_components - self.component_center_), axis=0) * 1.4826 + 1e-8
        )
        self.calibration_scores_ = self._raw_score(z_cal)
        self.low_threshold_ = float(np.quantile(self.calibration_scores_, 0.95))
        self.high_threshold_ = float(np.quantile(self.calibration_scores_, 0.99))
        self.n_reference_ = len(X)
        return self

    def _components(self, z):
        robust_distance = np.sqrt(np.mean(((z - self.center_) / self.scale_) ** 2, axis=1))
        oc = -self.ocsvm_.score_samples(z)
        lof = -self.lof_.score_samples(z)
        return np.column_stack([robust_distance, oc, lof])

    def _raw_score(self, z):
        components = self._components(z)
        standardized = (components - self.component_center_) / self.component_scale_
        return standardized.mean(axis=1)

    def score_samples(self, X):
        if not hasattr(self, "pca_"):
            raise RuntimeError("fit must be called before score_samples")
        z = self.pca_.transform(np.asarray(X, dtype=float))
        score = self._raw_score(z)
        cal = self.calibration_scores_
        # Higher anomaly score -> smaller finite-sample conformal p-value.
        p = (1.0 + (cal[:, None] >= score[None, :]).sum(axis=0)) / (len(cal) + 1.0)
        percentile = 100.0 * (cal[:, None] <= score[None, :]).mean(axis=0)
        tier = np.where(score >= self.high_threshold_, "high",
                        np.where(score >= self.low_threshold_, "moderate", "normal"))
        return {"score": score, "conformal_p": p, "reference_percentile": percentile, "tier": tier}

    def bootstrap_scores(self, X, n_bootstrap=200):
        """Return calibration-residual uncertainty without refitting on query samples."""
        X = np.asarray(X, dtype=float)
        base = self.score_samples(X)["score"]
        resid = self.calibration_scores_ - np.median(self.calibration_scores_)
        rng = np.random.default_rng(self.random_state + 1)
        draws = base[:, None] + rng.choice(resid, size=(len(base), n_bootstrap), replace=True)
        return np.quantile(draws, [0.025, 0.975], axis=1).T
