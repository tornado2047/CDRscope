"""Hard/soft prototype assignment, multiscale aggregation and compositional transforms."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence
import numpy as np

_EPS = 1e-12


def _row_l1(x):
    den = np.sum(x, axis=1, keepdims=True)
    return np.divide(x, den, out=np.zeros_like(x, dtype=float), where=den > 0)


def transform_counts(x: np.ndarray, method: str = "hellinger", pseudocount: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or np.any(x < 0):
        raise ValueError("count matrix must be non-negative and 2-dimensional")
    method = method.lower()
    if method == "relative":
        return _row_l1(x)
    if method == "hellinger":
        return np.sqrt(_row_l1(x))
    if method == "l2":
        den = np.linalg.norm(x, axis=1, keepdims=True)
        return np.divide(x, den, out=np.zeros_like(x), where=den > 0)
    if method == "clr":
        z = np.log(x + pseudocount)
        return z - z.mean(axis=1, keepdims=True)
    if method == "tfidf":
        tf = _row_l1(x)
        idf = np.log1p(x.shape[0] / (1.0 + np.count_nonzero(x, axis=0)))
        return tf * idf
    raise ValueError(f"Unknown normalization: {method}")


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * a @ b.T,
        0.0,
    )

@dataclass
class SpectrumTransformer:
    centroids: np.ndarray
    assignment: str = "soft"
    top_k: int = 5
    temperature: float = 0.2
    normalization: str = "hellinger"
    batch_size: int = 4096

    def __post_init__(self):
        self.centroids = np.asarray(self.centroids, dtype=float)
        if self.centroids.ndim != 2 or len(self.centroids) < 1:
            raise ValueError("centroids must be a non-empty 2D matrix")
        if self.assignment not in {"hard", "soft"}:
            raise ValueError("assignment must be hard or soft")
        if not 1 <= self.top_k <= len(self.centroids):
            raise ValueError("top_k must be between 1 and number of centroids")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def assign(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.centroids.shape[1]:
            raise ValueError("embedding dimension does not match panel")
        out = np.zeros((len(embeddings), len(self.centroids)), dtype=float)
        for start in range(0, len(embeddings), self.batch_size):
            stop = min(start + self.batch_size, len(embeddings))
            d2 = _squared_distances(embeddings[start:stop], self.centroids)
            if self.assignment == "hard":
                idx = np.argmin(d2, axis=1)
                out[np.arange(start, stop), idx] = 1.0
                continue
            k = min(self.top_k, d2.shape[1])
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            local = np.take_along_axis(d2, idx, axis=1)
            local -= local.min(axis=1, keepdims=True)
            weights = np.exp(-local / self.temperature)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), _EPS)
            rows = np.arange(start, stop)[:, None]
            out[rows, idx] = weights
        return out

    def sample_spectrum(self, embeddings: np.ndarray, counts: Optional[np.ndarray] = None) -> np.ndarray:
        weights = self.assign(embeddings)
        if counts is None:
            counts = np.ones(len(weights), dtype=float)
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (len(weights),) or np.any(counts < 0):
            raise ValueError("counts must be a non-negative vector aligned to embeddings")
        raw = (weights * counts[:, None]).sum(axis=0, keepdims=True)
        return transform_counts(raw, self.normalization)[0]


def aggregate_scale(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups, dtype=int)
    if x.shape[1] != len(groups) or np.any(groups < 0):
        raise ValueError("groups must map every fine prototype to a non-negative group")
    out = np.zeros((x.shape[0], int(groups.max()) + 1), dtype=float)
    for j, group in enumerate(groups):
        out[:, group] += x[:, j]
    return out


def multiscale_spectra(
    fine_counts: np.ndarray,
    scale_groups: Mapping[str, np.ndarray],
    normalization: str = "hellinger",
    include_fine: bool = True,
) -> Dict[str, np.ndarray]:
    result = {}
    if include_fine:
        result["fine"] = transform_counts(fine_counts, normalization)
    for name, groups in scale_groups.items():
        result[str(name)] = transform_counts(aggregate_scale(fine_counts, groups), normalization)
    return result
