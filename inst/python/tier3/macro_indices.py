#!/usr/bin/env python3
"""
Macro Indices Layer (Direction 3, L4)
=====================================
Sample-level macroscopic diversity and clonality metrics,
forming an independent representation layer orthogonal to the prototype spectrum.

Metrics:
  1. Shannon entropy (clone size distribution)
  2. Simpson diversity index (1 - D)
  3. Gini coefficient (clone size inequality)
  4. Oligoclonality index (top-k clone fraction)
  5. Chao1 richness estimate
  6. Pielou's evenness (normalized Shannon)
  7. Berger-Parker dominance (max clone fraction)
  8. Clonality (1 - normalized Shannon, per Adaptive Biotech)
  9. Unique sequence count (log10)
  10. Total clone count (log10)

These are scalar features, naturally comparable across individuals —
they don't require prototype alignment.
"""
import numpy as np
from collections import Counter
from scipy import stats


def shannon_entropy(counts):
    """Shannon entropy of clone size distribution."""
    c = np.array(counts, dtype=float)
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def simpson_diversity(counts):
    """Simpson diversity (1 - D), where D = sum(pi^2)."""
    c = np.array(counts, dtype=float)
    total = c.sum()
    if total == 0:
        return 0.0
    p = c / total
    return 1.0 - np.sum(p ** 2)


def gini_coefficient(counts):
    """Gini coefficient of clone size inequality."""
    c = np.sort(np.array(counts, dtype=float))
    n = len(c)
    if n == 0 or c.sum() == 0:
        return 0.0
    cumsum = np.cumsum(c)
    return (2.0 * np.sum((np.arange(1, n + 1)) * c)) / (n * c.sum()) - (n + 1.0) / n


def oligoclonality_index(counts, k=10):
    """Fraction of sequences in the top-k largest clones."""
    c = np.sort(np.array(counts, dtype=float))[::-1]
    total = c.sum()
    if total == 0:
        return 0.0
    k = min(k, len(c))
    return np.sum(c[:k]) / total


def chao1_richness(counts):
    """Chao1 non-parametric richness estimator."""
    c = np.array(counts, dtype=int)
    s_obs = len(c)
    if s_obs == 0:
        return 0.0
    s1 = np.sum(c == 1)
    s2 = np.sum(c == 2)
    return s_obs + (s1 ** 2) / max(2 * s2, 1)


def pielou_evenness(counts):
    """Pielou's evenness: Shannon / log(S)."""
    c = np.array(counts, dtype=float)
    s = len(c)
    if s <= 1 or c.sum() == 0:
        return 0.0
    h = shannon_entropy(c)
    return h / np.log2(s)


def berger_parker(counts):
    """Berger-Parker dominance: fraction in the largest clone."""
    c = np.array(counts, dtype=float)
    total = c.sum()
    if total == 0:
        return 0.0
    return np.max(c) / total


def clonality_metric(counts):
    """Clonality as defined by Adaptive Biotech: 1 - (Shannon / log2(S))."""
    c = np.array(counts, dtype=float)
    s = len(c)
    if s <= 1 or c.sum() == 0:
        return 0.0
    h = shannon_entropy(c)
    return 1.0 - (h / np.log2(s))


class MacroIndexExtractor:
    """L4: Macroscopic diversity and clonality layer."""

    def __init__(self, top_k=10):
        self.top_k = top_k
        self.feature_names = [
            'shannon_entropy', 'simpson_diversity', 'gini_coefficient',
            f'oligoclonality_top{top_k}', 'chao1_richness',
            'pielou_evenness', 'berger_parker_dominance', 'clonality',
            'log10_unique_clones', 'log10_total_clones',
        ]

    def transform(self, counts):
        """Extract macro indices from clone count vector.

        Args:
            counts: array-like of clone sizes (per-clone counts),
                    or a prototype spectrum (m-dim count vector)

        Returns:
            (10,) numpy array of macro indices
        """
        c = np.array(counts, dtype=float)
        c = c[c > 0]  # remove zero-count prototypes

        if len(c) == 0:
            return np.zeros(10, dtype=np.float32)

        n_unique = len(c)
        n_total = c.sum()

        result = np.array([
            shannon_entropy(c),
            simpson_diversity(c),
            gini_coefficient(c),
            oligoclonality_index(c, k=self.top_k),
            chao1_richness(c.astype(int)),
            pielou_evenness(c),
            berger_parker(c),
            clonality_metric(c),
            np.log10(n_unique) if n_unique > 0 else 0.0,
            np.log10(n_total) if n_total > 0 else 0.0,
        ], dtype=np.float32)

        return result

    def transform_many(self, count_vectors):
        """Extract macro indices for multiple samples.

        Args:
            count_vectors: list of count arrays, or (n_samples, m) array

        Returns:
            (n_samples, 10) numpy array
        """
        results = []
        for cv in count_vectors:
            results.append(self.transform(cv))
        return np.array(results, dtype=np.float32)

    def get_feature_names(self):
        return self.feature_names


if __name__ == '__main__':
    # Smoke test with realistic prototype spectrum
    np.random.seed(42)

    # Simulate a polyclonal (healthy) sample
    healthy_counts = np.random.negative_binomial(2, 0.1, size=10000)
    healthy_counts[healthy_counts > 0].sort()

    # Simulate an oligoclonal (disease) sample
    disease_counts = np.zeros(10000, dtype=float)
    disease_counts[:3] = [500, 300, 200]
    disease_counts[3:20] = np.random.randint(10, 50, 17)
    disease_counts[20:100] = np.random.randint(1, 5, 80)

    ext = MacroIndexExtractor()
    healthy_vec = ext.transform(healthy_counts)
    disease_vec = ext.transform(disease_counts)

    print("=== Macro Indices Smoke Test ===\n")
    print(f"{'Metric':<30} {'Healthy':>10} {'Disease':>10}")
    print("-" * 52)
    for name, h, d in zip(ext.feature_names, healthy_vec, disease_vec):
        print(f"{name:<30} {h:>10.4f} {d:>10.4f}")

    print(f"\nHealthy: diverse (high Shannon, low clonality)")
    print(f"Disease: oligoclonal (low Shannon, high clonality)")

    # Test transform_many
    multi = ext.transform_many([healthy_counts, disease_counts])
    print(f"\nBatch transform shape: {multi.shape}")
    assert multi.shape == (2, 10), f"Expected (2,10), got {multi.shape}"
    print("Smoke test passed.")
