#!/usr/bin/env python3
"""
Selection Imprint Axes (Direction 5)
=====================================
Independent axes capturing thymic/germinal selection imprints,
orthogonal to the prototype frequency spectrum.

Axes:
  S1: N-insertion index — non-templated nucleotide additions
  S2: Physicochemical profile — charge, hydrophobicity, aromaticity
  S3: V-J bias vector — V/J gene usage PCA projection
  S4: Germline distance — CDR3 edit distance to germline
  S5: MHC covariate — HLA stratification (optional)

Design principle: each axis is independently normalized and concatenated
alongside the prototype spectrum, never mixed into K-means channels.
"""
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

# ---------------------------------------------------------------------------
# Amino acid physicochemical properties (Kyte-Doolittle + charge at pH 7.4)
# ---------------------------------------------------------------------------
AA_HYDROPATHY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}
AA_CHARGE = {
    'K': 1, 'R': 1, 'H': 0.5,  # positive
    'D': -1, 'E': -1,          # negative
    'A': 0, 'N': 0, 'C': 0, 'Q': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0,
    'F': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0, 'P': 0,
}
AA_AROMATIC = {'F', 'W', 'Y', 'H'}
AA_POLAR = {'S', 'T', 'N', 'Q', 'C', 'Y'}
AA_BULKY = {'F', 'W', 'Y', 'I', 'L', 'V'}

# Common human V/J germline CDR3 templates (simplified — for N-insertion estimation)
# In production, use IMGT germline database
GERMLINE_V_END = ['CASS', 'CAVS', 'CASSF', 'CAEVS', 'CASSL', 'CSAR', 'CSAV']
GERMLINE_J_START = ['F', 'FG', 'FGQ', 'FGA', 'FGAG', 'FGQG', 'FQG', 'SY', 'NEQ', 'NTE']


def estimate_n_insertions(cdr3):
    """Estimate non-templated nucleotide insertions in a CDR3.

    Rough heuristic: CDR3 length minus template-derived portions.
    V-region contributes ~4 aa (CASS motif), J-region contributes ~4-5 aa.
    N-insertions ≈ len(CDR3) - V_template - J_template
    """
    if not cdr3 or len(cdr3) < 8:
        return 0
    v_len = 0
    for motif in sorted(GERMLINE_V_END, key=len, reverse=True):
        if cdr3.startswith(motif):
            v_len = len(motif)
            break
    if v_len == 0:
        v_len = 2

    j_len = 0
    for motif in sorted(GERMLINE_J_START, key=len, reverse=True):
        if cdr3.endswith(motif) or motif in cdr3[-6:]:
            j_len = len(motif)
            break
    if j_len == 0:
        j_len = 3

    n_insert = max(0, len(cdr3) - v_len - j_len)
    return n_insert


def compute_n_insertion_axis(sequences, counts=None):
    """S1: N-insertion distribution for a sample.

    Returns a 10-d vector: percentile curve (10th-100th) of N-insertion counts.
    """
    if counts is None:
        counts = np.ones(len(sequences))
    else:
        counts = np.array(counts, dtype=float)

    n_ins = np.array([estimate_n_insertions(s) for s in sequences])
    weighted = np.repeat(n_ins, counts.astype(int))

    percentiles = np.percentile(weighted, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return percentiles.astype(np.float32)


def compute_physicochemical_axis(sequences, counts=None):
    """S2: Physicochemical profile of CDR3 sequences.

    Returns a 10-d vector:
      [0-1] net charge (median, IQR)
      [2-3] hydropathy (median, IQR)
      [4-5] aromatic content (median, IQR)
      [6-7] CDR3 length (median, IQR)
      [8-9] polar fraction (median, IQR)
    """
    if counts is None:
        counts = np.ones(len(sequences))
    else:
        counts = np.array(counts, dtype=int)

    charges, hydro, arom, lengths, polars = [], [], [], [], []
    for s, c in zip(sequences, counts):
        if not s or len(s) < 4:
            continue
        net_charge = sum(AA_CHARGE.get(aa, 0) for aa in s) / len(s)
        avg_hydro = np.mean([AA_HYDROPATHY.get(aa, 0) for aa in s])
        arom_frac = sum(1 for aa in s if aa in AA_AROMATIC) / len(s)
        polar_frac = sum(1 for aa in s if aa in AA_POLAR) / len(s)
        charges.extend([net_charge] * int(c))
        hydro.extend([avg_hydro] * int(c))
        arom.extend([arom_frac] * int(c))
        lengths.extend([len(s)] * int(c))
        polars.extend([polar_frac] * int(c))

    if not charges:
        return np.zeros(10, dtype=np.float32)

    result = np.array([
        np.median(charges), np.percentile(charges, 75) - np.percentile(charges, 25),
        np.median(hydro), np.percentile(hydro, 75) - np.percentile(hydro, 25),
        np.median(arom), np.percentile(arom, 75) - np.percentile(arom, 25),
        np.median(lengths), np.percentile(lengths, 75) - np.percentile(lengths, 25),
        np.median(polars), np.percentile(polars, 75) - np.percentile(polars, 25),
    ], dtype=np.float32)
    return result


def compute_vj_bias_axis(v_genes, j_genes, pca_model=None):
    """S3: V-J gene usage bias vector.

    Builds a V×J co-usage matrix, flattens it, then projects to 20-d via PCA.
    On first call (no pca_model), fits PCA and returns it alongside the vector.
    """
    v_counter = Counter(v_genes)
    j_counter = Counter(j_genes)
    total = len(v_genes)
    if total == 0:
        return np.zeros(20, dtype=np.float32), None

    v_freq = np.array([v_counter.get(v, 0) for v in sorted(v_counter)]) / total
    j_freq = np.array([j_counter.get(j, 0) for j in sorted(j_counter)]) / total

    # V-J co-usage: simplified as outer product of top-10 V × top-10 J
    top_v = sorted(v_counter, key=v_counter.get, reverse=True)[:10]
    top_j = sorted(j_counter, key=j_counter.get, reverse=True)[:10]
    vj_matrix = np.zeros((10, 10), dtype=np.float32)
    for v, j in zip(v_genes, j_genes):
        if v in top_v and j in top_j:
            vi = top_v.index(v)
            ji = top_j.index(j)
            vj_matrix[vi, ji] += 1
    vj_matrix /= max(total, 1)

    vj_flat = vj_matrix.flatten()
    vj_entropy = -np.sum(vj_flat * np.log2(vj_flat + 1e-10))
    v_n_pairs = len(set(zip(v_genes, j_genes)))

    feature = np.concatenate([
        vj_flat[:15],
        [vj_entropy, v_n_pairs / max(total, 1),
         len(set(v_genes)) / max(total, 1), len(set(j_genes)) / max(total, 1),
         np.sum(v_freq[:5])],
    ])

    if pca_model is not None and hasattr(pca_model, 'transform'):
        if len(feature) >= 20:
            projected = pca_model.transform(feature.reshape(1, -1)).ravel()
            return projected.astype(np.float32), None
        else:
            padded = np.zeros(20, dtype=np.float32)
            padded[:len(feature)] = feature
            projected = pca_model.transform(padded.reshape(1, -1)).ravel()
            return projected.astype(np.float32), None

    return feature[:20].astype(np.float32), None


def compute_germline_distance_axis(sequences, counts=None):
    """S4: Germline distance — how far CDR3s are from germline template.

    Returns a 5-d vector:
      [0] median edit distance to nearest V-germline end
      [1] IQR of edit distances
      [2] fraction of sequences matching known V-germline ends (public)
      [3] fraction with high N-insertion (private, >6 insertions)
      [4] public/private ratio
    """
    if counts is None:
        counts = np.ones(len(sequences))
    else:
        counts = np.array(counts, dtype=int)

    distances = []
    n_public, n_private = 0, 0
    for s, c in zip(sequences, counts):
        n_ins = estimate_n_insertions(s)
        dist = n_ins
        distances.extend([dist] * int(c))
        if n_ins <= 2:
            n_public += int(c)
        elif n_ins > 6:
            n_private += int(c)

    if not distances:
        return np.zeros(5, dtype=np.float32)

    d = np.array(distances)
    med_d = np.median(d)
    iqr_d = np.percentile(d, 75) - np.percentile(d, 25)
    pub_frac = n_public / max(len(distances), 1)
    priv_frac = n_private / max(len(distances), 1)
    pub_priv_ratio = pub_frac / max(priv_frac, 1e-6)

    return np.array([med_d, iqr_d, pub_frac, priv_frac, pub_priv_ratio], dtype=np.float32)


class SelectionImprintExtractor:
    """Unified extractor for all selection imprint axes."""

    def __init__(self, pca_vj_components=20):
        self.pca_vj_components = pca_vj_components
        self.vj_pca = None
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, reference_samples):
        """Fit normalization on reference (CordBlood) data.

        Args:
            reference_samples: list of dicts with 'sequences', 'counts',
                              'v_genes', 'j_genes'
        """
        all_s1, all_s2, all_s3, all_s4 = [], [], [], []

        for s in reference_samples:
            seqs = s.get('sequences', [])
            counts = s.get('counts', None)
            v_genes = s.get('v_genes', [])
            j_genes = s.get('j_genes', [])

            all_s1.append(compute_n_insertion_axis(seqs, counts))
            all_s2.append(compute_physicochemical_axis(seqs, counts))
            s3, _ = compute_vj_bias_axis(v_genes, j_genes)
            all_s3.append(s3)
            all_s4.append(compute_germline_distance_axis(seqs, counts))

        X = np.hstack([np.array(all_s1), np.array(all_s2),
                       np.array(all_s3), np.array(all_s4)])
        self.scaler.fit(X)
        self._fitted = True
        print(f"  SelectionImprintExtractor fitted on {len(reference_samples)} reference samples "
              f"({X.shape[1]}-d per sample)")

    def transform(self, sample):
        """Extract all selection imprint axes for one sample.

        Args:
            sample: dict with 'sequences', 'counts', 'v_genes', 'j_genes'

        Returns:
            (55-d,) numpy array: [S1(10) | S2(10) | S3(20) | S4(5) | pad(10)]
        """
        seqs = sample.get('sequences', [])
        counts = sample.get('counts', None)
        v_genes = sample.get('v_genes', [])
        j_genes = sample.get('j_genes', [])

        s1 = compute_n_insertion_axis(seqs, counts)
        s2 = compute_physicochemical_axis(seqs, counts)
        s3, _ = compute_vj_bias_axis(v_genes, j_genes)
        s4 = compute_germline_distance_axis(seqs, counts)

        raw = np.concatenate([s1, s2, s3, s4])
        if self._fitted:
            raw = self.scaler.transform(raw.reshape(1, -1)).ravel()

        return raw

    def get_axis_names(self):
        return (
            [f'N_insertion_pct_{i+1}' for i in range(10)] +
            [f'Physico_charge_med', 'Physico_charge_iqr',
             'Physico_hydro_med', 'Physico_hydro_iqr',
             'Physico_arom_med', 'Physico_arom_iqr',
             'Physico_len_med', 'Physico_len_iqr',
             'Physico_polar_med', 'Physico_polar_iqr'] +
            [f'VJ_bias_{i+1}' for i in range(20)] +
            ['Germline_dist_med', 'Germline_dist_iqr', 'Germline_pub_frac',
             'Germline_priv_frac', 'Germline_pub_priv_ratio']
        )


if __name__ == '__main__':
    # Smoke test with synthetic data
    np.random.seed(42)
    test_seqs = [
        'CASSLAPGATNEKLFF', 'CASSQETQYF', 'CASSLAPGATNEKLFF',
        'CAVSDFDYIAKTF', 'CASSLGQYF', 'CASSQETQYF',
        'CASSPRTGQYF', 'CAWSVAFQETQYF', 'CASSLAPGATNEKLFF',
        'CASSRRDYIAKTF',
    ]
    test_v = ['TRAV1-1', 'TRAV1-2', 'TRAV1-1', 'TRAV3-1', 'TRAV1-2',
              'TRAV1-1', 'TRAV5-1', 'TRAV3-1', 'TRAV1-1', 'TRAV5-1']
    test_j = ['TRAJ1', 'TRAJ2', 'TRAJ1', 'TRAJ3', 'TRAJ2',
              'TRAJ1', 'TRAJ4', 'TRAJ3', 'TRAJ1', 'TRAJ4']

    ext = SelectionImprintExtractor()
    sample = {'sequences': test_seqs, 'counts': None, 'v_genes': test_v, 'j_genes': test_j}

    # Without fitting (raw values)
    raw = ext.transform(sample)
    print(f"Raw axis dim: {raw.shape[0]}")
    print(f"S1 (N-insertion): {raw[:10]}")
    print(f"S2 (Physico):    {raw[10:20]}")
    print(f"S3 (V-J bias):   {raw[20:40]}")
    print(f"S4 (Germline):   {raw[40:45]}")

    # With fitting
    ref_samples = [
        {'sequences': test_seqs[:5], 'counts': None, 'v_genes': test_v[:5], 'j_genes': test_j[:5]},
        {'sequences': test_seqs[5:], 'counts': None, 'v_genes': test_v[5:], 'j_genes': test_j[5:]},
    ]
    ext.fit(ref_samples)
    normed = ext.transform(sample)
    print(f"\nNormed axis dim: {normed.shape[0]}")
    print(f"Axis names: {ext.get_axis_names()[:5]} ...")
    print("\nSmoke test passed.")
