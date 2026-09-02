#!/usr/bin/env python3
"""
Unsupervised Multi-Layer Spectrum Test on RA-TRA
=================================================
Tests whether adding L2/L4/S layers to L1 improves unsupervised
deviation scoring and anomaly detection.

Key hypothesis: L4 (macro indices) and S (selection imprints) may be
MORE useful in unsupervised settings than in supervised — because they
capture interpretable biological variation without needing prototype alignment.

Methods (matching v2 protocol):
  1. Reference deviation (unsup: from mean of all; semi: from control mean)
  2. OCSVM (RBF, trained on controls)
  3. IsolationForest (trained on controls)
  4. LOF (unsupervised)
  5. Ensemble (rank-sum of above)

Layer combinations tested:
  - L1 only (v2 baseline)
  - L4 only
  - S only
  - L2 only
  - L1 + L2
  - L1 + L4
  - L1 + S
  - L1 + L2 + L4 + S (full multi-layer)
"""
import os, sys, json, time, warnings
import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

BASE = os.path.expanduser("~/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8")
E2E_DIR = os.path.join(BASE, "tier3_e2e_results")
OUT_DIR = os.path.join(BASE, "tier3_unsup_results")
os.makedirs(OUT_DIR, exist_ok=True)

# Layer offsets in the saved multi-layer matrix
L1_END = 10000
L2_END = 10400
L4_END = 10410
S_END = 10455


def load_multilayer():
    """Load saved multi-layer matrix and split into layers."""
    mat = np.load(os.path.join(E2E_DIR, "ra_tra_multilayer.npy"))
    labels = np.load(os.path.join(E2E_DIR, "ra_tra_multilayer_labels.npy")).astype(int)

    layers = {
        'L1': mat[:, :L1_END],
        'L2': mat[:, L1_END:L2_END],
        'L4': mat[:, L2_END:L4_END],
        'S': mat[:, L4_END:S_END],
    }

    print(f"  Multi-layer matrix: {mat.shape}")
    print(f"  Labels: {Counter(labels.tolist())}")
    print(f"  Layers: {', '.join(f'{k}({v.shape[1]})' for k, v in layers.items())}")
    return mat, labels, layers


# =========================================================================
# Method 1: Reference Deviation Scoring
# =========================================================================
def reference_deviation(X, labels):
    """Compute deviation scores from reference origin.

    Unsupervised: reference = mean of ALL samples (no labels)
    Semi: reference = mean of controls (labels used only to define reference)
    """
    # Unsupervised (ref = all)
    ref_all = X.mean(axis=0)
    scores_unsup = np.linalg.norm(X - ref_all, axis=1)

    # Semi-supervised (ref = controls)
    ctrl_mask = labels == 0
    ref_ctrl = X[ctrl_mask].mean(axis=0)
    scores_semi = np.linalg.norm(X - ref_ctrl, axis=1)

    auc_unsup = roc_auc_score(labels, scores_unsup)
    auc_semi = roc_auc_score(labels, scores_semi)

    # Mann-Whitney U
    _, p_unsup = mannwhitneyu(scores_unsup[labels == 0], scores_unsup[labels == 1])
    _, p_semi = mannwhitneyu(scores_semi[labels == 0], scores_semi[labels == 1])

    return {
        'scores_unsup': scores_unsup,
        'scores_semi': scores_semi,
        'auc_unsup': auc_unsup,
        'auc_semi': auc_semi,
        'p_unsup': p_unsup,
        'p_semi': p_semi,
    }


# =========================================================================
# Method 2: One-Class SVM (RBF, trained on controls)
# =========================================================================
def ocsvm_detection(X, labels, pca_dim=50):
    """OCSVM trained on controls, score all samples."""
    ctrl_mask = labels == 0

    # PCA reduce
    n_components = min(pca_dim, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    ocsvm.fit(X_pca[ctrl_mask])
    scores = -ocsvm.score_samples(X_pca)
    auc = roc_auc_score(labels, scores)
    return {'scores': scores, 'auc': auc}


# =========================================================================
# Method 3: Isolation Forest (trained on controls)
# =========================================================================
def isoforest_detection(X, labels):
    """IsolationForest trained on controls, score all samples."""
    ctrl_mask = labels == 0
    iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    iso.fit(X[ctrl_mask])
    scores = -iso.score_samples(X)
    auc = roc_auc_score(labels, scores)
    return {'scores': scores, 'auc': auc}


# =========================================================================
# Method 4: LOF (unsupervised)
# =========================================================================
def lof_detection(X, labels, pca_dim=50):
    """Local Outlier Factor, unsupervised."""
    n_components = min(pca_dim, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit(X_pca)
    scores = -lof.negative_outlier_factor_
    auc = roc_auc_score(labels, scores)
    return {'scores': scores, 'auc': auc}


# =========================================================================
# Method 5: Ensemble (rank-sum of unsup methods)
# =========================================================================
def ensemble_scoring(scores_dict, labels):
    """Ensemble by rank-sum of multiple unsupervised scores."""
    # Use unsupervised scores only (no semi-sup)
    rank_sum = np.zeros(len(labels))
    n_methods = 0
    for name, scores in scores_dict.items():
        if 'unsup' in name or 'lof' in name or name == 'iso_all':
            ranks = rankdata(scores)
            rank_sum += ranks
            n_methods += 1
    rank_sum /= n_methods
    auc = roc_auc_score(labels, rank_sum)
    return {'scores': rank_sum, 'auc': auc}


# =========================================================================
# Main test
# =========================================================================
def run_test():
    print("=" * 70)
    print("Unsupervised Multi-Layer Spectrum Test on RA-TRA")
    print("=" * 70)

    mat, labels, layers = load_multilayer()

    # Define layer combinations to test
    combos = {
        'L1_only (v2)': layers['L1'],
        'L2_only': layers['L2'],
        'L4_only': layers['L4'],
        'S_only': layers['S'],
        'L1+L2': np.hstack([layers['L1'], layers['L2']]),
        'L1+L4': np.hstack([layers['L1'], layers['L4']]),
        'L1+S': np.hstack([layers['L1'], layers['S']]),
        'L1+L2+L4+S (v3)': mat,
    }

    results = {}

    for name, X in combos.items():
        print(f"\n{'='*60}")
        print(f"  Config: {name} (dim={X.shape[1]})")
        print(f"{'='*60}")

        # L2 normalize (matching v2 protocol)
        X_normed = normalize(X, norm='l2', axis=1) if X.shape[1] > 1 else X

        config_results = {}

        # Method 1: Reference deviation
        print("  [1] Reference deviation...", end=' ', flush=True)
        dev = reference_deviation(X_normed, labels)
        config_results['deviation'] = dev
        print(f"unsup AUC={dev['auc_unsup']:.4f}, semi AUC={dev['auc_semi']:.4f}")

        # Method 2: OCSVM
        print("  [2] OCSVM (RBF, ctrl-trained)...", end=' ', flush=True)
        ocsvm = ocsvm_detection(X_normed, labels)
        config_results['ocsvm'] = ocsvm
        print(f"AUC={ocsvm['auc']:.4f}")

        # Method 3: Isolation Forest
        print("  [3] IsolationForest (ctrl-trained)...", end=' ', flush=True)
        iso = isoforest_detection(X_normed, labels)
        config_results['isoforest'] = iso
        print(f"AUC={iso['auc']:.4f}")

        # Method 4: LOF (unsupervised)
        print("  [4] LOF (unsupervised)...", end=' ', flush=True)
        lof = lof_detection(X_normed, labels)
        config_results['lof'] = lof
        print(f"AUC={lof['auc']:.4f}")

        # Method 5: Ensemble
        print("  [5] Ensemble (rank-sum)...", end=' ', flush=True)
        ens_scores = {
            'dev_unsup': dev['scores_unsup'],
            'ocsvm': ocsvm['scores'],
            'iso_all': iso['scores'],
            'lof': lof['scores'],
        }
        ens = ensemble_scoring(ens_scores, labels)
        config_results['ensemble'] = ens
        print(f"AUC={ens['auc']:.4f}")

        results[name] = {
            'dim': X.shape[1],
            'deviation_unsup_auc': dev['auc_unsup'],
            'deviation_semi_auc': dev['auc_semi'],
            'ocsvm_auc': ocsvm['auc'],
            'isoforest_auc': iso['auc'],
            'lof_auc': lof['auc'],
            'ensemble_auc': ens['auc'],
        }

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Unsupervised Multi-Layer Spectrum AUC Comparison")
    print(f"{'='*70}")
    print(f"{'Config':<22s} {'Dim':>6s} {'DevUnsup':>9s} {'DevSemi':>9s} "
          f"{'OCSVM':>9s} {'IsoFor':>9s} {'LOF':>9s} {'Ens':>9s}")
    print("-" * 82)
    for name, r in results.items():
        print(f"{name:<22s} {r['dim']:>6d} "
              f"{r['deviation_unsup_auc']:>9.4f} {r['deviation_semi_auc']:>9.4f} "
              f"{r['ocsvm_auc']:>9.4f} {r['isoforest_auc']:>9.4f} "
              f"{r['lof_auc']:>9.4f} {r['ensemble_auc']:>9.4f}")

    # Key comparisons
    print(f"\n{'='*70}")
    print("KEY COMPARISONS")
    print(f"{'='*70}")
    l1 = results['L1_only (v2)']
    v3 = results['L1+L2+L4+S (v3)']
    l4 = results['L4_only']
    s = results['S_only']

    print(f"\n  L1-only → Multi-layer (ensemble):")
    print(f"    Deviation: {l1['deviation_semi_auc']:.4f} → {v3['deviation_semi_auc']:.4f} "
          f"(Δ={v3['deviation_semi_auc']-l1['deviation_semi_auc']:+.4f})")
    print(f"    OCSVM:     {l1['ocsvm_auc']:.4f} → {v3['ocsvm_auc']:.4f} "
          f"(Δ={v3['ocsvm_auc']-l1['ocsvm_auc']:+.4f})")
    print(f"    Ensemble:  {l1['ensemble_auc']:.4f} → {v3['ensemble_auc']:.4f} "
          f"(Δ={v3['ensemble_auc']-l1['ensemble_auc']:+.4f})")

    print(f"\n  Low-dimensional layers (standalone):")
    print(f"    L4 (10-d):  dev={l4['deviation_semi_auc']:.4f}, ens={l4['ensemble_auc']:.4f}")
    print(f"    S  (45-d):  dev={s['deviation_semi_auc']:.4f}, ens={s['ensemble_auc']:.4f}")

    # Save
    output = {
        'test': 'Unsupervised multi-layer spectrum test',
        'dataset': 'RA-TRA (545 samples)',
        'results': results,
        'layer_dims': {'L1': 10000, 'L2': 400, 'L4': 10, 'S': 45},
    }
    out_path = os.path.join(OUT_DIR, 'unsup_multilayer_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    results = run_test()
