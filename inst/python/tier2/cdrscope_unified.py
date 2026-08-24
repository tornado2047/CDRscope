#!/usr/bin/env python3
"""
CDRscope v2.0 — Unified Pipeline
==================================
A single entry point that handles both supervised and unsupervised analysis.

Route A (Supervised): If labels are provided →
  - Linear SVM 5-fold CV classification
  - Differential prototype identification
  - ROC/AUC, confusion matrix

Route B (Unsupervised): All samples treated as unlabeled →
  - Reference deviation scoring (magnitude + direction)
  - One-Class SVM anomaly detection
  - LOF pure unsupervised anomaly detection
  - Diversity analysis (Shannon, Simpson, Pielou, Chao1)
  - Multi-scale analysis (2000 functional groups)

Route B always runs. Route A runs additionally if labels exist.

Quantitative outputs:
  - Deviation magnitude: ||sample - reference|| (how far)
  - Deviation direction: which prototypes deviate most (where)
  - Anomaly score: OCSVM/LOF probability of being abnormal
  - Diversity indices: repertoire richness/evenness

Visualization:
  - PCA/UMAP with deviation coloring
  - Deviation radar chart (top differential prototypes)
  - Deviation heatmap (samples × top prototypes)
  - Score distribution + ROC (if labels)
  - Summary dashboard
"""
import os, sys, json, time, pickle, warnings, base64
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, entropy
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, average_precision_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from umap import UMAP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

warnings.filterwarnings('ignore')

WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
TIER2_DIR = os.path.join(WORK_DIR, "CDRscope-v2", "inst", "python", "tier2")
sys.path.insert(0, TIER2_DIR)
PANEL_DIR = os.path.join(WORK_DIR, "cordblood_tra_panel")
OUTPUT_DIR = os.path.join(WORK_DIR, "cdrscope_unified_results")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"
EMBED_DIM = 480
M_TARGET = 10000
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Color system
C_CTRL = '#4a90d9'
C_PAT = '#ff6b6b'
C_ACCENT = '#5e5ce6'
C_GREEN = '#00a389'
C_ORANGE = '#ff9f0a'
C_PURPLE = '#bf5af2'
C_TEAL = '#64d2ff'
C_GRAY = '#8e8e93'

# Deviation colormap (white → blue for low, white → red for high)
DEV_CMAP = LinearSegmentedColormap.from_list('deviation',
    ['#4a90d9', '#ffffff', '#ff6b6b'], N=256)


def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def shannon(counts):
    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def simpson(counts):
    p = counts / counts.sum() if counts.sum() > 0 else counts
    return 1 - np.sum(p**2)


def pielou(counts):
    H = shannon(counts)
    S = np.sum(counts > 0)
    return H / np.log(S) if S > 1 else 0


def chao1(counts):
    S_obs = np.sum(counts > 0)
    S_single = np.sum(counts == 1)
    S_double = np.sum(counts == 2)
    if S_double == 0:
        return S_obs + S_single * (S_single - 1) / 2
    return S_obs + (S_single**2) / (2 * S_double)


# =========================================================================
# Data Loading
# =========================================================================
def load_data():
    """Load pre-computed RA-TRA matrix and labels."""
    mat_path = os.path.join(PANEL_DIR, "ra_tra_cb_matrix.npy")
    lbl_path = os.path.join(PANEL_DIR, "ra_tra_cb_labels.npy")

    print("Loading RA-TRA matrix...", flush=True)
    X_raw = np.load(mat_path).astype(np.float64)
    labels = np.load(lbl_path).astype(int)
    print(f"  Matrix: {X_raw.shape} | Labels: {Counter(labels.tolist())}", flush=True)

    X = normalize(X_raw, norm='l2')
    return X, X_raw, labels


# =========================================================================
# Route A: Supervised Analysis
# =========================================================================
def route_a_supervised(X, labels):
    """Supervised classification — only runs if labels are provided."""
    print("\n" + "=" * 60, flush=True)
    print("ROUTE A: Supervised Analysis", flush=True)
    print("=" * 60, flush=True)

    results = {}

    # 5-fold CV Linear SVM
    print("  Running 5-fold CV Linear SVM...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_pred = np.zeros(len(labels))
    all_prob = np.zeros(len(labels))
    fold_aucs = []

    for fold, (tr, te) in enumerate(cv.split(X, labels)):
        clf = LinearSVC(C=0.1, dual=False, max_iter=5000)
        clf.fit(X[tr], labels[tr])
        all_pred[te] = clf.predict(X[te])
        all_prob[te] = clf.decision_function(X[te])
        auc = roc_auc_score(labels[te], all_prob[te])
        fold_aucs.append(auc)
        print(f"    Fold {fold+1}: AUC={auc:.4f}", flush=True)

    auc = roc_auc_score(labels, all_prob)
    ap = average_precision_score(labels, all_prob)
    acc = accuracy_score(labels, all_pred)
    f1 = f1_score(labels, all_pred)
    mcc = matthews_corrcoef(labels, all_pred)
    sens = recall_score(labels, all_pred)
    spec = recall_score(1 - labels, 1 - all_pred)

    results['cv_aucs'] = fold_aucs
    results['auc'] = auc
    results['ap'] = ap
    results['accuracy'] = acc
    results['f1'] = f1
    results['mcc'] = mcc
    results['sensitivity'] = sens
    results['specificity'] = spec
    results['y_pred'] = all_pred
    results['y_score'] = all_prob

    print(f"  Overall: AUC={auc:.4f} | AP={ap:.4f} | Acc={acc:.4f} | "
          f"F1={f1:.4f} | Sens={sens:.4f} | Spec={spec:.4f}", flush=True)

    # Train final model on all data for prototype importance
    clf_final = LinearSVC(C=0.1, dual=False, max_iter=5000)
    clf_final.fit(X, labels)
    coef = clf_final.coef_[0]
    results['svm_coef'] = coef

    # Top differential prototypes
    top_pos = np.argsort(coef)[::-1][:30]  # patient-enriched
    top_neg = np.argsort(coef)[:30]  # control-enriched
    results['top_pos'] = top_pos
    results['top_neg'] = top_neg

    print(f"  Top 5 patient-enriched prototypes: {top_pos[:5]}", flush=True)
    print(f"  Top 5 control-enriched prototypes: {top_neg[:5]}", flush=True)

    return results


# =========================================================================
# Route B: Unsupervised Analysis
# =========================================================================
def route_b_unsupervised(X, X_raw, labels=None):
    """
    Unsupervised analysis — always runs.
    Core: deviation magnitude + deviation direction.
    """
    print("\n" + "=" * 60, flush=True)
    print("ROUTE B: Unsupervised Analysis", flush=True)
    print("=" * 60, flush=True)

    results = {}

    # --- Reference definition ---
    # Primary: CordBlood panel mean (built-in healthy reference)
    # If labels exist: also compute control mean for comparison
    ref_all = X.mean(axis=0)
    results['ref_all'] = ref_all

    if labels is not None and 0 in labels:
        ref_ctrl = X[labels == 0].mean(axis=0)
        results['ref_ctrl'] = ref_ctrl
        has_ctrl = True
    else:
        ref_ctrl = ref_all
        has_ctrl = False
    results['has_control_ref'] = has_ctrl

    # === Deviation Vector (per sample) ===
    print("\n  [B.1] Deviation Vector Analysis...", flush=True)
    dev_vectors = X - ref_ctrl  # (n_samples, 10000)
    dev_magnitude = np.linalg.norm(dev_vectors, axis=1)  # scalar per sample
    results['dev_vectors'] = dev_vectors
    results['dev_magnitude'] = dev_magnitude

    # Deviation direction: unit vector
    norms = dev_magnitude.copy()
    norms[norms == 0] = 1
    dev_direction = dev_vectors / norms[:, np.newaxis]
    results['dev_direction'] = dev_direction

    # Mean deviation per prototype (group level)
    dev_per_proto = dev_vectors.mean(axis=0)
    results['dev_per_proto'] = dev_per_proto

    # Top deviating prototypes (by absolute mean deviation)
    top_dev_idx = np.argsort(np.abs(dev_per_proto))[::-1][:50]
    results['top_dev_idx'] = top_dev_idx

    print(f"  Deviation magnitude: mean={dev_magnitude.mean():.4f}, "
          f"std={dev_magnitude.std():.4f}", flush=True)
    print(f"  Top 5 deviating prototypes: {top_dev_idx[:5]}", flush=True)

    # Validation if labels exist
    if labels is not None:
        auc_dev = roc_auc_score(labels, dev_magnitude)
        results['auc_dev'] = auc_dev
        stat, p = mannwhitneyu(dev_magnitude[labels == 0],
                               dev_magnitude[labels == 1])
        results['dev_mw_p'] = p
        print(f"  Deviation AUC={auc_dev:.4f} (MW p={p:.2e})", flush=True)

    # === PCA + UMAP ===
    print("\n  [B.2] Dimensionality Reduction...", flush=True)
    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X)
    results['X_pca'] = X_pca
    results['pca_var'] = pca.explained_variance_ratio_

    umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap.fit_transform(X_pca)
    results['X_umap'] = X_umap
    print(f"  PCA: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}", flush=True)

    # === One-Class SVM (anomaly detection) ===
    print("\n  [B.3] One-Class SVM Anomaly Detection...", flush=True)
    pca50 = PCA(n_components=50)
    X_pca50 = pca50.fit_transform(X)

    if has_ctrl:
        # Semi-supervised: train on controls
        ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        ocsvm.fit(X_pca50[labels == 0])
        anomaly_score = -ocsvm.score_samples(X_pca50)
        results['anomaly_score'] = anomaly_score
        if labels is not None:
            auc_ocsvm = roc_auc_score(labels, anomaly_score)
            results['auc_ocsvm'] = auc_ocsvm
            print(f"  OCSVM (ref=ctrl) AUC={auc_ocsvm:.4f}", flush=True)
    else:
        # Unsupervised: train on all
        ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        ocsvm.fit(X_pca50)
        anomaly_score = -ocsvm.score_samples(X_pca50)
        results['anomaly_score'] = anomaly_score

    # === LOF (pure unsupervised) ===
    print("  [B.4] Local Outlier Factor...", flush=True)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit(X_pca50)
    lof_score = -lof.negative_outlier_factor_
    results['lof_score'] = lof_score
    if labels is not None:
        auc_lof = roc_auc_score(labels, lof_score)
        results['auc_lof'] = auc_lof
        print(f"  LOF AUC={auc_lof:.4f}", flush=True)

    # === Diversity Analysis ===
    print("  [B.5] Diversity Analysis...", flush=True)
    diversity = []
    for i in range(X_raw.shape[0]):
        counts = X_raw[i]
        diversity.append({
            'sample_idx': i,
            'richness': int(np.sum(counts > 0)),
            'shannon': shannon(counts),
            'simpson': simpson(counts),
            'pielou': pielou(counts),
            'chao1': chao1(counts),
        })
    results['diversity'] = pd.DataFrame(diversity)
    print(f"  Richness: {results['diversity']['richness'].mean():.0f} ± "
          f"{results['diversity']['richness'].std():.0f}", flush=True)

    # === Multi-scale (2000 groups) ===
    print("\n  [B.6] Multi-scale Analysis (2000 groups)...", flush=True)
    panel_path = os.path.join(PANEL_DIR, "cb_tra_reference_panel_m10000.pkl")
    with open(panel_path, 'rb') as f:
        panel_data = pickle.load(f)
    centroids = panel_data['centroids']

    km_groups = KMeans(n_clusters=2000, random_state=42, n_init=10)
    groups = km_groups.fit_predict(centroids)
    X_group = np.zeros((X_raw.shape[0], 2000))
    for g in range(2000):
        mask = groups == g
        if mask.sum() > 0:
            X_group[:, g] = X_raw[:, mask].sum(axis=1)
    X_group_norm = normalize(X_group, norm='l2')

    pca_g = PCA(n_components=50)
    X_group_pca50 = pca_g.fit_transform(X_group_norm)

    if has_ctrl:
        ocsvm_g = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        ocsvm_g.fit(X_group_pca50[labels == 0])
        anomaly_score_g = -ocsvm_g.score_samples(X_group_pca50)
        results['anomaly_score_2000'] = anomaly_score_g
        if labels is not None:
            auc_g = roc_auc_score(labels, anomaly_score_g)
            results['auc_ocsvm_2000'] = auc_g
            print(f"  OCSVM (2000 groups) AUC={auc_g:.4f}", flush=True)

    # === JS Divergence Deviation ===
    print("  [B.7] JS Divergence Deviation...", flush=True)
    X_prob = X_raw / X_raw.sum(axis=1, keepdims=True)
    ref_prob = X_prob[labels == 0].mean(axis=0) if has_ctrl else X_prob.mean(axis=0)
    js_dev = np.array([jensenshannon(p, ref_prob, base=2) for p in X_prob])
    results['js_dev'] = js_dev
    if labels is not None:
        auc_js = roc_auc_score(labels, js_dev)
        results['auc_js'] = auc_js
        print(f"  JS deviation AUC={auc_js:.4f}", flush=True)

    # === Composite Score ===
    print("\n  [B.8] Composite Anomaly Score...", flush=True)
    # Z-normalize each score and average
    scores_to_combine = {
        'deviation': dev_magnitude,
        'ocsvm': anomaly_score,
        'lof': lof_score,
        'js_divergence': js_dev,
    }
    if 'anomaly_score_2000' in results:
        scores_to_combine['ocsvm_2000'] = results['anomaly_score_2000']

    z_scores = {}
    for name, s in scores_to_combine.items():
        z = (s - s.mean()) / (s.std() + 1e-10)
        z_scores[name] = z

    composite = np.mean(list(z_scores.values()), axis=0)
    results['composite_score'] = composite
    results['z_scores'] = z_scores

    if labels is not None:
        auc_composite = roc_auc_score(labels, composite)
        results['auc_composite'] = auc_composite
        print(f"  Composite AUC={auc_composite:.4f}", flush=True)

    # === Sample Classification (unsupervised) ===
    # Rank samples by composite score
    ranked = np.argsort(composite)[::-1]
    results['ranked_samples'] = ranked

    # Define anomaly tiers
    n_total = len(composite)
    n_high = max(1, int(n_total * 0.1))  # top 10%
    n_moderate = max(1, int(n_total * 0.25))  # top 25%

    tier = np.zeros(n_total, dtype=int)  # 0=normal, 1=moderate, 2=high
    tier[ranked[:n_high]] = 2
    tier[ranked[n_high:n_moderate]] = 1
    results['tier'] = tier

    print(f"  Tiers: Normal={np.sum(tier==0)}, "
          f"Moderate={np.sum(tier==1)}, High={np.sum(tier==2)}", flush=True)

    if labels is not None:
        for t, name in [(2, 'High'), (1, 'Moderate'), (0, 'Normal')]:
            mask = tier == t
            if mask.sum() > 0:
                pat_rate = labels[mask].mean()
                print(f"    {name}: {mask.sum()} samples, "
                      f"patient rate={pat_rate:.1%}", flush=True)

    return results


# =========================================================================
# Visualization
# =========================================================================
def plot_dashboard(sup_results, unsup_results, X, labels, has_labels):
    """Figure 1: Master dashboard — 2x3 grid."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # --- (0,0) PCA colored by deviation magnitude ---
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(unsup_results['X_pca'][:, 0],
                    unsup_results['X_pca'][:, 1],
                    c=unsup_results['dev_magnitude'],
                    cmap='YlOrRd', s=30, alpha=0.7, edgecolors='white',
                    linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Deviation', shrink=0.8)
    ax.set_xlabel(f'PC1 ({unsup_results["pca_var"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({unsup_results["pca_var"][1]:.1%})')
    ax.set_title('PCA — Deviation Magnitude')

    # --- (0,1) UMAP colored by anomaly score ---
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(unsup_results['X_umap'][:, 0],
                    unsup_results['X_umap'][:, 1],
                    c=unsup_results['anomaly_score'],
                    cmap='YlOrRd', s=30, alpha=0.7, edgecolors='white',
                    linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Anomaly', shrink=0.8)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP — Anomaly Score')

    # --- (0,2) PCA colored by tier ---
    ax = fig.add_subplot(gs[0, 2])
    tier_colors = {0: C_CTRL, 1: C_ORANGE, 2: C_PAT}
    tier_names = {0: 'Normal', 1: 'Moderate', 2: 'High'}
    tier = unsup_results['tier']
    for t in [0, 1, 2]:
        mask = tier == t
        ax.scatter(unsup_results['X_pca'][mask, 0],
                   unsup_results['X_pca'][mask, 1],
                   c=tier_colors[t], s=30, alpha=0.7,
                   edgecolors='white', linewidth=0.3,
                   label=f'{tier_names[t]} (n={mask.sum()})')
    ax.set_xlabel(f'PC1 ({unsup_results["pca_var"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({unsup_results["pca_var"][1]:.1%})')
    ax.set_title('PCA — Anomaly Tiers')
    ax.legend(fontsize=8)

    # --- (1,0) Score distribution ---
    ax = fig.add_subplot(gs[1, 0])
    if has_labels:
        for lv, color, name in [(0, C_CTRL, 'Control'), (1, C_PAT, 'Patient')]:
            mask = labels == lv
            ax.hist(unsup_results['composite_score'][mask], bins=30,
                    color=color, alpha=0.6, label=f'{name} (n={mask.sum()})',
                    edgecolor='white')
        ax.set_title('Composite Score Distribution')
        ax.legend()
    else:
        ax.hist(unsup_results['composite_score'], bins=30, color=C_ACCENT,
                alpha=0.7, edgecolor='white')
        ax.set_title('Composite Score Distribution')
    ax.set_xlabel('Composite Anomaly Score')
    ax.set_ylabel('Count')

    # --- (1,1) ROC curves ---
    ax = fig.add_subplot(gs[1, 1])
    if has_labels:
        score_methods = [
            ('Deviation', unsup_results['dev_magnitude'],
             unsup_results.get('auc_dev', 0), C_ACCENT),
            ('OCSVM', unsup_results['anomaly_score'],
             unsup_results.get('auc_ocsvm', 0), C_PAT),
            ('LOF', unsup_results['lof_score'],
             unsup_results.get('auc_lof', 0), C_ORANGE),
            ('JS Diverg.', unsup_results['js_dev'],
             unsup_results.get('auc_js', 0), C_GREEN),
            ('Composite', unsup_results['composite_score'],
             unsup_results.get('auc_composite', 0), C_PURPLE),
        ]
        if 'anomaly_score_2000' in unsup_results:
            score_methods.append(
                ('OCSVM-2000', unsup_results['anomaly_score_2000'],
                 unsup_results.get('auc_ocsvm_2000', 0), C_TEAL))
        if sup_results:
            score_methods.append(
                ('SVM (sup.)', sup_results['y_score'],
                 sup_results.get('auc', 0), '#ff453a'))

        for name, scores, auc, color in score_methods:
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, color=color, linewidth=1.8,
                    label=f'{name} ({auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No labels\nfor ROC', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color=C_GRAY)
        ax.set_title('ROC Curves (N/A)')

    # --- (1,2) Method comparison bar chart ---
    ax = fig.add_subplot(gs[1, 2])
    if has_labels:
        methods = [
            ('Dev.', unsup_results.get('auc_dev', 0), C_ACCENT),
            ('OCSVM', unsup_results.get('auc_ocsvm', 0), C_PAT),
            ('OCSVM\n2000', unsup_results.get('auc_ocsvm_2000', 0), C_TEAL),
            ('LOF', unsup_results.get('auc_lof', 0), C_ORANGE),
            ('JS', unsup_results.get('auc_js', 0), C_GREEN),
            ('Compos.', unsup_results.get('auc_composite', 0), C_PURPLE),
        ]
        if sup_results:
            methods.append(('SVM\n(sup.)', sup_results.get('auc', 0), '#ff453a'))

        names = [m[0] for m in methods]
        aucs = [m[1] for m in methods]
        colors = [m[2] for m in methods]
        bars = ax.bar(names, aucs, color=colors, alpha=0.8, edgecolor='white')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
        ax.set_ylabel('AUC-ROC')
        ax.set_title('Method Comparison')
        ax.set_ylim(0.4, 1.05)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No labels\nfor comparison', ha='center',
                va='center', transform=ax.transAxes, fontsize=14, color=C_GRAY)
        ax.set_title('Method Comparison (N/A)')

    plt.suptitle('CDRscope v2.0 — Unified Dashboard', fontsize=18,
                fontweight='bold', y=1.01)
    path = os.path.join(IMG_DIR, 'fig_dashboard.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig_dashboard.png saved", flush=True)


def plot_deviation_radar(unsup_results, labels, has_labels):
    """Figure 2: Radar chart of top differential prototypes."""
    dev_per_proto = unsup_results['dev_per_proto']
    top_idx = unsup_results['top_dev_idx'][:20]

    # Group-level deviation: aggregate top 20 into 4 quadrants
    n_per_quad = 5
    quadrants = []
    for q in range(4):
        q_idx = top_idx[q * n_per_quad:(q + 1) * n_per_quad]
        q_dev = dev_per_proto[q_idx]
        quadrants.append({
            'name': f'Proto Group {q+1}',
            'indices': q_idx,
            'mean_dev': np.mean(np.abs(q_dev)),
            'values': np.abs(q_dev),
        })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             subplot_kw=dict(projection='polar'))

    # Left: Radar for mean deviation
    ax = axes[0]
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False).reshape(5, 4)

    if has_labels:
        ctrl_dev = unsup_results['dev_vectors'][labels == 0].mean(axis=0)
        pat_dev = unsup_results['dev_vectors'][labels == 1].mean(axis=0)

        ctrl_vals = np.abs(ctrl_dev[top_idx])
        pat_vals = np.abs(pat_dev[top_idx])

        # Normalize
        max_val = max(ctrl_vals.max(), pat_vals.max())
        ctrl_norm = ctrl_vals / max_val
        pat_norm = pat_vals / max_val

        angles_flat = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        angles_closed = np.concatenate([angles_flat, [angles_flat[0]]])

        ax.plot(angles_closed, np.concatenate([ctrl_norm, [ctrl_norm[0]]]),
                color=C_CTRL, linewidth=2, label='Control mean')
        ax.fill(angles_closed, np.concatenate([ctrl_norm, [ctrl_norm[0]]]),
                color=C_CTRL, alpha=0.15)
        ax.plot(angles_closed, np.concatenate([pat_norm, [pat_norm[0]]]),
                color=C_PAT, linewidth=2, label='Patient mean')
        ax.fill(angles_closed, np.concatenate([pat_norm, [pat_norm[0]]]),
                color=C_PAT, alpha=0.15)

        ax.set_xticks(angles_flat)
        ax.set_xticklabels([f'P{i}' for i in top_idx], fontsize=7)
        ax.set_title('Deviation Radar\n(Top 20 Prototypes)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    else:
        all_dev = np.abs(dev_per_proto[top_idx])
        angles_flat = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        angles_closed = np.concatenate([angles_flat, [angles_flat[0]]])
        max_val = all_dev.max()
        ax.plot(angles_closed,
                np.concatenate([all_dev / max_val, [all_dev[-1] / max_val]]),
                color=C_ACCENT, linewidth=2, label='All samples')
        ax.fill(angles_closed,
                np.concatenate([all_dev / max_val, [all_dev[-1] / max_val]]),
                color=C_ACCENT, alpha=0.15)
        ax.set_xticks(angles_flat)
        ax.set_xticklabels([f'P{i}' for i in top_idx], fontsize=7)
        ax.set_title('Deviation Radar\n(Top 20 Prototypes)', pad=20)
        ax.legend()

    # Right: Bar chart of top 20 prototype deviation
    ax = axes[1]
    ax.barh(range(20), np.abs(dev_per_proto[top_idx]),
            color=[C_PAT if dev_per_proto[i] > 0 else C_CTRL
                   for i in top_idx],
            alpha=0.8, edgecolor='white', height=0.7)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f'Proto #{i}' for i in top_idx], fontsize=8)
    ax.set_xlabel('|Mean Deviation|')
    ax.set_title('Top 20 Prototypes by Deviation')
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_deviation_radar.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig_deviation_radar.png saved", flush=True)


def plot_deviation_heatmap(X, unsup_results, labels, has_labels, n_top=50):
    """Figure 3: Heatmap of top deviating prototypes across samples."""
    top_idx = unsup_results['top_dev_idx'][:n_top]
    dev_vectors = unsup_results['dev_vectors']

    # Extract submatrix
    sub = dev_vectors[:, top_idx]

    # Sort samples: by deviation magnitude
    sorted_idx = np.argsort(unsup_results['dev_magnitude'])
    sub_sorted = sub[sorted_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                             gridspec_kw={'width_ratios': [50, 1]})

    # Heatmap
    ax = axes[0]
    vmax = np.percentile(np.abs(sub_sorted), 95)
    im = ax.imshow(sub_sorted, aspect='auto', cmap=DEV_CMAP,
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Top Deviating Prototypes (ranked)')
    ax.set_ylabel('Samples (sorted by deviation)')
    ax.set_title(f'Deviation Heatmap (Top {n_top} Prototypes)')
    plt.colorbar(im, ax=ax, label='Deviation', shrink=0.6)

    # Add label bar on right
    ax2 = axes[1]
    if has_labels:
        n_samples = len(sorted_idx)
        for i, idx in enumerate(sorted_idx):
            ax2.add_patch(plt.Rectangle((0, i), 1, 1,
                                       facecolor=C_CTRL if labels[idx] == 0 else C_PAT,
                                       edgecolor='none'))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(n_samples, 0)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Label')
    else:
        ax2.set_visible(False)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_deviation_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig_deviation_heatmap.png saved", flush=True)


def plot_diversity(unsup_results, labels, has_labels):
    """Figure 4: Diversity indices."""
    div = unsup_results['diversity']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('richness', 'Richness (unique prototypes)', axes[0, 0]),
        ('shannon', 'Shannon Entropy', axes[0, 1]),
        ('pielou', 'Pielou Evenness', axes[1, 0]),
        ('chao1', 'Chao1 Richness', axes[1, 1]),
    ]

    for col, name, ax in metrics:
        if has_labels:
            for lv, color, label_name in [(0, C_CTRL, 'Control'), (1, C_PAT, 'Patient')]:
                mask = labels == lv
                vals = div[col].values[mask]
                ax.hist(vals, bins=25, color=color, alpha=0.6,
                        label=f'{label_name} (n={mask.sum()})',
                        edgecolor='white')
            # Mann-Whitney U
            stat, p = mannwhitneyu(div[col].values[labels == 0],
                                   div[col].values[labels == 1])
            ax.set_title(f'{name}\n(MW p={p:.2e})')
            ax.legend(fontsize=8)
        else:
            ax.hist(div[col].values, bins=25, color=C_ACCENT,
                    alpha=0.7, edgecolor='white')
            ax.set_title(name)
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel('Count')

    plt.suptitle('Repertoire Diversity Analysis', fontsize=16,
                fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_diversity.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig_diversity.png saved", flush=True)


def plot_supervised_detail(sup_results, labels):
    """Figure 5: Supervised analysis detail (if labels exist)."""
    if not sup_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Left: SVM score distribution
    ax = axes[0]
    for lv, color, name in [(0, C_CTRL, 'Control'), (1, C_PAT, 'Patient')]:
        mask = labels == lv
        ax.hist(sup_results['y_score'][mask], bins=30, color=color, alpha=0.6,
                label=f'{name} (n={mask.sum()})', edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('SVM Decision Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Supervised SVM Score\nAUC={sup_results["auc"]:.4f}')
    ax.legend()

    # Middle: Top prototype coefficients
    ax = axes[1]
    coef = sup_results['svm_coef']
    top_pos = sup_results['top_pos'][:15]
    top_neg = sup_results['top_neg'][:15]
    combined = np.concatenate([top_pos[::-1], top_neg])
    vals = coef[combined]
    colors = [C_PAT if v > 0 else C_CTRL for v in vals]

    ax.barh(range(len(combined)), vals, color=colors, alpha=0.8,
            edgecolor='white', height=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels([f'P{i}' for i in combined], fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('SVM Coefficient')
    ax.set_title('Top Differential Prototypes\n(Red=Patient, Blue=Control)')
    ax.invert_yaxis()

    # Right: Confusion matrix
    ax = axes[2]
    tp = np.sum((sup_results['y_pred'] == 1) & (labels == 1))
    fp = np.sum((sup_results['y_pred'] == 1) & (labels == 0))
    fn = np.sum((sup_results['y_pred'] == 0) & (labels == 1))
    tn = np.sum((sup_results['y_pred'] == 0) & (labels == 0))
    cm = np.array([[tn, fp], [fn, tp]])

    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Patient'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Control', 'Patient'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix\nAcc={sup_results["accuracy"]:.4f}')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_supervised.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig_supervised.png saved", flush=True)


def plot_sample_report(unsup_results, labels, has_labels):
    """Figure 6: Per-sample summary — ranked anomaly with tier coloring."""
    composite = unsup_results['composite_score']
    tier = unsup_results['tier']
    dev = unsup_results['dev_magnitude']
    ranked = unsup_results['ranked_samples']

    n = len(ranked)
    fig, ax = plt.subplots(figsize=(16, 8))

    # Sort by composite score (descending)
    y = np.arange(n)
    colors = [C_PAT if tier[i] == 2 else C_ORANGE if tier[i] == 1 else C_CTRL
              for i in range(n)]

    # Need to sort in ascending for display (least anomalous at bottom)
    sort_idx = np.argsort(composite)
    ax.barh(y, composite[sort_idx], color=[colors[i] for i in range(n)],
            alpha=0.7, edgecolor='white', height=0.8)

    # Add label markers if available
    if has_labels:
        for i, idx in enumerate(sort_idx):
            if labels[idx] == 1:
                ax.plot(composite[idx], i, '>', color=C_PAT, markersize=4)

    ax.set_yticks(y[::max(1, n // 30)])
    ax.set_yticklabels([f'S{sort_idx[i]}' for i in range(0, n, max(1, n // 30))],
                       fontsize=7)
    ax.set_xlabel('Composite Anomaly Score')
    ax.set_title('Sample Ranking by Anomaly Score')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_PAT, label='High anomaly (top 10%)'),
        Patch(facecolor=C_ORANGE, label='Moderate (top 25%)'),
        Patch(facecolor=C_CTRL, label='Normal'),
    ]
    if has_labels:
        legend_elements.append(plt.Line2D([0], [0], marker='>',
                                          color='w', markerfacecolor=C_PAT,
                                          markersize=8, label='Patient'))
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_sample_ranking.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig_sample_ranking.png saved", flush=True)


# =========================================================================
# HTML Report
# =========================================================================
def generate_html_report(sup_results, unsup_results, X, labels, has_labels,
                         output_path):
    """Generate comprehensive HTML report."""
    figures = [
        ('fig_dashboard.png', 'CDRscope Dashboard',
         '''<p>综合仪表板：<br>
         左上：PCA 按偏离度着色；中上：UMAP 按异常评分着色；右上：PCA 按异常等级着色<br>
         左下：综合评分分布（蓝=对照，红=患者）；中下：ROC 曲线对比；右下：方法 AUC 对比</p>'''),
        ('fig_deviation_radar.png', 'Deviation Direction Analysis',
         f'''<p>左：雷达图展示对照组与患者组在 Top 20 偏离原型上的平均偏离方向。<br>
         右：Top 20 原型的平均偏离量柱状图，红色=患者富集，蓝色=对照富集。<br>
         偏离方向告诉我们"哪些原型差异最大"，而偏离度告诉我们"差异有多大"。</p>'''),
        ('fig_deviation_heatmap.png', 'Deviation Heatmap',
         '''<p>每个样本在 Top 50 偏离原型上的偏差向量热图。
         样本按偏离度排序（从低到高），右侧色条标注真实标签。
         可以看到偏离度高的样本（上方）主要集中在患者组。</p>'''),
        ('fig_diversity.png', 'Repertoire Diversity',
         '''<p>四项多样性指标：Richness（独特原型数）、Shannon 熵、Pielou 均匀度、Chao1 估计。
         如果对照组与患者组在多样性上有显著差异，说明疾病影响了 TCR 库的整体结构。</p>'''),
        ('fig_sample_ranking.png', 'Sample Anomaly Ranking',
         '''<p>所有样本按综合异常评分排序。红色=高异常（top 10%），
         橙色=中异常（top 25%），蓝色=正常。
         ▶ 标记为真实患者样本（后验验证）。
         无监督方法可以在没有标签的情况下识别异常样本。</p>'''),
    ]

    if has_labels and sup_results:
        figures.append(
            ('fig_supervised.png', 'Supervised Analysis Detail',
             f'''<p>有监督路线结果：<br>
             左：SVM 决策分数分布（AUC={sup_results["auc"]:.4f}）<br>
             中：Top 差异原型（SVM 系数最大的原型）<br>
             右：混淆矩阵（准确率={sup_results["accuracy"]:.4f}）</p>''')
        )

    # Summary table
    sup_auc = sup_results['auc'] if sup_results else None
    rows = []
    rows.append(('Route A: Supervised SVM', f'AUC={sup_auc:.4f}' if sup_auc else 'N/A'))
    rows.append(('Route B: Deviation Score',
                 f'AUC={unsup_results.get("auc_dev", "N/A")}' if has_labels else 'computed'))
    rows.append(('Route B: One-Class SVM',
                 f'AUC={unsup_results.get("auc_ocsvm", "N/A")}' if has_labels else 'computed'))
    rows.append(('Route B: OCSVM (2000 groups)',
                 f'AUC={unsup_results.get("auc_ocsvm_2000", "N/A")}' if has_labels else 'computed'))
    rows.append(('Route B: LOF',
                 f'AUC={unsup_results.get("auc_lof", "N/A")}' if has_labels else 'computed'))
    rows.append(('Route B: JS Divergence',
                 f'AUC={unsup_results.get("auc_js", "N/A")}' if has_labels else 'computed'))
    rows.append(('Route B: Composite Score',
                 f'AUC={unsup_results.get("auc_composite", "N/A")}' if has_labels else 'computed'))

    tier = unsup_results['tier']
    tier_info = f"High={np.sum(tier==2)}, Moderate={np.sum(tier==1)}, Normal={np.sum(tier==0)}"

    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>CDRscope v2.0 — Unified Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #1a1a1a; }}
h1 {{ color: #1a1a1a; border-bottom: 3px solid #5e5ce6; padding-bottom: 10px; }}
h2 {{ color: #5e5ce6; margin-top: 40px; }}
.figure {{ background: white; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.figure img {{ width: 100%; border-radius: 8px; }}
.figure p {{ color: #555; line-height: 1.6; font-size: 14px; }}
.hero {{ background: linear-gradient(135deg, #5e5ce6 0%, #4a90d9 100%); color: white; border-radius: 16px; padding: 28px; margin: 20px 0; }}
.hero h1 {{ color: white; border: none; margin: 0; }}
.hero p {{ color: rgba(255,255,255,0.9); font-size: 16px; line-height: 1.6; }}
.summary {{ background: white; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.summary table {{ width: 100%; border-collapse: collapse; }}
.summary th, .summary td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
.summary th {{ background: #f5f5f7; font-weight: 600; }}
.method-box {{ background: white; border-left: 4px solid #5e5ce6; border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0; }}
.route-tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.route-a {{ background: #ffe0e0; color: #cc0000; }}
.route-b {{ background: #e0e8ff; color: #4a90d9; }}
</style>
</head><body>

<div class="hero">
<h1>CDRscope v2.0 — Unified Analysis Report</h1>
<p>CDRscope v2.0 统一流水线：根据输入样本是否携带标签，自动选择分析路线。
所有样本均进行无监督分析（Route B），计算与 CordBlood 参考集的偏离度和偏离方向。
如携带标签，额外执行有监督分类（Route A）。</p>
<p><b>样本数</b>：{len(labels)} | <b>有标签</b>：{"是" if has_labels else "否"} |
<b>异常等级</b>：{tier_info}</p>
</div>

<div class="summary">
<h2>Results Summary</h2>
<table>
<tr><th>Route</th><th>Method</th><th>Performance</th></tr>
'''

    for method, perf in rows:
        html += f'<tr><td>{"A" if "Supervised" in method else "B"}</td><td>{method}</td><td>{perf}</td></tr>\n'

    html += '''</table>
</div>

<div class="method-box">
<h3><span class="route-tag route-a">Route A</span> Supervised Analysis</h3>
<p>当样本携带标签（健康/疾病）时，运行 Linear SVM 5-fold 交叉验证分类。
识别差异原型（SVM 系数最大的原型），输出 AUC、F1、敏感度、特异度。</p>
</div>

<div class="method-box">
<h3><span class="route-tag route-b">Route B</span> Unsupervised Analysis</h3>
<p>所有样本均执行，不依赖标签。核心输出：<br>
<b>1. 偏离度（Deviation Magnitude）</b>：每个样本与 CordBlood 参考集均值的 L2 距离。<br>
<b>2. 偏离方向（Deviation Direction）</b>：每个原型上的偏差向量，指出哪些 CDR3 模式偏离最多。<br>
<b>3. 异常评分（Anomaly Score）</b>：One-Class SVM + LOF + JS 散度的综合评分。<br>
<b>4. 多样性指标</b>：Richness、Shannon、Simpson、Pielou、Chao1。<br>
<b>5. 异常等级</b>：Normal / Moderate / High（基于综合评分排名）。</p>
</div>
'''

    for img_name, title, desc in figures:
        img_path = os.path.join(IMG_DIR, img_name)
        if os.path.exists(img_path):
            b64 = img_to_b64(img_path)
            html += f'''
<div class="figure">
<h2>{title}</h2>
<img src="data:image/png;base64,{b64}" alt="{title}">
{desc}
</div>
'''

    # Key findings
    dev_top = unsup_results['top_dev_idx'][:10]
    html += f'''
<div class="method-box">
<h3>Key Quantitative Outputs</h3>
<p><b>Top 10 Deviating Prototypes</b>（偏离最大的原型，按平均偏差排序）：<br>
{", ".join([f"#{i}" for i in dev_top])}</p>
<p><b>Deviation Magnitude Statistics</b>：<br>
Mean = {unsup_results["dev_magnitude"].mean():.4f}, 
Std = {unsup_results["dev_magnitude"].std():.4f}, 
Range = [{unsup_results["dev_magnitude"].min():.4f}, {unsup_results["dev_magnitude"].max():.4f}]</p>
'''

    if has_labels and sup_results:
        html += f'''<p><b>Supervised Route</b>：<br>
AUC = {sup_results["auc"]:.4f}, 
Accuracy = {sup_results["accuracy"]:.4f}, 
F1 = {sup_results["f1"]:.4f}, 
Sensitivity = {sup_results["sensitivity"]:.4f}, 
Specificity = {sup_results["specificity"]:.4f}</p>'''

    html += '''
</div>

<div class="method-box">
<h3>方法论说明</h3>
<ol>
<li><b>双路线设计</b>：Route B（无监督）始终运行，Route A（有监督）仅在有标签时运行。
    两条路线互补：有监督找差异点，无监督量化偏离。</li>
<li><b>参考集锚定</b>：以 CordBlood TRA 参考面板（m=10,000 原型）作为健康基线，
    所有偏离度均相对于此参考计算。</li>
<li><b>偏离度 + 偏离方向</b>：偏离度（标量）衡量"有多远"，
    偏离方向（向量）指出"在哪里偏离"。两者结合提供了完整的定量描述。</li>
<li><b>综合评分</b>：融合偏离度、OCSVM、LOF、JS 散度的 z-score 均值，
    提供比单一方法更稳健的异常评估。</li>
<li><b>可视化</b>：仪表板（PCA/UMAP + 着色）、雷达图（偏离方向）、
    热图（样本×原型偏差）、多样性分布、样本排名。</li>
</ol>
</div>

</body></html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\n  HTML report: {output_path}", flush=True)


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 60, flush=True)
    print("CDRscope v2.0 — Unified Pipeline", flush=True)
    print("=" * 60, flush=True)

    # Load data
    X, X_raw, labels = load_data()
    has_labels = labels is not None and len(set(labels)) > 1
    print(f"  Mode: {'Supervised + Unsupervised' if has_labels else 'Unsupervised only'}",
          flush=True)

    # Route A: Supervised (if labels)
    sup_results = None
    if has_labels:
        sup_results = route_a_supervised(X, labels)

    # Route B: Unsupervised (always)
    unsup_results = route_b_unsupervised(X, X_raw, labels if has_labels else None)

    # Visualization
    print("\n" + "=" * 60, flush=True)
    print("Generating Visualizations...", flush=True)
    print("=" * 60, flush=True)

    plot_dashboard(sup_results, unsup_results, X, labels, has_labels)
    plot_deviation_radar(unsup_results, labels, has_labels)
    plot_deviation_heatmap(X, unsup_results, labels, has_labels)
    plot_diversity(unsup_results, labels, has_labels)
    plot_sample_report(unsup_results, labels, has_labels)
    if has_labels and sup_results:
        plot_supervised_detail(sup_results, labels)

    # HTML Report
    report_path = os.path.join(OUTPUT_DIR, "cdrscope_v2_report.html")
    generate_html_report(sup_results, unsup_results, X, labels, has_labels,
                         report_path)

    # Save results JSON
    results_json = {
        'mode': 'supervised+unsupervised' if has_labels else 'unsupervised',
        'n_samples': len(labels),
        'n_prototypes': X.shape[1],
        'route_a_supervised': {
            'auc': float(sup_results['auc']) if sup_results else None,
            'accuracy': float(sup_results['accuracy']) if sup_results else None,
            'f1': float(sup_results['f1']) if sup_results else None,
            'sensitivity': float(sup_results['sensitivity']) if sup_results else None,
            'specificity': float(sup_results['specificity']) if sup_results else None,
        } if sup_results else None,
        'route_b_unsupervised': {
            'deviation_magnitude': {
                'mean': float(unsup_results['dev_magnitude'].mean()),
                'std': float(unsup_results['dev_magnitude'].std()),
                'min': float(unsup_results['dev_magnitude'].min()),
                'max': float(unsup_results['dev_magnitude'].max()),
            },
            'auc_deviation': float(unsup_results.get('auc_dev', 0)) if has_labels else None,
            'auc_ocsvm': float(unsup_results.get('auc_ocsvm', 0)) if has_labels else None,
            'auc_ocsvm_2000': float(unsup_results.get('auc_ocsvm_2000', 0)) if has_labels else None,
            'auc_lof': float(unsup_results.get('auc_lof', 0)) if has_labels else None,
            'auc_js': float(unsup_results.get('auc_js', 0)) if has_labels else None,
            'auc_composite': float(unsup_results.get('auc_composite', 0)) if has_labels else None,
            'top_dev_prototypes': unsup_results['top_dev_idx'][:20].tolist(),
            'tier_counts': {
                'high': int(np.sum(unsup_results['tier'] == 2)),
                'moderate': int(np.sum(unsup_results['tier'] == 1)),
                'normal': int(np.sum(unsup_results['tier'] == 0)),
            },
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "cdrscope_v2_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results JSON: {json_path}", flush=True)

    # Save sample-level results CSV
    sample_df = unsup_results['diversity'].copy()
    sample_df['deviation_magnitude'] = unsup_results['dev_magnitude']
    sample_df['anomaly_score'] = unsup_results['anomaly_score']
    sample_df['lof_score'] = unsup_results['lof_score']
    sample_df['js_deviation'] = unsup_results['js_dev']
    sample_df['composite_score'] = unsup_results['composite_score']
    sample_df['tier'] = unsup_results['tier']
    if has_labels:
        sample_df['label'] = labels
        sample_df['label_name'] = ['Control' if l == 0 else 'Patient' for l in labels]
    csv_path = os.path.join(OUTPUT_DIR, "sample_results.csv")
    sample_df.to_csv(csv_path, index=False)
    print(f"  Sample CSV: {csv_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("CDRscope v2.0 — Complete", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
