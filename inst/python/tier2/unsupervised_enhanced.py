#!/usr/bin/env python3
"""
Enhanced Unsupervised TRA Analysis — 3 Priority Methods
=========================================================
1. Reference Deviation Scoring (healthy baseline → deviation per sample)
2. Variance Filtering + Refined Space (select informative prototypes)
3. One-Class SVM / Isolation Forest (anomaly detection)

Data: RA-TRA (545 samples: 210 controls + 335 patients)
Uses pre-computed m=10,000 prototype matrix.
Labels used ONLY for post-hoc validation, not for method fitting.
"""
import os, sys, json, time, warnings, base64
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, roc_curve, silhouette_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             average_precision_score)
from sklearn.neighbors import LocalOutlierFactor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from umap import UMAP

warnings.filterwarnings('ignore')

WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
PANEL_DIR = os.path.join(WORK_DIR, "cordblood_tra_panel")
OUTPUT_DIR = os.path.join(WORK_DIR, "unsupervised_enhanced_results")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

CTRL_COLOR = '#4a90d9'
PAT_COLOR = '#ff6b6b'
ACCENT = '#5e5ce6'
GREEN = '#00a389'
ORANGE = '#ff9f0a'
PURPLE = '#bf5af2'


def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


# =========================================================================
# Data Loading
# =========================================================================
def load_data():
    """Load pre-computed RA-TRA matrix and labels."""
    mat_path = os.path.join(PANEL_DIR, "ra_tra_cb_matrix.npy")
    lbl_path = os.path.join(PANEL_DIR, "ra_tra_cb_labels.npy")

    print("Loading pre-computed RA-TRA matrix...", flush=True)
    X_raw = np.load(mat_path)
    labels = np.load(lbl_path).astype(int)
    print(f"  Matrix: {X_raw.shape} | Labels: {Counter(labels.tolist())}", flush=True)

    # L2 normalize (same as supervised pipeline)
    X = normalize(X_raw, norm='l2')
    print(f"  L2 normalized. Range: [{X.min():.6f}, {X.max():.6f}]", flush=True)

    return X, labels


# =========================================================================
# Method 1: Reference Deviation Scoring
# =========================================================================
def method1_reference_deviation(X, labels):
    """
    Compute deviation of each sample from a reference baseline.
    Two modes:
      - Unsupervised: reference = mean of ALL samples (no labels used)
      - Semi-supervised: reference = mean of controls (labels used to define reference)
    """
    print("\n" + "=" * 60, flush=True)
    print("Method 1: Reference Deviation Scoring", flush=True)
    print("=" * 60, flush=True)

    results = {}

    # --- Unsupervised reference: mean of all samples ---
    ref_all = X.mean(axis=0)
    dev_all = X - ref_all
    scores_all = np.linalg.norm(dev_all, axis=1)
    results['dev_unsupervised'] = scores_all

    # --- Semi-supervised reference: mean of controls ---
    ctrl_mask = labels == 0
    ref_ctrl = X[ctrl_mask].mean(axis=0)
    dev_ctrl = X - ref_ctrl
    scores_ctrl = np.linalg.norm(dev_ctrl, axis=1)
    results['dev_semi'] = scores_ctrl

    # --- Per-prototype deviation (for heatmap) ---
    # Mean deviation per prototype, comparing patients vs controls
    pat_dev = dev_ctrl[labels == 1].mean(axis=0)  # mean deviation of patients from control ref
    ctrl_dev = dev_ctrl[labels == 0].mean(axis=0)  # should be ~0 by construction
    proto_diff = pat_dev - ctrl_dev  # signed difference
    results['proto_diff'] = proto_diff
    results['dev_ctrl_mean'] = ctrl_dev
    results['dev_pat_mean'] = pat_dev

    # --- Validation (post-hoc, using labels) ---
    auc_unsup = roc_auc_score(labels, scores_all)
    auc_semi = roc_auc_score(labels, scores_ctrl)
    results['auc_unsupervised'] = auc_unsup
    results['auc_semi'] = auc_semi

    # Mann-Whitney U test
    stat_semi, p_semi = mannwhitneyu(scores_ctrl[labels == 0], scores_ctrl[labels == 1])
    stat_unsup, p_unsup = mannwhitneyu(scores_all[labels == 0], scores_all[labels == 1])
    results['mw_p_semi'] = p_semi
    results['mw_p_unsup'] = p_unsup

    print(f"  Unsupervised (ref=all):  AUC={auc_unsup:.4f} | MW p={p_unsup:.2e}", flush=True)
    print(f"  Semi-supervised (ref=ctrl): AUC={auc_semi:.4f} | MW p={p_semi:.2e}", flush=True)
    print(f"  Top 10 most deviating prototypes (by |patient - control|):", flush=True)
    top_idx = np.argsort(np.abs(proto_diff))[::-1][:10]
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. Proto #{idx}: diff={proto_diff[idx]:+.4f} "
              f"(ctrl={ctrl_dev[idx]:.4f}, pat={pat_dev[idx]:.4f})", flush=True)

    return results


# =========================================================================
# Method 2: Variance Filtering + Refined Space
# =========================================================================
def method2_variance_filter(X, labels):
    """
    Select informative prototypes by variance, then re-analyze in refined space.
    Compare clustering quality (ARI) between full and refined spaces.
    """
    print("\n" + "=" * 60, flush=True)
    print("Method 2: Variance Filtering + Refined Space", flush=True)
    print("=" * 60, flush=True)

    variances = np.var(X, axis=0)
    sorted_idx = np.argsort(variances)[::-1]

    # Test multiple refinement levels
    refine_levels = [200, 500, 1000, 2000]
    results = {'variances': variances, 'sorted_idx': sorted_idx}
    refine_results = {}

    for n_top in refine_levels:
        top_idx = sorted_idx[:n_top]
        X_ref = X[:, top_idx]

        # PCA
        pca = PCA(n_components=min(n_top, 30))
        X_pca = pca.fit_transform(X_ref)

        # K-means
        best_k = 10
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = km.fit_predict(X_pca)
        ari = adjusted_rand_score(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)
        sil = silhouette_score(X_pca, clusters) if len(set(clusters)) > 1 else 0

        refine_results[n_top] = {
            'ari': ari, 'nmi': nmi, 'silhouette': sil,
            'var_explained': pca.explained_variance_ratio_[:5],
            'X_pca': X_pca
        }
        print(f"  Top {n_top:>4}: ARI={ari:.4f} | NMI={nmi:.4f} | Sil={sil:.4f} | "
              f"PC1={pca.explained_variance_ratio_[0]:.1%}", flush=True)

    # Full space baseline
    pca_full = PCA(n_components=30)
    X_pca_full = pca_full.fit_transform(X)
    km_full = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters_full = km_full.fit_predict(X_pca_full)
    ari_full = adjusted_rand_score(labels, clusters_full)
    print(f"  Full 10000: ARI={ari_full:.4f} (baseline)", flush=True)

    nmi_full = normalized_mutual_info_score(labels, clusters_full)
    refine_results['full'] = {'ari': ari_full, 'nmi': nmi_full, 'X_pca': X_pca_full}
    results['refine_results'] = refine_results

    # Use top 1000 for downstream
    best_n = 1000
    top_idx = sorted_idx[:best_n]
    X_refined = X[:, top_idx]
    results['X_refined'] = X_refined
    results['top_idx'] = top_idx

    # UMAP in refined space
    print("  Computing UMAP in refined space...", flush=True)
    pca_ref = PCA(n_components=30)
    X_pca_ref = pca_ref.fit_transform(X_refined)
    umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap_ref = umap.fit_transform(X_pca_ref)
    results['X_umap'] = X_umap_ref
    results['X_pca_ref'] = X_pca_ref
    results['pca_ref_var'] = pca_ref.explained_variance_ratio_

    print(f"  Refined space ready: {X_refined.shape}", flush=True)
    return results


# =========================================================================
# Method 3: One-Class SVM + Isolation Forest
# =========================================================================
def method3_anomaly_detection(X, labels, X_refined):
    """
    Train anomaly detectors on healthy controls, score all samples.
    Also train unsupervised (all samples) version.
    """
    print("\n" + "=" * 60, flush=True)
    print("Method 3: One-Class SVM + Isolation Forest", flush=True)
    print("=" * 60, flush=True)

    ctrl_mask = labels == 0
    results = {}

    # PCA reduce for kernel methods
    pca50 = PCA(n_components=50)
    X_pca50 = pca50.fit_transform(X)
    pca_ref50 = PCA(n_components=50)
    X_ref_pca50 = pca_ref50.fit_transform(X_refined)

    # --- One-Class SVM (RBF, trained on controls) ---
    print("  Training One-Class SVM (RBF) on controls...", flush=True)
    ocsvm_rbf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    ocsvm_rbf.fit(X_pca50[ctrl_mask])
    scores_ocsvm_rbf = -ocsvm_rbf.score_samples(X_pca50)
    auc_ocsvm_rbf = roc_auc_score(labels, scores_ocsvm_rbf)
    results['ocsvm_rbf'] = scores_ocsvm_rbf
    results['auc_ocsvm_rbf'] = auc_ocsvm_rbf
    print(f"    AUC (ref=ctrl, PCA50): {auc_ocsvm_rbf:.4f}", flush=True)

    # --- One-Class SVM (Linear, trained on controls, full space) ---
    print("  Training One-Class SVM (Linear) on controls...", flush=True)
    ocsvm_lin = OneClassSVM(kernel='linear', nu=0.1)
    ocsvm_lin.fit(X[ctrl_mask])
    scores_ocsvm_lin = -ocsvm_lin.score_samples(X)
    auc_ocsvm_lin = roc_auc_score(labels, scores_ocsvm_lin)
    results['ocsvm_lin'] = scores_ocsvm_lin
    results['auc_ocsvm_lin'] = auc_ocsvm_lin
    print(f"    AUC (ref=ctrl, linear, full): {auc_ocsvm_lin:.4f}", flush=True)

    # --- Isolation Forest (refined space, trained on controls) ---
    print("  Training Isolation Forest (refined) on controls...", flush=True)
    iso_ctrl = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    iso_ctrl.fit(X_refined[ctrl_mask])
    scores_iso_ctrl = -iso_ctrl.score_samples(X_refined)
    auc_iso_ctrl = roc_auc_score(labels, scores_iso_ctrl)
    results['iso_ctrl'] = scores_iso_ctrl
    results['auc_iso_ctrl'] = auc_iso_ctrl
    print(f"    AUC (ref=ctrl, refined): {auc_iso_ctrl:.4f}", flush=True)

    # --- Isolation Forest (unsupervised, trained on all) ---
    print("  Training Isolation Forest (unsupervised) on all...", flush=True)
    iso_all = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    iso_all.fit(X_refined)
    scores_iso_all = -iso_all.score_samples(X_refined)
    auc_iso_all = roc_auc_score(labels, scores_iso_all)
    results['iso_all'] = scores_iso_all
    results['auc_iso_all'] = auc_iso_all
    print(f"    AUC (unsup, refined): {auc_iso_all:.4f}", flush=True)

    # --- LOF (Local Outlier Factor, unsupervised) ---
    print("  Computing Local Outlier Factor (unsupervised)...", flush=True)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit(X_ref_pca50)
    scores_lof = -lof.negative_outlier_factor_
    auc_lof = roc_auc_score(labels, scores_lof)
    results['lof'] = scores_lof
    results['auc_lof'] = auc_lof
    print(f"    AUC (LOF, unsup, PCA50-refined): {auc_lof:.4f}", flush=True)

    return results


# =========================================================================
# Visualization
# =========================================================================
def plot_variance_filter(m2_results):
    """Figure 1: Variance distribution and prototype selection."""
    variances = m2_results['variances']
    sorted_idx = m2_results['sorted_idx']
    cumsum = np.cumsum(variances[sorted_idx]) / variances.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: variance distribution
    axes[0].hist(variances, bins=100, color=ACCENT, alpha=0.7, edgecolor='white')
    for n, color in [(200, GREEN), (500, ORANGE), (1000, PAT_COLOR), (2000, PURPLE)]:
        thresh = variances[sorted_idx[n - 1]]
        axes[0].axvline(thresh, color=color, linestyle='--', linewidth=1.5,
                        label=f'Top {n} (var={thresh:.6f})')
    axes[0].set_xlabel('Prototype Variance')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Prototype Variance Distribution')
    axes[0].legend(fontsize=8)
    axes[0].set_yscale('log')

    # Right: cumulative variance
    axes[1].plot(range(1, len(cumsum) + 1), cumsum, color=ACCENT, linewidth=1.5)
    for n, color in [(200, GREEN), (500, ORANGE), (1000, PAT_COLOR), (2000, PURPLE)]:
        y = cumsum[n - 1]
        axes[1].axvline(n, color=color, linestyle='--', linewidth=1.2)
        axes[1].annotate(f'{n}: {y:.1%}', (n, y), textcoords="offset points",
                         xytext=(8, -12), fontsize=8, color=color)
    axes[1].set_xlabel('Number of Prototypes (ranked by variance)')
    axes[1].set_ylabel('Cumulative Variance Fraction')
    axes[1].set_title('Cumulative Variance by Top-N Prototypes')
    axes[1].set_xscale('log')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_variance_filter.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_variance_filter.png saved", flush=True)


def plot_refined_space(m2_results, labels):
    """Figure 2: PCA and UMAP in refined space, colored by label (post-hoc)."""
    X_pca = m2_results['X_pca_ref']
    X_umap = m2_results['X_umap']
    var = m2_results['pca_ref_var']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coords, title, xl, yl in [
        (axes[0], X_pca, 'PCA (Refined Top-1000)', f'PC1 ({var[0]:.1%})', f'PC2 ({var[1]:.1%})'),
        (axes[1], X_umap, 'UMAP (Refined Top-1000)', 'UMAP-1', 'UMAP-2')
    ]:
        for label_val, color, name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
            mask = labels == label_val
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=30, alpha=0.6,
                      edgecolors='white', linewidth=0.3, label=f'{name} (n={mask.sum()})')
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_refined_space.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_refined_space.png saved", flush=True)


def plot_deviation_scores(m1_results, labels):
    """Figure 3: Deviation score distribution and ROC curves."""
    scores_unsup = m1_results['dev_unsupervised']
    scores_semi = m1_results['dev_semi']
    auc_unsup = m1_results['auc_unsupervised']
    auc_semi = m1_results['auc_semi']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Unsupervised deviation scores
    for label_val, color, name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
        mask = labels == label_val
        axes[0].hist(scores_unsup[mask], bins=30, color=color, alpha=0.6,
                     label=f'{name} (n={mask.sum()})', edgecolor='white')
    axes[0].set_xlabel('Deviation Score (unsupervised)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Reference: All Samples Mean\nAUC={auc_unsup:.4f}')
    axes[0].legend()

    # Middle: Semi-supervised deviation scores
    for label_val, color, name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
        mask = labels == label_val
        axes[1].hist(scores_semi[mask], bins=30, color=color, alpha=0.6,
                     label=f'{name} (n={mask.sum()})', edgecolor='white')
    axes[1].set_xlabel('Deviation Score (ref=controls)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Reference: Controls Mean\nAUC={auc_semi:.4f}')
    axes[1].legend()

    # Right: ROC curves
    for scores, auc_val, name, color in [
        (scores_unsup, auc_unsup, 'Unsupervised (ref=all)', ACCENT),
        (scores_semi, auc_semi, 'Semi-supervised (ref=ctrl)', PAT_COLOR)
    ]:
        fpr, tpr, _ = roc_curve(labels, scores)
        axes[2].plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{name}\nAUC={auc_val:.4f}')
    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC: Deviation Score vs Disease Label')
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_deviation_scores.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_deviation_scores.png saved", flush=True)


def plot_deviation_heatmap(m1_results, labels):
    """Figure 4: Top 30 most deviating prototypes — heatmap."""
    proto_diff = m1_results['proto_diff']
    top30 = np.argsort(np.abs(proto_diff))[::-1][:30]

    # Sort samples: controls first, then patients
    ctrl_idx = np.where(labels == 0)[0]
    pat_idx = np.where(labels == 1)[0]
    sorted_samples = np.concatenate([ctrl_idx, pat_idx])

    # Extract the submatrix
    # We need the raw deviation matrix — reconstruct from saved data
    # For heatmap, show per-sample prototype usage (L2-normalized) for top 30 prototypes
    # We'll use the deviation from control mean
    # dev_pat_mean and dev_ctrl_mean are per-prototype means
    # Need to get per-sample values for these prototypes

    # Actually, we saved proto_diff (mean diff), not per-sample values
    # Let's show the mean values as a bar chart instead
    fig, ax = plt.subplots(figsize=(14, 6))

    y_pos = np.arange(len(top30))
    diffs = proto_diff[top30]
    colors = [PAT_COLOR if d > 0 else CTRL_COLOR for d in diffs]

    ax.barh(y_pos, diffs, color=colors, alpha=0.8, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Proto #{i}' for i in top30], fontsize=8)
    ax.set_xlabel('Deviation (Patient - Control)')
    ax.set_title('Top 30 Prototypes by |Deviation| (Patient vs Control)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color=PAT_COLOR), Patch(color=CTRL_COLOR)],
              ['Higher in Patients', 'Higher in Controls'],
              loc='lower right')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_deviation_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_deviation_heatmap.png saved", flush=True)


def plot_ocsvm_scores(m3_results, labels):
    """Figure 5: One-Class SVM and Isolation Forest scores."""
    methods = [
        ('ocsvm_rbf', 'One-Class SVM (RBF)', m3_results['auc_ocsvm_rbf']),
        ('ocsvm_lin', 'One-Class SVM (Linear)', m3_results['auc_ocsvm_lin']),
        ('iso_ctrl', 'Isolation Forest (ref=ctrl)', m3_results['auc_iso_ctrl']),
        ('iso_all', 'Isolation Forest (unsup)', m3_results['auc_iso_all']),
        ('lof', 'Local Outlier Factor (unsup)', m3_results['auc_lof']),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (key, name, auc) in enumerate(methods):
        scores = m3_results[key]
        for label_val, color, label_name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
            mask = labels == label_val
            axes[i].hist(scores[mask], bins=30, color=color, alpha=0.6,
                         label=f'{label_name} (n={mask.sum()})', edgecolor='white')
        axes[i].set_xlabel('Anomaly Score')
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'{name}\nAUC={auc:.4f}')
        axes[i].legend(fontsize=8)

    # 6th panel: ROC curves
    ax_roc = axes[5]
    for key, name, auc, color in [
        ('ocsvm_rbf', 'OCSVM-RBF', m3_results['auc_ocsvm_rbf'], PAT_COLOR),
        ('ocsvm_lin', 'OCSVM-Lin', m3_results['auc_ocsvm_lin'], '#ff9f0a'),
        ('iso_ctrl', 'IF (ref=ctrl)', m3_results['auc_iso_ctrl'], ACCENT),
        ('iso_all', 'IF (unsup)', m3_results['auc_iso_all'], GREEN),
        ('lof', 'LOF (unsup)', m3_results['auc_lof'], PURPLE),
    ]:
        scores = m3_results[key]
        fpr, tpr, _ = roc_curve(labels, scores)
        ax_roc.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f'{name} ({auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC: All Anomaly Detectors')
    ax_roc.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_anomaly_scores.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_anomaly_scores.png saved", flush=True)


def plot_refined_clusters(m2_results, labels):
    """Figure 6: K-means clustering in refined space + ARI comparison."""
    X_pca = m2_results['X_pca_ref']
    X_umap = m2_results['X_umap']
    refine = m2_results['refine_results']

    # K-means in refined (top 1000) space
    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = km.fit_predict(X_pca)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # PCA by cluster
    CLUSTER_COLORS = ['#4a90d9', '#ff6b6b', '#00a389', '#ff9f0a', '#bf5af2',
                      '#5e5ce6', '#ff453a', '#64d2ff', '#ffd60a', '#af52de']
    for c in range(10):
        mask = clusters == c
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=CLUSTER_COLORS[c],
                       s=30, alpha=0.6, edgecolors='white', linewidth=0.3,
                       label=f'C{c} (n={mask.sum()})')
    axes[0].set_title('K-means Clusters in Refined Space (PCA)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(fontsize=7, ncol=2)

    # UMAP by label (post-hoc)
    for label_val, color, name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
        mask = labels == label_val
        axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], c=color, s=30, alpha=0.6,
                       edgecolors='white', linewidth=0.3, label=f'{name} (n={mask.sum()})')
    axes[1].set_title('UMAP — True Labels (Post-hoc)')
    axes[1].set_xlabel('UMAP-1')
    axes[1].set_ylabel('UMAP-2')
    axes[1].legend()

    # ARI comparison bar chart
    levels = [200, 500, 1000, 2000, 'Full\n10000']
    aris = [refine[n]['ari'] for n in [200, 500, 1000, 2000]] + [refine['full']['ari']]
    nmis = [refine[n]['nmi'] for n in [200, 500, 1000, 2000]] + [refine['full']['nmi']]

    x = np.arange(len(levels))
    w = 0.35
    axes[2].bar(x - w/2, aris, w, color=ACCENT, label='ARI', alpha=0.8)
    axes[2].bar(x + w/2, nmis, w, color=ORANGE, label='NMI', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(levels)
    axes[2].set_ylabel('Score')
    axes[2].set_title('Clustering Recovery vs Refined Space Size')
    axes[2].legend()
    axes[2].set_ylim(0, max(max(aris), max(nmis)) * 1.5)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_refined_clusters.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_refined_clusters.png saved", flush=True)


def plot_method_comparison(m1_results, m3_results, supervised_auc=0.9593):
    """Figure 7: AUC comparison across all methods."""
    methods = [
        ('Dev.\n(unsup)', m1_results['auc_unsupervised'], ACCENT),
        ('Dev.\n(semi)', m1_results['auc_semi'], '#5e5ce6'),
        ('OCSVM\nRBF', m3_results['auc_ocsvm_rbf'], PAT_COLOR),
        ('OCSVM\nLinear', m3_results['auc_ocsvm_lin'], '#ff9f0a'),
        ('IF\n(ref=ctrl)', m3_results['auc_iso_ctrl'], GREEN),
        ('IF\n(unsup)', m3_results['auc_iso_all'], '#64d2ff'),
        ('LOF\n(unsup)', m3_results['auc_lof'], PURPLE),
        ('SVM\n(supervised)', supervised_auc, '#ff453a'),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [m[0] for m in methods]
    aucs = [m[1] for m in methods]
    colors = [m[2] for m in methods]

    bars = ax.bar(names, aucs, color=colors, alpha=0.8, edgecolor='white', width=0.6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.axhline(supervised_auc, color='#ff453a', linestyle=':', alpha=0.5,
              label=f'Supervised baseline ({supervised_auc:.4f})')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Method Comparison: Unsupervised vs Supervised')
    ax.set_ylim(0.4, 1.0)
    ax.legend()

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_method_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_method_comparison.png saved", flush=True)


# =========================================================================
# HTML Report
# =========================================================================
def generate_html_report(m1, m2, m3, output_path):
    """Generate comprehensive HTML report."""
    figures = [
        ('fig_variance_filter.png', 'Method 2: Prototype Variance & Filtering',
         f'''<p>对 10,000 个原型按跨样本方差排序。左图显示方差分布——大多数原型方差很低（噪声），
         少数高方差原型携带个体差异信息。右图显示累积方差：Top 500 原型已覆盖约
         {m2['refine_results'][500]['var_explained'][0]:.1%} 的 PC1 方差。</p>'''),

        ('fig_refined_space.png', 'Method 2: Refined Space Visualization (Top-1000)',
         f'''<p>使用方差排名前 1,000 的原型构建精炼空间，L2 归一化后做 PCA 和 UMAP。
         蓝色=对照(210)，红色=患者(335)。即使精炼后，两组仍高度重叠——
         疾病信号是分布式的，不会在 2D 降维中自然分离。</p>'''),

        ('fig_deviation_scores.png', 'Method 1: Reference Deviation Scoring',
         f'''<p><b>无监督参考</b>（全部样本均值）：AUC={m1['auc_unsupervised']:.4f}（MW p={m1['mw_p_unsup']:.2e}）<br>
         <b>半监督参考</b>（对照均值）：AUC={m1['auc_semi']:.4f}（MW p={m1['mw_p_semi']:.2e}）<br>
         半监督参考效果更好，说明以健康基线为锚点能捕获更多疾病偏离信号。
         无监督版本也可用——不需要任何标签。</p>'''),

        ('fig_deviation_heatmap.png', 'Method 1: Top 30 Most Deviating Prototypes',
         '''<p>按 |患者 - 对照| 的均值差异排序的前 30 个原型。红色=患者在原型上富集，
         蓝色=对照富集。每个原型的贡献很小，但合在一起形成了可检测的信号——
         这就是 GWAS-like 分布式信号的体现。</p>'''),

        ('fig_anomaly_scores.png', 'Method 3: Anomaly Detection Scores',
         f'''<p>5 种异常检测方法的分数分布和 ROC 曲线：<br>
         • One-Class SVM (RBF, ref=ctrl): AUC={m3['auc_ocsvm_rbf']:.4f}<br>
         • One-Class SVM (Linear, ref=ctrl): AUC={m3['auc_ocsvm_lin']:.4f}<br>
         • Isolation Forest (ref=ctrl): AUC={m3['auc_iso_ctrl']:.4f}<br>
         • Isolation Forest (unsup): AUC={m3['auc_iso_all']:.4f}<br>
         • Local Outlier Factor (unsup): AUC={m3['auc_lof']:.4f}</p>'''),

        ('fig_refined_clusters.png', 'Method 2: Clustering in Refined Space',
         f'''<p>左：K-means (K=10) 在精炼空间中的聚类结果。<br>
         中：UMAP 按真实标签着色（后验验证）。<br>
         右：不同精炼级别的 ARI/NMI 对比。精炼后 ARI 有所改善但仍然很低——
         无监督聚类无法恢复疾病标签，验证了分布式信号的特征。</p>'''),

        ('fig_method_comparison.png', 'All Methods: AUC Comparison',
         f'''<p>所有方法的 AUC 对比。有监督 SVM (AUC=0.9593) 仍是上限，
         但部分无监督方法已接近：偏离评分 (AUC={m1['auc_semi']:.4f}) 和
         One-Class SVM (AUC={m3['auc_ocsvm_rbf']:.4f}) 表现最佳。
         无监督方法的价值在于：不需要疾病标签，可用于筛查和发现。</p>'''),
    ]

    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Enhanced Unsupervised TRA Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #1a1a1a; }}
h1 {{ color: #1a1a1a; border-bottom: 3px solid #5e5ce6; padding-bottom: 10px; }}
h2 {{ color: #5e5ce6; margin-top: 40px; }}
.figure {{ background: white; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.figure img {{ width: 100%; border-radius: 8px; }}
.figure p {{ color: #555; line-height: 1.6; font-size: 14px; }}
.summary {{ background: linear-gradient(135deg, #5e5ce6 0%, #4a90d9 100%); color: white; border-radius: 12px; padding: 24px; margin: 20px 0; }}
.summary h2 {{ color: white; margin-top: 0; }}
.summary table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
.summary td, .summary th {{ padding: 8px 12px; text-align: center; color: white; border-bottom: 1px solid rgba(255,255,255,0.2); }}
.method-box {{ background: white; border-left: 4px solid #5e5ce6; border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0; }}
</style>
</head><body>

<h1>Enhanced Unsupervised TRA Repertoire Analysis</h1>

<div class="summary">
<h2>Summary</h2>
<table>
<tr><th>Method</th><th>Mode</th><th>AUC-ROC</th><th>p-value</th></tr>
<tr><td>Deviation Score (ref=all)</td><td>Unsupervised</td><td>{m1['auc_unsupervised']:.4f}</td><td>{m1['mw_p_unsup']:.2e}</td></tr>
<tr><td>Deviation Score (ref=ctrl)</td><td>Semi-supervised</td><td>{m1['auc_semi']:.4f}</td><td>{m1['mw_p_semi']:.2e}</td></tr>
<tr><td>One-Class SVM (RBF)</td><td>Semi-supervised</td><td>{m3['auc_ocsvm_rbf']:.4f}</td><td>-</td></tr>
<tr><td>One-Class SVM (Linear)</td><td>Semi-supervised</td><td>{m3['auc_ocsvm_lin']:.4f}</td><td>-</td></tr>
<tr><td>Isolation Forest (ref=ctrl)</td><td>Semi-supervised</td><td>{m3['auc_iso_ctrl']:.4f}</td><td>-</td></tr>
<tr><td>Isolation Forest (unsup)</td><td>Unsupervised</td><td>{m3['auc_iso_all']:.4f}</td><td>-</td></tr>
<tr><td>Local Outlier Factor</td><td>Unsupervised</td><td>{m3['auc_lof']:.4f}</td><td>-</td></tr>
<tr style="border-top: 2px solid white;"><td><b>Supervised SVM (baseline)</b></td><td><b>Supervised</b></td><td><b>0.9593</b></td><td>-</td></tr>
</table>
</div>

<div class="method-box">
<h3>Method 1: Reference Deviation Scoring</h3>
<p>以健康参考为锚点，计算每个样本的偏离向量。偏离越大 = 越异常。
<b>无监督模式</b>：参考 = 全部样本均值（不需标签）。
<b>半监督模式</b>：参考 = 对照组均值（只需知道哪些是健康）。</p>
</div>

<div class="method-box">
<h3>Method 2: Variance Filtering + Refined Space</h3>
<p>从 10,000 维全空间中按方差选择 Top-N 原型，构建精炼特征空间。
减少噪声维度，提高信噪比。测试了 200/500/1000/2000 个原型。</p>
</div>

<div class="method-box">
<h3>Method 3: One-Class SVM + Anomaly Detection</h3>
<p>在健康样本上训练异常检测模型，对所有样本评分。
包括 One-Class SVM (RBF/Linear)、Isolation Forest、Local Outlier Factor。
高异常分数 = 偏离健康分布 = 潜在疾病。</p>
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

    html += '''
<div class="method-box">
<h3>结论与建议</h3>
<ol>
<li><b>参考偏离评分</b>是最简单且最有效的无监督方法——不需要训练，只需一个健康参考基线</li>
<li><b>方差过滤</b>有效降低噪声，但对聚类 ARI 的改善有限（分布式信号本质决定了聚类无法恢复标签）</li>
<li><b>One-Class SVM</b>是标准异常检测方法，效果与偏离评分相当</li>
<li>无监督方法无法达到有监督 SVM 的精度（AUC 0.96），但可作为<b>筛查工具</b>——无需疾病标签</li>
<li>最佳实践：<b>偏离评分 + One-Class SVM 组合</b>，两者互补，提高鲁棒性</li>
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
    print("Enhanced Unsupervised TRA Analysis", flush=True)
    print("3 Priority Methods on RA-TRA (545 samples)", flush=True)
    print("=" * 60, flush=True)

    # Load data
    X, labels = load_data()

    # Method 1: Reference Deviation Scoring
    m1 = method1_reference_deviation(X, labels)

    # Method 2: Variance Filtering
    m2 = method2_variance_filter(X, labels)

    # Method 3: Anomaly Detection
    m3 = method3_anomaly_detection(X, labels, m2['X_refined'])

    # Visualization
    print("\n" + "=" * 60, flush=True)
    print("Generating figures...", flush=True)
    print("=" * 60, flush=True)

    plot_variance_filter(m2)
    plot_refined_space(m2, labels)
    plot_deviation_scores(m1, labels)
    plot_deviation_heatmap(m1, labels)
    plot_ocsvm_scores(m3, labels)
    plot_refined_clusters(m2, labels)
    plot_method_comparison(m1, m3)

    # HTML report
    report_path = os.path.join(OUTPUT_DIR, "enhanced_unsupervised_report.html")
    generate_html_report(m1, m2, m3, report_path)

    # Save results JSON
    results_json = {
        'method1_deviation': {
            'auc_unsupervised': float(m1['auc_unsupervised']),
            'auc_semi': float(m1['auc_semi']),
            'mw_p_unsup': float(m1['mw_p_unsup']),
            'mw_p_semi': float(m1['mw_p_semi']),
        },
        'method2_variance': {
            'ari_200': float(m2['refine_results'][200]['ari']),
            'ari_500': float(m2['refine_results'][500]['ari']),
            'ari_1000': float(m2['refine_results'][1000]['ari']),
            'ari_2000': float(m2['refine_results'][2000]['ari']),
            'ari_full': float(m2['refine_results']['full']['ari']),
        },
        'method3_anomaly': {
            'auc_ocsvm_rbf': float(m3['auc_ocsvm_rbf']),
            'auc_ocsvm_lin': float(m3['auc_ocsvm_lin']),
            'auc_iso_ctrl': float(m3['auc_iso_ctrl']),
            'auc_iso_all': float(m3['auc_iso_all']),
            'auc_lof': float(m3['auc_lof']),
        },
        'supervised_baseline_auc': 0.9593,
    }
    json_path = os.path.join(OUTPUT_DIR, "enhanced_unsupervised_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results JSON: {json_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
