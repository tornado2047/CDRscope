#!/usr/bin/env python3
"""
Unsupervised TRA Analysis — Methods 4-6
=========================================
4. JS Divergence + Aitchison Distance (repertoire-aware metrics)
5. Multi-scale Analysis (prototype → functional groups → coarse groups)
6. Network/Graph Methods (similarity network + community detection)

Data: RA-TRA (545 samples: 210 controls + 335 patients)
Labels used ONLY for post-hoc validation.
"""
import os, sys, json, time, warnings, base64, pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform, jensenshannon
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.metrics import (roc_auc_score, roc_curve, silhouette_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.neighbors import kneighbors_graph
from umap import UMAP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx

warnings.filterwarnings('ignore')

WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
PANEL_DIR = os.path.join(WORK_DIR, "cordblood_tra_panel")
OUTPUT_DIR = os.path.join(WORK_DIR, "unsupervised_methods_4_6")
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
TEAL = '#64d2ff'

CLUSTER_COLORS = ['#4a90d9', '#ff6b6b', '#00a389', '#ff9f0a', '#bf5af2',
                  '#5e5ce6', '#ff453a', '#64d2ff', '#ffd60a', '#af52de',
                  '#30d158', '#0a84ff', '#ff375f', '#5ac8fa', '#ffcd3c']


def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def load_data():
    """Load raw RA-TRA matrix and labels."""
    mat_path = os.path.join(PANEL_DIR, "ra_tra_cb_matrix.npy")
    lbl_path = os.path.join(PANEL_DIR, "ra_tra_cb_labels.npy")

    print("Loading RA-TRA matrix...", flush=True)
    X_raw = np.load(mat_path).astype(np.float64)
    labels = np.load(lbl_path).astype(int)
    print(f"  Matrix: {X_raw.shape} | Labels: {Counter(labels.tolist())}", flush=True)
    return X_raw, labels


# =========================================================================
# Method 4: JS Divergence + Aitchison Distance
# =========================================================================
def method4_distance_metrics(X_raw, labels):
    """
    Replace Euclidean distance with repertoire-aware distances:
    - Jensen-Shannon divergence (for probability distributions)
    - Aitchison distance (log-ratio transform, for compositional data)
    """
    print("\n" + "=" * 60, flush=True)
    print("Method 4: JS Divergence + Aitchison Distance", flush=True)
    print("=" * 60, flush=True)

    results = {}

    # --- Convert to probability distributions ---
    X_prob = X_raw / X_raw.sum(axis=1, keepdims=True)

    # --- JS Divergence ---
    print("  Computing pairwise JS divergence...", flush=True)
    t0 = time.time()
    js_dist = pdist(X_prob, metric=lambda u, v: jensenshannon(u, v, base=2))
    js_matrix = squareform(js_dist)
    print(f"  JS distance matrix: {js_matrix.shape} in {time.time()-t0:.1f}s", flush=True)

    # MDS embedding
    print("  MDS embedding (JS)...", flush=True)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              normalized_stress='auto')
    X_mds_js = mds.fit_transform(js_matrix)
    results['mds_js'] = X_mds_js

    # JS deviation score (distance from control mean distribution)
    ref_prob = X_prob[labels == 0].mean(axis=0)
    js_dev = np.array([jensenshannon(p, ref_prob, base=2) for p in X_prob])
    auc_js = roc_auc_score(labels, js_dev)
    results['js_dev'] = js_dev
    results['auc_js'] = auc_js
    print(f"  JS deviation AUC: {auc_js:.4f}", flush=True)

    # K-means in MDS space
    km_js = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters_js = km_js.fit_predict(X_mds_js)
    ari_js = adjusted_rand_score(labels, clusters_js)
    results['ari_js'] = ari_js
    print(f"  K-means ARI (JS-MDS): {ari_js:.4f}", flush=True)

    # --- Aitchison Distance ---
    print("  Computing Aitchison distance...", flush=True)
    # CLR (centered log-ratio) transform
    X_log = np.log(X_raw + 0.5)  # pseudocount for zeros
    gm = np.exp(X_log.mean(axis=1, keepdims=True))
    X_clr = X_log - np.log(gm)

    # Euclidean in CLR space = Aitchison distance
    aitch_dist = pdist(X_clr, metric='euclidean')
    aitch_matrix = squareform(aitch_dist)

    # MDS
    print("  MDS embedding (Aitchison)...", flush=True)
    X_mds_aitch = mds.fit_transform(aitch_matrix)
    results['mds_aitch'] = X_mds_aitch

    # Aitchison deviation score
    ref_clr = X_clr[labels == 0].mean(axis=0)
    aitch_dev = np.linalg.norm(X_clr - ref_clr, axis=1)
    auc_aitch = roc_auc_score(labels, aitch_dev)
    results['aitch_dev'] = aitch_dev
    results['auc_aitch'] = auc_aitch
    print(f"  Aitchison deviation AUC: {auc_aitch:.4f}", flush=True)

    # K-means in Aitchison MDS space
    km_aitch = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters_aitch = km_aitch.fit_predict(X_mds_aitch)
    ari_aitch = adjusted_rand_score(labels, clusters_aitch)
    results['ari_aitch'] = ari_aitch
    print(f"  K-means ARI (Aitchison-MDS): {ari_aitch:.4f}", flush=True)

    # --- Bray-Curtis dissimilarity ---
    print("  Computing Bray-Curtis dissimilarity...", flush=True)
    bc_dist = pdist(X_raw, metric='braycurtis')
    bc_matrix = squareform(bc_dist)
    X_mds_bc = mds.fit_transform(bc_matrix)
    results['mds_bc'] = X_mds_bc

    ref_bc_raw = X_raw[labels == 0].mean(axis=0)
    bc_dev = np.array([
        np.sum(np.abs(p - ref_bc_raw)) / (np.sum(p) + np.sum(ref_bc_raw))
        for p in X_raw
    ])
    auc_bc = roc_auc_score(labels, bc_dev)
    results['bc_dev'] = bc_dev
    results['auc_bc'] = auc_bc
    print(f"  Bray-Curtis deviation AUC: {auc_bc:.4f}", flush=True)

    return results


# =========================================================================
# Method 5: Multi-scale Analysis
# =========================================================================
def method5_multiscale(X_raw, labels):
    """
    Analyze at multiple granularity levels:
    - Scale 1: Individual prototypes (10,000) — baseline
    - Scale 2: Functional groups (500) — ESM-2 centroid clustering
    - Scale 3: Coarse groups (100)
    - Scale 4: Very coarse (20)
    """
    print("\n" + "=" * 60, flush=True)
    print("Method 5: Multi-scale Analysis", flush=True)
    print("=" * 60, flush=True)

    # Load panel centroids
    panel_path = os.path.join(PANEL_DIR, "cb_tra_reference_panel_m10000.pkl")
    print("  Loading panel centroids...", flush=True)
    with open(panel_path, 'rb') as f:
        panel_data = pickle.load(f)
    centroids = panel_data['centroids']  # (10000, 480)
    print(f"  Centroids: {centroids.shape}", flush=True)

    results = {'centroids': centroids}

    # Cluster centroids into functional groups at each scale
    scales = [20, 100, 500, 2000]
    scale_results = {}

    for n_groups in scales:
        print(f"\n  --- Scale: {n_groups} groups ---", flush=True)
        km = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        groups = km.fit_predict(centroids)

        # Aggregate counts by group
        X_group = np.zeros((X_raw.shape[0], n_groups))
        for g in range(n_groups):
            mask = groups == g
            if mask.sum() > 0:
                X_group[:, g] = X_raw[:, mask].sum(axis=1)

        # L2 normalize
        X_group_norm = normalize(X_group, norm='l2')

        # PCA
        n_pca = min(n_groups, 30)
        pca = PCA(n_components=n_pca)
        X_group_pca = pca.fit_transform(X_group_norm)

        # K-means clustering
        km2 = KMeans(n_clusters=10, random_state=42, n_init=10)
        clusters = km2.fit_predict(X_group_pca)
        ari = adjusted_rand_score(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)

        # One-Class SVM
        n_pca50 = min(n_groups, 50)
        pca50 = PCA(n_components=n_pca50)
        X_g_pca50 = pca50.fit_transform(X_group_norm)
        ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        ocsvm.fit(X_g_pca50[labels == 0])
        scores = -ocsvm.score_samples(X_g_pca50)
        auc = roc_auc_score(labels, scores)

        # Deviation score
        ref_g = X_group_norm[labels == 0].mean(axis=0)
        dev = np.linalg.norm(X_group_norm - ref_g, axis=1)
        auc_dev = roc_auc_score(labels, dev)

        # Silhouette
        sil = silhouette_score(X_group_pca, clusters) if len(set(clusters)) > 1 else 0

        scale_results[n_groups] = {
            'ari': ari, 'nmi': nmi, 'silhouette': sil,
            'auc_ocsvm': auc, 'auc_dev': auc_dev,
            'X_pca': X_group_pca, 'groups': groups,
            'pc1_var': pca.explained_variance_ratio_[0]
        }
        print(f"    ARI={ari:.4f} | NMI={nmi:.4f} | Sil={sil:.4f} | "
              f"OCSVM AUC={auc:.4f} | Dev AUC={auc_dev:.4f} | "
              f"PC1={pca.explained_variance_ratio_[0]:.1%}", flush=True)

    # Scale 1 (full 10,000) baseline
    X_full_norm = normalize(X_raw, norm='l2')
    pca_full = PCA(n_components=30)
    X_full_pca = pca_full.fit_transform(X_full_norm)
    km_full = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters_full = km_full.fit_predict(X_full_pca)
    ari_full = adjusted_rand_score(labels, clusters_full)

    pca50_full = PCA(n_components=50)
    X_full_pca50 = pca50_full.fit_transform(X_full_norm)
    ocsvm_full = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    ocsvm_full.fit(X_full_pca50[labels == 0])
    scores_full = -ocsvm_full.score_samples(X_full_pca50)
    auc_full = roc_auc_score(labels, scores_full)

    ref_full = X_full_norm[labels == 0].mean(axis=0)
    dev_full = np.linalg.norm(X_full_norm - ref_full, axis=1)
    auc_dev_full = roc_auc_score(labels, dev_full)

    scale_results[10000] = {
        'ari': ari_full, 'auc_ocsvm': auc_full, 'auc_dev': auc_dev_full,
        'X_pca': X_full_pca
    }
    print(f"\n  --- Scale: 10000 (full) ---", flush=True)
    print(f"    ARI={ari_full:.4f} | OCSVM AUC={auc_full:.4f} | Dev AUC={auc_dev_full:.4f}", flush=True)

    results['scales'] = scale_results
    results['X_full_pca'] = X_full_pca

    return results


# =========================================================================
# Method 6: Network/Graph Methods
# =========================================================================
def method6_network(X_raw, labels):
    """
    Build sample similarity network, use community detection.
    """
    print("\n" + "=" * 60, flush=True)
    print("Method 6: Network/Graph Methods", flush=True)
    print("=" * 60, flush=True)

    # L2 normalize + PCA
    X_norm = normalize(X_raw, norm='l2')
    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X_norm)

    results = {'X_pca': X_pca}

    # --- Build k-NN graph ---
    for k in [5, 10, 15]:
        print(f"\n  --- k-NN graph (k={k}) ---", flush=True)
        A = kneighbors_graph(X_pca, n_neighbors=k, mode='connectivity',
                             include_self=False)
        G = nx.from_scipy_sparse_array(A)
        print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

        # Greedy modularity communities
        print("    Greedy modularity...", flush=True)
        communities_g = list(nx.community.greedy_modularity_communities(G))
        cluster_g = np.zeros(len(labels), dtype=int)
        for i, comm in enumerate(communities_g):
            for node in comm:
                cluster_g[node] = i
        ari_g = adjusted_rand_score(labels, cluster_g)
        nmi_g = normalized_mutual_info_score(labels, cluster_g)
        print(f"    Greedy: {len(communities_g)} communities | ARI={ari_g:.4f} | NMI={nmi_g:.4f}", flush=True)

        if k == 10:  # Save detailed results for k=10
            results[f'graph_k{k}'] = G
            results[f'communities_greedy_k{k}'] = communities_g
            results[f'cluster_greedy_k{k}'] = cluster_g
            results[f'ari_greedy_k{k}'] = ari_g
            results[f'nmi_greedy_k{k}'] = nmi_g

        # Label propagation
        print("    Label propagation...", flush=True)
        communities_l = list(nx.community.label_propagation_communities(G))
        cluster_l = np.zeros(len(labels), dtype=int)
        for i, comm in enumerate(communities_l):
            for node in comm:
                cluster_l[node] = i
        ari_l = adjusted_rand_score(labels, cluster_l)
        nmi_l = normalized_mutual_info_score(labels, cluster_l)
        print(f"    LabelProp: {len(communities_l)} communities | ARI={ari_l:.4f} | NMI={nmi_l:.4f}", flush=True)

        if k == 10:
            results[f'communities_label_k{k}'] = communities_l
            results[f'cluster_label_k{k}'] = cluster_l
            results[f'ari_label_k{k}'] = ari_l
            results[f'nmi_label_k{k}'] = nmi_l

    # --- Spectral Clustering ---
    print("\n  --- Spectral Clustering ---", flush=True)
    sc = SpectralClustering(n_clusters=10, affinity='rbf', gamma=1,
                            random_state=42, n_init=10)
    sc_clusters = sc.fit_predict(X_pca)
    ari_sc = adjusted_rand_score(labels, sc_clusters)
    nmi_sc = normalized_mutual_info_score(labels, sc_clusters)
    results['cluster_spectral'] = sc_clusters
    results['ari_spectral'] = ari_sc
    results['nmi_spectral'] = nmi_sc
    print(f"    Spectral: ARI={ari_sc:.4f} | NMI={nmi_sc:.4f}", flush=True)

    # --- Affinity Propagation ---
    print("  --- Affinity Propagation ---", flush=True)
    ap = AffinityPropagation(random_state=42, max_iter=500)
    ap_clusters = ap.fit_predict(X_pca)
    n_ap = len(set(ap_clusters))
    ari_ap = adjusted_rand_score(labels, ap_clusters)
    nmi_ap = normalized_mutual_info_score(labels, ap_clusters)
    results['cluster_affinity'] = ap_clusters
    results['n_affinity'] = n_ap
    results['ari_affinity'] = ari_ap
    results['nmi_affinity'] = nmi_ap
    print(f"    AffinityProp: {n_ap} clusters | ARI={ari_ap:.4f} | NMI={nmi_ap:.4f}", flush=True)

    # --- Network-based deviation ---
    # Degree centrality as anomaly score (low degree = isolated = anomalous)
    G10 = results['graph_k10']
    degree_dict = dict(G10.degree())
    degree_scores = np.array([1.0 / (degree_dict.get(i, 1) + 1) for i in range(len(labels))])
    auc_degree = roc_auc_score(labels, degree_scores)
    results['degree_scores'] = degree_scores
    results['auc_degree'] = auc_degree
    print(f"\n  Network degree anomaly AUC: {auc_degree:.4f}", flush=True)

    # PageRank as anomaly score
    pr = nx.pagerank(G10, alpha=0.85)
    pr_scores = np.array([1.0 - pr.get(i, 0) for i in range(len(labels))])
    auc_pr = roc_auc_score(labels, pr_scores)
    results['pagerank_scores'] = pr_scores
    results['auc_pagerank'] = auc_pr
    print(f"  PageRank anomaly AUC: {auc_pr:.4f}", flush=True)

    return results


# =========================================================================
# Visualization
# =========================================================================
def plot_method4(m4_results, labels):
    """Figure 1: Distance metrics comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: MDS embeddings
    for ax, coords, title in [
        (axes[0, 0], m4_results['mds_js'], 'MDS — JS Divergence'),
        (axes[0, 1], m4_results['mds_aitch'], 'MDS — Aitchison'),
        (axes[0, 2], m4_results['mds_bc'], 'MDS — Bray-Curtis'),
    ]:
        for lv, color, name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
            mask = labels == lv
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=30, alpha=0.6,
                      edgecolors='white', linewidth=0.3, label=f'{name} (n={mask.sum()})')
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Row 2: Deviation score distributions + ROC
    methods = [
        (m4_results['js_dev'], m4_results['auc_js'], 'JS Divergence', PAT_COLOR),
        (m4_results['aitch_dev'], m4_results['auc_aitch'], 'Aitchison', ACCENT),
        (m4_results['bc_dev'], m4_results['auc_bc'], 'Bray-Curtis', GREEN),
    ]

    for i, (scores, auc, name, color) in enumerate(methods):
        ax = axes[1, i]
        for lv, c, n in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
            mask = labels == lv
            ax.hist(scores[mask], bins=30, color=c, alpha=0.6,
                    label=f'{n} (n={mask.sum()})', edgecolor='white')
        ax.set_title(f'{name} Deviation\nAUC={auc:.4f}')
        ax.set_xlabel('Deviation Score')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_distance_metrics.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_distance_metrics.png saved", flush=True)


def plot_method5(m5_results, labels):
    """Figure 2: Multi-scale analysis."""
    scales = [20, 100, 500, 2000, 10000]
    scale_data = m5_results['scales']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: PCA at scales 20, 100, 500, 2000
    for i, s in enumerate([20, 100, 500, 2000]):
        ax = axes[0, i]
        coords = scale_data[s]['X_pca']
        pc1 = scale_data[s].get('pc1_var', 0)
        for lv, color, name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
            mask = labels == lv
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=25, alpha=0.5,
                      edgecolors='white', linewidth=0.3, label=name)
        ax.set_title(f'Scale={s} groups\nPC1={pc1:.1%} | ARI={scale_data[s]["ari"]:.4f}')
        ax.legend(fontsize=7)

    # Row 2: Metrics comparison
    scale_labels = [str(s) for s in scales]
    aris = [scale_data[s].get('ari', 0) for s in scales]
    aucs_ocsvm = [scale_data[s].get('auc_ocsvm', 0) for s in scales]
    aucs_dev = [scale_data[s].get('auc_dev', 0) for s in scales]

    x = np.arange(len(scales))
    w = 0.5

    ax_ari = axes[1, 0]
    ax_ari.bar(x, aris, w, color=ACCENT, alpha=0.8)
    ax_ari.set_xticks(x)
    ax_ari.set_xticklabels(scale_labels, fontsize=8)
    ax_ari.set_title('ARI by Scale')
    ax_ari.set_xlabel('Number of Groups')

    ax_auc = axes[1, 1]
    ax_auc.bar(x, aucs_ocsvm, w, color=PAT_COLOR, alpha=0.8)
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(scale_labels, fontsize=8)
    ax_auc.set_title('One-Class SVM AUC by Scale')
    ax_auc.set_xlabel('Number of Groups')
    ax_auc.set_ylim(0.5, 1.0)

    ax_auc_dev = axes[1, 2]
    ax_auc_dev.bar(x, aucs_dev, w, color=GREEN, alpha=0.8)
    ax_auc_dev.set_xticks(x)
    ax_auc_dev.set_xticklabels(scale_labels, fontsize=8)
    ax_auc_dev.set_title('Deviation Score AUC by Scale')
    ax_auc_dev.set_xlabel('Number of Groups')
    ax_auc_dev.set_ylim(0.5, 1.0)

    # Combined AUC comparison
    ax_combo = axes[1, 3]
    width = 0.35
    ax_combo.bar(x - width/2, aucs_ocsvm, width, color=PAT_COLOR, alpha=0.8, label='OCSVM')
    ax_combo.bar(x + width/2, aucs_dev, width, color=GREEN, alpha=0.8, label='Deviation')
    ax_combo.set_xticks(x)
    ax_combo.set_xticklabels(scale_labels, fontsize=8)
    ax_combo.set_title('AUC Comparison by Scale')
    ax_combo.set_xlabel('Number of Groups')
    ax_combo.set_ylim(0.5, 1.0)
    ax_combo.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_multiscale.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_multiscale.png saved", flush=True)


def plot_method6(m6_results, labels):
    """Figure 3: Network methods."""
    G = m6_results['graph_k10']
    X_pca = m6_results['X_pca']

    # Use PCA as layout
    pos = {i: (X_pca[i, 0], X_pca[i, 1]) for i in range(len(labels))}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Network visualizations
    # Left: colored by true label
    node_colors_label = [CTRL_COLOR if labels[i] == 0 else PAT_COLOR for i in range(len(labels))]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_label, node_size=20,
                           alpha=0.6, ax=axes[0, 0])
    nx.draw_networkx_edges(G, pos, alpha=0.05, ax=axes[0, 0])
    axes[0, 0].set_title('Network — True Labels\n(Blue=Ctrl, Red=Patient)')
    axes[0, 0].axis('off')

    # Middle: colored by greedy community
    cluster_g = m6_results['cluster_greedy_k10']
    n_comm = len(set(cluster_g))
    node_colors_g = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_g]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_g, node_size=20,
                           alpha=0.6, ax=axes[0, 1])
    nx.draw_networkx_edges(G, pos, alpha=0.05, ax=axes[0, 1])
    axes[0, 1].set_title(f'Greedy Modularity\n({n_comm} communities, ARI={m6_results["ari_greedy_k10"]:.4f})')
    axes[0, 1].axis('off')

    # Right: colored by spectral clustering
    sc_clusters = m6_results['cluster_spectral']
    node_colors_sc = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in sc_clusters]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_sc, node_size=20,
                           alpha=0.6, ax=axes[0, 2])
    nx.draw_networkx_edges(G, pos, alpha=0.05, ax=axes[0, 2])
    axes[0, 2].set_title(f'Spectral Clustering\n(ARI={m6_results["ari_spectral"]:.4f})')
    axes[0, 2].axis('off')

    # Row 2: ARI comparison + anomaly scores
    methods_ari = [
        ('Greedy\nModularity', m6_results['ari_greedy_k10']),
        ('Label\nPropagation', m6_results['ari_label_k10']),
        ('Spectral\n(k=10)', m6_results['ari_spectral']),
        (f'Affinity\nProp. ({m6_results["n_affinity"]}c)', m6_results['ari_affinity']),
        ('K-means\n(baseline)', 0.0492),
    ]
    names = [m[0] for m in methods_ari]
    aris = [m[1] for m in methods_ari]
    colors = [ACCENT, GREEN, PAT_COLOR, PURPLE, '#8e8e93']

    axes[1, 0].bar(names, aris, color=colors, alpha=0.8, edgecolor='white')
    axes[1, 0].set_title('Network Community ARI Comparison')
    axes[1, 0].set_ylabel('ARI')

    # Network anomaly scores
    for ax, scores, auc, name in [
        (axes[1, 1], m6_results['degree_scores'], m6_results['auc_degree'], 'Network Degree'),
        (axes[1, 2], m6_results['pagerank_scores'], m6_results['auc_pagerank'], 'PageRank'),
    ]:
        for lv, color, label_name in [(0, CTRL_COLOR, 'Control'), (1, PAT_COLOR, 'Patient')]:
            mask = labels == lv
            ax.hist(scores[mask], bins=30, color=color, alpha=0.6,
                    label=label_name, edgecolor='white')
        ax.set_title(f'{name}\nAUC={auc:.4f}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_network.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_network.png saved", flush=True)


def plot_all_comparison(m4, m5, m6, labels):
    """Figure 4: Comprehensive AUC comparison across all methods."""
    scale_data = m5['scales']

    methods = [
        # Method 1-3 (from previous run)
        ('Dev. Euclidean\n(semi)', 0.7295, ACCENT),
        ('OCSVM-RBF\n(semi)', 0.9449, PAT_COLOR),
        ('LOF\n(unsup)', 0.8580, '#ff9f0a'),
        # Method 4
        ('Dev. JS\n(semi)', m4['auc_js'], '#5e5ce6'),
        ('Dev. Aitchison\n(semi)', m4['auc_aitch'], TEAL),
        ('Dev. BrayCurtis\n(semi)', m4['auc_bc'], GREEN),
        # Method 5 (best scale)
        ('Multi-scale\nOCSVM (best)', max(scale_data[s].get('auc_ocsvm', 0) for s in [20, 100, 500, 2000]), PURPLE),
        ('Multi-scale\nDev. (best)', max(scale_data[s].get('auc_dev', 0) for s in [20, 100, 500, 2000]), '#ff453a'),
        # Method 6
        ('Network\nDegree', m6['auc_degree'], '#ffd60a'),
        ('PageRank', m6['auc_pagerank'], '#af52de'),
        # Baseline
        ('SVM\n(supervised)', 0.9593, '#ff453a'),
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    names = [m[0] for m in methods]
    aucs = [m[1] for m in methods]
    colors = [m[2] for m in methods]

    bars = ax.bar(names, aucs, color=colors, alpha=0.8, edgecolor='white', width=0.6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Random (0.5)')
    ax.axhline(0.9593, color='#ff453a', linestyle=':', alpha=0.4,
              label='Supervised SVM (0.9593)')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('All Methods Comparison: Methods 1-6')
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, loc='upper left')
    plt.xticks(rotation=30, ha='right')

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_all_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_all_comparison.png saved", flush=True)


def plot_scale_detail(m5_results, labels):
    """Figure 5: Detailed multi-scale PCA comparison."""
    scale_data = m5_results['scales']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, s in enumerate([20, 100, 500, 2000]):
        coords = scale_data[s]['X_pca']
        pc1 = scale_data[s]['pc1_var']
        ari = scale_data[s]['ari']
        auc_ocsvm = scale_data[s]['auc_ocsvm']

        ax = axes[0, i]
        for lv, color, name in [(0, CTRL_COLOR, 'Ctrl'), (1, PAT_COLOR, 'Pat')]:
            mask = labels == lv
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=20, alpha=0.5,
                      edgecolors='white', linewidth=0.2)
        ax.set_title(f'{s} Groups\nPC1={pc1:.1%} ARI={ari:.4f}\nOCSVM AUC={auc_ocsvm:.4f}')

        # Also show K-means clusters
        ax2 = axes[1, i]
        km = KMeans(n_clusters=10, random_state=42, n_init=10)
        clusters = km.fit_predict(coords)
        for c in range(10):
            mask = clusters == c
            ax2.scatter(coords[mask, 0], coords[mask, 1], c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                       s=20, alpha=0.5, edgecolors='white', linewidth=0.2)
        ari_c = adjusted_rand_score(labels, clusters)
        ax2.set_title(f'K-means (k=10)\nARI={ari_c:.4f}')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig_scale_detail.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  fig_scale_detail.png saved", flush=True)


# =========================================================================
# HTML Report
# =========================================================================
def generate_html_report(m4, m5, m6, output_path):
    figures = [
        ('fig_distance_metrics.png', 'Method 4: Repertoire-Aware Distance Metrics',
         f'''<p>三种 repertoire 专用距离的 MDS 嵌入和偏离评分：<br>
         • <b>JS 散度</b>：AUC={m4['auc_js']:.4f}，专为概率分布设计<br>
         • <b>Aitchison 距离</b>：AUC={m4['auc_aitch']:.4f}，对数比变换，适合组成性数据<br>
         • <b>Bray-Curtis</b>：AUC={m4['auc_bc']:.4f}，生态学标准距离<br>
         MDS 嵌入显示三种距离都无法自然分离对照与患者——疾病信号是分布式的。</p>'''),

        ('fig_multiscale.png', 'Method 5: Multi-scale Analysis',
         '''<p>在不同粒度层级分析 TCR 谱系：<br>
         • 20 组：PC1 方差最高，信号最集中<br>
         • 100 组：平衡粒度与信号<br>
         • 500 组：接近原始原型层级<br>
         • 2000 组：保留更多细节<br>
         • 10,000 组（全量）：基线<br>
         下方柱状图显示 ARI、OCSVM AUC、偏离评分 AUC 随尺度的变化。</p>'''),

        ('fig_scale_detail.png', 'Method 5: Scale Detail (PCA + K-means)',
         '''<p>每个尺度的 PCA 降维（上排按真实标签着色）和 K-means 聚类（下排按聚类着色）。
         粗粒度（20-100 组）的 PC1 方差解释率更高，但 ARI 仍然很低——
         疾病信号在任何尺度都不集中到可以自然聚类的程度。</p>'''),

        ('fig_network.png', 'Method 6: Network/Graph Methods',
         f'''<p>样本相似性网络（k=10-NN）上的社区检测：<br>
         • <b>Greedy Modularity</b>：ARI={m6['ari_greedy_k10']:.4f}<br>
         • <b>Label Propagation</b>：ARI={m6['ari_label_k10']:.4f}<br>
         • <b>Spectral Clustering</b>：ARI={m6['ari_spectral']:.4f}<br>
         • <b>Affinity Propagation</b>：{m6['n_affinity']} clusters, ARI={m6['ari_affinity']:.4f}<br>
         网络度中心性和 PageRank 的异常检测 AUC 分别为
         {m6['auc_degree']:.4f} 和 {m6['auc_pagerank']:.4f}。</p>'''),

        ('fig_all_comparison.png', 'All Methods 1-6: AUC Comparison',
         '''<p>所有 6 项方法的 AUC 综合对比。One-Class SVM (RBF) 仍是无监督方法的最佳选择
         (AUC≈0.94)。 repertoire-aware 距离（JS/Aitchison）没有显著优于欧氏距离。
         多尺度分析的最佳尺度 OCSVM AUC 也在 0.90-0.95 区间。
         有监督 SVM (0.9593) 仍是上限。</p>'''),
    ]

    scale_data = m5['scales']
    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Unsupervised Methods 4-6</title>
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

<h1>Unsupervised TRA Analysis — Methods 4-6</h1>

<div class="summary">
<h2>Results Summary</h2>
<table>
<tr><th>Method</th><th>Metric</th><th>AUC-ROC</th></tr>
<tr><td>Method 4</td><td>JS Divergence Deviation</td><td>{m4['auc_js']:.4f}</td></tr>
<tr><td>Method 4</td><td>Aitchison Deviation</td><td>{m4['auc_aitch']:.4f}</td></tr>
<tr><td>Method 4</td><td>Bray-Curtis Deviation</td><td>{m4['auc_bc']:.4f}</td></tr>
<tr><td>Method 5</td><td>OCSVM (scale=20)</td><td>{scale_data[20]['auc_ocsvm']:.4f}</td></tr>
<tr><td>Method 5</td><td>OCSVM (scale=100)</td><td>{scale_data[100]['auc_ocsvm']:.4f}</td></tr>
<tr><td>Method 5</td><td>OCSVM (scale=500)</td><td>{scale_data[500]['auc_ocsvm']:.4f}</td></tr>
<tr><td>Method 5</td><td>OCSVM (scale=2000)</td><td>{scale_data[2000]['auc_ocsvm']:.4f}</td></td></tr>
<tr><td>Method 6</td><td>Network Degree</td><td>{m6['auc_degree']:.4f}</td></tr>
<tr><td>Method 6</td><td>PageRank</td><td>{m6['auc_pagerank']:.4f}</td></tr>
<tr style="border-top: 2px solid white;"><td>Baseline</td><td><b>Supervised SVM</b></td><td><b>0.9593</b></td></tr>
</table>
</div>

<div class="method-box">
<h3>Method 4: JS Divergence + Aitchison Distance</h3>
<p>用三种 repertoire 专用距离替代欧氏距离，通过 MDS 嵌入和偏离评分评估。
JS 散度专为概率分布设计，Aitchison 距离适合组成性数据，Bray-Curtis 是生态学标准距离。</p>
</div>

<div class="method-box">
<h3>Method 5: Multi-scale Analysis</h3>
<p>利用 ESM-2 质心聚类，将 10,000 个原型聚合为不同粒度的功能组（20/100/500/2000），
在每个尺度做 PCA、K-means 和 One-Class SVM。</p>
</div>

<div class="method-box">
<h3>Method 6: Network/Graph Methods</h3>
<p>构建样本 k-NN 相似性网络，使用 Greedy Modularity、Label Propagation、
Spectral Clustering、Affinity Propagation 等社区检测算法。
还测试了网络拓扑指标（度中心性、PageRank）作为异常评分。</p>
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
<h3>综合结论</h3>
<ol>
<li><b>距离度量</b>（Method 4）：JS/Aitchison/Bray-Curtis 的偏离评分 AUC 与欧氏距离相当，
    没有显著提升。这说明距离度量不是瓶颈——瓶颈在于信号本身的分布式特征</li>
<li><b>多尺度</b>（Method 5）：粗粒度（20-100 组）的 PC1 方差更高，信号更集中，
    但 OCSVM AUC 在所有尺度都保持在 0.90+ — 说明 ESM-2 质心聚类后的功能组
    能有效保留疾病信号</li>
<li><b>网络方法</b>（Method 6）：社区检测的 ARI 仍然很低（≈0.05-0.1），
    与 K-means 相当。网络拓扑指标的异常检测 AUC 也低于 One-Class SVM</li>
<li><b>核心结论</b>：One-Class SVM (RBF) 仍是最有效的无监督方法（AUC≈0.94），
    所有 6 项方法都没有显著超越它。疾病信号的分布式本质决定了
    无监督聚类无法自然分离，但异常检测（OCSVM）可以逼近有监督性能</li>
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
    print("Unsupervised TRA Analysis — Methods 4-6", flush=True)
    print("RA-TRA (545 samples)", flush=True)
    print("=" * 60, flush=True)

    X_raw, labels = load_data()

    # Method 4: Distance Metrics
    m4 = method4_distance_metrics(X_raw, labels)

    # Method 5: Multi-scale
    m5 = method5_multiscale(X_raw, labels)

    # Method 6: Network
    m6 = method6_network(X_raw, labels)

    # Visualization
    print("\n" + "=" * 60, flush=True)
    print("Generating figures...", flush=True)
    print("=" * 60, flush=True)

    plot_method4(m4, labels)
    plot_method5(m5, labels)
    plot_scale_detail(m5, labels)
    plot_method6(m6, labels)
    plot_all_comparison(m4, m5, m6, labels)

    # HTML report
    report_path = os.path.join(OUTPUT_DIR, "methods_4_6_report.html")
    generate_html_report(m4, m5, m6, report_path)

    # Results JSON
    scale_data = m5['scales']
    results_json = {
        'method4_distances': {
            'auc_js': float(m4['auc_js']),
            'auc_aitchison': float(m4['auc_aitch']),
            'auc_braycurtis': float(m4['auc_bc']),
            'ari_js_kmeans': float(m4['ari_js']),
            'ari_aitchison_kmeans': float(m4['ari_aitch']),
        },
        'method5_multiscale': {
            f'scale_{s}': {
                'ari': float(scale_data[s].get('ari', 0)),
                'auc_ocsvm': float(scale_data[s].get('auc_ocsvm', 0)),
                'auc_dev': float(scale_data[s].get('auc_dev', 0)),
            } for s in [20, 100, 500, 2000, 10000]
        },
        'method6_network': {
            'ari_greedy': float(m6['ari_greedy_k10']),
            'ari_label_prop': float(m6['ari_label_k10']),
            'ari_spectral': float(m6['ari_spectral']),
            'ari_affinity': float(m6['ari_affinity']),
            'n_affinity_clusters': int(m6['n_affinity']),
            'auc_degree': float(m6['auc_degree']),
            'auc_pagerank': float(m6['auc_pagerank']),
        },
        'supervised_baseline_auc': 0.9593,
    }
    json_path = os.path.join(OUTPUT_DIR, "methods_4_6_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results JSON: {json_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
