#!/usr/bin/env python3
"""
Unsupervised TRA Repertoire Analysis Pipeline
===============================================
Replaces supervised SVM with unsupervised methods:
  - Clustering (K-means + HDBSCAN + GMM)
  - Diversity analysis (Shannon, Simpson, Pielou)
  - Outlier detection (Isolation Forest)
  - If labels exist → post-hoc validation (ARI, NMI, enrichment)

Works in two scenarios:
  Scenario 1: All samples unlabeled or all healthy → find individual variation
  Scenario 2: Labeled samples (e.g., RA) → validate unsupervised clusters

Uses the same CB TRA panel projection (m=10,000) — already unsupervised.
"""
import os, sys, json, time, pickle, warnings, glob
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import entropy, mannwhitneyu, chi2_contingency
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, davies_bouldin_score,
                             calinski_harabasz_score, roc_auc_score, roc_curve,
                             f1_score, accuracy_score)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
TIER2_DIR = os.path.join(WORK_DIR, "CDRscope-v2", "inst", "python", "tier2")
sys.path.insert(0, TIER2_DIR)

PANEL_DIR = os.path.join(WORK_DIR, "cordblood_tra_panel")
OUTPUT_DIR = os.path.join(WORK_DIR, "unsupervised_tra_results")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")  # Will be set per-dataset in main()

ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"
EMBED_DIM = 480
M_TARGET = 10000
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Color palette for clusters
CLUSTER_COLORS = ['#4a90d9', '#ff6b6b', '#00a389', '#ff9f0a', '#bf5af2',
                   '#5e5ce6', '#ff453a', '#64d2ff', '#ffd60a', '#af52de',
                   '#30d158', '#0a84ff', '#ff9f0a', '#bf5af2', '#ff375f']


def img_to_b64(path):
    import base64
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def shannon_entropy(counts):
    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def simpson_index(counts):
    p = counts / counts.sum() if counts.sum() > 0 else counts
    return 1 - np.sum(p**2)


def pielou_evenness(counts):
    H = shannon_entropy(counts)
    S = np.sum(counts > 0)
    return H / np.log(S) if S > 1 else 0


# =========================================================================
# ESM-2 Embedding (same as supervised, cached)
# =========================================================================
def compute_esm2_embeddings(sequences, batch_size=256):
    import torch
    from transformers import AutoTokenizer, AutoModel

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"  Loading ESM-2 ({ESM2_MODEL}) on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModel.from_pretrained(ESM2_MODEL)
    model = model.to(device)
    model.eval()

    n = len(sequences)
    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
    n_batches = (n + batch_size - 1) // batch_size
    start = time.time()

    for b in range(n_batches):
        i0 = b * batch_size
        i1 = min(i0 + batch_size, n)
        batch_seqs = sequences[i0:i1]
        spaced = [" ".join(list(s)) for s in batch_seqs]
        inputs = tokenizer(spaced, return_tensors='pt', padding=True, truncation=True, max_length=50)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        masked = hidden * mask
        summed = masked.sum(dim=1).cpu().numpy()
        counts = mask.sum(dim=1).cpu().numpy()
        embeddings[i0:i1] = summed / counts
        if (b + 1) % 50 == 0 or b == n_batches - 1:
            elapsed = time.time() - start
            rate = i1 / elapsed if elapsed > 0 else 0
            eta = (n - i1) / rate if rate > 0 else 0
            print(f"    Batch {b+1}/{n_batches} | {i1:,}/{n:,} | {rate:.0f} seq/s | ETA {eta:.0f}s", flush=True)

    print(f"  Embedding done: {n:,} seqs in {time.time()-start:.0f}s", flush=True)
    return embeddings


def assign_to_centroids(embeddings, centroids, batch_size=10000):
    from scipy.spatial.distance import cdist as _cdist
    n = embeddings.shape[0]
    assignments = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch_size):
        batch = embeddings[i:i+batch_size]
        dists = _cdist(batch, centroids, metric='sqeuclidean')
        assignments[i:i+batch_size] = np.argmin(dists, axis=1)
    return assignments


# =========================================================================
# Dataset loaders
# =========================================================================
def load_ra_tra():
    """Load RA-TRA dataset (210 ctrl + 335 patients)"""
    sys.path.insert(0, TIER2_DIR)
    from cross_disease_benchmark import load_ra_dataset
    samples = load_ra_dataset(chain='TRA')
    for s in samples:
        s['name'] = s.get('sample_id', s.get('name', 'unknown'))
    return samples, "RA-TRA"


def load_ms_pbmc():
    """MS PBMC: 4 MS vs 4 IIH controls"""
    path = os.path.join(WORK_DIR, "geo_ms_tcr", "ms_tra_pseudobulk.csv")
    df = pd.read_csv(path)
    df = df[df['tissue'] == 'PBMC']
    samples = []
    for sample_name in df['sample'].unique():
        sdf = df[df['sample'] == sample_name]
        disease = sdf['disease'].iloc[0]
        label = 1 if disease == 'MS' else 0
        samples.append({
            'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
            'label': label,
            'name': sample_name
        })
    return samples, "MS-PBMC"


def load_zenodo_ra_hd():
    """Zenodo RA (13) vs HD (17)"""
    path = os.path.join(WORK_DIR, "zenodo_scTCR", "zenodo_autoimmune_tra.csv")
    df = pd.read_csv(path)
    samples = []
    for disease, label in [('RA', 1), ('HD', 0)]:
        dis_df = df[df['disease'] == disease]
        for sample_name in dis_df['sample'].unique():
            sdf = dis_df[dis_df['sample'] == sample_name]
            samples.append({
                'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
                'label': label,
                'name': f"{disease}_{sample_name}"
            })
    return samples, "Zenodo-RA-vs-HD"


def load_sle_hd():
    """SLE (18) vs Zenodo HD (17)"""
    sle_path = os.path.join(WORK_DIR, "zenodo_scTCR", "sle_tra_pseudobulk.csv")
    hd_path = os.path.join(WORK_DIR, "zenodo_scTCR", "zenodo_autoimmune_tra.csv")
    samples = []
    sle_df = pd.read_csv(sle_path)
    for sample_name in sle_df['sample'].unique():
        sdf = sle_df[sle_df['sample'] == sample_name]
        count_col = 'count' if 'count' in sdf.columns else None
        if count_col is None:
            sdf = sdf.copy(); sdf['count'] = 1; count_col = 'count'
        samples.append({
            'df': sdf[['cdr3', count_col]].rename(columns={'cdr3': 'junction_aa', count_col: 'duplicate_count'}),
            'label': 1, 'name': f"SLE_{sample_name}"
        })
    hd_df = pd.read_csv(hd_path)
    hd_df = hd_df[hd_df['disease'] == 'HD']
    for sample_name in hd_df['sample'].unique():
        sdf = hd_df[hd_df['sample'] == sample_name]
        samples.append({
            'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
            'label': 0, 'name': f"HD_{sample_name}"
        })
    return samples, "SLE-vs-HD"


def load_hd_only():
    """Load only HD samples from Zenodo (Scenario 1: all healthy)"""
    path = os.path.join(WORK_DIR, "zenodo_scTCR", "zenodo_autoimmune_tra.csv")
    df = pd.read_csv(path)
    df = df[df['disease'] == 'HD']
    samples = []
    for sample_name in df['sample'].unique():
        sdf = df[df['sample'] == sample_name]
        samples.append({
            'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
            'label': -1,  # Unknown label
            'name': sample_name
        })
    return samples, "HD-Only (Scenario 1)"


# =========================================================================
# Projection
# =========================================================================
def project_dataset(samples, centroids, dataset_name, cached_embeddings=None):
    print(f"\n{'='*60}", flush=True)
    print(f"  Projecting {dataset_name} -> m={centroids.shape[0]}", flush=True)
    print(f"{'='*60}", flush=True)

    all_seqs = set()
    for s in samples:
        df = s['df']
        seqs = df['junction_aa'].dropna().values
        for seq in seqs:
            if isinstance(seq, str) and len(seq) >= 8 and set(seq) <= STANDARD_AA:
                all_seqs.add(seq)
    all_seqs = sorted(all_seqs)
    print(f"  Unique valid sequences: {len(all_seqs):,}", flush=True)

    new_seqs = [s for s in all_seqs if s not in cached_embeddings] if cached_embeddings else all_seqs
    if new_seqs:
        print(f"  New sequences to embed: {len(new_seqs):,}", flush=True)
        new_emb = compute_esm2_embeddings(new_seqs)
        for seq, emb in zip(new_seqs, new_emb):
            cached_embeddings[seq] = emb
    else:
        print(f"  All sequences cached", flush=True)

    embeddings = np.array([cached_embeddings[s] for s in all_seqs])
    print(f"  Assigning to {centroids.shape[0]} centroids...", flush=True)
    assignments = assign_to_centroids(embeddings, centroids)
    seq_to_centroid = {seq: assignments[i] for i, seq in enumerate(all_seqs)}

    m = centroids.shape[0]
    n = len(samples)
    count_matrix = np.zeros((n, m), dtype=np.float32)
    sample_names = []
    labels = np.zeros(n, dtype=np.int32) - 1  # -1 = unknown

    for i, s in enumerate(samples):
        df = s['df']
        seqs = df['junction_aa'].dropna().values
        count_col = 'duplicate_count' if 'duplicate_count' in df.columns else None
        counts = df[count_col].fillna(1).values if count_col else np.ones(len(seqs))
        for seq, cnt in zip(seqs, counts):
            if isinstance(seq, str) and seq in seq_to_centroid:
                count_matrix[i, seq_to_centroid[seq]] += float(cnt)
        labels[i] = s['label']
        sample_names.append(s['name'])

    print(f"  Matrix: {count_matrix.shape} | Labels: {Counter(labels.tolist())}", flush=True)
    return count_matrix, labels, sample_names


# =========================================================================
# Step 1: Unsupervised Dimensionality Reduction
# =========================================================================
def step1_dim_reduction(X_norm, sample_names, dataset_name):
    print("\n[Step 1] Unsupervised dimensionality reduction...", flush=True)

    # PCA
    pca = PCA(n_components=min(30, X_norm.shape[0]-1, X_norm.shape[1]))
    pca_coords = pca.fit_transform(X_norm)
    var_explained = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_explained)
    n90 = int(np.searchsorted(cumvar, 0.9) + 1)
    print(f"  PCA: {pca_coords.shape[1]} components | 90% var at PC{n90} | "
          f"PC1={var_explained[0]:.4f} PC2={var_explained[1]:.4f}", flush=True)

    # UMAP (unsupervised)
    try:
        import umap
        umap_coords = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, X_norm.shape[0]-1),
                                min_dist=0.1).fit_transform(X_norm)
        has_umap = True
        print("  UMAP done", flush=True)
    except Exception as e:
        print(f"  UMAP failed ({e}), using t-SNE", flush=True)
        from sklearn.manifold import TSNE
        umap_coords = TSNE(n_components=2, random_state=42, perplexity=min(30, X_norm.shape[0]//4)).fit_transform(X_norm)
        has_umap = False

    # PCA plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(pca_coords[:, 0], pca_coords[:, 1], c='#5e5ce6', s=40, alpha=0.7,
                    edgecolors='white', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    axes[0].set_title(f'{dataset_name} — PCA (Unsupervised)')

    axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1], c='#5e5ce6', s=40, alpha=0.7,
                    edgecolors='white', linewidth=0.5)
    axes[1].set_xlabel('UMAP-1')
    axes[1].set_ylabel('UMAP-2')
    axes[1].set_title(f'{dataset_name} — UMAP (Unsupervised)')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_dim_reduction.png'), bbox_inches='tight')
    plt.close()
    print("  fig_dim_reduction.png done", flush=True)

    # PCA scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, min(21, len(var_explained)+1)), var_explained[:20], color='#5e5ce6', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(1, min(21, len(cumvar)+1)), cumvar[:20], 'o-', color='#ff6b6b', lw=2, markersize=4)
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Individual Variance')
    ax2.set_ylabel('Cumulative Variance')
    ax.set_title('PCA Scree Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_pca_scree.png'), bbox_inches='tight')
    plt.close()
    print("  fig_pca_scree.png done", flush=True)

    return pca_coords, umap_coords, var_explained


# =========================================================================
# Step 2: Unsupervised Clustering
# =========================================================================
def step2_clustering(X_norm, pca_coords, umap_coords, sample_names, labels, dataset_name):
    print("\n[Step 2] Unsupervised clustering...", flush=True)

    n = X_norm.shape[0]
    max_k = min(10, n - 1)
    results = {}

    # --- K-means with silhouette analysis ---
    print("  K-means silhouette analysis...", flush=True)
    k_range = range(2, max_k + 1)
    sil_scores = []
    db_scores = []
    ch_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(X_norm)
        if len(np.unique(clusters)) > 1:
            sil = silhouette_score(X_norm, clusters)
            db = davies_bouldin_score(X_norm, clusters)
            ch = calinski_harabasz_score(X_norm, clusters)
        else:
            sil, db, ch = 0, 0, 0
        sil_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"  Best K (silhouette): {best_k} (sil={max(sil_scores):.4f})", flush=True)

    # Run K-means with best K
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_clusters = km_best.fit_predict(X_norm)
    results['kmeans'] = {'clusters': km_clusters, 'k': best_k, 'silhouette': max(sil_scores)}

    # Silhouette plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(list(k_range), sil_scores, 'o-', color='#5e5ce6', lw=2, markersize=6)
    axes[0].axvline(x=best_k, color='#ff6b6b', linestyle='--', label=f'Best K={best_k}')
    axes[0].set_xlabel('K (clusters)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('K-means Selection')
    axes[0].legend()

    axes[1].plot(list(k_range), db_scores, 's-', color='#00a389', lw=2, markersize=6)
    axes[1].set_xlabel('K (clusters)')
    axes[1].set_ylabel('Davies-Bouldin (lower=better)')
    axes[1].set_title('Davies-Bouldin Index')

    axes[2].plot(list(k_range), ch_scores, '^-', color='#ff9f0a', lw=2, markersize=6)
    axes[2].set_xlabel('K (clusters)')
    axes[2].set_ylabel('Calinski-Harabasz (higher=better)')
    axes[2].set_title('Calinski-Harabasz Index')
    plt.suptitle(f'{dataset_name} — Clustering Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_cluster_selection.png'), bbox_inches='tight')
    plt.close()
    print("  fig_cluster_selection.png done", flush=True)

    # --- GMM ---
    print("  GMM clustering...", flush=True)
    bic_scores = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='diag')
        gmm.fit(X_norm)
        bic_scores.append(gmm.bic(X_norm))
    best_k_gmm = list(k_range)[np.argmin(bic_scores)]
    gmm_best = GaussianMixture(n_components=best_k_gmm, random_state=42, covariance_type='diag')
    gmm_clusters = gmm_best.fit_predict(X_norm)
    results['gmm'] = {'clusters': gmm_clusters, 'k': best_k_gmm, 'bic': min(bic_scores)}
    print(f"  GMM best K (BIC): {best_k_gmm}", flush=True)

    # --- HDBSCAN (if available) ---
    try:
        import hdbscan
        print("  HDBSCAN clustering...", flush=True)
        hdb = hdbscan.HDBSCAN(min_cluster_size=max(2, n//10), min_samples=1,
                              metric='euclidean', cluster_selection_method='eom')
        hdb_clusters = hdb.fit_predict(X_norm)
        results['hdbscan'] = {'clusters': hdb_clusters,
                              'k': len(np.unique(hdb_clusters[hdb_clusters >= 0])),
                              'n_noise': int(np.sum(hdb_clusters < 0))}
        print(f"  HDBSCAN: {results['hdbscan']['k']} clusters, "
              f"{results['hdbscan']['n_noise']} noise", flush=True)
    except ImportError:
        print("  HDBSCAN not available, skipping", flush=True)

    # --- Cluster visualization (K-means best) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in [(axes[0], pca_coords, 'PCA'), (axes[1], umap_coords, 'UMAP')]:
        for c in range(best_k):
            mask = km_clusters == c
            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=50, alpha=0.8,
                      edgecolors='white', linewidth=0.5, label=f'Cluster {c} (n={mask.sum()})')
        ax.set_title(f'{title} — K-means (K={best_k})')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_clusters.png'), bbox_inches='tight')
    plt.close()
    print("  fig_clusters.png done", flush=True)

    return results


# =========================================================================
# Step 3: Diversity & Individual Variation Analysis
# =========================================================================
def step3_diversity(X, labels, sample_names, dataset_name):
    print("\n[Step 3] Diversity & individual variation analysis...", flush=True)

    n = X.shape[0]
    metrics = []

    for i in range(n):
        row = X[i]
        # Convert counts to integers for diversity
        counts = row.astype(int)
        total = counts.sum()

        # Shannon entropy (on prototype distribution)
        H = shannon_entropy(counts)
        # Simpson diversity
        D = simpson_index(counts)
        # Pielou evenness
        J = pielou_evenness(counts)
        # Richness (number of non-zero prototypes)
        S = int(np.sum(counts > 0))
        # Total sequences
        N = int(total)
        # Berger-Parker dominance (max proportion)
        bp = counts.max() / total if total > 0 else 0
        # Chao1 estimator (on prototypes)
        f1 = int(np.sum(counts == 1))
        f2 = int(np.sum(counts == 2))
        chao1 = S + (f1**2 / (2 * f2)) if f2 > 0 else S + f1 * (f1 - 1) / 2

        metrics.append({
            'sample': sample_names[i],
            'label': int(labels[i]),
            'total_seqs': N,
            'richness': S,
            'shannon': float(H),
            'simpson': float(D),
            'pielou': float(J),
            'berger_parker': float(bp),
            'chao1': float(chao1),
        })

    df_metrics = pd.DataFrame(metrics)

    # Plot diversity metrics
    div_cols = ['richness', 'shannon', 'simpson', 'pielou', 'berger_parker']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(div_cols):
        ax = axes[i]
        vals = df_metrics[col].values
        ax.barh(range(len(vals)), vals, color='#5e5ce6', alpha=0.7, height=0.6,
                edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(df_metrics['sample'], fontsize=7)
        ax.set_xlabel(col)
        ax.set_title(f'{col} distribution')

    # Pairwise distance heatmap
    ax = axes[5]
    from sklearn.metrics import pairwise_distances
    D_mat = pairwise_distances(X, metric='cosine')
    im = ax.imshow(D_mat, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [s[:10] for s in sample_names]
    ax.set_xticklabels(short_names, fontsize=6, rotation=90)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title('Pairwise Cosine Distance')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'{dataset_name} — Diversity & Variation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_diversity.png'), bbox_inches='tight')
    plt.close()
    print("  fig_diversity.png done", flush=True)

    # If labels exist, plot diversity by group
    if len(df_metrics[df_metrics['label'] >= 0]) > 0:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, col in enumerate(div_cols):
            ax = axes[i]
            labeled = df_metrics[df_metrics['label'] >= 0]
            for label_val in sorted(labeled['label'].unique()):
                vals = labeled[labeled['label'] == label_val][col].values
                label_name = 'Control' if label_val == 0 else 'Disease'
                color = '#4a90d9' if label_val == 0 else '#ff6b6b'
                ax.hist(vals, bins=15, alpha=0.6, color=color, label=f'{label_name} (n={len(vals)})',
                        edgecolor='white', linewidth=0.3)
            ax.set_title(col)
            ax.legend(fontsize=8)
        plt.suptitle(f'{dataset_name} — Diversity by Group (Post-hoc)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'fig_diversity_by_group.png'), bbox_inches='tight')
        plt.close()
        print("  fig_diversity_by_group.png done", flush=True)

    return df_metrics


# =========================================================================
# Step 4: Outlier Detection (Isolation Forest)
# =========================================================================
def step4_outlier_detection(X_norm, sample_names, umap_coords, dataset_name):
    print("\n[Step 4] Outlier detection (Isolation Forest)...", flush=True)

    # Isolation Forest
    iso_forest = IsolationForest(random_state=42, contamination='auto', n_estimators=200)
    outlier_labels = iso_forest.fit_predict(X_norm)
    anomaly_scores = -iso_forest.score_samples(X_norm)  # higher = more anomalous

    # KNN distance (average distance to 3 nearest neighbors)
    k = min(3, X_norm.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_norm)
    distances, _ = nn.kneighbors(X_norm)
    knn_scores = distances[:, 1:].mean(axis=1)

    # Sort by anomaly score
    sort_idx = np.argsort(anomaly_scores)[::-1]
    print(f"  Top 5 anomalous samples:", flush=True)
    for rank, idx in enumerate(sort_idx[:5]):
        print(f"    {rank+1}. {sample_names[idx]} (anomaly={anomaly_scores[idx]:.4f}, "
              f"knn_dist={knn_scores[idx]:.4f})", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # UMAP with anomaly coloring
    scatter = axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1],
                              c=anomaly_scores, cmap='YlOrRd', s=50, alpha=0.8,
                              edgecolors='white', linewidth=0.5)
    axes[0].set_title('Anomaly Score (Isolation Forest)')
    axes[0].set_xlabel('UMAP-1')
    axes[0].set_ylabel('UMAP-2')
    plt.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)

    # Bar chart of anomaly scores
    axes[1].barh(range(len(anomaly_scores)), anomaly_scores[sort_idx],
                 color=['#ff6b6b' if outlier_labels[i] == -1 else '#4a90d9'
                        for i in sort_idx], height=0.6, edgecolor='white', linewidth=0.3)
    axes[1].set_yticks(range(len(sort_idx)))
    axes[1].set_yticklabels([sample_names[i] for i in sort_idx], fontsize=7)
    axes[1].set_xlabel('Anomaly Score (higher = more anomalous)')
    axes[1].set_title('Anomaly Ranking')
    axes[1].axvline(x=np.median(anomaly_scores), color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f'{dataset_name} — Outlier Detection', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_outliers.png'), bbox_inches='tight')
    plt.close()
    print("  fig_outliers.png done", flush=True)

    return {'anomaly_scores': anomaly_scores.tolist(),
            'outlier_labels': outlier_labels.tolist(),
            'knn_distances': knn_scores.tolist()}


# =========================================================================
# Step 5: Post-hoc Label Validation (if labels exist)
# =========================================================================
def step5_label_validation(cluster_results, labels, sample_names, X_norm, dataset_name):
    print("\n[Step 5] Post-hoc label validation...", flush=True)

    # Only validate if we have real labels (>=0)
    mask = labels >= 0
    if mask.sum() == 0:
        print("  No labels available — pure unsupervised mode (Scenario 1)", flush=True)
        return None

    if len(np.unique(labels[mask])) < 2:
        print("  Only one label type — cannot validate clustering vs labels", flush=True)
        return None

    validation = {}

    for method, result in cluster_results.items():
        clusters = result['clusters']
        # ARI (Adjusted Rand Index)
        ari = adjusted_rand_score(labels[mask], clusters[mask])
        # NMI (Normalized Mutual Information)
        nmi = normalized_mutual_info_score(labels[mask], clusters[mask])

        # Cluster-label contingency table
        contingency = pd.crosstab(
            pd.Series(clusters[mask], name='Cluster'),
            pd.Series(labels[mask], name='Label')
        )

        # Chi-square test
        try:
            chi2, p_chi2, _, _ = chi2_contingency(contingency.values)
        except ValueError:
            chi2, p_chi2 = 0, 1.0

        # For 2-cluster vs 2-label: use cluster as binary classifier
        auc = None
        if len(np.unique(clusters[mask])) == 2 and len(np.unique(labels[mask])) == 2:
            # Map cluster to label that gives best AUC
            auc1 = roc_auc_score(labels[mask], clusters[mask])
            auc2 = roc_auc_score(labels[mask], 1 - clusters[mask])
            auc = max(auc1, auc2)

        validation[method] = {
            'ari': float(ari),
            'nmi': float(nmi),
            'chi2': float(chi2),
            'p_value': float(p_chi2),
            'auc': float(auc) if auc else None,
            'contingency': contingency.to_dict(),
            'k': result['k'],
        }

        print(f"  {method}: ARI={ari:.4f} | NMI={nmi:.4f} | "
              f"Chi2={chi2:.2f} (p={p_chi2:.4f})" +
              (f" | AUC={auc:.4f}" if auc else ""), flush=True)

    # Visualization: clusters colored by label
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # PCA by label
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_norm)

    for label_val in sorted(np.unique(labels[mask])):
        m = labels == label_val
        label_name = 'Control' if label_val == 0 else 'Disease'
        color = '#4a90d9' if label_val == 0 else '#ff6b6b'
        axes[0].scatter(coords[m, 0], coords[m, 1], c=color, s=50, alpha=0.8,
                       edgecolors='white', linewidth=0.5, label=f'{label_name} (n={m.sum()})')
    axes[0].set_title('PCA — True Labels (Post-hoc)')
    axes[0].legend()

    # UMAP by label (reuse from step1 if available, else recompute)
    try:
        import umap
        umap_coords = umap.UMAP(n_components=2, random_state=42,
                               n_neighbors=min(15, X_norm.shape[0]-1),
                               min_dist=0.1).fit_transform(X_norm)
    except:
        umap_coords = coords

    for label_val in sorted(np.unique(labels[mask])):
        m = labels == label_val
        label_name = 'Control' if label_val == 0 else 'Disease'
        color = '#4a90d9' if label_val == 0 else '#ff6b6b'
        axes[1].scatter(umap_coords[m, 0], umap_coords[m, 1], c=color, s=50, alpha=0.8,
                       edgecolors='white', linewidth=0.5, label=f'{label_name}')
    axes[1].set_title('UMAP — True Labels (Post-hoc)')
    axes[1].legend()

    # Contingency heatmap (K-means best)
    if 'kmeans' in cluster_results:
        km_clusters = cluster_results['kmeans']['clusters']
        contingency = pd.crosstab(
            pd.Series(km_clusters[mask], name='Cluster'),
            pd.Series(labels[mask], name='Label')
        )
        im = axes[2].imshow(contingency.values, cmap='Blues', aspect='auto')
        axes[2].set_xticks(range(len(contingency.columns)))
        axes[2].set_xticklabels(['Control', 'Disease'][:len(contingency.columns)], fontsize=10)
        axes[2].set_yticks(range(len(contingency.index)))
        axes[2].set_yticklabels([f'Cluster {c}' for c in contingency.index], fontsize=10)
        for r in range(contingency.shape[0]):
            for c in range(contingency.shape[1]):
                axes[2].text(c, r, str(contingency.values[r, c]),
                           ha='center', va='center', fontsize=14, fontweight='bold')
        axes[2].set_title('Cluster-Label Contingency (K-means)')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f'{dataset_name} — Post-hoc Label Validation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_label_validation.png'), bbox_inches='tight')
    plt.close()
    print("  fig_label_validation.png done", flush=True)

    return validation


# =========================================================================
# Step 6: Prototype Importance (Unsupervised)
# =========================================================================
def step6_prototype_analysis(X_norm, cluster_results, pca_coords, dataset_name):
    print("\n[Step 6] Unsupervised prototype importance...", flush=True)

    # PCA loadings — which prototypes drive PC1/PC2
    pca = PCA(n_components=2)
    pca.fit(X_norm)
    pc1_loadings = pca.components_[0]
    pc2_loadings = pca.components_[1]

    # Top prototypes by loading magnitude
    top_pc1 = np.argsort(np.abs(pc1_loadings))[::-1][:20]
    top_pc2 = np.argsort(np.abs(pc2_loadings))[::-1][:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(range(20), pc1_loadings[top_pc1][::-1],
                color=['#ff6b6b' if v > 0 else '#4a90d9' for v in pc1_loadings[top_pc1][::-1]],
                height=0.6, edgecolor='white', linewidth=0.3)
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels([f'Proto {i}' for i in top_pc1[::-1]], fontsize=8)
    axes[0].set_xlabel('PC1 Loading')
    axes[0].set_title('Top 20 Prototypes — PC1')

    axes[1].barh(range(20), pc2_loadings[top_pc2][::-1],
                color=['#ff6b6b' if v > 0 else '#4a90d9' for v in pc2_loadings[top_pc2][::-1]],
                height=0.6, edgecolor='white', linewidth=0.3)
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([f'Proto {i}' for i in top_pc2[::-1]], fontsize=8)
    axes[1].set_xlabel('PC2 Loading')
    axes[1].set_title('Top 20 Prototypes — PC2')

    plt.suptitle(f'{dataset_name} — Prototype Importance (PCA Loadings)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_prototype_importance.png'), bbox_inches='tight')
    plt.close()
    print("  fig_prototype_importance.png done", flush=True)

    # Cluster centroids — prototype usage per cluster
    if 'kmeans' in cluster_results:
        km = KMeans(n_clusters=cluster_results['kmeans']['k'], random_state=42, n_init=10)
        km.fit(X_norm)
        cluster_centers = km.cluster_centers_

        # Find discriminative prototypes (highest variance between cluster centers)
        proto_var = np.var(cluster_centers, axis=0)
        top_discrim = np.argsort(proto_var)[::-1][:30]

        fig, ax = plt.subplots(figsize=(14, 6))
        data = cluster_centers[:, top_discrim]
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
        ax.set_yticks(range(cluster_centers.shape[0]))
        ax.set_yticklabels([f'Cluster {c}' for c in range(cluster_centers.shape[0])], fontsize=10)
        ax.set_xticks(range(30))
        ax.set_xticklabels([f'P{top_discrim[i]}' for i in range(30)], fontsize=7, rotation=45)
        ax.set_title('Top 30 Discriminative Prototypes (Cluster Centroids)')
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'fig_cluster_prototypes.png'), bbox_inches='tight')
        plt.close()
        print("  fig_cluster_prototypes.png done", flush=True)

    return {'top_pc1': top_pc1.tolist(), 'top_pc2': top_pc2.tolist()}


# =========================================================================
# HTML Report
# =========================================================================
def generate_html_report(dataset_name, n_samples, has_labels, cluster_results,
                         diversity_df, outlier_results, validation_results,
                         pca_var, output_path):
    def img_b64(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    import base64

    imgs = {}
    for name in ['fig_dim_reduction', 'fig_pca_scree', 'fig_cluster_selection',
                 'fig_clusters', 'fig_diversity', 'fig_outliers',
                 'fig_prototype_importance', 'fig_cluster_prototypes',
                 'fig_diversity_by_group', 'fig_label_validation']:
        path = os.path.join(IMG_DIR, f'{name}.png')
        if os.path.exists(path):
            imgs[name] = img_b64(path)

    # Cluster summary
    cluster_summary = ""
    if 'kmeans' in cluster_results:
        km = cluster_results['kmeans']
        cluster_summary = f"""
        <div class="metric"><div class="value">{km['k']}</div><div class="label">K-means Clusters</div></div>
        <div class="metric"><div class="value">{km['silhouette']:.4f}</div><div class="label">Silhouette</div></div>"""

    # Validation summary
    validation_html = ""
    if validation_results:
        for method, v in validation_results.items():
            auc_str = f"<td style='text-align:center'>{v['auc']:.4f}</td>" if v['auc'] else "<td style='text-align:center'>—</td>"
            validation_html += f"""
            <tr>
                <td style="padding:8px;font-weight:600">{method}</td>
                <td style="text-align:center">{v['k']}</td>
                <td style="text-align:center;font-weight:700;color:{'#00a389' if v['ari'] > 0.3 else '#ff6b6b'}">{v['ari']:.4f}</td>
                <td style="text-align:center">{v['nmi']:.4f}</td>
                <td style="text-align:center">{v['chi2']:.2f}</td>
                <td style="text-align:center">{v['p_value']:.4e}</td>
                {auc_str}
            </tr>"""
    else:
        validation_html = "<tr><td colspan='7' style='text-align:center;padding:20px;color:#8e8e93'>No labels available — pure unsupervised mode (Scenario 1)</td></tr>"

    # Diversity table
    div_table = diversity_df[['sample', 'label', 'total_seqs', 'richness', 'shannon',
                             'simpson', 'pielou', 'berger_parker']].to_html(index=False, float_format='%.4f',
                                                                            classes='div-table')

    # Outlier ranking
    outlier_html = ""
    if outlier_results:
        scores = np.array(outlier_results['anomaly_scores'])
        names = diversity_df['sample'].values
        sort_idx = np.argsort(scores)[::-1]
        for rank, idx in enumerate(sort_idx[:10]):
            outlier_html += f"""
            <tr>
                <td style="padding:6px;font-weight:600">{rank+1}</td>
                <td style="padding:6px">{names[idx]}</td>
                <td style="text-align:center;padding:6px">{scores[idx]:.4f}</td>
            </tr>"""

    scenario = "Scenario 2 (labeled samples as post-hoc validation)" if has_labels else "Scenario 1 (all unlabeled — find individual variation)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unsupervised TRA Repertoire Analysis — {dataset_name}</title>
<style>
:root {{
    --bg: #f5f5f7; --card: #ffffff; --text: #1d1d1f; --green: #00a389;
    --red: #ff6b6b; --blue: #4a90d9; --purple: #5e5ce6; --orange: #ff9f0a;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
.container {{ max-width:1200px; margin:0 auto; padding:24px; }}
.header {{ text-align:center; margin-bottom:32px; }}
.header h1 {{ font-size:28px; font-weight:700; margin-bottom:8px; }}
.header p {{ color:#6e6e73; font-size:15px; }}
.badge {{ display:inline-block; background:var(--purple); color:white; padding:4px 12px; border-radius:12px; font-size:12px; font-weight:600; margin-top:8px; }}
.card {{ background:var(--card); border-radius:16px; padding:24px; margin-bottom:24px; box-shadow:0 2px 12px rgba(0,0,0,0.06); }}
.card h2 {{ font-size:20px; margin-bottom:16px; padding-bottom:12px; border-bottom:2px solid var(--bg); }}
.summary-metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }}
.metric {{ background:#f8f8f8; border-radius:12px; padding:16px; text-align:center; }}
.metric .value {{ font-size:24px; font-weight:700; }}
.metric .label {{ font-size:11px; color:#6e6e73; text-transform:uppercase; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#f8f8f8; padding:12px 8px; text-align:center; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; color:#6e6e73; }}
td {{ padding:8px; }}
img {{ max-width:100%; border-radius:12px; margin:12px 0; }}
.note {{ background:#e8f4fd; border-left:4px solid var(--blue); padding:12px 16px; border-radius:8px; margin:16px 0; font-size:14px; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Unsupervised TRA Repertoire Analysis</h1>
        <p>{dataset_name} | CB TRA Panel (m=10,000) | ESM-2 + K-means + Unsupervised Clustering</p>
        <div class="badge">{scenario}</div>
    </div>

    <div class="summary-metrics">
        <div class="metric"><div class="value">{n_samples}</div><div class="label">Samples</div></div>
        {cluster_summary}
        <div class="metric"><div class="value">{pca_var[0]:.1%}</div><div class="label">PC1 Variance</div></div>
    </div>

    <div class="card">
        <h2>1. Unsupervised Dimensionality Reduction</h2>
        <p>PCA and UMAP projection of samples in CB TRA panel space — no labels used.</p>
        <img src="data:image/png;base64,{imgs.get('fig_dim_reduction','')}">
        <img src="data:image/png;base64,{imgs.get('fig_pca_scree','')}">
    </div>

    <div class="card">
        <h2>2. Unsupervised Clustering</h2>
        <p>K-means with silhouette/Davies-Bouldin/Calinski-Harabasz selection. GMM with BIC. HDBSCAN density-based.</p>
        <img src="data:image/png;base64,{imgs.get('fig_cluster_selection','')}">
        <img src="data:image/png;base64,{imgs.get('fig_clusters','')}">
    </div>

    <div class="card">
        <h2>3. Diversity & Individual Variation</h2>
        <p>Per-sample diversity metrics computed on prototype usage distribution.</p>
        <img src="data:image/png;base64,{imgs.get('fig_diversity','')}">
        {'<img src="data:image/png;base64,' + imgs.get('fig_diversity_by_group','') + '">' if 'fig_diversity_by_group' in imgs else ''}
        {div_table}
    </div>

    <div class="card">
        <h2>4. Outlier Detection (Isolation Forest)</h2>
        <p>Unsupervised anomaly detection — identifies samples that deviate from the group.</p>
        <img src="data:image/png;base64,{imgs.get('fig_outliers','')}">
        <table>
            <thead><tr><th>Rank</th><th>Sample</th><th>Anomaly Score</th></tr></thead>
            <tbody>{outlier_html}</tbody>
        </table>
    </div>

    <div class="card">
        <h2>5. Prototype Importance (Unsupervised)</h2>
        <p>PCA loadings identify which prototypes drive sample separation — without labels.</p>
        <img src="data:image/png;base64,{imgs.get('fig_prototype_importance','')}">
        <img src="data:image/png;base64,{imgs.get('fig_cluster_prototypes','')}">
    </div>

    <div class="card">
        <h2>6. Post-hoc Label Validation</h2>
        {'<p>Labels used to validate unsupervised clusters (ARI, NMI, Chi-square, AUC).</p><img src="data:image/png;base64,' + imgs.get('fig_label_validation','') + '">' if 'fig_label_validation' in imgs else '<p>No labels available — pure unsupervised analysis.</p>'}
        <table>
            <thead>
                <tr><th>Method</th><th>K</th><th>ARI</th><th>NMI</th><th>Chi²</th><th>p-value</th><th>AUC</th></tr>
            </thead>
            <tbody>{validation_html}</tbody>
        </table>
    </div>

    <div class="card">
        <h2>Methods</h2>
        <div style="font-size:14px; line-height:1.8;">
            <p><strong>Projection:</strong> CDR3 → ESM-2 (480 dim) → nearest CB TRA centroid → count matrix (m=10,000)</p>
            <p><strong>Normalization:</strong> L2 normalization</p>
            <p><strong>Dimensionality reduction:</strong> PCA (30 components) + UMAP (2D, unsupervised)</p>
            <p><strong>Clustering:</strong> K-means (silhouette selection), GMM (BIC), HDBSCAN (density)</p>
            <p><strong>Diversity:</strong> Shannon entropy, Simpson, Pielou evenness, Berger-Parker, Chao1</p>
            <p><strong>Outlier detection:</strong> Isolation Forest (200 trees, auto contamination) + KNN distance</p>
            <p><strong>Post-hoc validation:</strong> ARI, NMI, Chi-square test, cluster-label contingency table</p>
        </div>
    </div>
</div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\n  HTML report: {output_path}", flush=True)


# =========================================================================
# Main
# =========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unsupervised TRA Pipeline')
    parser.add_argument('--dataset', type=str, default='ra_tra',
                       choices=['ra_tra', 'ms_pbmc', 'zenodo_ra_hd', 'sle_hd', 'hd_only'],
                       help='Dataset to analyze')
    args = parser.parse_args()

    print("="*60, flush=True)
    print("  Unsupervised TRA Repertoire Analysis Pipeline", flush=True)
    print("="*60, flush=True)

    # Load dataset
    loaders = {
        'ra_tra': load_ra_tra,
        'ms_pbmc': load_ms_pbmc,
        'zenodo_ra_hd': load_zenodo_ra_hd,
        'sle_hd': load_sle_hd,
        'hd_only': load_hd_only,
    }
    samples, dataset_name = loaders[args.dataset]()
    n_samples = len(samples)
    print(f"  Dataset: {dataset_name} ({n_samples} samples)", flush=True)

    # Dataset-specific output
    global IMG_DIR
    IMG_DIR = os.path.join(OUTPUT_DIR, args.dataset, "imgs")
    os.makedirs(IMG_DIR, exist_ok=True)

    has_labels = any(s['label'] >= 0 for s in samples)
    n_labeled = sum(1 for s in samples if s['label'] >= 0)
    print(f"  Labeled samples: {n_labeled}/{n_samples}", flush=True)
    if has_labels:
        print(f"  Mode: Scenario 2 (labeled samples for post-hoc validation)", flush=True)
    else:
        print(f"  Mode: Scenario 1 (all unlabeled — find individual variation)", flush=True)

    # Fast path: load pre-computed RA-TRA matrix if available
    mat_path = os.path.join(PANEL_DIR, "ra_tra_cb_matrix.npy")
    lbl_path = os.path.join(PANEL_DIR, "ra_tra_cb_labels.npy")
    if args.dataset == 'ra_tra' and os.path.exists(mat_path):
        print(f"\nLoading pre-computed RA-TRA matrix...", flush=True)
        X = np.load(mat_path)
        labels = np.load(lbl_path).astype(int)
        sample_names = [s['name'] for s in samples]
        print(f"  Matrix: {X.shape} | Labels: {Counter(labels.tolist())}", flush=True)
    else:
        # Load CB TRA panel and project
        panel_path = os.path.join(PANEL_DIR, f"cb_tra_reference_panel_m{M_TARGET}.pkl")
        print(f"\nLoading CB TRA panel...", flush=True)
        with open(panel_path, 'rb') as f:
            panel_data = pickle.load(f)
        centroids = panel_data['centroids']
        print(f"  Panel: {centroids.shape}", flush=True)
        cached_emb = {}
        X, labels, sample_names = project_dataset(samples, centroids, dataset_name, cached_emb)

    # L2 normalize
    X_norm = normalize(X.astype(np.float64), norm='l2', axis=1)

    # Step 1: Dimensionality reduction
    pca_coords, umap_coords, pca_var = step1_dim_reduction(X_norm, sample_names, dataset_name)

    # Step 2: Clustering
    cluster_results = step2_clustering(X_norm, pca_coords, umap_coords, sample_names, labels, dataset_name)

    # Step 3: Diversity
    diversity_df = step3_diversity(X, labels, sample_names, dataset_name)

    # Step 4: Outlier detection
    outlier_results = step4_outlier_detection(X_norm, sample_names, umap_coords, dataset_name)

    # Step 5: Post-hoc validation (if labels exist)
    validation_results = step5_label_validation(cluster_results, labels, sample_names, X_norm, dataset_name)

    # Step 6: Prototype importance
    step6_prototype_analysis(X_norm, cluster_results, pca_coords, dataset_name)

    # Save results
    results_json = {
        'dataset': dataset_name,
        'n_samples': n_samples,
        'has_labels': has_labels,
        'pca_variance': pca_var.tolist()[:10],
        'cluster_results': {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                 for kk, vv in v.items()}
                            for k, v in cluster_results.items()},
        'validation': validation_results,
    }
    json_path = os.path.join(OUTPUT_DIR, f'unsupervised_{args.dataset}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    # Diversity CSV
    diversity_df.to_csv(os.path.join(OUTPUT_DIR, f'unsupervised_{args.dataset}_diversity.csv'), index=False)

    # HTML report
    html_path = os.path.join(OUTPUT_DIR, f'unsupervised_{args.dataset}_report.html')
    generate_html_report(dataset_name, n_samples, has_labels, cluster_results,
                        diversity_df, outlier_results, validation_results,
                        pca_var, html_path)

    print(f"\n{'='*60}", flush=True)
    print(f"  DONE — {dataset_name}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
