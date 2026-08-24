#!/usr/bin/env python3
"""
Cross-Disease TRA Panel Testing
================================
Tests the CordBlood TRA reference panel (m=10,000) on multiple autoimmune
disease datasets with TRA chain data:

1. MS (GEO GSE232343) — 4 MS vs 4 IIH controls (PBMC)
2. MS (GEO GSE232343) — 8 MS vs 8 IIH controls (PBMC + CSF)
3. SLE (GEO GSE254176) — 18 SLE vs Zenodo HD controls
4. Zenodo RA vs HD — from pan-disease h5ad
5. Zenodo PsA vs HD — from pan-disease h5ad

For each: project to CB TRA panel → L2 norm → Linear SVM 5-fold CV → AUC
"""
import os, sys, json, time, pickle, warnings, glob
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_curve, accuracy_score, recall_score)
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
TIER2_DIR = os.path.join(WORK_DIR, "CDRscope-v2", "inst", "python", "tier2")
sys.path.insert(0, TIER2_DIR)

PANEL_DIR = os.path.join(WORK_DIR, "cordblood_tra_panel")
OUTPUT_DIR = os.path.join(WORK_DIR, "cross_disease_tra_results")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

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

DISEASE_COLORS = {
    'RA': '#ff6b6b', 'MS': '#5e5ce6', 'SLE': '#00a389',
    'AS': '#ff9f0a', 'PsA': '#bf5af2', 'Control': '#4a90d9',
}


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / (np.sqrt((np.std(a)**2 + np.std(b)**2) / 2) + 1e-10)


# =========================================================================
# ESM-2 Embedding
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
# Dataset loaders — return list of {'df': pd.DataFrame, 'label': 0/1, 'name': str}
# df must have 'junction_aa' column (CDR3 amino acid)
# =========================================================================
def load_ms_pbmc():
    """MS PBMC: 4 MS vs 4 IIH controls"""
    path = os.path.join(WORK_DIR, "geo_ms_tcr", "ms_tra_pseudobulk.csv")
    df = pd.read_csv(path)
    # Filter PBMC only
    df = df[df['tissue'] == 'PBMC']
    samples = []
    for sample_name in df['sample'].unique():
        sdf = df[df['sample'] == sample_name]
        disease = sdf['disease'].iloc[0]
        if disease == 'MS':
            label = 1
        else:
            label = 0
        samples.append({
            'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
            'label': label,
            'name': sample_name
        })
    return samples, "MS-PBMC (4vs4)"


def load_ms_all():
    """MS PBMC+CSF: 8 MS vs 8 IIH controls"""
    path = os.path.join(WORK_DIR, "geo_ms_tcr", "ms_tra_pseudobulk.csv")
    df = pd.read_csv(path)
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
    return samples, "MS-All (8vs8)"


def load_sle_with_hd():
    """SLE: 18 SLE vs Zenodo HD controls"""
    sle_path = os.path.join(WORK_DIR, "zenodo_scTCR", "sle_tra_pseudobulk.csv")
    hd_path = os.path.join(WORK_DIR, "zenodo_scTCR", "zenodo_autoimmune_tra.csv")

    sle_df = pd.read_csv(sle_path)
    samples = []

    # SLE samples
    for sample_name in sle_df['sample'].unique():
        sdf = sle_df[sle_df['sample'] == sample_name]
        count_col = 'count' if 'count' in sdf.columns else None
        if count_col is None:
            sdf = sdf.copy()
            sdf['count'] = 1
            count_col = 'count'
        samples.append({
            'df': sdf[['cdr3', count_col]].rename(columns={'cdr3': 'junction_aa', count_col: 'duplicate_count'}),
            'label': 1,
            'name': f"SLE_{sample_name}"
        })

    # HD controls from Zenodo (if available)
    if os.path.exists(hd_path):
        hd_df = pd.read_csv(hd_path)
        hd_df = hd_df[hd_df['disease'] == 'HD']
        for sample_name in hd_df['sample'].unique():
            sdf = hd_df[hd_df['sample'] == sample_name]
            samples.append({
                'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
                'label': 0,
                'name': f"HD_{sample_name}"
            })
    else:
        print("  WARNING: Zenodo HD not available, using only SLE samples (no controls)")
        return [], "SLE vs HD (pending)"

    return samples, f"SLE vs HD ({sum(1 for s in samples if s['label']==1)}vs{sum(1 for s in samples if s['label']==0)})"


def load_zenodo_disease(disease_name):
    """Load disease vs HD from Zenodo autoimmune extraction"""
    path = os.path.join(WORK_DIR, "zenodo_scTCR", "zenodo_autoimmune_tra.csv")
    if not os.path.exists(path):
        return [], f"{disease_name} vs HD (pending)"

    df = pd.read_csv(path)
    samples = []

    # Disease samples
    dis_df = df[df['disease'] == disease_name]
    for sample_name in dis_df['sample'].unique():
        sdf = dis_df[dis_df['sample'] == sample_name]
        samples.append({
            'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
            'label': 1,
            'name': f"{disease_name}_{sample_name}"
        })

    # HD controls
    hd_df = df[df['disease'] == 'HD']
    for sample_name in hd_df['sample'].unique():
        sdf = hd_df[hd_df['sample'] == sample_name]
        samples.append({
            'df': sdf[['cdr3', 'count']].rename(columns={'cdr3': 'junction_aa', 'count': 'duplicate_count'}),
            'label': 0,
            'name': f"HD_{sample_name}"
        })

    n_dis = sum(1 for s in samples if s['label'] == 1)
    n_ctrl = sum(1 for s in samples if s['label'] == 0)
    return samples, f"{disease_name} vs HD ({n_dis}vs{n_ctrl})"


# =========================================================================
# Projection: samples → CB TRA panel space
# =========================================================================
def project_dataset(samples, centroids, dataset_name, cached_embeddings=None):
    print(f"\n{'='*60}", flush=True)
    print(f"  Projecting {dataset_name} -> m={centroids.shape[0]} (CB TRA panel)", flush=True)
    print(f"{'='*60}", flush=True)

    # Collect all unique sequences
    all_seqs = set()
    for s in samples:
        df = s['df']
        seqs = df['junction_aa'].dropna().values
        for seq in seqs:
            if isinstance(seq, str) and len(seq) >= 8 and set(seq) <= STANDARD_AA:
                all_seqs.add(seq)
    all_seqs = sorted(all_seqs)
    print(f"  Unique valid sequences: {len(all_seqs):,}", flush=True)

    # Embed
    new_seqs = [s for s in all_seqs if s not in cached_embeddings] if cached_embeddings else all_seqs
    if new_seqs:
        print(f"  New sequences to embed: {len(new_seqs):,}", flush=True)
        new_emb = compute_esm2_embeddings(new_seqs)
        for seq, emb in zip(new_seqs, new_emb):
            cached_embeddings[seq] = emb
    else:
        print(f"  All sequences already embedded (cached)", flush=True)

    embeddings = np.array([cached_embeddings[s] for s in all_seqs])

    # Assign to centroids
    print(f"  Assigning to {centroids.shape[0]} centroids...", flush=True)
    assignments = assign_to_centroids(embeddings, centroids)
    seq_to_centroid = {seq: assignments[i] for i, seq in enumerate(all_seqs)}

    # Build count matrix
    m = centroids.shape[0]
    n = len(samples)
    count_matrix = np.zeros((n, m), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)

    for i, s in enumerate(samples):
        df = s['df']
        seqs = df['junction_aa'].dropna().values
        count_col = 'duplicate_count' if 'duplicate_count' in df.columns else None
        if count_col:
            counts = df[count_col].fillna(1).values
        else:
            counts = np.ones(len(seqs))
        for seq, cnt in zip(seqs, counts):
            if isinstance(seq, str) and seq in seq_to_centroid:
                count_matrix[i, seq_to_centroid[seq]] += float(cnt)
        labels[i] = s['label']

    print(f"  Matrix: {count_matrix.shape} | Labels: {np.bincount(labels)} (0=ctrl, 1=case)", flush=True)
    return count_matrix, labels


# =========================================================================
# Classification: Linear SVM, 5-fold CV
# =========================================================================
def classify(X, labels, dataset_name):
    X_norm = normalize(X.astype(np.float64), norm='l2', axis=1)
    n_splits = min(5, min(np.bincount(labels)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = np.zeros(len(labels))
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_norm, labels)):
        svm = LinearSVC(C=0.1, random_state=42, max_iter=5000, dual=False)
        svm.fit(X_norm[train_idx], labels[train_idx])
        scores[test_idx] = svm.decision_function(X_norm[test_idx])
        fold_auc = roc_auc_score(labels[test_idx], scores[test_idx])
        fold_aucs.append(fold_auc)

    auc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)
    fpr, tpr, thresh = roc_curve(labels, scores)
    best_thresh = thresh[np.argmax(tpr - fpr)]
    y_pred = (scores >= best_thresh).astype(int)

    f1 = f1_score(labels, y_pred)
    mcc = matthews_corrcoef(labels, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    acc = accuracy_score(labels, y_pred)
    sens = recall_score(labels, y_pred, zero_division=0)
    spec = recall_score(1 - labels, 1 - y_pred, zero_division=0)

    d_score = cohens_d(scores[labels == 1], scores[labels == 0])

    results = {
        'dataset': dataset_name,
        'n_samples': len(labels),
        'n_control': int(np.sum(labels == 0)),
        'n_case': int(np.sum(labels == 1)),
        'auc': float(auc),
        'auc_pr': float(auc_pr),
        'f1': float(f1),
        'mcc': float(mcc),
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'fold_aucs': [float(a) for a in fold_aucs],
        'mean_fold_auc': float(np.mean(fold_aucs)),
        'std_fold_auc': float(np.std(fold_aucs)),
        'cohens_d': float(d_score),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'scores': scores.tolist(),
        'labels': labels.tolist(),
    }

    print(f"\n  {'='*50}", flush=True)
    print(f"  {dataset_name} — Linear SVM (L2-norm, CB TRA m={X.shape[1]})", flush=True)
    print(f"  {'='*50}", flush=True)
    print(f"  AUC-ROC:     {auc:.4f}", flush=True)
    print(f"  AUC-PR:      {auc_pr:.4f}", flush=True)
    print(f"  F1:          {f1:.4f}", flush=True)
    print(f"  Cohen's d:   {d_score:.2f}", flush=True)
    print(f"  Fold AUCs:   {['%.4f' % a for a in fold_aucs]}", flush=True)

    return results, scores, X_norm


# =========================================================================
# Visualization
# =========================================================================
def plot_roc_overlay(all_results, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#ff6b6b', '#5e5ce6', '#00a389', '#ff9f0a', '#bf5af2', '#4a90d9']
    for i, (r, color) in enumerate(zip(all_results, colors)):
        fpr = r['fpr']
        tpr = r['tpr']
        auc = r['auc']
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{r['dataset']} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Cross-Disease TRA Classification — ROC Curves (CB TRA Panel m=10,000)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_auc_bar(all_results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r['dataset'] for r in all_results]
    aucs = [r['auc'] for r in all_results]
    n_samples = [r['n_samples'] for r in all_results]
    colors = [DISEASE_COLORS.get(r['dataset'].split()[0], '#5e5ce6') for r in all_results]
    bars = ax.barh(names, aucs, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    for bar, auc, n in zip(bars, aucs, n_samples):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{auc:.4f} (n={n})', va='center', fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('Cross-Disease TRA Classification Performance (CB TRA Panel)', fontsize=14)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_score_distributions(all_results, save_path):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    for i, r in enumerate(all_results):
        ax = axes[0, i]
        scores = np.array(r['scores'])
        labels = np.array(r['labels'])
        ctrl = labels == 0
        case = labels == 1
        bins = np.linspace(min(scores.min(), -3), max(scores.max(), 3), 25)
        ax.hist(scores[ctrl], bins=bins, color='#4a90d9', alpha=0.7, label='Control', edgecolor='white', linewidth=0.3)
        ax.hist(scores[case], bins=bins, color='#ff6b6b', alpha=0.7, label='Disease', edgecolor='white', linewidth=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f"{r['dataset']}\n(AUC={r['auc']:.4f}, d={r['cohens_d']:.2f})", fontsize=11)
        ax.set_xlabel('SVM Score')
        ax.legend(fontsize=9)
    plt.suptitle('SVM Score Distributions (CB TRA Panel m=10,000)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_combined_pca(all_results, all_X_norm, all_labels, save_path):
    """PCA of all datasets combined (shared panel space)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'Control': '#4a90d9', 'Disease': '#ff6b6b'}
    markers = {'MS-PBMC': 'o', 'MS-All': 's', 'SLE': 'D', 'RA': '^', 'PsA': 'P', 'AS': 'X'}

    for r, X, labels in zip(all_results, all_X_norm, all_labels):
        if X is None:
            continue
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        dataset_key = r['dataset'].split()[0]
        marker = markers.get(dataset_key, 'o')
        ctrl = labels == 0
        case = labels == 1
        ax.scatter(coords[ctrl, 0], coords[ctrl, 1], c='#4a90d9', marker=marker, s=40, alpha=0.6, edgecolors='white', linewidth=0.3,
                   label=f"{r['dataset']} Ctrl")
        ax.scatter(coords[case, 0], coords[case, 1], c='#ff6b6b', marker=marker, s=40, alpha=0.6, edgecolors='white', linewidth=0.3,
                   label=f"{r['dataset']} Case")

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA of All Datasets in CB TRA Panel Space', fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc='best')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


# =========================================================================
# HTML Report
# =========================================================================
def generate_html_report(all_results, img_dir, output_path):
    import base64
    def img_b64(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    roc_img = img_b64(os.path.join(img_dir, 'fig_roc_overlay.png'))
    bar_img = img_b64(os.path.join(img_dir, 'fig_auc_bar.png'))
    score_img = img_b64(os.path.join(img_dir, 'fig_score_dist.png'))
    pca_img = img_b64(os.path.join(img_dir, 'fig_combined_pca.png'))

    rows = ""
    for r in all_results:
        color = DISEASE_COLORS.get(r['dataset'].split()[0], '#5e5ce6')
        rows += f"""
        <tr style="border-bottom: 1px solid #e0e0e0;">
            <td style="padding:10px;font-weight:600;color:{color}">{r['dataset']}</td>
            <td style="text-align:center">{r['n_control']} vs {r['n_case']}</td>
            <td style="text-align:center;font-weight:700;color:{'#00a389' if r['auc'] > 0.8 else '#ff9f0a' if r['auc'] > 0.7 else '#ff6b6b'}">{r['auc']:.4f}</td>
            <td style="text-align:center">{r['auc_pr']:.4f}</td>
            <td style="text-align:center">{r['f1']:.4f}</td>
            <td style="text-align:center">{r['accuracy']:.4f}</td>
            <td style="text-align:center">{r['sensitivity']:.4f}</td>
            <td style="text-align:center">{r['specificity']:.4f}</td>
            <td style="text-align:center">{r['cohens_d']:.2f}</td>
            <td style="text-align:center;font-size:11px">{', '.join(f'{a:.3f}' for a in r['fold_aucs'])}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Disease TRA Panel Testing Report</title>
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
.card {{ background:var(--card); border-radius:16px; padding:24px; margin-bottom:24px; box-shadow:0 2px 12px rgba(0,0,0,0.06); }}
.card h2 {{ font-size:20px; margin-bottom:16px; padding-bottom:12px; border-bottom:2px solid var(--bg); }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#f8f8f8; padding:12px 8px; text-align:center; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; color:#6e6e73; }}
td {{ padding:8px; }}
.summary-metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }}
.metric {{ background:#f8f8f8; border-radius:12px; padding:16px; text-align:center; }}
.metric .value {{ font-size:24px; font-weight:700; }}
.metric .label {{ font-size:11px; color:#6e6e73; text-transform:uppercase; margin-top:4px; }}
img {{ max-width:100%; border-radius:12px; margin:12px 0; }}
.note {{ background:#fff3cd; border-left:4px solid #ffc107; padding:12px 16px; border-radius:8px; margin:16px 0; font-size:14px; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Cross-Disease TRA Classification Report</h1>
        <p>CordBlood TRA Reference Panel (m=10,000) | ESM-2 + K-means | Linear SVM (L2-norm, 5-fold CV)</p>
    </div>

    <div class="summary-metrics">
        <div class="metric"><div class="value">{len(all_results)}</div><div class="label">Datasets Tested</div></div>
        <div class="metric"><div class="value">{sum(r['n_samples'] for r in all_results)}</div><div class="label">Total Samples</div></div>
        <div class="metric"><div class="value">{np.mean([r['auc'] for r in all_results]):.4f}</div><div class="label">Mean AUC</div></div>
        <div class="metric"><div class="value">{max(r['auc'] for r in all_results):.4f}</div><div class="label">Best AUC</div></div>
    </div>

    <div class="card">
        <h2>Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Ctrl vs Case</th>
                    <th>AUC-ROC</th>
                    <th>AUC-PR</th>
                    <th>F1</th>
                    <th>Accuracy</th>
                    <th>Sens.</th>
                    <th>Spec.</th>
                    <th>Cohen's d</th>
                    <th>Fold AUCs</th>
                </tr>
            </thead>
            <tbody>{rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>ROC Curves — Cross-Disease Comparison</h2>
        <img src="data:image/png;base64,{roc_img}" alt="ROC Curves">
    </div>

    <div class="card">
        <h2>AUC-ROC Comparison</h2>
        <img src="data:image/png;base64,{bar_img}" alt="AUC Bar Chart">
    </div>

    <div class="card">
        <h2>SVM Score Distributions</h2>
        <img src="data:image/png;base64,{score_img}" alt="Score Distributions">
    </div>

    <div class="card">
        <h2>Combined PCA (Shared Panel Space)</h2>
        <img src="data:image/png;base64,{pca_img}" alt="Combined PCA">
    </div>

    <div class="card">
        <h2>Methods</h2>
        <div style="font-size:14px; line-height:1.8;">
            <p><strong>Reference Panel:</strong> CordBlood TRA (1,318,977 unique CDR3 sequences) → ESM-2 embedding (480 dim) → K-means quantization (m=10,000, 77.7% variance explained)</p>
            <p><strong>Sample Projection:</strong> CDR3 sequences → ESM-2 embedding → nearest centroid assignment → count per prototype → m=10,000 dimensional vector per sample</p>
            <p><strong>Normalization:</strong> L2 normalization of count vectors</p>
            <p><strong>Classification:</strong> Linear SVM (C=0.1), 5-fold stratified cross-validation</p>
            <p><strong>Datasets:</strong> MS (GSE232343, PBMC and CSF), SLE (GSE254176), Zenodo autoimmune panel (RA, PsA, AS vs HD)</p>
        </div>
    </div>
</div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\n  HTML report saved: {output_path}", flush=True)


# =========================================================================
# Main
# =========================================================================
def main():
    print("="*60, flush=True)
    print("  Cross-Disease TRA Panel Testing", flush=True)
    print("  CB TRA Panel (m=10,000) on multiple autoimmune diseases", flush=True)
    print("="*60, flush=True)

    # Load CB TRA panel
    panel_path = os.path.join(PANEL_DIR, f"cb_tra_reference_panel_m{M_TARGET}.pkl")
    print(f"\nLoading CB TRA panel from {panel_path}...", flush=True)
    with open(panel_path, 'rb') as f:
        panel_data = pickle.load(f)
    centroids = panel_data['centroids']
    print(f"  Panel: {centroids.shape} centroids, {centroids.shape[1]} dim", flush=True)

    # Define datasets to test
    dataset_configs = [
        ('ms_pbmc', load_ms_pbmc),
        ('ms_all', load_ms_all),
        ('sle_hd', load_sle_with_hd),
        ('zenodo_ra', lambda: load_zenodo_disease('RA')),
        ('zenodo_psa', lambda: load_zenodo_disease('PsA')),
    ]

    all_results = []
    all_X_norm = []
    all_labels = []
    cached_embeddings = {}  # Cache across datasets

    for key, loader in dataset_configs:
        print(f"\n{'#'*60}", flush=True)
        print(f"  Dataset: {key}", flush=True)
        print(f"{'#'*60}", flush=True)

        try:
            samples, dataset_name = loader()
            if len(samples) == 0:
                print(f"  SKIP: No samples loaded for {key}", flush=True)
                continue

            n_ctrl = sum(1 for s in samples if s['label'] == 0)
            n_case = sum(1 for s in samples if s['label'] == 1)
            if n_ctrl < 2 or n_case < 2:
                print(f"  SKIP: Too few samples ({n_ctrl} ctrl vs {n_case} case) for {key}", flush=True)
                continue

            X, labels = project_dataset(samples, centroids, dataset_name, cached_embeddings)
            results, scores, X_norm = classify(X, labels, dataset_name)
            all_results.append(results)
            all_X_norm.append(X_norm)
            all_labels.append(labels)

        except Exception as e:
            print(f"  ERROR for {key}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNo datasets were successfully tested!", flush=True)
        return

    # Generate visualizations
    print(f"\n{'='*60}", flush=True)
    print("  Generating visualizations...", flush=True)
    print(f"{'='*60}", flush=True)

    plot_roc_overlay(all_results, os.path.join(IMG_DIR, 'fig_roc_overlay.png'))
    plot_auc_bar(all_results, os.path.join(IMG_DIR, 'fig_auc_bar.png'))
    plot_score_distributions(all_results, os.path.join(IMG_DIR, 'fig_score_dist.png'))
    plot_combined_pca(all_results, all_X_norm, all_labels, os.path.join(IMG_DIR, 'fig_combined_pca.png'))

    # Save results JSON
    json_path = os.path.join(OUTPUT_DIR, 'cross_disease_tra_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results JSON: {json_path}", flush=True)

    # Generate HTML report
    html_path = os.path.join(OUTPUT_DIR, 'cross_disease_tra_report.html')
    generate_html_report(all_results, IMG_DIR, html_path)

    print(f"\n{'='*60}", flush=True)
    print(f"  DONE — {len(all_results)} datasets tested", flush=True)
    print(f"{'='*60}", flush=True)
    for r in all_results:
        print(f"  {r['dataset']}: AUC={r['auc']:.4f} | F1={r['f1']:.4f} | d={r['cohens_d']:.2f}", flush=True)


if __name__ == '__main__':
    main()
