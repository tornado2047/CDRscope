#!/usr/bin/env python3
"""
CordBlood TRA Reference Panel — Build & RA-TRA Validation
============================================================
Build a TCR reference panel using ONLY CordBlood TRA data,
then project RA-TRA samples onto it for classification.

Chain strategy: 7 chains (TRA/TRB/TRG/TRD/IGH/IGL/IGK) will each get
their own reference panel. This script handles TRA (alpha chain).

Rationale:
  Cord blood represents naive/unexposed TCR repertoires.
  Using it as the reference panel tests whether the quantization space
  is truly disease-agnostic (built from healthy baseline, not disease data).

Data:
  - Reference: CordBlood TRA (synthetic_CB_library.csv, ~2.3M rows)
  - Validation: RA-TRA (Aterido 2024, 210 control + 335 patient = 545 samples)

Flow:
  1. Load CordBlood TRA CDR3 sequences → reference pool
  2. ESM-2 embedding → 480-dim vectors
  3. K-means quantization → m=10,000 prototypes
  4. Saturation analysis (does m saturate with CB TRA data?)
  5. Project RA-TRA samples onto CB TRA panel
  6. Linear SVM classification (5-fold CV)
  7. HTML report
"""
import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, roc_curve, accuracy_score, recall_score
)
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

TIER2_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
sys.path.insert(0, TIER2_DIR)

CB_TRA_FILE = os.path.join(os.path.expanduser("~"), ".trae-cn", "attachments",
                            "6a7c2cace78ca95f7748ffbb",
                            "7ceaa732-c60d-455f-a748-f58a8e871501_6a62fcd8-b1c3-4cc6-9e81-d6e3469504e7_synthetic_CB_library.csv")
OUTPUT_DIR = os.path.join(WORK_DIR, "cordblood_tra_panel")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"
EMBED_DIM = 480
M_TARGET = 10000

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

CTRL_COLOR = '#4a90d9'
PAT_COLOR = '#ff6b6b'
ACCENT = '#5e5ce6'
GREEN = '#00a389'
ORANGE = '#ff9f0a'
CB_COLOR = '#8e8e93'


# =========================================================================
# Step 1: Load CordBlood TRA data → reference pool
# =========================================================================
def load_cordblood_tra_pool():
    """Load all unique CDR3 sequences from CordBlood TRA library."""
    print(f"  Loading CordBlood TRA: {os.path.basename(CB_TRA_FILE)}")

    if not os.path.exists(CB_TRA_FILE):
        print(f"  ERROR: {CB_TRA_FILE} not found")
        sys.exit(1)

    # Read in chunks for memory efficiency
    all_seqs = Counter()
    chunk_size = 200000
    n_rows = 0

    for chunk in pd.read_csv(CB_TRA_FILE, chunksize=chunk_size, dtype=str):
        n_rows += len(chunk)
        seqs = chunk['junction_aa'].dropna().values
        if 'duplicate_count' in chunk.columns:
            counts = chunk['duplicate_count'].fillna(1).astype(float).values
        else:
            counts = np.ones(len(seqs))

        for s, c in zip(seqs, counts):
            s = str(s).strip()
            if len(s) >= 8 and all(aa in STANDARD_AA for aa in s):
                all_seqs[s] += int(c)

        if n_rows % 500000 == 0:
            print(f"    {n_rows:,} rows read, {len(all_seqs):,} unique seqs so far")

    print(f"  Total rows: {n_rows:,}")
    print(f"  CordBlood TRA reference pool: {len(all_seqs):,} unique CDR3 sequences")
    return all_seqs


# =========================================================================
# Step 2: ESM-2 Embedding
# =========================================================================
def compute_esm2_embeddings(sequences, model_name=ESM2_MODEL, device='auto', batch_size=256):
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    print(f"  Loading ESM-2 model ({model_name}) on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
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

        inputs = tokenizer(spaced, return_tensors='pt', padding=True,
                           truncation=True, max_length=50)
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

    print(f"  Embedding complete: {n:,} sequences in {time.time()-start:.0f}s", flush=True)
    return embeddings


# =========================================================================
# Step 3: K-means Quantization
# =========================================================================
def assign_to_centroids(embeddings, centroids, batch_size=10000):
    from scipy.spatial.distance import cdist as _cdist
    n = embeddings.shape[0]
    assignments = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch_size):
        batch = embeddings[i:i+batch_size]
        dists = _cdist(batch, centroids, metric='sqeuclidean')
        assignments[i:i+batch_size] = np.argmin(dists, axis=1)
    return assignments


def train_reference_panel(embeddings, n_prototypes, random_state=42):
    n = embeddings.shape[0]
    print(f"  Training reference panel: {n:,} sequences -> {n_prototypes} prototypes", flush=True)

    kmeans = MiniBatchKMeans(
        n_clusters=n_prototypes,
        batch_size=min(10000, n),
        n_init=3,
        max_iter=100,
        random_state=random_state,
        verbose=0
    )
    labels = kmeans.fit_predict(embeddings)
    inertia = kmeans.inertia_

    total_var = np.sum(np.var(embeddings, axis=0)) * n
    variance_explained = 1 - inertia / total_var

    cluster_sizes = np.bincount(labels, minlength=n_prototypes)
    print(f"  Inertia: {inertia:,.0f}", flush=True)
    print(f"  Variance explained: {variance_explained:.4f} ({variance_explained*100:.1f}%)", flush=True)
    print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"median={np.median(cluster_sizes):.0f}", flush=True)

    return {
        'centroids': kmeans.cluster_centers_.astype(np.float32),
        'variance_explained': float(variance_explained),
        'cluster_sizes': cluster_sizes.tolist(),
        'n_prototypes': n_prototypes,
        'n_sequences': n,
    }


# =========================================================================
# Step 4: Saturation analysis (CB TRA only)
# =========================================================================
def saturation_analysis(embeddings, target_ve=0.80, random_state=42):
    n_total = embeddings.shape[0]
    subsample_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    m_candidates = [200, 500, 1000, 2000, 5000, 10000]

    results = []

    for frac in subsample_fracs:
        pool_size = int(n_total * frac)
        print(f"\n  Saturation test: pool_size = {pool_size:,} (frac={frac})", flush=True)

        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_total, size=pool_size, replace=False)
        sub_emb = embeddings[idx]

        scan = []
        max_m = min(10000, max(50, pool_size // 10))
        test_ms = [m for m in m_candidates if m <= max_m]

        for m in test_ms:
            km = MiniBatchKMeans(n_clusters=m, batch_size=min(10000, pool_size),
                                 n_init=3, max_iter=100, random_state=random_state)
            km.fit(sub_emb)
            ve = 1 - km.inertia_ / (np.sum(np.var(sub_emb, axis=0)) * pool_size)
            scan.append({'m': m, 'variance_explained': float(ve)})
            print(f"    m={m}: VE={ve:.4f}", flush=True)

        ms = np.array([r['m'] for r in scan])
        ves = np.array([r['variance_explained'] for r in scan])

        if np.max(ves) >= target_ve:
            idx_above = np.where(ves >= target_ve)[0][0]
            if idx_above > 0:
                m_lo, ve_lo = ms[idx_above - 1], ves[idx_above - 1]
                m_hi, ve_hi = ms[idx_above], ves[idx_above]
                frac_interp = (target_ve - ve_lo) / (ve_hi - ve_lo)
                m_needed = m_lo + frac_interp * (m_hi - m_lo)
            else:
                m_needed = float(ms[0])
        else:
            m_needed = float(ms[-1])

        results.append({
            'pool_size': pool_size,
            'fraction': frac,
            'm_needed': round(m_needed),
            'scan_results': scan,
        })
        print(f"  -> m needed for {target_ve*100:.0f}% VE: ~{round(m_needed)}", flush=True)

    return results


# =========================================================================
# Step 5: Load RA-TRA samples
# =========================================================================
def load_ra_tra_samples():
    """Load RA TRA per-sample data for validation."""
    from cross_disease_benchmark import load_ra_dataset
    samples = load_ra_dataset(chain='TRA')
    print(f"  RA-TRA: {len(samples)} samples")
    return samples


# =========================================================================
# Step 6: Project RA-TRA onto CB TRA panel
# =========================================================================
def project_dataset(samples, centroids, dataset_name):
    print(f"\n{'='*60}", flush=True)
    print(f"  Projecting {dataset_name} -> m={centroids.shape[0]} space (CB TRA panel)", flush=True)
    print(f"{'='*60}", flush=True)

    all_seqs = set()
    for s in samples:
        df = s['df']
        seqs = df['junction_aa'].dropna().values
        for seq in seqs:
            if isinstance(seq, str) and len(seq) >= 8:
                all_seqs.add(seq)
    all_seqs = sorted(all_seqs)
    print(f"  Unique sequences: {len(all_seqs):,}", flush=True)

    print(f"  Computing ESM-2 embeddings...", flush=True)
    embeddings = compute_esm2_embeddings(all_seqs)

    print(f"  Assigning to {centroids.shape[0]} centroids...", flush=True)
    assignments = assign_to_centroids(embeddings, centroids)

    seq_to_centroid = {seq: assignments[i] for i, seq in enumerate(all_seqs)}

    m = centroids.shape[0]
    n = len(samples)
    count_matrix = np.zeros((n, m), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)

    for i, s in enumerate(samples):
        df = s['df']
        seqs = df['junction_aa'].dropna().values
        count_col = None
        for col in ['duplicate_count', 'count', 'Count', 'freq']:
            if col in df.columns:
                count_col = col
                break
        if count_col:
            counts = df[count_col].fillna(1).values
        else:
            counts = np.ones(len(seqs))
        for seq, cnt in zip(seqs, counts):
            if isinstance(seq, str) and seq in seq_to_centroid:
                count_matrix[i, seq_to_centroid[seq]] += float(cnt)
        labels[i] = s['label']

    print(f"  Matrix: {count_matrix.shape}", flush=True)
    print(f"  Labels: {np.bincount(labels)} (0=control, 1=case)", flush=True)
    return count_matrix, labels


# =========================================================================
# Step 7: Classification
# =========================================================================
def classify(X, labels, dataset_name):
    X_norm = normalize(X.astype(np.float64), norm='l2', axis=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros(len(labels))
    all_fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_norm, labels)):
        svm = LinearSVC(C=0.1, random_state=42, max_iter=5000, dual=False)
        svm.fit(X_norm[train_idx], labels[train_idx])
        scores[test_idx] = svm.decision_function(X_norm[test_idx])
        fold_auc = roc_auc_score(labels[test_idx], scores[test_idx])
        all_fold_aucs.append(fold_auc)

    auc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)
    fpr, tpr, thresh = roc_curve(labels, scores)
    best_thresh = thresh[np.argmax(tpr - fpr)]
    y_pred = (scores >= best_thresh).astype(int)

    f1 = f1_score(labels, y_pred)
    mcc = matthews_corrcoef(labels, y_pred)
    acc = accuracy_score(labels, y_pred)
    sens = recall_score(labels, y_pred)
    spec = recall_score(1 - labels, 1 - y_pred)

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
        'fold_aucs': [float(a) for a in all_fold_aucs],
        'mean_fold_auc': float(np.mean(all_fold_aucs)),
        'std_fold_auc': float(np.std(all_fold_aucs)),
    }

    print(f"\n  {'='*50}", flush=True)
    print(f"  {dataset_name} -- Linear SVM (L2-norm, CB TRA panel m={X.shape[1]})", flush=True)
    print(f"  {'='*50}", flush=True)
    print(f"  AUC-ROC:     {auc:.4f}", flush=True)
    print(f"  AUC-PR:      {auc_pr:.4f}", flush=True)
    print(f"  F1:          {f1:.4f}", flush=True)
    print(f"  MCC:         {mcc:.4f}", flush=True)
    print(f"  Accuracy:   {acc:.4f}", flush=True)
    print(f"  Sensitivity: {sens:.4f}", flush=True)
    print(f"  Specificity: {spec:.4f}", flush=True)
    print(f"  Fold AUCs:  {['%.4f' % a for a in all_fold_aucs]}", flush=True)

    return results, scores, X_norm


# =========================================================================
# Step 8: Visualization
# =========================================================================
def make_plots(results, scores, labels, X_norm, cb_panel_info):
    dataset_name = "RA-TRA"

    # --- Fig 1: ROC curve ---
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = results['auc']
    ax.plot(fpr, tpr, color=PAT_COLOR, lw=2, label=f'{dataset_name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — CordBlood TRA Panel (m={M_TARGET})')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig1_roc_curve.png'), bbox_inches='tight')
    plt.close()
    print("  fig1_roc_curve.png done", flush=True)

    # --- Fig 2: Score distribution ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ctrl_scores = scores[labels == 0]
    case_scores = scores[labels == 1]
    bins = np.linspace(min(scores.min(), -3), max(scores.max(), 3), 30)
    ax.hist(ctrl_scores, bins=bins, color=CTRL_COLOR, alpha=0.7,
            label=f'Control (n={len(ctrl_scores)})', edgecolor='white', linewidth=0.3)
    ax.hist(case_scores, bins=bins, color=PAT_COLOR, alpha=0.7,
            label=f'Patient (n={len(case_scores)})', edgecolor='white', linewidth=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('SVM Disease Score')
    ax.set_ylabel('Sample Count')
    ax.set_title(f'{dataset_name} Score Distribution (AUC={auc:.4f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig2_score_distribution.png'), bbox_inches='tight')
    plt.close()
    print("  fig2_score_distribution.png done", flush=True)

    # --- Fig 3: PCA scatter ---
    fig, ax = plt.subplots(figsize=(7, 6))
    pca = PCA(n_components=2)
    coords = pca.fit(X_norm).transform(X_norm)
    ctrl = labels == 0
    case = labels == 1
    ax.scatter(coords[ctrl, 0], coords[ctrl, 1], c=CTRL_COLOR, s=20, alpha=0.7,
               edgecolors='white', linewidth=0.3, label=f'Control (n={ctrl.sum()})')
    ax.scatter(coords[case, 0], coords[case, 1], c=PAT_COLOR, s=20, alpha=0.7,
               edgecolors='white', linewidth=0.3, label=f'Patient (n={case.sum()})')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'{dataset_name} — PCA (CB TRA Panel)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig3_pca_scatter.png'), bbox_inches='tight')
    plt.close()
    print("  fig3_pca_scatter.png done", flush=True)

    # --- Fig 4: Saturation curve ---
    if 'saturation' in cb_panel_info:
        sat = cb_panel_info['saturation']
        fig, ax = plt.subplots(figsize=(7, 5))
        pool_sizes = [r['pool_size'] for r in sat]
        m_needed = [r['m_needed'] for r in sat]
        ax.plot(pool_sizes, m_needed, 'o-', color=CB_COLOR, lw=2, markersize=8)
        for ps, mn in zip(pool_sizes, m_needed):
            ax.annotate(f'm={mn}', (ps, mn), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9)
        ax.set_xlabel('CordBlood TRA Pool Size')
        ax.set_ylabel('m needed for 80% VE')
        ax.set_title('Saturation Analysis — CordBlood TRA')
        ax.axhline(y=M_TARGET, color=ORANGE, linestyle='--', lw=1.5, alpha=0.7,
                   label=f'target m={M_TARGET}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'fig4_saturation.png'), bbox_inches='tight')
        plt.close()
        print("  fig4_saturation.png done", flush=True)

    # --- Fig 5: VE scan ---
    if 'saturation' in cb_panel_info:
        sat = cb_panel_info['saturation']
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in sat:
            scan = r['scan_results']
            ms = [s['m'] for s in scan]
            ves = [s['variance_explained'] for s in scan]
            ax.plot(ms, ves, 'o-', label=f'pool={r["pool_size"]:,}', markersize=5)
        ax.axhline(y=0.80, color='gray', linestyle='--', lw=1, alpha=0.5, label='80% VE target')
        ax.set_xlabel('m (number of prototypes)')
        ax.set_ylabel('Variance Explained')
        ax.set_title('VE vs m — CordBlood TRA Reference Pool')
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'fig5_ve_scan.png'), bbox_inches='tight')
        plt.close()
        print("  fig5_ve_scan.png done", flush=True)

    # --- Fig 6: Fold variability ---
    fig, ax = plt.subplots(figsize=(6, 5))
    fold_aucs = results['fold_aucs']
    ax.scatter([1]*len(fold_aucs), fold_aucs, color=PAT_COLOR, s=40, zorder=5, alpha=0.8)
    ax.plot([0.8, 1.2], [np.mean(fold_aucs)]*2, color=PAT_COLOR, lw=2, zorder=4)
    ax.set_xticks([1])
    ax.set_xticklabels([dataset_name])
    ax.set_ylabel('Fold AUC-ROC')
    ax.set_title('5-Fold CV AUC Variability')
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig6_fold_variability.png'), bbox_inches='tight')
    plt.close()
    print("  fig6_fold_variability.png done", flush=True)


# =========================================================================
# Step 9: HTML Report
# =========================================================================
def generate_html(results, scores, labels, cb_panel_info):
    import base64

    def img_to_b64(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    imgs = {f: img_to_b64(os.path.join(IMG_DIR, f))
            for f in os.listdir(IMG_DIR) if f.endswith('.png')}

    ctrl_scores = scores[labels == 0]
    case_scores = scores[labels == 1]

    sat_section = ""
    if 'saturation' in cb_panel_info:
        sat = cb_panel_info['saturation']
        sat_rows = ""
        for r in sat:
            sat_rows += f"""
            <tr>
              <td>{r['pool_size']:,}</td>
              <td>{r['fraction']*100:.0f}%</td>
              <td><strong>{r['m_needed']}</strong></td>
            </tr>"""
        sat_section = f"""
        <section>
          <h2>Saturation Analysis (CordBlood TRA)</h2>
          <p>Tests whether m saturates as the CordBlood TRA pool grows.
          If m plateaus, the CB TRA pool captures the structural complexity
          of the alpha-chain TCR sequence space without needing disease data.</p>
          <table>
            <thead><tr><th>Pool Size</th><th>Fraction</th><th>m needed (80% VE)</th></tr></thead>
            <tbody>{sat_rows}</tbody>
          </table>
          <img src="data:image/png;base64,{imgs.get('fig4_saturation.png','')}" alt="Saturation" />
          <img src="data:image/png;base64,{imgs.get('fig5_ve_scan.png','')}" alt="VE Scan" />
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CordBlood TRA Reference Panel — RA-TRA Validation</title>
<style>
:root {{
  --bg:#fff; --bg2:#f8f9fc; --bg3:#eef1f8; --ink:#1a1d29; --ink2:#3d4255;
  --muted:#6b7390; --rule:#e1e5ef; --accent:#5e5ce6; --green:#00a389;
  --red:#ff453a; --orange:#ff9f0a; --cb:#8e8e93;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Display","Helvetica Neue",Arial,sans-serif;
  --mono:"SF Mono","Fira Code",monospace;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:var(--font); color:var(--ink); background:var(--bg); line-height:1.7; padding:2rem 1rem; }}
.page {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:1.8rem; margin-bottom:0.5rem; }}
h2 {{ font-size:1.3rem; margin:2rem 0 1rem; padding-bottom:0.5rem; border-bottom:2px solid var(--cb); }}
h3 {{ font-size:1.1rem; margin-bottom:0.5rem; color:var(--accent); }}
.subtitle {{ color:var(--muted); margin-bottom:2rem; font-size:0.95rem; }}
section {{ margin-bottom:2.5rem; }}
table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:0.9rem; }}
th,td {{ padding:0.5rem 0.75rem; text-align:center; border-bottom:1px solid var(--rule); }}
th {{ background:var(--bg3); font-weight:600; }}
tr:hover {{ background:var(--bg2); }}
img {{ max-width:100%; border-radius:8px; margin:1rem 0; }}
.card {{ background:var(--bg2); border:1px solid var(--rule); border-radius:10px; padding:1.5rem; margin:1rem 0; }}
.metrics-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.75rem; margin:1rem 0; }}
.metric {{ background:var(--bg3); border-radius:8px; padding:0.75rem; text-align:center; }}
.metric .label {{ display:block; font-size:0.75rem; color:var(--muted); margin-bottom:0.25rem; }}
.metric .value {{ font-size:1.1rem; font-weight:700; color:var(--ink); }}
.note {{ background:var(--bg3); border-left:3px solid var(--cb); padding:1rem 1.5rem; border-radius:0 8px 8px 0; margin:1rem 0; font-size:0.9rem; }}
</style>
</head>
<body>
<div class="page">
  <h1>CordBlood TRA Reference Panel</h1>
  <p class="subtitle">TCR Alpha-Chain Quantization — Built from CordBlood, Validated on RA-TRA</p>

  <div class="note">
    <strong>Method:</strong> A reference panel is built using ONLY CordBlood TRA CDR3 sequences
    (naive/unexposed alpha-chain repertoire). RA-TRA samples are projected onto this panel,
    L2-normalized, and classified with Linear SVM (5-fold CV).
    <br><br>
    <strong>Chain strategy:</strong> 7 chains (TRA/TRB/TRG/TRD/IGH/IGL/IGK) will each get
    their own reference panel. This report covers TRA (alpha chain).
  </div>

  <section>
    <h2>Reference Panel Info</h2>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Chain</td><td>TRA (alpha)</td></tr>
        <tr><td>Source</td><td>CordBlood TRA (synthetic library)</td></tr>
        <tr><td>Unique sequences</td><td>{cb_panel_info.get('n_sequences', '—'):,}</td></tr>
        <tr><td>Prototypes (m)</td><td>{cb_panel_info.get('m', M_TARGET)}</td></tr>
        <tr><td>Variance Explained</td><td>{cb_panel_info.get('variance_explained', 0):.4f}</td></tr>
        <tr><td>ESM-2 Model</td><td>{ESM2_MODEL}</td></tr>
        <tr><td>Embedding dim</td><td>{EMBED_DIM}</td></tr>
        <tr><td>Validation data</td><td>RA-TRA (Aterido 2024)</td></tr>
      </tbody>
    </table>
  </section>

  {sat_section}

  <section>
    <h2>Classification Results</h2>
    <div class="card">
      <h3>RA-TRA — Linear SVM (L2-norm, CB TRA panel m={M_TARGET})</h3>
      <div class="metrics-grid">
        <div class="metric"><span class="label">AUC-ROC</span><span class="value">{results['auc']:.4f}</span></div>
        <div class="metric"><span class="label">AUC-PR</span><span class="value">{results['auc_pr']:.4f}</span></div>
        <div class="metric"><span class="label">F1</span><span class="value">{results['f1']:.4f}</span></div>
        <div class="metric"><span class="label">MCC</span><span class="value">{results['mcc']:.4f}</span></div>
        <div class="metric"><span class="label">Sensitivity</span><span class="value">{results['sensitivity']:.4f}</span></div>
        <div class="metric"><span class="label">Specificity</span><span class="value">{results['specificity']:.4f}</span></div>
        <div class="metric"><span class="label">Samples</span><span class="value">{results['n_samples']}</span></div>
        <div class="metric"><span class="label">Fold AUC</span><span class="value">{results['mean_fold_auc']:.4f} ± {results['std_fold_auc']:.4f}</span></div>
      </div>
      <p>Control: {results['n_control']} samples | Patient: {results['n_case']} samples</p>
      <p>Control score: {ctrl_scores.mean():.3f} ± {ctrl_scores.std():.3f} | Patient score: {case_scores.mean():.3f} ± {case_scores.std():.3f}</p>
    </div>
    <table>
      <thead><tr>
        <th>Dataset</th><th>Samples</th><th>Control</th><th>Patient</th>
        <th>AUC-ROC</th><th>AUC-PR</th><th>F1</th><th>MCC</th>
        <th>Sens.</th><th>Spec.</th>
      </tr></thead>
      <tbody>
        <tr>
          <td>RA-TRA</td>
          <td>{results['n_samples']}</td>
          <td>{results['n_control']}</td>
          <td>{results['n_case']}</td>
          <td><strong>{results['auc']:.4f}</strong></td>
          <td>{results['auc_pr']:.4f}</td>
          <td>{results['f1']:.4f}</td>
          <td>{results['mcc']:.4f}</td>
          <td>{results['sensitivity']:.4f}</td>
          <td>{results['specificity']:.4f}</td>
        </tr>
      </tbody>
    </table>
    <img src="data:image/png;base64,{imgs.get('fig1_roc_curve.png','')}" alt="ROC Curve" />
  </section>

  <section>
    <h2>Score Distribution</h2>
    <img src="data:image/png;base64,{imgs.get('fig2_score_distribution.png','')}" alt="Score Distribution" />
  </section>

  <section>
    <h2>PCA Visualization</h2>
    <img src="data:image/png;base64,{imgs.get('fig3_pca_scatter.png','')}" alt="PCA Scatter" />
  </section>

  <section>
    <h2>5-Fold CV Variability</h2>
    <img src="data:image/png;base64,{imgs.get('fig6_fold_variability.png','')}" alt="Fold Variability" />
  </section>

  <section>
    <h2>Key Findings</h2>
    <div class="note">
      <p><strong>Interpretation:</strong> Using CordBlood TRA (naive alpha-chain repertoire)
      as the reference panel tests whether the "TCR transcriptome" quantization space
      is disease-agnostic at the chain level. Classification of RA-TRA on this panel
      demonstrates whether disease signals in the alpha chain can be captured by a
      reference built entirely from healthy cord blood data.</p>
      <br>
      <p><strong>Future work:</strong> Separate reference panels will be built for all 7 chains
      (TRA, TRB, TRG, TRD, IGH, IGL, IGK), enabling comprehensive AIRR-seq analysis
      across all receptor types.</p>
    </div>
  </section>
</div>
</body>
</html>"""

    report_path = os.path.join(OUTPUT_DIR, "cordblood_tra_panel_report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"\n  Report saved: {report_path}", flush=True)
    return report_path


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 70, flush=True)
    print("  CordBlood TRA Reference Panel — Build & RA-TRA Validation", flush=True)
    print("  Chain: TRA (alpha) | m=10,000 | ESM-2 + K-means", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Build reference pool from CordBlood TRA
    print("\n[1/7] Building CordBlood TRA reference pool...", flush=True)
    cb_pool = load_cordblood_tra_pool()
    cb_sequences = list(cb_pool.keys())
    print(f"  Pool: {len(cb_sequences):,} unique sequences", flush=True)

    # Step 2: ESM-2 embedding
    print("\n[2/7] Computing ESM-2 embeddings for CordBlood TRA sequences...", flush=True)
    cb_embeddings = compute_esm2_embeddings(cb_sequences)

    # Step 3: Train reference panel (m=10,000)
    print(f"\n[3/7] Training reference panel (m={M_TARGET})...", flush=True)
    panel = train_reference_panel(cb_embeddings, M_TARGET)
    centroids = panel['centroids']

    # Save CB TRA panel
    cb_panel_path = os.path.join(OUTPUT_DIR, f"cb_tra_reference_panel_m{M_TARGET}.pkl")
    with open(cb_panel_path, 'wb') as f:
        pickle.dump({
            'centroids': centroids,
            'variance_explained': panel['variance_explained'],
            'n_sequences': len(cb_sequences),
            'n_prototypes': M_TARGET,
            'sequences': cb_sequences,
            'embeddings': cb_embeddings,
            'chain': 'TRA',
        }, f)
    print(f"  CB TRA panel saved: {cb_panel_path}", flush=True)

    # Step 4: Saturation analysis
    print("\n[4/7] Saturation analysis (CordBlood TRA only)...", flush=True)
    sat_results = saturation_analysis(cb_embeddings)
    sat_path = os.path.join(OUTPUT_DIR, "cb_tra_saturation_analysis.json")
    with open(sat_path, 'w') as f:
        json.dump(sat_results, f, indent=2, default=str)
    print(f"  Saturation results saved: {sat_path}", flush=True)

    cb_panel_info = {
        'n_sequences': len(cb_sequences),
        'm': M_TARGET,
        'variance_explained': panel['variance_explained'],
        'saturation': sat_results,
        'chain': 'TRA',
    }

    # Step 5: Load RA-TRA samples
    print("\n[5/7] Loading RA-TRA dataset...", flush=True)
    ra_samples = load_ra_tra_samples()

    # Step 6: Project RA-TRA onto CB TRA panel
    print("\n[6/7] Projecting RA-TRA onto CordBlood TRA panel...", flush=True)
    ra_matrix, ra_labels = project_dataset(ra_samples, centroids, "RA-TRA")
    np.save(os.path.join(OUTPUT_DIR, "ra_tra_cb_matrix.npy"), ra_matrix)
    np.save(os.path.join(OUTPUT_DIR, "ra_tra_cb_labels.npy"), ra_labels)

    # Step 7: Classify
    print("\n[7/7] Classification (Linear SVM, L2-norm, 5-fold CV)...", flush=True)
    results, scores, X_norm = classify(ra_matrix, ra_labels, "RA-TRA")

    # Visualization & report
    print("\n[8/7] Generating plots and HTML report...", flush=True)
    make_plots(results, scores, ra_labels, X_norm, cb_panel_info)

    all_results = {
        'cb_panel': cb_panel_info,
        'classification': results,
        'config': {
            'reference': 'CordBlood TRA only',
            'chain': 'TRA',
            'm': M_TARGET,
            'classifier': 'LinearSVC(C=0.1, L2-norm)',
            'cv_folds': 5,
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "cordblood_tra_panel_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results JSON saved: {json_path}", flush=True)

    report_path = generate_html(results, scores, ra_labels, cb_panel_info)

    print(f"\n{'='*70}", flush=True)
    print("  CordBlood TRA Reference Panel Analysis Complete!", flush=True)
    print(f"  Report: {report_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
