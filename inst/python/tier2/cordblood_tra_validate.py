#!/usr/bin/env python3
"""
CordBlood TRA Panel — RA-TRA Validation (loads pre-built panel)
================================================================
Loads the saved CB TRA reference panel and projects RA-TRA samples
onto it for classification. Skips the slow saturation analysis.
"""
import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
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


def assign_to_centroids(embeddings, centroids, batch_size=10000):
    from scipy.spatial.distance import cdist as _cdist
    n = embeddings.shape[0]
    assignments = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch_size):
        batch = embeddings[i:i+batch_size]
        dists = _cdist(batch, centroids, metric='sqeuclidean')
        assignments[i:i+batch_size] = np.argmin(dists, axis=1)
    return assignments


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


def make_plots(results, scores, labels, X_norm, cb_panel_info):
    dataset_name = "RA-TRA"

    # Fig 1: ROC curve
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

    # Fig 2: Score distribution
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

    # Fig 3: PCA scatter
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

    # Fig 4: Fold variability
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


def generate_html(results, scores, labels, cb_panel_info):
    import base64

    def img_to_b64(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    imgs = {f: img_to_b64(os.path.join(IMG_DIR, f))
            for f in os.listdir(IMG_DIR) if f.endswith('.png')}

    ctrl_scores = scores[labels == 0]
    case_scores = scores[labels == 1]

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
        <tr><td>Variance Explained</td><td>{cb_panel_info.get('variance_explained', 0):.4f} ({cb_panel_info.get('variance_explained', 0)*100:.1f}%)</td></tr>
        <tr><td>ESM-2 Model</td><td>{ESM2_MODEL}</td></tr>
        <tr><td>Embedding dim</td><td>{EMBED_DIM}</td></tr>
        <tr><td>Validation data</td><td>RA-TRA (Aterido 2024)</td></tr>
      </tbody>
    </table>
  </section>

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
      <p><strong>Variance Explained:</strong> The CB TRA panel with m=10,000 captures
      {cb_panel_info.get('variance_explained', 0)*100:.1f}% of the variance in the
      CordBlood TRA embedding space, confirming that the quantization space is well-structured.</p>
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


def main():
    print("=" * 70, flush=True)
    print("  CordBlood TRA Panel — RA-TRA Validation (pre-built panel)", flush=True)
    print("  Chain: TRA (alpha) | m=10,000 | Linear SVM", flush=True)
    print("=" * 70, flush=True)

    # Load pre-built CB TRA panel
    panel_path = os.path.join(OUTPUT_DIR, f"cb_tra_reference_panel_m{M_TARGET}.pkl")
    print(f"\n[1/5] Loading pre-built CB TRA panel...", flush=True)
    with open(panel_path, 'rb') as f:
        panel_data = pickle.load(f)

    centroids = panel_data['centroids']
    cb_panel_info = {
        'n_sequences': panel_data['n_sequences'],
        'm': panel_data['n_prototypes'],
        'variance_explained': panel_data['variance_explained'],
        'chain': panel_data.get('chain', 'TRA'),
    }
    print(f"  Panel loaded: {cb_panel_info['n_sequences']:,} sequences, m={cb_panel_info['m']}", flush=True)
    print(f"  Variance explained: {cb_panel_info['variance_explained']:.4f} ({cb_panel_info['variance_explained']*100:.1f}%)", flush=True)

    # Load RA-TRA samples
    print("\n[2/5] Loading RA-TRA dataset...", flush=True)
    from cross_disease_benchmark import load_ra_dataset
    ra_samples = load_ra_dataset(chain='TRA')
    print(f"  RA-TRA: {len(ra_samples)} samples", flush=True)

    # Project RA-TRA onto CB TRA panel
    print("\n[3/5] Projecting RA-TRA onto CordBlood TRA panel...", flush=True)
    ra_matrix, ra_labels = project_dataset(ra_samples, centroids, "RA-TRA")
    np.save(os.path.join(OUTPUT_DIR, "ra_tra_cb_matrix.npy"), ra_matrix)
    np.save(os.path.join(OUTPUT_DIR, "ra_tra_cb_labels.npy"), ra_labels)

    # Classify
    print("\n[4/5] Classification (Linear SVM, L2-norm, 5-fold CV)...", flush=True)
    results, scores, X_norm = classify(ra_matrix, ra_labels, "RA-TRA")

    # Plots & report
    print("\n[5/5] Generating plots and HTML report...", flush=True)
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
    print("  CordBlood TRA Panel Analysis Complete!", flush=True)
    print(f"  Report: {report_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
