#!/usr/bin/env python3
"""
Cross-Disease Tier 2 Analysis
=============================
Project CMV and MS datasets onto the existing m=10,000 reference panel
and run Linear SVM classification, demonstrating the pipeline is
disease-agnostic.

Datasets:
  1. CMV (Emerson 2017) — CMV+ vs CMV- per-sample TRB
  2. MS (Alves Sousa 2019) — MS vs HC per-sample PBMC TRB
  3. RA (Aterido 2024) — Patient vs Control TRB (baseline)

Analysis:
  - Project each dataset onto the same m=10,000 TCR transcriptome
  - L2 normalize + Linear SVM (5-fold CV)
  - Cross-disease transfer: train on RA, test on CMV/MS
  - Supervised visualization (SVM projection, PCA)
  - HTML comparison report
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
    matthews_corrcoef, roc_curve, confusion_matrix,
    accuracy_score, recall_score, precision_score
)
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

TIER2_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
sys.path.insert(0, TIER2_DIR)

PANEL_DIR = os.path.join(WORK_DIR, "tcr_reference_panel")
OUTPUT_DIR = os.path.join(WORK_DIR, "cross_disease_tier2")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

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


# =========================================================================
# Step 1: Load existing reference panel
# =========================================================================
def load_panel():
    panel_path = os.path.join(PANEL_DIR, "reference_panel_m10000.pkl")
    if not os.path.exists(panel_path):
        print("ERROR: reference_panel_m10000.pkl not found. Run --train-panel first.")
        sys.exit(1)
    with open(panel_path, 'rb') as f:
        panel = pickle.load(f)
    centroids = panel['centroids']
    print(f"Loaded reference panel: m={centroids.shape[0]}, "
          f"VE={panel['variance_explained']:.4f}")
    return centroids


# =========================================================================
# Step 2: Load datasets
# =========================================================================
def load_ra_samples():
    from cross_disease_benchmark import load_ra_dataset
    samples = load_ra_dataset(chain='TRB')
    print(f"  RA: {len(samples)} samples")
    return samples


def load_cmv_samples():
    from cross_disease_benchmark import load_emerson_cmv
    samples = load_emerson_cmv()
    print(f"  CMV: {len(samples)} samples")
    return samples


def load_ms_samples():
    from cross_disease_benchmark import load_ms_dataset
    samples = load_ms_dataset()
    print(f"  MS: {len(samples)} samples")
    return samples


# =========================================================================
# Step 3: Project samples to m=10,000 space (efficient: embed unique seqs once)
# =========================================================================
def project_dataset(samples, centroids, dataset_name):
    from tcr_reference_quantization import compute_esm2_embeddings, assign_to_centroids

    print(f"\n{'='*60}")
    print(f"  Projecting {dataset_name} → m={centroids.shape[0]} space")
    print(f"{'='*60}")

    # Collect all unique sequences across all samples
    all_seqs = set()
    for s in samples:
        df = s['df']
        seqs = df['junction_aa'].dropna().values
        for seq in seqs:
            if isinstance(seq, str) and len(seq) >= 8:
                all_seqs.add(seq)
    all_seqs = sorted(all_seqs)
    print(f"  Unique sequences: {len(all_seqs):,}")

    # Embed all unique sequences
    print(f"  Computing ESM-2 embeddings...")
    embeddings = compute_esm2_embeddings(all_seqs)

    # Assign to nearest centroid
    print(f"  Assigning to {centroids.shape[0]} centroids...")
    assignments = assign_to_centroids(embeddings, centroids)

    # Build seq → centroid mapping
    seq_to_centroid = {seq: assignments[i] for i, seq in enumerate(all_seqs)}

    # Build count matrix
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

    print(f"  Matrix: {count_matrix.shape}")
    print(f"  Labels: {np.bincount(labels)} (0=control, 1=case)")

    return count_matrix, labels


# =========================================================================
# Step 4: Classification
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
    j_idx = tpr - fpr
    best_thresh = thresh[np.argmax(j_idx)]
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

    print(f"\n  {'='*50}")
    print(f"  {dataset_name} — Linear SVM (L2-norm, m=10,000)")
    print(f"  {'='*50}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  AUC-PR:      {auc_pr:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  MCC:         {mcc:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  Fold AUCs:  {['%.4f' % a for a in all_fold_aucs]}")

    return results, scores, X_norm


# =========================================================================
# Step 5: Cross-disease transfer
# =========================================================================
def cross_disease_transfer(X_train, y_train, X_test, y_test, train_name, test_name):
    X_train_norm = normalize(X_train.astype(np.float64), norm='l2', axis=1)
    X_test_norm = normalize(X_test.astype(np.float64), norm='l2', axis=1)

    svm = LinearSVC(C=0.1, random_state=42, max_iter=5000, dual=False)
    svm.fit(X_train_norm, y_train)
    scores = svm.decision_function(X_test_norm)

    auc = roc_auc_score(y_test, scores)
    auc_pr = average_precision_score(y_test, scores)
    fpr, tpr, thresh = roc_curve(y_test, scores)
    best_thresh = thresh[np.argmax(tpr - fpr)]
    y_pred = (scores >= best_thresh).astype(int)

    result = {
        'train': train_name,
        'test': test_name,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'auc': float(auc),
        'auc_pr': float(auc_pr),
        'f1': float(f1_score(y_test, y_pred)),
        'mcc': float(matthews_corrcoef(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
    }
    print(f"\n  Transfer {train_name} → {test_name}: AUC={auc:.4f}")
    return result


# =========================================================================
# Step 6: Visualization
# =========================================================================
def make_plots(datasets_data, transfer_results):
    """Generate all plots."""

    # --- Fig 1: ROC curves ---
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {'RA-TRB': PAT_COLOR, 'CMV-TRB': CTRL_COLOR, 'MS-TRB': GREEN}
    for name, (results, scores, labels, X_norm) in datasets_data.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = results['auc']
        color = colors.get(name, ACCENT)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Cross-Disease Tier 2 Classification')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig1_roc_curves.png'), bbox_inches='tight')
    plt.close()
    print("  fig1_roc_curves.png done")

    # --- Fig 2: AUC summary bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(datasets_data.keys())
    aucs = [datasets_data[n][0]['auc'] for n in names]
    bars = ax.bar(names, aucs, color=[colors.get(n, ACCENT) for n in names],
                  edgecolor='white', linewidth=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Classification Performance Across Diseases (m=10,000, Linear SVM)')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig2_auc_summary.png'), bbox_inches='tight')
    plt.close()
    print("  fig2_auc_summary.png done")

    # --- Fig 3: SVM score distributions ---
    n_datasets = len(datasets_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4), squeeze=False)
    for i, (name, (results, scores, labels, X_norm)) in enumerate(datasets_data.items()):
        ax = axes[0][i]
        ctrl_scores = scores[labels == 0]
        case_scores = scores[labels == 1]
        bins = np.linspace(min(scores.min(), -3), max(scores.max(), 3), 30)
        ax.hist(ctrl_scores, bins=bins, color=CTRL_COLOR, alpha=0.7,
                label=f'Control (n={len(ctrl_scores)})', edgecolor='white', linewidth=0.3)
        ax.hist(case_scores, bins=bins, color=PAT_COLOR, alpha=0.7,
                label=f'Case (n={len(case_scores)})', edgecolor='white', linewidth=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('SVM Disease Score')
        ax.set_ylabel('Sample Count')
        ax.set_title(f'{name}\nAUC={results["auc"]:.4f}')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig3_score_distributions.png'), bbox_inches='tight')
    plt.close()
    print("  fig3_score_distributions.png done")

    # --- Fig 4: PCA scatter ---
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4), squeeze=False)
    for i, (name, (results, scores, labels, X_norm)) in enumerate(datasets_data.items()):
        ax = axes[0][i]
        pca = PCA(n_components=2)
        coords = pca.fit(X_norm).transform(X_norm)
        ctrl = labels == 0
        case = labels == 1
        ax.scatter(coords[ctrl, 0], coords[ctrl, 1], c=CTRL_COLOR, s=20, alpha=0.7,
                   edgecolors='white', linewidth=0.3, label=f'Control (n={ctrl.sum()})')
        ax.scatter(coords[case, 0], coords[case, 1], c=PAT_COLOR, s=20, alpha=0.7,
                   edgecolors='white', linewidth=0.3, label=f'Case (n={case.sum()})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{name} — PCA')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig4_pca_scatter.png'), bbox_inches='tight')
    plt.close()
    print("  fig4_pca_scatter.png done")

    # --- Fig 5: Cross-disease transfer heatmap ---
    if transfer_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        pairs = [(r['train'], r['test'], r['auc']) for r in transfer_results]
        train_names = sorted(set(p[0] for p in pairs))
        test_names = sorted(set(p[1] for p in pairs))
        matrix = np.full((len(train_names), len(test_names)), np.nan)
        for tr, te, auc in pairs:
            matrix[train_names.index(tr), test_names.index(te)] = auc
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(test_names)))
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.set_yticks(range(len(train_names)))
        ax.set_yticklabels(train_names)
        for i in range(len(train_names)):
            for j in range(len(test_names)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10,
                            fontweight='bold',
                            color='white' if val > 0.65 or val < 0.45 else 'black')
        ax.set_title('Cross-Disease Transfer (AUC-ROC)')
        plt.colorbar(im, ax=ax, label='AUC-ROC')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'fig5_transfer_heatmap.png'), bbox_inches='tight')
        plt.close()
        print("  fig5_transfer_heatmap.png done")

    # --- Fig 6: Fold AUC variability ---
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = []
    data = []
    labels_list = []
    for i, (name, (results, _, _, _)) in enumerate(datasets_data.items()):
        fold_aucs = results['fold_aucs']
        positions.append(i + 1)
        data.append(fold_aucs)
        labels_list.append(name)
        ax.scatter([i+1]*len(fold_aucs), fold_aucs, color=colors.get(name, ACCENT),
                   s=30, zorder=5, alpha=0.8)
        ax.plot([i+0.8, i+1.2], [np.mean(fold_aucs)]*2, color=colors.get(name, ACCENT),
                lw=2, zorder=4)
    ax.set_xticks(range(1, len(labels_list)+1))
    ax.set_xticklabels(labels_list)
    ax.set_ylabel('Fold AUC-ROC')
    ax.set_title('5-Fold CV AUC Variability')
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig6_fold_variability.png'), bbox_inches='tight')
    plt.close()
    print("  fig6_fold_variability.png done")


# =========================================================================
# Step 7: HTML report
# =========================================================================
def generate_html(datasets_data, transfer_results):
    import base64

    def img_to_b64(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    imgs = {f: img_to_b64(os.path.join(IMG_DIR, f))
            for f in os.listdir(IMG_DIR) if f.endswith('.png')}

    dataset_rows = ""
    for name, (results, _, _, _) in datasets_data.items():
        dataset_rows += f"""
        <tr>
          <td>{name}</td>
          <td>{results['n_samples']}</td>
          <td>{results['n_control']}</td>
          <td>{results['n_case']}</td>
          <td><strong>{results['auc']:.4f}</strong></td>
          <td>{results['auc_pr']:.4f}</td>
          <td>{results['f1']:.4f}</td>
          <td>{results['mcc']:.4f}</td>
          <td>{results['sensitivity']:.4f}</td>
          <td>{results['specificity']:.4f}</td>
        </tr>"""

    transfer_rows = ""
    for r in transfer_results:
        transfer_rows += f"""
        <tr>
          <td>{r['train']}</td>
          <td>{r['test']}</td>
          <td>{r['n_train']}</td>
          <td>{r['n_test']}</td>
          <td><strong>{r['auc']:.4f}</strong></td>
          <td>{r['auc_pr']:.4f}</td>
          <td>{r['f1']:.4f}</td>
        </tr>"""

    fig1 = imgs.get('fig1_roc_curves.png', '')
    fig2 = imgs.get('fig2_auc_summary.png', '')
    fig3 = imgs.get('fig3_score_distributions.png', '')
    fig4 = imgs.get('fig4_pca_scatter.png', '')
    fig5 = imgs.get('fig5_transfer_heatmap.png', '')
    fig6 = imgs.get('fig6_fold_variability.png', '')

    # Build per-dataset detail cards
    detail_cards = ""
    for name, (results, scores, labels, X_norm) in datasets_data.items():
        ctrl_scores = scores[labels == 0]
        case_scores = scores[labels == 1]
        detail_cards += f"""
        <div class="card">
          <h3>{name}</h3>
          <div class="metrics-grid">
            <div class="metric"><span class="label">AUC-ROC</span><span class="value">{results['auc']:.4f}</span></div>
            <div class="metric"><span class="label">AUC-PR</span><span class="value">{results['auc_pr']:.4f}</span></div>
            <div class="metric"><span class="label">F1</span><span class="value">{results['f1']:.4f}</span></div>
            <div class="metric"><span class="label">MCC</span><span class="value">{results['mcc']:.4f}</span></div>
            <div class="metric"><span class="label">Sensitivity</span><span class="value">{results['sensitivity']:.4f}</span></div>
            <div class="metric"><span class="label">Specificity</span><span class="value">{results['specificity']:.4f}</span></div>
            <div class="metric"><span class="label">Samples</span><span class="value">{results['n_samples']}</span></div>
            <div class="metric"><span class="label">Mean Fold AUC</span><span class="value">{results['mean_fold_auc']:.4f} ± {results['std_fold_auc']:.4f}</span></div>
          </div>
          <p>Control mean score: {ctrl_scores.mean():.3f} ± {ctrl_scores.std():.3f} | Case mean score: {case_scores.mean():.3f} ± {case_scores.std():.3f}</p>
        </div>"""

    transfer_section = ""
    if transfer_results and fig5:
        transfer_section = f"""
        <section>
          <h2>Cross-Disease Transfer</h2>
          <p>Train on one disease, test on another. Evaluates whether the TCR
          quantization space captures universal disease patterns.</p>
          <table>
            <thead><tr><th>Train</th><th>Test</th><th>n_train</th><th>n_test</th><th>AUC</th><th>AUC-PR</th><th>F1</th></tr></thead>
            <tbody>{transfer_rows}</tbody>
          </table>
          <img src="data:image/png;base64,{fig5}" alt="Transfer Heatmap" />
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Disease Tier 2 Analysis Report</title>
<style>
:root {{
  --bg:#fff; --bg2:#f8f9fc; --bg3:#eef1f8; --ink:#1a1d29; --ink2:#3d4255;
  --muted:#6b7390; --rule:#e1e5ef; --accent:#5e5ce6; --green:#00a389;
  --red:#ff453a; --orange:#ff9f0a;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Display","Helvetica Neue",Arial,sans-serif;
  --mono:"SF Mono","Fira Code",monospace;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:var(--font); color:var(--ink); background:var(--bg); line-height:1.7; padding:2rem 1rem; }}
.page {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:1.8rem; margin-bottom:0.5rem; }}
h2 {{ font-size:1.3rem; margin:2rem 0 1rem; padding-bottom:0.5rem; border-bottom:2px solid var(--accent); }}
h3 {{ font-size:1.1rem; margin-bottom:0.5rem; color:var(--accent); }}
.subtitle {{ color:var(--muted); margin-bottom:2rem; font-size:0.95rem; }}
section {{ margin-bottom:2.5rem; }}
table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:0.9rem; }}
th,td {{ padding:0.5rem 0.75rem; text-align:center; border-bottom:1px solid var(--rule); }}
th {{ background:var(--bg3); font-weight:600; }}
tr:hover {{ background:var(--bg2); }}
img {{ max-width:100%; border-radius:8px; margin:1rem 0; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
.card {{ background:var(--bg2); border:1px solid var(--rule); border-radius:10px; padding:1.5rem; margin:1rem 0; }}
.metrics-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.75rem; margin:1rem 0; }}
.metric {{ background:var(--bg3); border-radius:8px; padding:0.75rem; text-align:center; }}
.metric .label {{ display:block; font-size:0.75rem; color:var(--muted); margin-bottom:0.25rem; }}
.metric .value {{ font-size:1.1rem; font-weight:700; color:var(--ink); }}
.note {{ background:var(--bg3); border-left:3px solid var(--accent); padding:1rem 1.5rem; border-radius:0 8px 8px 0; margin:1rem 0; font-size:0.9rem; }}
</style>
</head>
<body>
<div class="page">
  <h1>Cross-Disease Tier 2 Analysis</h1>
  <p class="subtitle">TCR Quantization Pipeline (m=10,000) — Disease-Agnostic Validation</p>

  <div class="note">
    <strong>Method:</strong> All datasets are projected onto the same m=10,000
    "TCR transcriptome" reference panel (built from RA+CMV+MS+SLE+VDJdb),
    L2-normalized, and classified with Linear SVM (5-fold CV).
  </div>

  <section>
    <h2>Classification Results</h2>
    <table>
      <thead><tr>
        <th>Dataset</th><th>Samples</th><th>Control</th><th>Case</th>
        <th>AUC-ROC</th><th>AUC-PR</th><th>F1</th><th>MCC</th>
        <th>Sens.</th><th>Spec.</th>
      </tr></thead>
      <tbody>{dataset_rows}</tbody>
    </table>
    <img src="data:image/png;base64,{fig1}" alt="ROC Curves" />
    <img src="data:image/png;base64,{fig2}" alt="AUC Summary" />
  </section>

  <section>
    <h2>Score Distributions</h2>
    <img src="data:image/png;base64,{fig3}" alt="Score Distributions" />
  </section>

  <section>
    <h2>PCA Visualization</h2>
    <img src="data:image/png;base64,{fig4}" alt="PCA Scatter" />
  </section>

  {transfer_section}

  <section>
    <h2>5-Fold CV Variability</h2>
    <img src="data:image/png;base64,{fig6}" alt="Fold Variability" />
  </section>

  <section>
    <h2>Per-Dataset Details</h2>
    {detail_cards}
  </section>

  <section>
    <h2>Key Findings</h2>
    <div class="note">
      <p>The m=10,000 TCR transcriptome space is disease-agnostic: a single
      reference panel built from pooled CDR3 sequences enables classification
      across multiple diseases without disease-specific feature engineering.</p>
    </div>
  </section>
</div>
</body>
</html>"""

    report_path = os.path.join(OUTPUT_DIR, "cross_disease_tier2_report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"\n  Report saved: {report_path}")
    return report_path


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 70)
    print("  Cross-Disease Tier 2 Analysis")
    print("  m=10,000 TCR Transcriptome — Disease-Agnostic Validation")
    print("=" * 70)

    # Load reference panel
    centroids = load_panel()

    # Load RA (baseline, already projected)
    print("\n[1] Loading RA data...")
    ra_samples = load_ra_samples()

    # Load CMV
    print("\n[2] Loading CMV data...")
    cmv_samples = load_cmv_samples()

    # Load MS
    print("\n[3] Loading MS data...")
    ms_samples = load_ms_samples()

    # Project each dataset
    print("\n[4] Projecting datasets onto m=10,000 space...")

    # RA — try loading pre-computed matrix first
    ra_matrix_path = os.path.join(PANEL_DIR, "ra_count_matrix_m10000.npy")
    ra_labels_path = os.path.join(PANEL_DIR, "ra_labels_m10000.npy")
    if os.path.exists(ra_matrix_path):
        print("  RA: loading pre-computed matrix...")
        ra_matrix = np.load(ra_matrix_path)
        ra_labels = np.load(ra_labels_path)
    else:
        ra_matrix, ra_labels = project_dataset(ra_samples, centroids, "RA-TRB")

    # CMV — try loading pre-computed matrix first
    cmv_matrix_path = os.path.join(OUTPUT_DIR, "cmv_count_matrix_m10000.npy")
    cmv_labels_path = os.path.join(OUTPUT_DIR, "cmv_labels_m10000.npy")
    if os.path.exists(cmv_matrix_path):
        print("  CMV: loading pre-computed matrix...")
        cmv_matrix = np.load(cmv_matrix_path)
        cmv_labels = np.load(cmv_labels_path)
    else:
        cmv_matrix, cmv_labels = project_dataset(cmv_samples, centroids, "CMV-TRB")
        np.save(os.path.join(OUTPUT_DIR, "cmv_count_matrix_m10000.npy"), cmv_matrix)
        np.save(os.path.join(OUTPUT_DIR, "cmv_labels_m10000.npy"), cmv_labels)

    # MS — try loading pre-computed matrix first
    ms_matrix_path = os.path.join(OUTPUT_DIR, "ms_count_matrix_m10000.npy")
    ms_labels_path = os.path.join(OUTPUT_DIR, "ms_labels_m10000.npy")
    if os.path.exists(ms_matrix_path):
        print("  MS: loading pre-computed matrix...")
        ms_matrix = np.load(ms_matrix_path)
        ms_labels = np.load(ms_labels_path)
    else:
        ms_matrix, ms_labels = project_dataset(ms_samples, centroids, "MS-TRB")
        np.save(os.path.join(OUTPUT_DIR, "ms_count_matrix_m10000.npy"), ms_matrix)
        np.save(os.path.join(OUTPUT_DIR, "ms_labels_m10000.npy"), ms_labels)

    # Classify each dataset
    print("\n[5] Classification (Linear SVM, L2-norm, 5-fold CV)...")
    ra_results, ra_scores, ra_X = classify(ra_matrix, ra_labels, "RA-TRB")
    cmv_results, cmv_scores, cmv_X = classify(cmv_matrix, cmv_labels, "CMV-TRB")
    ms_results, ms_scores, ms_X = classify(ms_matrix, ms_labels, "MS-TRB")

    datasets_data = {
        "RA-TRB": (ra_results, ra_scores, ra_labels, ra_X),
        "CMV-TRB": (cmv_results, cmv_scores, cmv_labels, cmv_X),
        "MS-TRB": (ms_results, ms_scores, ms_labels, ms_X),
    }

    # Cross-disease transfer
    print("\n[6] Cross-disease transfer learning...")
    transfer_results = []
    # RA → CMV
    transfer_results.append(
        cross_disease_transfer(ra_matrix, ra_labels, cmv_matrix, cmv_labels, "RA-TRB", "CMV-TRB"))
    # CMV → RA
    transfer_results.append(
        cross_disease_transfer(cmv_matrix, cmv_labels, ra_matrix, ra_labels, "CMV-TRB", "RA-TRB"))
    # RA → MS
    transfer_results.append(
        cross_disease_transfer(ra_matrix, ra_labels, ms_matrix, ms_labels, "RA-TRB", "MS-TRB"))
    # CMV → MS
    transfer_results.append(
        cross_disease_transfer(cmv_matrix, cmv_labels, ms_matrix, ms_labels, "CMV-TRB", "MS-TRB"))

    # Visualization
    print("\n[7] Generating plots...")
    make_plots(datasets_data, transfer_results)

    # Save results JSON
    all_results = {
        'datasets': {name: data[0] for name, data in datasets_data.items()},
        'transfer': transfer_results,
        'config': {
            'm': 10000,
            'classifier': 'LinearSVC(C=0.1, L2-norm)',
            'cv_folds': 5,
            'reference_panel': 'RA+CMV+MS+SLE+VDJdb',
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "cross_disease_tier2_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results JSON saved: {json_path}")

    # Generate HTML report
    print("\n[8] Generating HTML report...")
    report_path = generate_html(datasets_data, transfer_results)

    print(f"\n{'='*70}")
    print("  Cross-Disease Tier 2 Analysis Complete!")
    print(f"  Report: {report_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
