#!/usr/bin/env python3
"""
Supervised visualization approaches for RA TCR data.
The disease signal is a linear hyperplane in 10K-dim space.
UMAP can't capture it, but these methods can:

1. SVM projection (1D): project samples onto the SVM weight vector
2. SVM axis + orthogonal PC (2D): SVM direction as x-axis, top orthogonal PC as y-axis
3. PLS-DA (2D): supervised dimensionality reduction maximizing cov(X,y)
4. LDA (1D): maximize between-class / within-class variance
5. Supervised UMAP: uses labels to guide embedding
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import umap

warnings.filterwarnings('ignore')

BASE = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
OUTPUT_DIR = os.path.join(BASE, "seurat_analysis")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

CTRL = '#4a90d9'
PAT = '#ff6b6b'
ACCENT = '#5e5ce6'
GREEN = '#00a389'

# Load data
count = np.load(os.path.join(BASE, "tcr_reference_panel/ra_count_matrix_m10000.npy"))
labels = np.load(os.path.join(BASE, "tcr_reference_panel/ra_labels_m10000.npy"))
X = normalize(count.astype(np.float64), norm='l2', axis=1)
n = len(labels)
ctrl_mask = labels == 0
pat_mask = labels == 1

print(f"Data: {X.shape}, Control={ctrl_mask.sum()}, Patient={pat_mask.sum()}", flush=True)

# ============================================================
# 1. SVM projection (1D) — trained on full data for visualization
# ============================================================
print("\n[1] SVM projection (1D)...", flush=True)
svm = LinearSVC(C=0.1, random_state=42, max_iter=5000, dual=False)
svm.fit(X, labels)
w = svm.coef_[0]  # (10000,)
b = svm.intercept_[0]
svm_proj = X @ w + b  # (546,) — projection onto SVM direction

fig, ax = plt.subplots(figsize=(9, 4))
y_ctrl = np.random.normal(0, 0.05, ctrl_mask.sum())
y_pat = np.random.normal(1, 0.05, pat_mask.sum())
ax.scatter(svm_proj[ctrl_mask], y_ctrl, c=CTRL, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Control (n={ctrl_mask.sum()})')
ax.scatter(svm_proj[pat_mask], y_pat, c=PAT, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Patient (n={pat_mask.sum()})')
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Decision boundary (f(x)=0)')
ax.set_xlabel('SVM Projection Score')
ax.set_ylabel('')
ax.set_yticks([])
ax.set_title('SVM Weight Vector Projection (1D)')
ax.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig_svm_1d.png'), bbox_inches='tight')
plt.close()
print("  Done", flush=True)

# ============================================================
# 2. SVM axis + orthogonal PC (2D)
# ============================================================
print("[2] SVM axis + orthogonal PC (2D)...", flush=True)
# Project onto SVM direction
svm_axis = X @ w  # (546,)
# Remove SVM direction from X, then PCA for orthogonal component
X_resid = X - np.outer(svm_axis, w / np.dot(w, w))
scaler_r = StandardScaler()
X_r_scaled = scaler_r.fit_transform(X_resid)
pca_r = PCA(n_components=1, random_state=42)
orth_axis = pca_r.fit_transform(X_r_scaled).ravel()

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(svm_axis[ctrl_mask], orth_axis[ctrl_mask], c=CTRL, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Control (n={ctrl_mask.sum()})')
ax.scatter(svm_axis[pat_mask], orth_axis[pat_mask], c=PAT, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Patient (n={pat_mask.sum()})')
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
ax.set_xlabel('SVM Disease Axis (projection onto w)')
ax.set_ylabel('Orthogonal PC1 (residual variance)')
ax.set_title('SVM Axis + Orthogonal PC')
ax.legend(loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig_svm_orth.png'), bbox_inches='tight')
plt.close()
print("  Done", flush=True)

# ============================================================
# 3. PLS-DA (2D)
# ============================================================
print("[3] PLS-DA (2D)...", flush=True)
pls = PLSRegression(n_components=2, scale=True)
pls_scores = pls.fit_transform(X, labels)[0]  # (546, 2)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(pls_scores[ctrl_mask, 0], pls_scores[ctrl_mask, 1], c=CTRL, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Control (n={ctrl_mask.sum()})')
ax.scatter(pls_scores[pat_mask, 0], pls_scores[pat_mask, 1], c=PAT, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Patient (n={pat_mask.sum()})')
ax.set_xlabel('PLS Component 1 (disease direction)')
ax.set_ylabel('PLS Component 2')
ax.set_title('PLS-DA Projection')
ax.legend(loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig_plsda.png'), bbox_inches='tight')
plt.close()
print("  Done", flush=True)

# ============================================================
# 4. LDA (1D)
# ============================================================
print("[4] LDA (1D)...", flush=True)
lda = LinearDiscriminantAnalysis(n_components=1)
lda_proj = lda.fit_transform(X, labels).ravel()

fig, ax = plt.subplots(figsize=(9, 4))
y_c = np.random.normal(0, 0.05, ctrl_mask.sum())
y_p = np.random.normal(1, 0.05, pat_mask.sum())
ax.scatter(lda_proj[ctrl_mask], y_c, c=CTRL, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Control (n={ctrl_mask.sum()})')
ax.scatter(lda_proj[pat_mask], y_p, c=PAT, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Patient (n={pat_mask.sum()})')
ax.set_xlabel('LDA Projection')
ax.set_ylabel('')
ax.set_yticks([])
ax.set_title('Linear Discriminant Analysis (1D)')
ax.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig_lda.png'), bbox_inches='tight')
plt.close()
print("  Done", flush=True)

# ============================================================
# 5. Supervised UMAP
# ============================================================
print("[5] Supervised UMAP...", flush=True)
# First do PCA to 30 dims
scaler_s = StandardScaler()
X_s = scaler_s.fit_transform(X)
pca_s = PCA(n_components=30, random_state=42)
X_pca = pca_s.fit_transform(X_s)

reducer_sup = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, metric='euclidean', random_state=42)
sup_emb = reducer_sup.fit_transform(X_pca, y=labels)  # supervised!

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(sup_emb[ctrl_mask, 0], sup_emb[ctrl_mask, 1], c=CTRL, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Control (n={ctrl_mask.sum()})')
ax.scatter(sup_emb[pat_mask, 0], sup_emb[pat_mask, 1], c=PAT, s=25, alpha=0.7, edgecolors='white', linewidth=0.3, label=f'Patient (n={pat_mask.sum()})')
ax.set_xlabel('Supervised UMAP 1')
ax.set_ylabel('Supervised UMAP 2')
ax.set_title('Supervised UMAP (labels guide embedding)')
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig_supervised_umap.png'), bbox_inches='tight')
plt.close()
print("  Done", flush=True)

# ============================================================
# 6. Combined 2x3 panel
# ============================================================
print("[6] Combined panel...", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: Original UMAP (unsupervised)
coords = pd.read_csv(os.path.join(OUTPUT_DIR, 'unified_coordinates.csv'))
axes[0,0].scatter(coords.loc[ctrl_mask, 'UMAP1'], coords.loc[ctrl_mask, 'UMAP2'], c=CTRL, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[0,0].scatter(coords.loc[pat_mask, 'UMAP1'], coords.loc[pat_mask, 'UMAP2'], c=PAT, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[0,0].set_title('A. UMAP (unsupervised)', fontweight='bold')
axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
axes[0,0].legend(['Control', 'Patient'], loc='best', fontsize=8, markerscale=0.5)

# Panel B: SVM 1D
y_c = np.random.normal(0, 0.04, ctrl_mask.sum())
y_p = np.random.normal(1, 0.04, pat_mask.sum())
axes[0,1].scatter(svm_proj[ctrl_mask], y_c, c=CTRL, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[0,1].scatter(svm_proj[pat_mask], y_p, c=PAT, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[0,1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0,1].set_title('B. SVM Projection (1D)', fontweight='bold')
axes[0,1].set_yticks([])

# Panel C: SVM + Orthogonal PC
axes[0,2].scatter(svm_axis[ctrl_mask], orth_axis[ctrl_mask], c=CTRL, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[0,2].scatter(svm_axis[pat_mask], orth_axis[pat_mask], c=PAT, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[0,2].axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
axes[0,2].set_title('C. SVM Axis + Orth PC', fontweight='bold')
axes[0,2].set_xlabel('SVM direction')
axes[0,2].set_ylabel('Orthogonal PC1')

# Panel D: PLS-DA
axes[1,0].scatter(pls_scores[ctrl_mask, 0], pls_scores[ctrl_mask, 1], c=CTRL, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[1,0].scatter(pls_scores[pat_mask, 0], pls_scores[pat_mask, 1], c=PAT, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[1,0].set_title('D. PLS-DA', fontweight='bold')
axes[1,0].set_xlabel('PLS Comp 1')
axes[1,0].set_ylabel('PLS Comp 2')

# Panel E: LDA
y_c2 = np.random.normal(0, 0.04, ctrl_mask.sum())
y_p2 = np.random.normal(1, 0.04, pat_mask.sum())
axes[1,1].scatter(lda_proj[ctrl_mask], y_c2, c=CTRL, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[1,1].scatter(lda_proj[pat_mask], y_p2, c=PAT, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[1,1].set_title('E. LDA (1D)', fontweight='bold')
axes[1,1].set_yticks([])
axes[1,1].set_xlabel('LDA projection')

# Panel F: Supervised UMAP
axes[1,2].scatter(sup_emb[ctrl_mask, 0], sup_emb[ctrl_mask, 1], c=CTRL, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[1,2].scatter(sup_emb[pat_mask, 0], sup_emb[pat_mask, 1], c=PAT, s=15, alpha=0.7, edgecolors='white', linewidth=0.2)
axes[1,2].set_title('F. Supervised UMAP', fontweight='bold')
axes[1,2].set_xticks([]); axes[1,2].set_yticks([])

plt.suptitle('RA TCR Visualization: Unsupervised vs Supervised Methods (m=10,000 space)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig_combined_viz.png'), bbox_inches='tight')
plt.close()
print("  Done", flush=True)

# Print separation metrics
print("\n=== Separation Quality ===", flush=True)
# Cohen's d for each 1D projection
def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)

print(f"  SVM 1D:  d = {cohens_d(svm_proj[ctrl_mask], svm_proj[pat_mask]):.2f}, "
      f"AUC = {roc_auc_score(labels, svm_proj):.4f}", flush=True)
print(f"  LDA 1D:  d = {cohens_d(lda_proj[ctrl_mask], lda_proj[pat_mask]):.2f}, "
      f"AUC = {roc_auc_score(labels, lda_proj):.4f}", flush=True)
pls1 = pls_scores[:, 0]
print(f"  PLS-DA:  d = {cohens_d(pls1[ctrl_mask], pls1[pat_mask]):.2f}", flush=True)
print(f"\n  Images saved to: {IMG_DIR}", flush=True)
print("Done!", flush=True)
