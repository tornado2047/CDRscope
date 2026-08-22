#!/usr/bin/env python3
"""
Unified Pipeline: Seurat-style Analysis on m=10,000 Space
============================================================
Uses the validated unified pipeline:
  - m=10,000 TCR prototype space (disease-agnostic)
  - L2 normalization (removes sequencing depth bias)
  - Linear SVM (optimal for sparse high-dim data)

Pipeline:
  1. L2 normalize count matrix
  2. PCA → top 50 PCs
  3. UMAP 2D embedding
  4. KMeans clustering (k=2..8) + silhouette
  5. Linear SVM decision function as "disease score"
  6. Visualize: UMAP by label, UMAP by cluster, PCA, disease score, cluster composition
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    roc_auc_score, roc_curve, average_precision_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
from scipy.stats import fisher_exact
import umap

warnings.filterwarnings('ignore')

BASE = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
OUTPUT_DIR = os.path.join(BASE, "seurat_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70, flush=True)
    print("  Unified Pipeline: Seurat-style Analysis (L2 + Linear SVM)", flush=True)
    print("  m=10,000 TCR Prototype Space", flush=True)
    print("=" * 70, flush=True)

    # === Load data ===
    print("\n[1/6] Loading data...", flush=True)
    count = np.load(os.path.join(BASE, "tcr_reference_panel/ra_count_matrix_m10000.npy"))
    labels = np.load(os.path.join(BASE, "tcr_reference_panel/ra_labels_m10000.npy"))
    n_samples = len(labels)
    print(f"  Matrix: {count.shape}", flush=True)
    print(f"  Labels: Control={np.sum(labels==0)}, Patient={np.sum(labels==1)}", flush=True)

    # === L2 Normalize ===
    print("\n[2/6] L2 normalization...", flush=True)
    X = normalize(count.astype(np.float64), norm='l2', axis=1)
    print(f"  L2 norm per sample: mean={np.linalg.norm(X, axis=1).mean():.4f} (should be 1.0)", flush=True)
    print(f"  Sparsity: {np.mean(X == 0):.1%}", flush=True)

    # === PCA ===
    print("\n[3/6] PCA (50 components)...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=50, random_state=42)
    pca_scores = pca.fit_transform(X_scaled)
    ve = pca.explained_variance_ratio_
    cumve = np.cumsum(ve)
    print(f"  PC1: {ve[0]:.1%}, PC5: {cumve[4]:.1%}, PC10: {cumve[9]:.1%}, PC20: {cumve[19]:.1%}", flush=True)

    # === UMAP ===
    print("\n[4/6] UMAP (on top 30 PCs)...", flush=True)
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                         metric='euclidean', random_state=42)
    umap_emb = reducer.fit_transform(pca_scores[:, :30])
    print(f"  UMAP done in {time.time()-t0:.1f}s", flush=True)

    # === KMeans clustering ===
    print("\n[5/6] KMeans clustering (k=2..8)...", flush=True)
    cluster_results = {}
    best_k, best_sil = 2, -1
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cl = km.fit_predict(pca_scores[:, :30])
        sil = silhouette_score(pca_scores[:, :30], cl, sample_size=min(500, n_samples))
        ari = adjusted_rand_score(labels, cl)
        nmi = normalized_mutual_info_score(labels, cl)
        cluster_results[k] = {'labels': cl, 'silhouette': sil, 'ari': ari, 'nmi': nmi}
        print(f"  k={k}: sil={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}", flush=True)
        if sil > best_sil:
            best_k, best_sil = k, sil
    print(f"  Best k={best_k} (silhouette={best_sil:.4f})", flush=True)

    # Use best_k for downstream analysis
    best_cluster = cluster_results[best_k]['labels']

    # === Linear SVM disease score (5-fold CV) ===
    print("\n[6/6] Linear SVM disease score (5-fold CV)...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_scores = np.zeros(n_samples)
    lr_scores = np.zeros(n_samples)
    fold_aucs_svm, fold_aucs_lr = [], []

    for train_idx, test_idx in skf.split(X, labels):
        # Linear SVM
        svm = LinearSVC(C=0.1, random_state=42, max_iter=5000, dual=False)
        svm.fit(X[train_idx], labels[train_idx])
        svm_scores[test_idx] = svm.decision_function(X[test_idx])
        fold_aucs_svm.append(roc_auc_score(labels[test_idx], svm_scores[test_idx]))

        # L2 Logistic Regression
        lr = LogisticRegression(penalty='l2', C=1.0, max_iter=2000, random_state=42, solver='lbfgs')
        lr.fit(X[train_idx], labels[train_idx])
        lr_scores[test_idx] = lr.predict_proba(X[test_idx])[:, 1]
        fold_aucs_lr.append(roc_auc_score(labels[test_idx], lr_scores[test_idx]))

    svm_auc = roc_auc_score(labels, svm_scores)
    svm_auc_pr = average_precision_score(labels, svm_scores)
    lr_auc = roc_auc_score(labels, lr_scores)
    lr_auc_pr = average_precision_score(labels, lr_scores)

    fpr_svm, tpr_svm, thresh_svm = roc_curve(labels, svm_scores)
    j_idx = tpr_svm - fpr_svm
    best_thresh_idx = np.argmax(j_idx)
    svm_y_pred = (svm_scores >= thresh_svm[best_thresh_idx]).astype(int)
    svm_f1 = f1_score(labels, svm_y_pred)
    svm_mcc = matthews_corrcoef(labels, svm_y_pred)
    tn, fp, fn, tp = confusion_matrix(labels, svm_y_pred).ravel()

    print(f"  Linear SVM: AUC={svm_auc:.4f} (PR={svm_auc_pr:.4f})", flush=True)
    print(f"    F1={svm_f1:.4f}, MCC={svm_mcc:.4f}", flush=True)
    print(f"    Sens={tp/(tp+fn):.4f}, Spec={tn/(tn+fp):.4f}", flush=True)
    print(f"    Fold AUCs: {[round(a,4) for a in fold_aucs_svm]}", flush=True)
    print(f"  L2 LR:      AUC={lr_auc:.4f} (PR={lr_auc_pr:.4f})", flush=True)

    # === Cluster annotation ===
    print("\n  Cluster annotation (best k)...", flush=True)
    annotations = []
    for c in range(best_k):
        mask = best_cluster == c
        n_in = mask.sum()
        pat_in = labels[mask].sum()
        ctrl_in = n_in - pat_in
        pat_frac = pat_in / n_in if n_in > 0 else 0
        overall_pat = np.mean(labels == 1)
        enrichment = pat_frac / (overall_pat + 1e-6)
        oddsr, pval = fisher_exact([[pat_in, ctrl_in],
                                    [np.sum(labels==1)-pat_in, np.sum(labels==0)-ctrl_in]])
        ann_label = "Patient-enriched" if pat_frac > overall_pat else "Control-enriched"
        annotations.append({
            'cluster': int(c), 'n_samples': int(n_in),
            'patient': int(pat_in), 'control': int(ctrl_in),
            'patient_frac': round(float(pat_frac), 3),
            'enrichment': round(float(enrichment), 3),
            'odds_ratio': round(float(oddsr), 3),
            'p_value': float(pval),
            'label': ann_label,
        })
        print(f"    Cluster {c}: {n_in} samples, {pat_frac:.0%} patient "
              f"(enrich={enrichment:.2f}, p={pval:.2e}) [{ann_label}]", flush=True)

    ari_best = cluster_results[best_k]['ari']
    nmi_best = cluster_results[best_k]['nmi']
    print(f"\n  ARI (best k vs labels): {ari_best:.4f}", flush=True)
    print(f"  NMI (best k vs labels): {nmi_best:.4f}", flush=True)

    # === Save coordinates ===
    coords_df = pd.DataFrame({
        'sample_id': [f'S{i}' for i in range(n_samples)],
        'label': labels,
        'group': ['Patient' if l == 1 else 'Control' for l in labels],
        'cluster': best_cluster,
        'UMAP1': umap_emb[:, 0],
        'UMAP2': umap_emb[:, 1],
        'PC1': pca_scores[:, 0],
        'PC2': pca_scores[:, 1],
        'PC3': pca_scores[:, 2] if pca_scores.shape[1] > 2 else 0,
        'svm_score': svm_scores,
        'lr_score': lr_scores,
    })
    coords_path = os.path.join(OUTPUT_DIR, 'unified_coordinates.csv')
    coords_df.to_csv(coords_path, index=False)
    print(f"\n  Coordinates saved: {coords_path}", flush=True)

    # === Save results JSON ===
    results = {
        'pipeline': 'L2_normalize + LinearSVM',
        'm': 10000,
        'n_samples': int(n_samples),
        'n_control': int(np.sum(labels == 0)),
        'n_patient': int(np.sum(labels == 1)),
        'sparsity': float(np.mean(X == 0)),
        'pca_ve': [round(float(v), 4) for v in ve[:20]],
        'pca_cumve': [round(float(v), 4) for v in cumve[:20]],
        'best_k': int(best_k),
        'best_silhouette': float(best_sil),
        'ari': float(ari_best),
        'nmi': float(nmi_best),
        'svm_auc': round(float(svm_auc), 4),
        'svm_auc_pr': round(float(svm_auc_pr), 4),
        'svm_f1': round(float(svm_f1), 4),
        'svm_mcc': round(float(svm_mcc), 4),
        'svm_sensitivity': round(float(tp/(tp+fn)), 4),
        'svm_specificity': round(float(tn/(tn+fp)), 4),
        'svm_fold_aucs': [round(float(a), 4) for a in fold_aucs_svm],
        'lr_auc': round(float(lr_auc), 4),
        'lr_auc_pr': round(float(lr_auc_pr), 4),
        'lr_fold_aucs': [round(float(a), 4) for a in fold_aucs_lr],
        'cluster_annotations': annotations,
        'silhouette_by_k': {str(k): round(float(v['silhouette']), 4) for k, v in cluster_results.items()},
        'ari_by_k': {str(k): round(float(v['ari']), 4) for k, v in cluster_results.items()},
        'nmi_by_k': {str(k): round(float(v['nmi']), 4) for k, v in cluster_results.items()},
    }
    results_path = os.path.join(OUTPUT_DIR, 'unified_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}", flush=True)

    # === Generate chart data for HTML report ===
    print("\n  Generating chart data...", flush=True)
    control = coords_df[coords_df['label'] == 0]
    patient = coords_df[coords_df['label'] == 1]

    chart_data = {
        'umap_label': {
            'control': [[float(r['UMAP1']), float(r['UMAP2']), r['sample_id']] for _, r in control.iterrows()],
            'patient': [[float(r['UMAP1']), float(r['UMAP2']), r['sample_id']] for _, r in patient.iterrows()],
        },
        'umap_svm': [],
        'umap_cluster': {},
        'pca_scatter': {
            'control': [[float(r['PC1']), float(r['PC2']), r['sample_id']] for _, r in control.iterrows()],
            'patient': [[float(r['PC1']), float(r['PC2']), r['sample_id']] for _, r in patient.iterrows()],
        },
        'pca_variance': {
            'individual': [round(v * 100, 2) for v in ve[:20]],
            'cumulative': [round(v * 100, 2) for v in cumve[:20]],
        },
        'silhouette': {
            'k_values': list(cluster_results.keys()),
            'scores': [round(float(cluster_results[k]['silhouette']), 4) for k in cluster_results],
            'ari': [round(float(cluster_results[k]['ari']), 4) for k in cluster_results],
            'nmi': [round(float(cluster_results[k]['nmi']), 4) for k in cluster_results],
        },
        'cluster_composition': annotations,
        'best_k': best_k,
        'roc': {
            'fpr': [round(float(v), 4) for v in fpr_svm],
            'tpr': [round(float(v), 4) for v in tpr_svm],
            'auc': round(float(svm_auc), 4),
        },
        'svm_hist': {
            'control': [float(v) for v in svm_scores[labels == 0]],
            'patient': [float(v) for v in svm_scores[labels == 1]],
        },
        'metrics': {
            'svm_auc': round(float(svm_auc), 4),
            'svm_auc_pr': round(float(svm_auc_pr), 4),
            'svm_f1': round(float(svm_f1), 4),
            'svm_mcc': round(float(svm_mcc), 4),
            'svm_sens': round(float(tp/(tp+fn)), 4),
            'svm_spec': round(float(tn/(tn+fp)), 4),
            'lr_auc': round(float(lr_auc), 4),
            'ari': float(ari_best),
            'nmi': float(nmi_best),
        },
    }

    # UMAP by cluster
    for c in range(best_k):
        mask = best_cluster == c
        chart_data['umap_cluster'][str(c)] = [
            [float(umap_emb[i, 0]), float(umap_emb[i, 1]), f'S{i}']
            for i in np.where(mask)[0]
        ]

    # UMAP colored by SVM score
    chart_data['umap_svm'] = [
        [float(umap_emb[i, 0]), float(umap_emb[i, 1]), float(svm_scores[i]), int(labels[i])]
        for i in range(n_samples)
    ]

    chart_path = os.path.join(OUTPUT_DIR, 'unified_chart_data.json')
    with open(chart_path, 'w') as f:
        json.dump(chart_data, f)
    print(f"  Chart data saved: {chart_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  Done!", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
