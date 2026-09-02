#!/usr/bin/env python3
"""
End-to-End Test: Multi-Layer Spectrum on Real RA-TRA Data
==========================================================
Compares classification performance:
  1. L1-only (prototype spectrum, 10,000-d) — current CDRscope v2
  2. Multi-layer (L1 + L2 + L4 + S-axes) — CDRscope v3

Uses:
  - Pre-computed L1 matrix (545 samples × 10,000 prototypes)
  - Raw RA-TRA CSV files for L2/L4/S-axes extraction
  - Linear SVM with 5-fold CV (matching v2 protocol)
"""
import os, sys, glob, json, time, warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats

warnings.filterwarnings('ignore')

BASE = os.path.expanduser("~/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8")
TIER2_DIR = os.path.join(BASE, "CDRscope-v2", "inst", "python", "tier2")
TIER3_DIR = os.path.join(BASE, "CDRscope-v2", "inst", "python", "tier3")
PANEL_DIR = os.path.join(BASE, "cordblood_tra_panel")
RA_CTRL_DIR = os.path.join(BASE, "CDRscope-analysis", "RA_data", "RA_Control_Files")
RA_PAT_DIR = os.path.join(BASE, "CDRscope-analysis", "RA_data", "RA_Patient_Files")
OUT_DIR = os.path.join(BASE, "tier3_e2e_results")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, TIER2_DIR)
sys.path.insert(0, os.path.dirname(TIER3_DIR))

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def is_valid_seq(s):
    s = str(s).strip()
    return len(s) >= 8 and all(aa in STANDARD_AA for aa in s)


# ---------------------------------------------------------------------------
# Step 1: Load pre-computed L1 matrix
# ---------------------------------------------------------------------------
def load_l1_matrix():
    """Load pre-computed RA-TRA prototype spectrum matrix."""
    mat_path = os.path.join(PANEL_DIR, "ra_tra_cb_matrix.npy")
    lbl_path = os.path.join(PANEL_DIR, "ra_tra_cb_labels.npy")
    X = np.load(mat_path)
    y = np.load(lbl_path).astype(int)
    print(f"  L1 matrix: {X.shape}, labels: {y.shape}")
    print(f"  Controls: {(y==0).sum()}, Patients: {(y==1).sum()}")
    return X, y


# ---------------------------------------------------------------------------
# Step 2: Load raw RA-TRA CSV files (for L2/L4/S-axes)
# ---------------------------------------------------------------------------
def load_ra_tra_raw():
    """Load raw RA-TRA CSV files, returning per-sample sequences + V/J genes."""
    samples = []
    for group, label, directory in [('Control', 0, RA_CTRL_DIR), ('Patient', 1, RA_PAT_DIR)]:
        files = sorted(glob.glob(os.path.join(directory, '*_TRA.csv')))
        for f in files:
            try:
                df = pd.read_csv(f)
                if len(df) == 0:
                    continue
                seq_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'
                df = df[df[seq_col].apply(is_valid_seq)]
                if len(df) == 0:
                    continue

                seqs = df[seq_col].values
                if 'duplicate_count' in df.columns:
                    counts = df['duplicate_count'].fillna(1).astype(int).values
                else:
                    counts = np.ones(len(seqs), dtype=int)

                v_genes = df['v_call'].fillna('unknown').values if 'v_call' in df.columns else ['unknown'] * len(seqs)
                j_genes = df['j_call'].fillna('unknown').values if 'j_call' in df.columns else ['unknown'] * len(seqs)

                sample_id = os.path.basename(f).replace('__TRA.csv', '').replace('_r__TRA.csv', '')
                samples.append({
                    'sample_id': sample_id,
                    'group': group,
                    'label': label,
                    'sequences': seqs.tolist(),
                    'counts': counts.tolist(),
                    'v_genes': v_genes.tolist(),
                    'j_genes': j_genes.tolist(),
                })
            except Exception as e:
                continue

    print(f"  Loaded {len(samples)} raw samples "
          f"({sum(1 for s in samples if s['label']==0)} ctrl + "
          f"{sum(1 for s in samples if s['label']==1)} pat)")
    return samples


# ---------------------------------------------------------------------------
# Step 3: Extract multi-layer features
# ---------------------------------------------------------------------------
def extract_multilayer_features(raw_samples, l1_matrix, fit_motif_on=None):
    """Extract L2, L4, S-axes for each sample and assemble with L1.

    Args:
        raw_samples: list of sample dicts from load_ra_tra_raw()
        l1_matrix: (n, 10000) pre-computed L1 matrix
        fit_motif_on: indices of samples to use for motif dictionary (default: first 100)

    Returns:
        (X_multilayer, y, layer_info)
    """
    from tier3.macro_indices import MacroIndexExtractor
    from tier3.selection_imprints import SelectionImprintExtractor
    from tier3.motif_layer import MotifSpectrumExtractor

    n_samples = len(raw_samples)
    print(f"\n  Extracting multi-layer features for {n_samples} samples...")

    # Build motif dictionary from reference subset
    if fit_motif_on is None:
        fit_motif_on = list(range(min(100, n_samples)))
    ref_seqs = []
    for idx in fit_motif_on:
        ref_seqs.extend(raw_samples[idx]['sequences'][:200])  # subsample for speed
    print(f"  Building motif dictionary from {len(ref_seqs)} reference sequences...")
    motif_ext = MotifSpectrumExtractor(top_k_3mer=200, top_k_4mer=200)
    motif_ext.fit(ref_seqs)

    # Fit selection imprint scaler on reference subset
    ref_samples_for_imprint = []
    for idx in fit_motif_on:
        s = raw_samples[idx]
        ref_samples_for_imprint.append({
            'sequences': s['sequences'][:500],
            'counts': s['counts'][:500],
            'v_genes': s['v_genes'][:500],
            'j_genes': s['j_genes'][:500],
        })
    imprint_ext = SelectionImprintExtractor()
    imprint_ext.fit(ref_samples_for_imprint)

    # Extractors
    macro_ext = MacroIndexExtractor()

    # Extract per-sample features
    l2_list, l4_list, s_list = [], [], []

    t0 = time.time()
    for i, sample in enumerate(raw_samples):
        seqs = sample['sequences']
        counts = sample['counts']
        v_genes = sample['v_genes']
        j_genes = sample['j_genes']

        # L2: Motif spectrum
        l2 = motif_ext.transform(seqs, counts)
        l2_list.append(l2)

        # L4: Macro indices (from counts)
        l4 = macro_ext.transform(counts)
        l4_list.append(l4)

        # S-axes: Selection imprints
        s_sample = {
            'sequences': seqs,
            'counts': counts,
            'v_genes': v_genes,
            'j_genes': j_genes,
        }
        s_vec = imprint_ext.transform(s_sample)
        s_list.append(s_vec)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_samples} ({elapsed:.0f}s)")

    l2_arr = np.array(l2_list, dtype=np.float32)
    l4_arr = np.array(l4_list, dtype=np.float32)
    s_arr = np.array(s_list, dtype=np.float32)

    # L1: L2-normalize
    l1_normed = normalize(l1_matrix, norm='l2', axis=1)

    # Per-layer L2 normalization
    l2_normed = normalize(l2_arr, norm='l2', axis=1)
    l4_normed = normalize(l4_arr.reshape(-1, 1), norm='l2', axis=0).ravel().reshape(n_samples, -1)
    s_normed = normalize(s_arr, norm='l2', axis=1)

    # Assemble multi-layer spectrum
    X_multilayer = np.hstack([l1_normed, l2_normed, l4_normed, s_normed])

    y = np.array([s['label'] for s in raw_samples])

    layer_info = {
        'L1_prototype': l1_normed.shape[1],
        'L2_motif': l2_normed.shape[1],
        'L4_macro': l4_normed.shape[1],
        'S_imprints': s_normed.shape[1],
        'total': X_multilayer.shape[1],
    }

    return X_multilayer, y, layer_info, (l1_normed, l2_normed, l4_normed, s_normed)


# ---------------------------------------------------------------------------
# Step 4: Classification comparison
# ---------------------------------------------------------------------------
def classify_compare(X_l1, X_multilayer, y, layer_components=None):
    """Compare L1-only vs multi-layer classification.

    Args:
        X_l1: (n, 10000) L1-only features
        X_multilayer: (n, d) multi-layer features
        y: (n,) labels
        layer_components: optional dict of individual layers for ablation

    Returns:
        dict with AUC and accuracy for each configuration
    """
    results = {}
    configs = {
        'L1_only (v2)': X_l1,
        'Multi-layer (v3)': X_multilayer,
    }
    if layer_components:
        configs.update({
            'L2_only': layer_components['L2'],
            'L4_only': layer_components['L4'],
            'S_only': layer_components['S'],
            'L1+L4': np.hstack([X_l1, layer_components['L4']]),
            'L1+S': np.hstack([X_l1, layer_components['S']]),
            'L1+L2': np.hstack([X_l1, layer_components['L2']]),
        })

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, X in configs.items():
        aucs, accs = [], []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LinearSVC(C=0.1, max_iter=5000, random_state=42, dual='auto')
            clf.fit(X_train, y_train)
            scores = clf.decision_function(X_test)
            preds = clf.predict(X_test)

            aucs.append(roc_auc_score(y_test, scores))
            accs.append(accuracy_score(y_test, preds))

        results[name] = {
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'n_features': X.shape[1],
        }
        print(f"  {name:<20s} | dim={X.shape[1]:>6d} | "
              f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f} | "
              f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CDRscope v3 End-to-End Test: Multi-Layer Spectrum on RA-TRA")
    print("=" * 70)

    # Step 1: Load pre-computed L1
    print("\n[1] Loading pre-computed L1 matrix...")
    X_l1_raw, y = load_l1_matrix()

    # Step 2: Load raw samples
    print("\n[2] Loading raw RA-TRA CSV files...")
    raw_samples = load_ra_tra_raw()

    # Match L1 matrix with raw samples
    n_match = min(len(raw_samples), X_l1_raw.shape[0])
    print(f"  Matching: {n_match} raw samples, {X_l1_raw.shape[0]} L1 rows")
    X_l1 = X_l1_raw[:n_match]
    y = y[:n_match]
    raw_samples = raw_samples[:n_match]

    # Step 3: Extract multi-layer features
    print("\n[3] Extracting multi-layer features...")
    X_multi, y_multi, layer_info, components = extract_multilayer_features(
        raw_samples, X_l1
    )

    print(f"\n  Layer dimensions:")
    for name, dim in layer_info.items():
        print(f"    {name}: {dim}")
    print(f"  Multi-layer spectrum: {X_multi.shape}")

    # Step 4: Classification comparison
    print("\n[4] Classification comparison (Linear SVM, 5-fold CV):")
    print("-" * 70)

    l1_normed = normalize(X_l1, norm='l2', axis=1)
    layer_components = {
        'L2': components[1],
        'L4': components[2],
        'S': components[3],
    }

    results = classify_compare(l1_normed, X_multi, y, layer_components)

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    l1_auc = results['L1_only (v2)']['auc_mean']
    multi_auc = results['Multi-layer (v3)']['auc_mean']
    delta = multi_auc - l1_auc

    print(f"\n  L1-only (v2):     AUC = {l1_auc:.4f}")
    print(f"  Multi-layer (v3): AUC = {multi_auc:.4f}")
    print(f"  Δ AUC:            {delta:+.4f} ({'improvement' if delta > 0 else 'degradation'})")

    # Per-layer contribution
    print(f"\n  Per-layer ablation (vs L1 baseline):")
    for name in ['L2_only', 'L4_only', 'S_only', 'L1+L2', 'L1+L4', 'L1+S']:
        if name in results:
            r = results[name]
            delta_r = r['auc_mean'] - l1_auc
            print(f"    {name:<12s} | AUC={r['auc_mean']:.4f} | Δ={delta_r:+.4f} | dim={r['n_features']}")

    # Save results
    output = {
        'test': 'CDRscope v3 multi-layer spectrum end-to-end test',
        'dataset': 'RA-TRA (Aterido 2024)',
        'n_samples': int(n_match),
        'layer_info': layer_info,
        'classification_results': results,
        'summary': {
            'l1_auc': l1_auc,
            'multi_auc': multi_auc,
            'delta_auc': delta,
        },
    }
    out_path = os.path.join(OUT_DIR, 'e2e_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Also save the multi-layer matrix for future use
    np.save(os.path.join(OUT_DIR, 'ra_tra_multilayer.npy'), X_multi)
    np.save(os.path.join(OUT_DIR, 'ra_tra_multilayer_labels.npy'), y)
    print(f"  Multi-layer matrix saved ({X_multi.shape})")

    return results


if __name__ == '__main__':
    results = main()
