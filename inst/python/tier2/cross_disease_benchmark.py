#!/usr/bin/env python3
"""
CDRscope v2 — Cross-Disease Benchmark Pipeline
================================================
Unified analysis pipeline testing CDRscope v2 across multiple disease
fields: RA, CMV, SLE, and VDJdb multi-disease specificity.

Datasets:
  1. RA (Aterido 2024) — per-sample bulk repertoire (TRA, TRB, TRA+TRB)
  2. CMV (Emerson 2017) — per-sample bulk repertoire (TRB)
  3. SLE (Liu 2019, via DeepTAPE) — pseudo-sample from aggregated CDR3s (TRB)
  4. VDJdb multi-disease — sequence-level specificity (CMV, SARS-CoV-2, HIV, EBV, Flu)

Each dataset is analyzed with the full CDRscope v2 feature set:
  - Physicochemical (8 features)
  - Diversity (7 features)
  - Convergence (3 features)
  - V/J gene one-hot (variable)
  - V-J pairing (11 features)

Classification: 5-fold CV Random Forest, AUC-ROC metric.
"""
import os, sys, json, glob, time, warnings, random, pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, f1_score, matthews_corrcoef,
                              accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "cross_disease_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "model_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
KD = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,
      'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,
      'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
CHARGE = {'K':+1,'R':+1,'H':+0.5,'D':-1,'E':-1}
AROMATIC = set('FWY')

N_TOP = 500  # top clones per sample

# =========================================================================
# IMGT Standardized V/J Gene Vocabulary (unified feature space)
# Ensures all datasets of the same chain produce identical feature columns,
# enabling cross-disease model transfer (Tier 1 → Tier 2).
# =========================================================================
IMGT_TRBV = [
    'TRBV1','TRBV2','TRBV3-1','TRBV4-1','TRBV5-1','TRBV5-2','TRBV5-3',
    'TRBV5-4','TRBV5-5','TRBV5-6','TRBV5-7','TRBV5-8','TRBV6-1','TRBV6-2',
    'TRBV6-3','TRBV6-4','TRBV6-5','TRBV6-6','TRBV6-7','TRBV6-8','TRBV6-9',
    'TRBV7-1','TRBV7-2','TRBV7-3','TRBV7-4','TRBV7-6','TRBV7-7','TRBV7-8',
    'TRBV7-9','TRBV9','TRBV10-1','TRBV10-2','TRBV10-3','TRBV11-1','TRBV11-2',
    'TRBV11-3','TRBV12-3','TRBV12-4','TRBV12-5','TRBV13','TRBV14','TRBV15',
    'TRBV16','TRBV17','TRBV18','TRBV19','TRBV20-1','TRBV21-1','TRBV22-1',
    'TRBV23-1','TRBV24-1','TRBV25-1','TRBV26','TRBV27','TRBV28','TRBV29-1',
    'TRBV30','TRBV31','TRBV32','TRBV33-1','TRBV34','TRBV35','TRBV36','TRBV37',
]

IMGT_TRBJ = [
    'TRBJ1-1','TRBJ1-2','TRBJ1-3','TRBJ1-4','TRBJ1-5','TRBJ1-6',
    'TRBJ2-1','TRBJ2-2','TRBJ2-3','TRBJ2-4','TRBJ2-5','TRBJ2-6','TRBJ2-7',
]

IMGT_TRAV = [
    'TRAV1-1','TRAV1-2','TRAV2','TRAV3-1','TRAV3-2','TRAV4','TRAV5','TRAV6',
    'TRAV7','TRAV8-1','TRAV8-2','TRAV8-3','TRAV8-4','TRAV8-5','TRAV8-6',
    'TRAV8-7','TRAV9-1','TRAV9-2','TRAV10','TRAV11','TRAV12-1','TRAV12-2',
    'TRAV12-3','TRAV13-1','TRAV13-2','TRAV14','TRAV15','TRAV16','TRAV17',
    'TRAV18','TRAV19','TRAV20','TRAV21','TRAV22','TRAV23','TRAV24','TRAV25',
    'TRAV26-1','TRAV26-2','TRAV27','TRAV28','TRAV29','TRAV30','TRAV31',
    'TRAV32','TRAV33','TRAV34','TRAV35','TRAV36','TRAV37','TRAV38-1',
    'TRAV38-2','TRAV39','TRAV40','TRAV41','TRAV42-1','TRAV42-2',
]

IMGT_TRAJ = [
    'TRAJ1','TRAJ2','TRAJ3','TRAJ4','TRAJ5','TRAJ6','TRAJ7','TRAJ8',
    'TRAJ9','TRAJ10','TRAJ11','TRAJ12','TRAJ13','TRAJ14','TRAJ15','TRAJ16',
    'TRAJ17','TRAJ18','TRAJ19','TRAJ20','TRAJ21','TRAJ22','TRAJ23','TRAJ24',
    'TRAJ25','TRAJ26','TRAJ27','TRAJ28','TRAJ29','TRAJ30','TRAJ31','TRAJ32',
    'TRAJ33','TRAJ34','TRAJ35','TRAJ36','TRAJ37','TRAJ38','TRAJ39','TRAJ40',
    'TRAJ41','TRAJ42','TRAJ43','TRAJ44','TRAJ45','TRAJ46','TRAJ47','TRAJ48',
    'TRAJ49','TRAJ50','TRAJ51','TRAJ52','TRAJ53','TRAJ54',
]

GENE_VOCAB = {
    'TRB': {'V': IMGT_TRBV, 'J': IMGT_TRBJ},
    'TRA': {'V': IMGT_TRAV, 'J': IMGT_TRAJ},
    'TRA+TRB': {'V': IMGT_TRAV + IMGT_TRBV, 'J': IMGT_TRAJ + IMGT_TRBJ},
}


def is_valid_seq(seq):
    if not isinstance(seq, str) or len(seq) < 4:
        return False
    return all(a in STANDARD_AA for a in seq)


# =========================================================================
# Feature Engineering (CDRscope v2 unified)
# =========================================================================

def compute_features(df, seq_col='junction_aa', v_col='v_call', j_col='j_call',
                     count_col='duplicate_count', chain='TRB'):
    """Compute all CDRscope v2 features for a sample using standardized IMGT gene vocabulary."""
    seqs = df[seq_col].values
    counts = df[count_col].values if count_col in df.columns else np.ones(len(seqs))
    counts = np.maximum(counts, 1)
    freqs = counts / counts.sum()
    n = len(seqs)

    lengths = [len(s) for s in seqs]
    charges = [sum(CHARGE.get(a, 0) for a in s) for s in seqs]
    hydros = [np.mean([KD.get(a, 0) for a in s]) if s else 0 for s in seqs]
    aroms = [sum(1 for a in s if a in AROMATIC)/len(s) if s else 0 for s in seqs]
    w = freqs / (freqs.sum() + 1e-10)

    p = counts / counts.sum()
    sorted_p = np.sort(p)[::-1]
    cum = np.cumsum(sorted_p)

    feats = {
        # Physicochemical
        'phys_len_mean': np.average(lengths, weights=w),
        'phys_len_std': np.std(lengths),
        'phys_charge_mean': np.average(charges, weights=w),
        'phys_charge_std': np.std(charges),
        'phys_hydro_mean': np.average(hydros, weights=w),
        'phys_hydro_std': np.std(hydros),
        'phys_arom_mean': np.average(aroms, weights=w),
        'phys_arom_std': np.std(aroms),
        # Diversity
        'div_simpson': 1 - np.sum(p**2),
        'div_shannon': entropy(p),
        'div_clonality': 1 - (entropy(p) / np.log(n)) if n > 1 else 0,
        'div_d50': (np.searchsorted(cum, 0.5) + 1) / n,
        'div_pielou': entropy(p) / np.log(n) if n > 1 else 0,
        'div_n_clones': n,
        # Convergence
        'conv_top1_freq': sorted_p[0] if len(sorted_p) > 0 else 0,
        'conv_top5_freq': sum(sorted_p[:5]) / (sum(sorted_p) + 1e-10),
        'conv_unique_ratio': len(set(seqs)) / n,
    }

    # V/J gene one-hot using standardized IMGT vocabulary
    v_genes = df[v_col].values if v_col in df.columns else ['?']*n
    j_genes = df[j_col].values if j_col in df.columns else ['?']*n
    v_counts = Counter(v_genes)
    j_counts = Counter(j_genes)
    vocab = GENE_VOCAB.get(chain, GENE_VOCAB['TRB'])
    for vg in vocab['V']:
        feats[f'vgene_{vg}'] = v_counts.get(vg, 0) / n
    for jg in vocab['J']:
        feats[f'jgene_{jg}'] = j_counts.get(jg, 0) / n

    # V-J pairing features
    pair_counts = Counter()
    v_w_counts = Counter()
    j_w_counts = Counter()
    for v, j, c in zip(v_genes, j_genes, counts):
        pair_counts[(v, j)] += c
        v_w_counts[v] += c
        j_w_counts[j] += c

    n_unique_pairs = len(pair_counts)
    n_v = len(v_w_counts)
    n_j = len(j_w_counts)

    pair_vals = np.array(list(pair_counts.values()), dtype=float)
    pair_p = pair_vals / pair_vals.sum()
    v_vals = np.array(list(v_w_counts.values()), dtype=float)
    v_p = v_vals / v_vals.sum()
    j_vals = np.array(list(j_w_counts.values()), dtype=float)
    j_p = j_vals / j_vals.sum()

    feats.update({
        'vj_n_unique_pairs': n_unique_pairs,
        'vj_pair_entropy': entropy(pair_p) if len(pair_p) > 0 else 0,
        'vj_pair_simpson': 1 - np.sum(pair_p**2) if len(pair_p) > 0 else 0,
        'vj_max_pair_frac': pair_p.max() if len(pair_p) > 0 else 0,
        'vj_v_entropy': entropy(v_p),
        'vj_v_simpson': 1 - np.sum(v_p**2),
        'vj_j_entropy': entropy(j_p),
        'vj_j_simpson': 1 - np.sum(j_p**2),
        'vj_n_v_genes': n_v,
        'vj_n_j_genes': n_j,
        'vj_pair_ratio': n_unique_pairs / (n_v * n_j) if n_v > 0 and n_j > 0 else 0,
    })

    return feats


# =========================================================================
# Classification — Unified CV with comprehensive metrics + model checkpoint
# =========================================================================

def run_cv_full(X, y, feature_names, n_folds=5, sample_ids=None):
    """Single 5-fold CV pass returning AUC, ROC curve, AUC-PR, F1, MCC,
    feature importances, per-sample predictions, and the final trained model."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_probs, all_true, all_ids = [], [], []
    importances = np.zeros(X.shape[1])
    models = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        prob = rf.predict_proba(X[test_idx])[:, 1]
        all_probs.extend(prob)
        all_true.extend(y[test_idx].tolist())
        importances += rf.feature_importances_
        models.append(rf)
        if sample_ids is not None:
            all_ids.extend([sample_ids[i] for i in test_idx])

    importances /= n_folds
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    auc_roc = roc_auc_score(all_true, all_probs)
    fpr, tpr, roc_thresh = roc_curve(all_true, all_probs)
    auc_pr = average_precision_score(all_true, all_probs)

    # Youden's J threshold for binary metrics
    j_index = tpr - fpr
    best_idx = np.argmax(j_index)
    best_thresh = roc_thresh[best_idx]
    y_pred = (all_probs >= best_thresh).astype(int)
    f1 = f1_score(all_true, y_pred)
    mcc = matthews_corrcoef(all_true, y_pred)
    acc = accuracy_score(all_true, y_pred)
    tp = int(np.sum((y_pred == 1) & (all_true == 1)))
    fp = int(np.sum((y_pred == 1) & (all_true == 0)))
    fn = int(np.sum((y_pred == 0) & (all_true == 1)))
    tn = int(np.sum((y_pred == 0) & (all_true == 0)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Retrain final model on all data for checkpoint
    final_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    return {
        'auc_roc': round(auc_roc, 4),
        'auc_pr': round(auc_pr, 4),
        'f1': round(f1, 4),
        'mcc': round(mcc, 4),
        'accuracy': round(acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'youden_j': round(j_index[best_idx], 4),
        'best_threshold': round(float(best_thresh), 4),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'importances': dict(zip(feature_names, importances.tolist())),
        'sample_predictions': list(zip(all_ids, all_true.tolist(), all_probs.tolist())) if sample_ids is not None else None,
        'model': final_model,
        'feature_names': feature_names,
    }


def save_checkpoint(result, dataset_name, chain='TRB'):
    """Save model checkpoint for Tier 2 integration."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.replace('/', '_')}.pkl")
    ckpt = {
        'dataset': dataset_name,
        'chain': chain,
        'model': result['model'],
        'feature_names': result['feature_names'],
        'gene_vocab': GENE_VOCAB.get(chain, GENE_VOCAB['TRB']),
        'auc_roc': result['auc_roc'],
        'n_samples': result.get('n_samples'),
        'n_features': result.get('n_features'),
        'category_importance': result.get('category_importance'),
        'metrics': {k: v for k, v in result.items() if k in
                    ('auc_roc', 'auc_pr', 'f1', 'mcc', 'accuracy',
                     'sensitivity', 'specificity', 'youden_j', 'best_threshold')},
    }
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt, f)
    print(f"  Model checkpoint saved: {ckpt_path}")
    return ckpt_path


def load_checkpoint(dataset_name):
    """Load model checkpoint for Tier 2 sequence-level analysis."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.replace('/', '_')}.pkl")
    with open(ckpt_path, 'rb') as f:
        return pickle.load(f)


# =========================================================================
# Dataset Loaders
# =========================================================================

def load_ra_dataset(chain='TRB'):
    """Load RA dataset (Aterido 2024)."""
    ctrl_dir = os.path.join(BASE, "CDRscope-analysis/RA_data/RA_Control_Files")
    pat_dir = os.path.join(BASE, "CDRscope-analysis/RA_data/RA_Patient_Files")
    samples = []
    for group, label, directory in [('Control', 0, ctrl_dir), ('Patient', 1, pat_dir)]:
        files = sorted(glob.glob(os.path.join(directory, f'*{chain}*.csv')))
        for f in files:
            try:
                df = pd.read_csv(f)
                if len(df) == 0: continue
                seq_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'
                df = df[df[seq_col].apply(is_valid_seq)]
                if len(df) == 0: continue
                count_col = 'duplicate_count' if 'duplicate_count' in df.columns else None
                if count_col:
                    df = df.sort_values(count_col, ascending=False).head(N_TOP)
                df = df.rename(columns={seq_col: 'junction_aa'})
                samples.append({
                    'sample_id': os.path.basename(f).replace(f'__{chain}.csv', '').replace(f'_r__{chain}.csv', ''),
                    'group': group, 'label': label, 'df': df, 'chain': chain
                })
            except: continue
    return samples


def load_ra_joint():
    """Load RA TRA+TRB joint samples."""
    tra = {s['sample_id']: s for s in load_ra_dataset('TRA')}
    trb = {s['sample_id']: s for s in load_ra_dataset('TRB')}
    common = set(tra.keys()) & set(trb.keys())
    samples = []
    for sid in sorted(common):
        df = pd.concat([tra[sid]['df'], trb[sid]['df']], ignore_index=True)
        samples.append({
            'sample_id': sid, 'group': tra[sid]['group'], 'label': tra[sid]['label'],
            'df': df, 'chain': 'TRA+TRB'
        })
    return samples


def load_ms_dataset():
    """Load Multiple Sclerosis dataset (Alves Sousa 2019, GEO GSE121082).
    Real per-sample bulk TCRbeta sequencing from PBMCs."""
    ms_dir = os.path.join(BASE, "ms_tcr_data")
    samples = []
    seen_patients = set()

    for f in sorted(glob.glob(os.path.join(ms_dir, "*.txt.gz"))):
        basename = os.path.basename(f)
        # Parse group from filename: GSM{ID}_{Group}_{S}_{CellType}[_L{n}].txt.gz
        parts = basename.replace('.txt.gz', '').split('_')
        if len(parts) < 4:
            continue
        gsm_id = parts[0]
        group_label = parts[1]  # HC, MS, or HAM
        sample_id = parts[2]  # S1, S2, etc.
        cell_type = parts[3] if len(parts) > 3 else 'PBMC'

        # Only use PBMC samples (exclude CD4/CD8 sorted)
        if cell_type != 'PBMC':
            continue

        # Only use MS and HC (exclude HAM/TSP)
        if group_label == 'HAM':
            continue

        # Skip longitudinal replicates (L1, L2, L3) - take only first
        if len(parts) > 5 and parts[4].startswith('L') and parts[4] != 'L1':
            continue
        if len(parts) > 5 and parts[4].startswith('L'):
            pass  # Keep L1, skip L2/L3 handled above

        # Skip if already seen this patient (avoid duplicates)
        patient_key = f"{group_label}_{sample_id}"
        if patient_key in seen_patients:
            continue
        seen_patients.add(patient_key)

        try:
            df = pd.read_csv(f, sep='\t', compression='gzip')
            if len(df) == 0:
                continue

            # Map columns to standard format
            seq_col = 'CDR3 amino acid sequence'
            v_col = 'V segments'
            j_col = 'J segments'
            count_col = 'Count'

            if seq_col not in df.columns:
                continue

            df = df.rename(columns={
                seq_col: 'junction_aa',
                v_col: 'v_call',
                j_col: 'j_call',
                count_col: 'duplicate_count'
            })

            # Filter valid sequences
            df = df[df['junction_aa'].apply(is_valid_seq)]
            if len(df) == 0:
                continue

            # Normalize gene names
            if 'v_call' in df.columns:
                df['v_call'] = df['v_call'].apply(normalize_gene_name)
            if 'j_call' in df.columns:
                df['j_call'] = df['j_call'].apply(normalize_gene_name)

            # Sort by count and take top N
            if 'duplicate_count' in df.columns:
                df = df.sort_values('duplicate_count', ascending=False).head(N_TOP)

            label = 1 if group_label == 'MS' else 0
            samples.append({
                'sample_id': f'{gsm_id}_{group_label}_{sample_id}',
                'group': 'MS' if group_label == 'MS' else 'Control',
                'label': label,
                'df': df,
                'chain': 'TRB'
            })
        except Exception as e:
            continue

    return samples


def load_emerson_cmv():
    """Load Emerson 2017 CMV dataset."""
    samples = []
    emerson_dir = os.path.join(BASE, "emerson_cdrscope")
    for group, label in [('negative', 0), ('positive', 1)]:
        group_dir = os.path.join(emerson_dir, group)
        if not os.path.exists(group_dir): continue
        for f in sorted(glob.glob(os.path.join(group_dir, '*.csv'))):
            try:
                df = pd.read_csv(f)
                if len(df) == 0: continue
                df = df[df['cdr3_aa'].apply(is_valid_seq)]
                if len(df) == 0: continue
                count_col = 'count' if 'count' in df.columns else 'duplicate_count'
                if count_col in df.columns:
                    df = df.sort_values(count_col, ascending=False).head(N_TOP)
                df = df.rename(columns={'cdr3_aa': 'junction_aa'})
                if 'v_call' not in df.columns and 'v_gene' in df.columns:
                    df = df.rename(columns={'v_gene': 'v_call'})
                if 'j_call' not in df.columns and 'j_gene' in df.columns:
                    df = df.rename(columns={'j_gene': 'j_call'})
                # Normalize gene names (TCRBV29 -> TRBV29)
                if 'v_call' in df.columns:
                    df['v_call'] = df['v_call'].apply(normalize_gene_name)
                if 'j_call' in df.columns:
                    df['j_call'] = df['j_call'].apply(normalize_gene_name)
                samples.append({
                    'sample_id': os.path.basename(f).replace('__TRB.csv', ''),
                    'group': group, 'label': label, 'df': df, 'chain': 'TRB'
                })
            except: continue
    return samples


def normalize_gene_name(gene):
    """Normalize gene names: TCRBV29 -> TRBV29, TRBV20-1*07 -> TRBV20-1."""
    if not isinstance(gene, str) or gene == '?':
        return '?'
    gene = gene.split('*')[0].strip()
    if gene.startswith('TCR'):
        gene = gene.replace('TCR', 'TR', 1)
    return gene


def load_sle_pseudo(n_samples_per_group=50, seqs_per_sample=500):
    """Load SLE dataset (Liu 2019, via DeepTAPE) as pseudo-samples."""
    sle_dir = os.path.join(BASE, "pird_sle_tcr_data")
    # Load all SLE and HI CDR3s with V gene info
    sle_data = []  # list of (cdr3, v_gene)
    hi_data = []
    for f in sorted(glob.glob(os.path.join(sle_dir, "SLE_[1-5].csv"))):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            seq = row.get('CDR3AA', '')
            vg = normalize_gene_name(row.get('V_Gene', '?'))
            if is_valid_seq(seq):
                sle_data.append((seq, vg))
    for f in sorted(glob.glob(os.path.join(sle_dir, "HI_[1-5].csv"))):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            seq = row.get('CDR3AA', '')
            vg = normalize_gene_name(row.get('V_Gene', '?'))
            if is_valid_seq(seq):
                hi_data.append((seq, vg))

    random.seed(42)
    samples = []
    for i in range(n_samples_per_group):
        n_sle = min(seqs_per_sample, len(sle_data))
        n_hi = min(seqs_per_sample, len(hi_data))
        sle_sample = random.sample(sle_data, n_sle)
        hi_sample = random.sample(hi_data, n_hi)

        sle_df = pd.DataFrame({
            'junction_aa': [x[0] for x in sle_sample],
            'v_call': [x[1] for x in sle_sample],
            'j_call': '?',
            'duplicate_count': 1
        })
        hi_df = pd.DataFrame({
            'junction_aa': [x[0] for x in hi_sample],
            'v_call': [x[1] for x in hi_sample],
            'j_call': '?',
            'duplicate_count': 1
        })

        samples.append({
            'sample_id': f'SLE_{i}', 'group': 'SLE', 'label': 1,
            'df': sle_df, 'chain': 'TRB'
        })
        samples.append({
            'sample_id': f'HC_{i}', 'group': 'Control', 'label': 0,
            'df': hi_df, 'chain': 'TRB'
        })
    return samples


def load_sle_persample():
    """Load real per-sample SLE TCR data from PIRD/CNGB.

    Expected directory structure:
      pird_sle_persample_data/
        SLE/  *.tsv or *.csv  (one file per patient)
        HC/   *.tsv or *.csv  (one file per healthy control)

    Each file should contain CDR3 amino acid sequences with V/J gene info.
    Column name variants are auto-mapped.
    """
    data_dir = os.path.join(BASE, "pird_sle_persample_data")
    samples = []

    col_maps = [
        {'seq': 'CDR3AA', 'v': 'V_Gene', 'j': 'J_Gene', 'freq': 'Frequency'},
        {'seq': 'junction_aa', 'v': 'v_call', 'j': 'j_call', 'freq': 'duplicate_count'},
        {'seq': 'CDR3aa', 'v': 'vGeneName', 'j': 'jGeneName', 'freq': 'cloneFraction'},
        {'seq': 'amino_acid', 'v': 'v_gene', 'j': 'j_gene', 'freq': 'count'},
    ]

    for group_label, label, subdir in [('SLE', 1, 'SLE'), ('Control', 0, 'HC')]:
        group_dir = os.path.join(data_dir, subdir)
        if not os.path.isdir(group_dir):
            continue
        for f in sorted(glob.glob(os.path.join(group_dir, "*.tsv")) +
                        glob.glob(os.path.join(group_dir, "*.csv")) +
                        glob.glob(os.path.join(group_dir, "*.txt"))):
            try:
                sep = '\t' if f.endswith('.tsv') or f.endswith('.txt') else ','
                df = pd.read_csv(f, sep=sep)
                if len(df) == 0:
                    continue

                cmap = None
                for m in col_maps:
                    if m['seq'] in df.columns:
                        cmap = m
                        break
                if cmap is None:
                    seq_cols = [c for c in df.columns if 'cdr3' in c.lower() or 'aa' in c.lower()]
                    if not seq_cols:
                        continue
                    cmap = {'seq': seq_cols[0], 'v': 'V_Gene', 'j': 'J_Gene', 'freq': 'Frequency'}
                    for c in df.columns:
                        cl = c.lower()
                        if 'v_gene' in cl or 'v_call' in cl or c == 'V_Gene':
                            cmap['v'] = c
                        if 'j_gene' in cl or 'j_call' in cl or c == 'J_Gene':
                            cmap['j'] = c

                df = df.rename(columns={
                    cmap['seq']: 'junction_aa',
                    cmap.get('v', 'V_Gene'): 'v_call',
                    cmap.get('j', 'J_Gene'): 'j_call',
                    cmap.get('freq', 'Frequency'): 'duplicate_count'
                })

                df = df[df['junction_aa'].apply(lambda x: is_valid_seq(str(x)) if pd.notna(x) else False)]
                if len(df) == 0:
                    continue

                if 'v_call' in df.columns:
                    df['v_call'] = df['v_call'].apply(lambda x: normalize_gene_name(str(x)) if pd.notna(x) else '?')
                else:
                    df['v_call'] = '?'
                if 'j_call' in df.columns:
                    df['j_call'] = df['j_call'].apply(lambda x: normalize_gene_name(str(x)) if pd.notna(x) else '?')
                else:
                    df['j_call'] = '?'

                if 'duplicate_count' in df.columns:
                    df = df.sort_values('duplicate_count', ascending=False).head(N_TOP)

                sample_id = os.path.splitext(os.path.basename(f))[0]
                samples.append({
                    'sample_id': sample_id,
                    'group': group_label,
                    'label': label,
                    'df': df,
                    'chain': 'TRB'
                })
            except Exception as e:
                print(f"  Warning: Failed to load {f}: {e}")
                continue

    print(f"  Loaded {len([s for s in samples if s['label']==1])} SLE + "
          f"{len([s for s in samples if s['label']==0])} HC per-sample data")
    return samples


def load_vdjdb_disease(disease_name, n_per_group=2000):
    """Load VDJdb disease-specific CDR3s as sequence-level classification."""
    disease_file = os.path.join(BASE, f"vdjdb_data/disease_specific/{disease_name}_trb.tsv")
    if not os.path.exists(disease_file):
        return None

    df = pd.read_csv(disease_file, sep='\t')
    pos_seqs = df['cdr3'].dropna().tolist()
    pos_seqs = [s for s in pos_seqs if is_valid_seq(s)]

    # Load background (healthy) sequences from SLE HI data as negative
    hi_dir = os.path.join(BASE, "pird_sle_tcr_data")
    neg_seqs = []
    for f in sorted(glob.glob(os.path.join(hi_dir, "HI_[1-5].csv"))):
        df_hi = pd.read_csv(f)
        neg_seqs.extend(df_hi['CDR3AA'].dropna().tolist())
    neg_seqs = [s for s in neg_seqs if is_valid_seq(s)]

    random.seed(42)
    n_pos = min(n_per_group, len(pos_seqs))
    n_neg = min(n_per_group, len(neg_seqs))
    pos_sample = random.sample(pos_seqs, n_pos)
    neg_sample = random.sample(neg_seqs, n_neg)

    # Create per-sequence features
    pos_df = pd.DataFrame({
        'junction_aa': pos_sample,
        'v_call': df['v_gene'].values[:n_pos] if 'v_gene' in df.columns else ['?']*n_pos,
        'j_call': df['j_gene'].values[:n_pos] if 'j_gene' in df.columns else ['?']*n_pos,
        'duplicate_count': 1,
        'label': 1
    })
    neg_df = pd.DataFrame({
        'junction_aa': neg_sample,
        'v_call': '?',
        'j_call': '?',
        'duplicate_count': 1,
        'label': 0
    })

    return pd.concat([pos_df, neg_df], ignore_index=True)


def compute_sequence_features(seq, v_gene='?', j_gene='?'):
    """Compute per-sequence features (not per-sample)."""
    n = len(seq)
    charge = sum(CHARGE.get(a, 0) for a in seq)
    hydro = np.mean([KD.get(a, 0) for a in seq])
    arom = sum(1 for a in seq if a in AROMATIC) / n

    return {
        'seq_len': n,
        'seq_charge': charge,
        'seq_charge_per_aa': charge / n,
        'seq_hydro': hydro,
        'seq_arom': arom,
        'seq_n_aromatic': sum(1 for a in seq if a in AROMATIC),
        'seq_n_positive': sum(1 for a in seq if CHARGE.get(a, 0) > 0),
        'seq_n_negative': sum(1 for a in seq if CHARGE.get(a, 0) < 0),
        'seq_net_charge': charge,
    }


# =========================================================================
# Main Benchmark
# =========================================================================

def benchmark_per_sample(name, samples, published_auc=None, published_ref="", chain='TRB'):
    """Run per-sample classification benchmark with unified feature space,
    comprehensive metrics, model checkpoint, and per-sample predictions."""
    print(f"\n{'='*70}")
    print(f"  Dataset: {name}")
    print(f"  Samples: {len(samples)}", end="")
    n_ctrl = sum(1 for s in samples if s['label'] == 0)
    n_case = sum(1 for s in samples if s['label'] == 1)
    print(f" (Control: {n_ctrl}, Case: {n_case})")

    # Determine chain from samples or parameter
    use_chain = chain
    if samples and 'chain' in samples[0]:
        use_chain = samples[0]['chain']

    # Compute features with standardized gene vocabulary
    all_features = []
    for sample in samples:
        sample_chain = sample.get('chain', use_chain)
        feats = compute_features(sample['df'], chain=sample_chain)
        feats['sample_id'] = sample['sample_id']
        feats['label'] = sample['label']
        all_features.append(feats)

    feat_df = pd.DataFrame(all_features)
    sample_ids = feat_df['sample_id'].tolist()
    feature_cols = [c for c in feat_df.columns if c not in ('sample_id', 'label')]
    y = feat_df['label'].values

    X = np.nan_to_num(feat_df[feature_cols].values, nan=0.0)
    mask = np.std(X, axis=0) > 0
    X = X[:, mask]
    cols = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]

    n_onehot = sum(1 for c in cols if c.startswith('vgene_') or c.startswith('jgene_'))
    print(f"  Features: {X.shape[1]} (base=17, onehot={n_onehot}, pairing=11)")

    # Single CV pass with all metrics
    cv_result = run_cv_full(X, y, cols, n_folds=5, sample_ids=sample_ids)

    auc = cv_result['auc_roc']
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  AUC-PR:  {cv_result['auc_pr']:.4f}")
    print(f"  F1:      {cv_result['f1']:.4f}  MCC: {cv_result['mcc']:.4f}")
    print(f"  Sens:    {cv_result['sensitivity']:.4f}  Spec: {cv_result['specificity']:.4f}")
    print(f"  Youden J: {cv_result['youden_j']:.4f} (threshold={cv_result['best_threshold']:.3f})")

    if published_auc:
        diff = auc - published_auc
        print(f"  Published AUC: {published_auc:.4f} ({published_ref})")
        print(f"  Difference: {diff:+.4f}")

    imp = cv_result['importances']
    base_imp = sum(v for k, v in imp.items() if k.startswith('phys_') or k.startswith('div_') or k.startswith('conv_'))
    onehot_imp = sum(v for k, v in imp.items() if k.startswith('vgene_') or k.startswith('jgene_'))
    pairing_imp = sum(v for k, v in imp.items() if k.startswith('vj_'))

    print(f"\n  Feature category importance:")
    print(f"    Physicochemical+Diversity: {base_imp:.4f} ({base_imp*100/sum(imp.values()):.1f}%)")
    print(f"    V/J gene one-hot:          {onehot_imp:.4f} ({onehot_imp*100/sum(imp.values()):.1f}%)")
    print(f"    V-J pairing:               {pairing_imp:.4f} ({pairing_imp*100/sum(imp.values()):.1f}%)")

    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 features:")
    for fname, fimp in imp_sorted[:10]:
        cat = '[VJ-pair]' if fname.startswith('vj_') else ('[onehot]' if fname.startswith('vgene_') or fname.startswith('jgene_') else '[base]')
        print(f"    {fname:35s} {fimp:.4f} {cat}")

    # Export per-sample predictions
    pred_rows = []
    for sid, label, prob in cv_result['sample_predictions']:
        pred_rows.append({'sample_id': sid, 'label': label, 'P(disease)': round(prob, 4),
                          'prediction': int(prob >= cv_result['best_threshold'])})
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(OUTPUT_DIR, f"predictions_{name.replace('/', '_')}.csv")
    pred_df.to_csv(pred_path, index=False)

    # Build result dict
    result = {
        'dataset': name,
        'n_samples': len(samples),
        'n_control': n_ctrl,
        'n_case': n_case,
        'n_features': int(X.shape[1]),
        'auc': auc,
        'auc_roc': auc,
        'auc_pr': cv_result['auc_pr'],
        'f1': cv_result['f1'],
        'mcc': cv_result['mcc'],
        'accuracy': cv_result['accuracy'],
        'sensitivity': cv_result['sensitivity'],
        'specificity': cv_result['specificity'],
        'youden_j': cv_result['youden_j'],
        'best_threshold': cv_result['best_threshold'],
        'published_auc': published_auc,
        'published_ref': published_ref,
        'category_importance': {
            'base': round(base_imp, 4),
            'onehot': round(onehot_imp, 4),
            'pairing': round(pairing_imp, 4),
        },
        'fpr': cv_result['fpr'],
        'tpr': cv_result['tpr'],
        'top_features': [(k, round(v, 4)) for k, v in imp_sorted[:10]],
        'model': cv_result['model'],
        'feature_names': cols,
    }

    # Save model checkpoint for Tier 2
    ckpt_path = save_checkpoint(result, name, chain=use_chain)
    result['checkpoint_path'] = ckpt_path
    result['predictions_path'] = pred_path

    print(f"  Predictions exported: {pred_path}")

    return result


def benchmark_vdjdb(name, disease_file, n_per_group=2000):
    """Run VDJdb sequence-level specificity benchmark."""
    print(f"\n{'='*70}")
    print(f"  VDJdb Disease: {name}")

    data = load_vdjdb_disease(disease_file, n_per_group)
    if data is None or len(data) == 0:
        print(f"  No data found for {disease_file}")
        return None

    pos = data[data['label'] == 1]
    neg = data[data['label'] == 0]
    print(f"  Sequences: {len(data)} (Positive: {len(pos)}, Background: {len(neg)})")

    # Compute per-sequence features
    all_feats = []
    for _, row in data.iterrows():
        feats = compute_sequence_features(row['junction_aa'],
                                          row.get('v_call', '?'),
                                          row.get('j_call', '?'))
        feats['label'] = row['label']
        all_feats.append(feats)

    feat_df = pd.DataFrame(all_feats)
    feature_cols = [c for c in feat_df.columns if c != 'label']
    y = feat_df['label'].values
    X = np.nan_to_num(feat_df[feature_cols].values, nan=0.0)

    mask = np.std(X, axis=0) > 0
    X = X[:, mask]
    cols = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]

    cv_result = run_cv_full(X, y, cols, n_folds=5, sample_ids=None)
    print(f"  AUC-ROC (sequence-level): {cv_result['auc_roc']:.4f}")

    return {
        'dataset': f"VDJdb-{name}",
        'n_sequences': len(data),
        'n_positive': len(pos),
        'n_background': len(neg),
        'auc': cv_result['auc_roc'],
        'auc_pr': cv_result['auc_pr'],
        'f1': cv_result['f1'],
        'mcc': cv_result['mcc'],
        'fpr': cv_result['fpr'],
        'tpr': cv_result['tpr'],
    }


def main():
    print("=" * 70)
    print("  CDRscope v2 — Cross-Disease Benchmark Pipeline")
    print("  Testing generalizability across disease fields")
    print("=" * 70)

    all_results = []

    # =====================================================================
    # 1. RA (Aterido 2024)
    # =====================================================================
    print("\n\n" + "#"*70)
    print("#  1. Rheumatoid Arthritis (Aterido et al. 2024)")
    print("#"*70)

    for chain in ['TRA', 'TRB', 'TRA+TRB']:
        if chain == 'TRA+TRB':
            samples = load_ra_joint()
        else:
            samples = load_ra_dataset(chain)
        if len(samples) < 10:
            print(f"  Skipping {chain} (too few samples)")
            continue
        result = benchmark_per_sample(
            f"RA-{chain}", samples,
            published_auc=0.961 if chain == 'TRB' else None,
            published_ref="Aterido 2024 (CDRscope v2.1)" if chain == 'TRB' else "",
            chain=chain
        )
        result['disease'] = 'Rheumatoid Arthritis'
        result['chain'] = chain
        result['data_source'] = 'Aterido et al. Genome Biology 2024'
        all_results.append(result)

    # =====================================================================
    # 2. CMV (Emerson 2017)
    # =====================================================================
    print("\n\n" + "#"*70)
    print("#  2. CMV Serostatus (Emerson et al. 2017)")
    print("#"*70)

    emerson_samples = load_emerson_cmv()
    if len(emerson_samples) >= 10:
        result = benchmark_per_sample(
            "CMV-TRB", emerson_samples,
            published_auc=0.99,
            published_ref="Emerson et al. Nat Genet 2017 (TCR-based CMV prediction)",
            chain='TRB'
        )
        result['disease'] = 'CMV Infection'
        result['chain'] = 'TRB'
        result['data_source'] = 'Emerson et al. Nature Genetics 2017'
        all_results.append(result)

    # =====================================================================
    # 3. SLE (Liu 2019)
    # =====================================================================
    print("\n\n" + "#"*70)
    print("#  3. Systemic Lupus Erythematosus (Liu et al. 2019)")
    print("#"*70)

    sle_samples = load_sle_persample()
    if sle_samples and len(sle_samples) >= 10:
        print("  Using REAL per-sample data from PIRD/CNGB")
        result = benchmark_per_sample(
            "SLE-TRB", sle_samples,
            published_auc=0.99,
            published_ref="Liu et al. Ann Rheum Dis 2019 (V-gene usage, AUROC>0.99)",
            chain='TRB'
        )
        result['disease'] = 'Systemic Lupus Erythematosus'
        result['chain'] = 'TRB'
        result['data_source'] = 'Liu et al. Ann Rheum Dis 2019 (PIRD/CNGB)'
        result['note'] = 'Real per-sample data from PIRD/CNGB'
    else:
        print("  WARNING: Real SLE per-sample data not found, using pseudo-samples")
        sle_samples = load_sle_pseudo(n_samples_per_group=50, seqs_per_sample=500)
        result = benchmark_per_sample(
            "SLE-TRB", sle_samples,
            published_auc=0.99,
            published_ref="Liu et al. Ann Rheum Dis 2019 (V-gene usage, AUROC>0.99)",
            chain='TRB'
        )
        result['disease'] = 'Systemic Lupus Erythematosus'
        result['chain'] = 'TRB'
        result['data_source'] = 'Liu et al. Ann Rheum Dis 2019 (DeepTAPE aggregated)'
        result['note'] = 'Pseudo-samples from aggregated CDR3 pool (AUC inflated)'
    all_results.append(result)

    # =====================================================================
    # 4. Multiple Sclerosis (Alves Sousa 2019)
    # =====================================================================
    print("\n\n" + "#"*70)
    print("#  4. Multiple Sclerosis (Alves Sousa et al. 2019)")
    print("#"*70)

    ms_samples = load_ms_dataset()
    if len(ms_samples) >= 10:
        result = benchmark_per_sample(
            "MS-TRB", ms_samples,
            published_auc=None,
            published_ref="Alves Sousa et al. Sci Rep 2019 (TCR-beta repertoire in MS)",
            chain='TRB'
        )
        result['disease'] = 'Multiple Sclerosis'
        result['chain'] = 'TRB'
        result['data_source'] = 'Alves Sousa et al. Sci Rep 2019 (GEO GSE121082)'
        all_results.append(result)

    # =====================================================================
    # 5. VDJdb Multi-Disease Specificity
    # =====================================================================
    print("\n\n" + "#"*70)
    print("#  5. VDJdb Multi-Disease Specificity Benchmark")
    print("#"*70)

    vdjdb_diseases = [
        ('CMV', 'cmv'),
        ('SARS-CoV-2', 'sars_cov_2'),
        ('HIV', 'hiv'),
        ('EBV', 'ebv'),
        ('InfluenzaA', 'influenzaa'),
        ('HCV', 'hcv'),
        ('M.tuberculosis', 'mtuberculosis'),
        ('YFV', 'yfv'),
    ]

    vdjdb_results = []
    for display_name, file_key in vdjdb_diseases:
        result = benchmark_vdjdb(display_name, file_key)
        if result:
            result['disease'] = f"{display_name} (antigen-specific)"
            result['data_source'] = 'VDJdb'
            vdjdb_results.append(result)
            all_results.append(result)

    # =====================================================================
    # Summary and Visualizations
    # =====================================================================
    print("\n\n" + "#"*70)
    print("#  Summary")
    print("#"*70)

    # Per-sample results table
    print(f"\n  Per-Sample Classification Results:")
    print(f"  {'Dataset':<20} {'N':>5} {'Feat':>5} {'AUC':>7} {'PR':>7} {'F1':>7} {'MCC':>7} {'Sens':>7} {'Spec':>7}")
    print(f"  {'-'*82}")
    for r in all_results:
        if 'n_samples' in r:
            print(f"  {r['dataset']:<20} {r['n_samples']:>5} {r['n_features']:>5} "
                  f"{r['auc']:>7.4f} {r.get('auc_pr', 0):>7.4f} {r.get('f1', 0):>7.4f} "
                  f"{r.get('mcc', 0):>7.4f} {r.get('sensitivity', 0):>7.4f} {r.get('specificity', 0):>7.4f}")

    # VDJdb results table
    print(f"\n  VDJdb Sequence-Level Specificity Results:")
    print(f"  {'Disease':<20} {'N_seqs':>8} {'Positive':>10} {'Background':>10} {'AUC':>8}")
    print(f"  {'-'*56}")
    for r in vdjdb_results:
        print(f"  {r['dataset']:<20} {r['n_sequences']:>8} {r['n_positive']:>10} {r['n_background']:>10} {r['auc']:>8.4f}")

    # Save results JSON (exclude non-serializable: model, feature_names, fpr/tpr)
    serializable_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items()
                  if k not in ('fpr', 'tpr', 'model', 'feature_names', 'sample_predictions')}
        serializable_results.append(r_copy)
    with open(os.path.join(OUTPUT_DIR, 'benchmark_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    # =====================================================================
    # Generate Figures
    # =====================================================================

    # Figure 1: ROC curves for per-sample datasets
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    per_sample = [r for r in all_results if 'n_samples' in r]
    colors = plt.cm.Set1(np.linspace(0, 1, len(per_sample)))
    for r, color in zip(per_sample, colors):
        ax.plot(r['fpr'], r['tpr'], color=color, linewidth=2,
                label=f"{r['dataset']} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Per-Sample Classification: Disease vs Control', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    # Figure 1b: VDJdb ROC curves
    ax = axes[1]
    for r, color in zip(vdjdb_results, colors[:len(vdjdb_results)]):
        ax.plot(r['fpr'], r['tpr'], color=color, linewidth=2,
                label=f"{r['dataset']} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('VDJdb Sequence-Level Specificity', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    plt.suptitle('CDRscope v2: Cross-Disease Benchmark', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Feature category importance across diseases
    per_sample_with_imp = [r for r in all_results if 'category_importance' in r and 'n_samples' in r]
    if per_sample_with_imp:
        fig, ax = plt.subplots(figsize=(12, 6))
        names = [r['dataset'] for r in per_sample_with_imp]
        base_imp = [r['category_importance']['base'] for r in per_sample_with_imp]
        onehot_imp = [r['category_importance']['onehot'] for r in per_sample_with_imp]
        pairing_imp = [r['category_importance']['pairing'] for r in per_sample_with_imp]

        x = np.arange(len(names))
        w = 0.25
        ax.bar(x - w, base_imp, w, label='Physicochemical + Diversity', color='#3D6A8C')
        ax.bar(x, onehot_imp, w, label='V/J Gene One-hot', color='#5B9279')
        ax.bar(x + w, pairing_imp, w, label='V-J Pairing', color='#D97742')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Feature Importance (sum)', fontsize=12)
        ax.set_title('Feature Category Importance Across Diseases', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: AUC comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    all_names = [r['dataset'] for r in all_results]
    all_aucs = [r['auc'] for r in all_results]
    bar_colors = ['#3D6A8C' if 'n_samples' in r else '#D97742' for r in all_results]
    bars = ax.barh(all_names, all_aucs, color=bar_colors, height=0.6)
    for bar, auc in zip(bars, all_aucs):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('CDRscope v2: Cross-Disease AUC Summary', fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_auc_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print(f"    benchmark_results.json")
    print(f"    fig1_roc_curves.png")
    print(f"    fig2_feature_importance.png")
    print(f"    fig3_auc_summary.png")
    print(f"    predictions_*.csv (per-sample P(disease))")
    print(f"    model_checkpoints/ (Tier 2 input: RF + feature schema + gene vocab)")
    print("\nDone!")


if __name__ == '__main__':
    main()
