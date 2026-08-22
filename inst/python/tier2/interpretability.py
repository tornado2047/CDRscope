#!/usr/bin/env python3
"""
Multi-layer Interpretability Analysis
======================================
Instead of Seurat's FindMarkers (univariate), we use the SVM classifier's
weight vector as the ground truth for feature importance, then annotate
through multiple biological layers:

1. SVM Weight Ranking — which prototypes matter most for classification
2. V/J Gene Enrichment — aggregate V/J gene usage in top-weighted prototypes
3. CDR3 Motif Analysis — shared k-mer patterns in top prototypes
4. Physicochemical Profiling — biophysical properties of CDR3 sequences
5. Convergence Score — how many V/J combinations converge to same prototype
6. VDJdb Cross-reference — match against known disease-associated TCRs
"""
import os, sys, json, time, pickle, warnings, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

BASE = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
OUTPUT_DIR = os.path.join(BASE, "seurat_analysis")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

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
ORANGE = '#ff9f0a'

# Amino acid properties
AA_HYDROPHOBIC = set('AVILMFWYC')
AA_POSITIVE = set('KRH')
AA_NEGATIVE = set('DE')
AA_AROMATIC = set('FWYH')
AA_POLAR = set('STNQ')
AA_SMALL = set('AGCS')
AA_TINY = set('AGS')
AA_WEIGHT = {'A':89,'R':174,'N':132,'D':133,'C':121,'E':147,'Q':146,'G':75,'H':155,
             'I':131,'L':131,'K':146,'M':149,'F':165,'P':115,'S':105,'T':119,
             'W':204,'Y':181,'V':117}


def main():
    print("=" * 70, flush=True)
    print("  Multi-layer Interpretability Analysis", flush=True)
    print("=" * 70, flush=True)

    # Load data
    count = np.load(os.path.join(BASE, "tcr_reference_panel/ra_count_matrix_m10000.npy"))
    labels = np.load(os.path.join(BASE, "tcr_reference_panel/ra_labels_m10000.npy"))
    X = normalize(count.astype(np.float64), norm='l2', axis=1)
    n_proto = count.shape[1]
    ctrl_mask = labels == 0
    pat_mask = labels == 1

    # ============================================================
    # 1. SVM Weight Analysis
    # ============================================================
    print("\n[1/6] SVM weight analysis (5-fold CV)...", flush=True)
    # 5-fold CV to get stable weights
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    weights = np.zeros((5, n_proto))
    for fold, (train_idx, _) in enumerate(skf.split(X, labels)):
        svm = LinearSVC(C=0.1, random_state=42, max_iter=5000, dual=False)
        svm.fit(X[train_idx], labels[train_idx])
        weights[fold] = svm.coef_[0]

    mean_weights = weights.mean(axis=0)
    weight_std = weights.std(axis=0)
    weight_t = mean_weights / (weight_std + 1e-10)  # t-like statistic

    # Rank by absolute weight (importance)
    importance = np.abs(mean_weights)
    ranked = np.argsort(importance)[::-1]

    # Top positive (patient) and negative (control) weights
    top_pat_idx = np.argsort(mean_weights)[-50:][::-1]  # top 50 patient
    top_ctrl_idx = np.argsort(mean_weights)[:50]  # top 50 control

    print(f"  Weight range: [{mean_weights.min():.4f}, {mean_weights.max():.4f}]", flush=True)
    print(f"  Top patient weight: P{top_pat_idx[0]} = {mean_weights[top_pat_idx[0]]:.4f}", flush=True)
    print(f"  Top control weight: P{top_ctrl_idx[0]} = {mean_weights[top_ctrl_idx[0]]:.4f}", flush=True)

    # Cumulative importance
    sorted_imp = importance[ranked]
    cum_imp = np.cumsum(sorted_imp) / sorted_imp.sum()
    n_for_50 = np.searchsorted(cum_imp, 0.5) + 1
    n_for_80 = np.searchsorted(cum_imp, 0.8) + 1
    n_for_90 = np.searchsorted(cum_imp, 0.9) + 1
    print(f"  Prototypes for 50% cumulative importance: {n_for_50}", flush=True)
    print(f"  Prototypes for 80% cumulative importance: {n_for_80}", flush=True)
    print(f"  Prototypes for 90% cumulative importance: {n_for_90}", flush=True)

    # Plot: weight distribution + cumulative importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(mean_weights, bins=100, color=ACCENT, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('SVM Weight')
    ax.set_ylabel('Prototype Count')
    ax.set_title('A. SVM Weight Distribution')
    ax.annotate(f'Top patient (n=50)', xy=(mean_weights[top_pat_idx[0]], 50),
                xytext=(mean_weights[top_pat_idx[0]]*0.5, 300),
                arrowprops=dict(arrowstyle='->', color=PAT), fontsize=9, color=PAT)
    ax.annotate(f'Top control (n=50)', xy=(mean_weights[top_ctrl_idx[0]], 50),
                xytext=(mean_weights[top_ctrl_idx[0]]*0.5, 300),
                arrowprops=dict(arrowstyle='->', color=CTRL), fontsize=9, color=CTRL)

    ax = axes[1]
    ax.plot(range(1, 201), cum_imp[:200], color=ACCENT, linewidth=2)
    ax.axhline(y=0.5, color=ORANGE, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.8, color=GREEN, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.9, color=PAT, linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=n_for_50, color=ORANGE, linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=n_for_80, color=GREEN, linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Number of Top Prototypes')
    ax.set_ylabel('Cumulative Importance (%)')
    ax.set_title('B. Cumulative Feature Importance')
    ax.set_ylim(0, 1.05)
    ax.annotate(f'{n_for_50}→50%', xy=(n_for_50, 0.5), fontsize=9, color=ORANGE,
                xytext=(n_for_50+10, 0.4), arrowprops=dict(arrowstyle='->', color=ORANGE))
    ax.annotate(f'{n_for_80}→80%', xy=(n_for_80, 0.8), fontsize=9, color=GREEN,
                xytext=(n_for_80+10, 0.7), arrowprops=dict(arrowstyle='->', color=GREEN))

    plt.suptitle('SVM Classifier Feature Importance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_svm_weights.png'), bbox_inches='tight')
    plt.close()
    print("  Plot saved", flush=True)

    # ============================================================
    # 2. Load reference panel + build prototype→sequences mapping
    # ============================================================
    print("\n[2/6] Loading reference panel and prototype annotations...", flush=True)
    with open(os.path.join(BASE, "tcr_reference_panel/reference_panel_m10000.pkl"), 'rb') as f:
        panel = pickle.load(f)
    centroids = panel['centroids']
    ref_seqs = panel['sequences']
    ref_emb = panel['embeddings']

    # Assign sequences to prototypes
    print("  Computing prototype assignments...", flush=True)
    proto_map = defaultdict(list)
    batch_size = 5000
    for i in range(0, len(ref_seqs), batch_size):
        batch = ref_emb[i:i+batch_size]
        sims = np.dot(batch, centroids.T)
        assignments = np.argmax(sims, axis=1)
        for j, seq in enumerate(ref_seqs[i:i+batch_size]):
            proto_map[int(assignments[j])].append(seq)

    # Load RA data for V/J gene mapping
    print("  Loading RA data for V/J genes...", flush=True)
    sys.path.insert(0, BASE)
    import cross_disease_benchmark as cdb
    ra_samples = cdb.load_ra_dataset('TRB')

    # Build sequence → (V_gene, J_gene, count) mapping
    seq_vj = {}
    for s in ra_samples:
        df = s['df']
        if 'v_call' in df.columns:
            for _, row in df.iterrows():
                seq = str(row.get('junction_aa', ''))
                v = str(row.get('v_call', ''))
                j = str(row.get('j_call', ''))
                c = int(row.get('duplicate_count', 1))
                if seq and v and v != 'nan' and v != 'None':
                    if seq not in seq_vj:
                        seq_vj[seq] = []
                    seq_vj[seq].append((v, j, c))
    print(f"  V/J gene mapping: {len(seq_vj)} unique sequences", flush=True)

    # ============================================================
    # 3. V/J Gene Enrichment in Top-Weighted Prototypes
    # ============================================================
    print("\n[3/6] V/J gene enrichment analysis...", flush=True)

    def get_vj_for_protos(proto_indices):
        v_genes = []
        j_genes = []
        for p in proto_indices:
            seqs = proto_map.get(p, [])
            for s in seqs:
                if s in seq_vj:
                    for v, j, c in seq_vj[s]:
                        v_genes.extend([v] * min(c, 10))
                        j_genes.extend([j] * min(c, 10))
        return Counter(v_genes), Counter(j_genes)

    # Top 100 patient-weighted and control-weighted prototypes
    top_pat_100 = ranked[:100][mean_weights[ranked[:100]] > 0]
    top_ctrl_100 = ranked[:100][mean_weights[ranked[:100]] < 0]
    # Use all top 50 positive and top 50 negative
    top_pat_50 = top_pat_idx
    top_ctrl_50 = top_ctrl_idx

    pat_v, pat_j = get_vj_for_protos(top_pat_50)
    ctrl_v, ctrl_j = get_vj_for_protos(top_ctrl_50)

    # Compute enrichment (patient vs control)
    all_v = set(list(pat_v.keys()) + list(ctrl_v.keys()))
    total_pat = sum(pat_v.values()) + 1
    total_ctrl = sum(ctrl_v.values()) + 1

    v_enrichment = []
    for gene in all_v:
        pat_freq = pat_v.get(gene, 0) / total_pat
        ctrl_freq = ctrl_v.get(gene, 0) / total_ctrl
        log_enrich = np.log2((pat_freq + 1e-6) / (ctrl_freq + 1e-6))
        v_enrichment.append({'gene': gene, 'pat_count': pat_v.get(gene, 0),
                           'ctrl_count': ctrl_v.get(gene, 0),
                           'pat_freq': pat_freq, 'ctrl_freq': ctrl_freq,
                           'log2_enrichment': log_enrich})
    v_enr_df = pd.DataFrame(v_enrichment).sort_values('log2_enrichment', ascending=False)

    print(f"  V genes in patient-prototypes: {len(pat_v)}, control-prototypes: {len(ctrl_v)}", flush=True)
    print(f"  Top 5 patient-enriched V genes:", flush=True)
    for _, r in v_enr_df.head(5).iterrows():
        print(f"    {r['gene']}: log2FC={r['log2_enrichment']:.2f} "
              f"(pat={r['pat_count']}, ctrl={r['ctrl_count']})", flush=True)

    # Plot V gene enrichment
    top_v_show = pd.concat([v_enr_df.head(15), v_enr_df.tail(15)])
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PAT if v > 0 else CTRL for v in top_v_show['log2_enrichment']]
    ax.barh(range(len(top_v_show)), top_v_show['log2_enrichment'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_v_show)))
    ax.set_yticklabels(top_v_show['gene'], fontsize=8)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('log2 Enrichment (Patient / Control)')
    ax.set_title('V Gene Enrichment in Top-Weighted Prototypes')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_vgene_enrichment.png'), bbox_inches='tight')
    plt.close()
    print("  Plot saved", flush=True)

    # ============================================================
    # 4. CDR3 Motif Analysis (k-mer)
    # ============================================================
    print("\n[4/6] CDR3 motif analysis...", flush=True)

    def extract_kmers(seqs, k=4):
        kmers = Counter()
        for s in seqs:
            for i in range(len(s) - k + 1):
                kmers[s[i:i+k]] += 1
        return kmers

    # Get sequences from top patient vs control prototypes
    pat_seqs = []
    for p in top_pat_50:
        pat_seqs.extend(proto_map.get(p, []))
    ctrl_seqs = []
    for p in top_ctrl_50:
        ctrl_seqs.extend(proto_map.get(p, []))

    print(f"  Patient prototype sequences: {len(pat_seqs)}", flush=True)
    print(f"  Control prototype sequences: {len(ctrl_seqs)}", flush=True)

    pat_kmers = extract_kmers(pat_seqs, k=4)
    ctrl_kmers = extract_kmers(ctrl_seqs, k=4)

    # Find enriched motifs
    total_pat_k = sum(pat_kmers.values()) + 1
    total_ctrl_k = sum(ctrl_kmers.values()) + 1
    motif_enrichment = []
    for kmer, count in pat_kmers.items():
        ctrl_count = ctrl_kmers.get(kmer, 0)
        pat_freq = count / total_pat_k
        ctrl_freq = ctrl_count / total_ctrl_k
        log_enrich = np.log2((pat_freq + 1e-8) / (ctrl_freq + 1e-8))
        if count >= 5:
            motif_enrichment.append({'motif': kmer, 'pat_count': count,
                                   'ctrl_count': ctrl_count,
                                   'log2_enrichment': log_enrich})
    motif_df = pd.DataFrame(motif_enrichment).sort_values('log2_enrichment', ascending=False)

    print(f"  Top 5 patient-enriched motifs (k=4):", flush=True)
    for _, r in motif_df.head(5).iterrows():
        print(f"    {r['motif']}: log2FC={r['log2_enrichment']:.2f} "
              f"(pat={r['pat_count']}, ctrl={r['ctrl_count']})", flush=True)

    # Plot motif enrichment
    top_motifs = pd.concat([motif_df.head(20), motif_df.tail(20)])
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PAT if v > 0 else CTRL for v in top_motifs['log2_enrichment']]
    ax.barh(range(len(top_motifs)), top_motifs['log2_enrichment'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_motifs)))
    ax.set_yticklabels(top_motifs['motif'], fontsize=8, fontfamily='monospace')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('log2 Enrichment (Patient / Control)')
    ax.set_title('CDR3 4-mer Motif Enrichment in Top-Weighted Prototypes')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_motif_enrichment.png'), bbox_inches='tight')
    plt.close()
    print("  Plot saved", flush=True)

    # ============================================================
    # 5. Physicochemical Profiling
    # ============================================================
    print("\n[5/6] Physicochemical profiling...", flush=True)

    def compute_physchem(seqs):
        results = []
        for s in seqs:
            n = len(s)
            if n == 0:
                continue
            charge = sum(1 for a in s if a in AA_POSITIVE) - sum(1 for a in s if a in AA_NEGATIVE)
            hyd = sum(1 for a in s if a in AA_HYDROPHOBIC) / n
            aro = sum(1 for a in s if a in AA_AROMATIC) / n
            mw = sum(AA_WEIGHT.get(a, 100) for a in s) / n
            gly = sum(1 for a in s if a == 'G') / n
            pro = sum(1 for a in s if a == 'P') / n
            cys = sum(1 for a in s if a == 'C') / n
            results.append({
                'length': n, 'charge': charge, 'charge_density': charge / n,
                'hydrophobicity': hyd, 'aromaticity': aro,
                'avg_mw': mw, 'glycine': gly, 'proline': pro, 'cysteine': cys
            })
        return pd.DataFrame(results)

    pat_pc = compute_physchem(pat_seqs[:5000])
    ctrl_pc = compute_physchem(ctrl_seqs[:5000])

    # Plot physicochemical comparison
    props = ['length', 'charge', 'hydrophobicity', 'aromaticity', 'glycine', 'proline']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, prop in enumerate(props):
        ax = axes[i // 3, i % 3]
        data = [ctrl_pc[prop].dropna(), pat_pc[prop].dropna()]
        bp = ax.boxplot(data, labels=['Control', 'Patient'], patch_artist=True,
                       widths=0.5, showfliers=False)
        bp['boxes'][0].set_facecolor(CTRL)
        bp['boxes'][1].set_facecolor(PAT)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_alpha(0.6)
        ax.set_title(prop, fontweight='bold')

        # Statistical test
        from scipy.stats import mannwhitneyu
        try:
            _, p = mannwhitneyu(ctrl_pc[prop].dropna(), pat_pc[prop].dropna())
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax.text(0.5, 0.95, f'p={p:.2e} {sig}', transform=ax.transAxes,
                   ha='center', va='top', fontsize=8)
        except:
            pass

    plt.suptitle('CDR3 Physicochemical Properties: Patient vs Control Prototypes',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_physchem.png'), bbox_inches='tight')
    plt.close()
    print("  Plot saved", flush=True)

    # ============================================================
    # 6. Convergence Analysis
    # ============================================================
    print("\n[6/6] Convergence analysis...", flush=True)

    # For each top prototype, count how many unique V/J combinations produce
    # sequences that map to it
    convergence_data = []
    for proto in list(top_pat_50[:20]) + list(top_ctrl_50[:20]):
        seqs = proto_map.get(proto, [])
        vj_pairs = set()
        v_genes_p = set()
        n_seqs = len(seqs)
        for s in seqs:
            if s in seq_vj:
                for v, j, c in seq_vj[s]:
                    vj_pairs.add((v, j))
                    v_genes_p.add(v)
        convergence_score = len(vj_pairs) / max(n_seqs, 1) if n_seqs > 0 else 0
        w = mean_weights[proto]
        convergence_data.append({
            'prototype': int(proto),
            'n_seqs': n_seqs,
            'n_vj_pairs': len(vj_pairs),
            'n_v_genes': len(v_genes_p),
            'convergence_score': round(convergence_score, 4),
            'svm_weight': round(float(w), 6),
            'direction': 'Patient' if w > 0 else 'Control',
        })

    conv_df = pd.DataFrame(convergence_data)
    print(f"  Analyzed {len(conv_df)} top prototypes", flush=True)
    print(f"  Mean convergence score: {conv_df['convergence_score'].mean():.3f}", flush=True)
    print(f"  Patient protos mean V genes: {conv_df[conv_df['direction']=='Patient']['n_v_genes'].mean():.1f}", flush=True)
    print(f"  Control protos mean V genes: {conv_df[conv_df['direction']=='Control']['n_v_genes'].mean():.1f}", flush=True)

    # Plot convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    pat_conv = conv_df[conv_df['direction'] == 'Patient']
    ctrl_conv = conv_df[conv_df['direction'] == 'Control']
    ax.scatter(pat_conv['n_v_genes'], pat_conv['convergence_score'],
              c=PAT, s=50, alpha=0.7, edgecolors='white', linewidth=0.3, label='Patient-enriched')
    ax.scatter(ctrl_conv['n_v_genes'], ctrl_conv['convergence_score'],
              c=CTRL, s=50, alpha=0.7, edgecolors='white', linewidth=0.3, label='Control-enriched')
    ax.set_xlabel('Number of Unique V Genes per Prototype')
    ax.set_ylabel('Convergence Score (V/J pairs / sequences)')
    ax.set_title('TCR Convergence in Top-Weighted Prototypes')
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_convergence.png'), bbox_inches='tight')
    plt.close()
    print("  Plot saved", flush=True)

    # ============================================================
    # 7. VDJdb Cross-reference
    # ============================================================
    print("\n[7/6] VDJdb cross-reference...", flush=True)
    vdjdb_path = os.path.join(BASE, "tcr_reference_panel/vdjdb_seqs.txt")
    if os.path.exists(vdjdb_path):
        with open(vdjdb_path) as f:
            vdjdb_seqs = set(line.strip() for line in f if line.strip())
        print(f"  VDJdb sequences: {len(vdjdb_seqs)}", flush=True)

        # Check how many sequences in top prototypes match VDJdb
        pat_matches = set(pat_seqs) & vdjdb_seqs
        ctrl_matches = set(ctrl_seqs) & vdjdb_seqs
        print(f"  Patient prototype matches: {len(pat_matches)}", flush=True)
        print(f"  Control prototype matches: {len(ctrl_matches)}", flush=True)

        # Per-prototype match rate
        proto_match_data = []
        for proto in list(top_pat_50[:20]) + list(top_ctrl_50[:20]):
            seqs = proto_map.get(proto, [])
            matches = set(seqs) & vdjdb_seqs
            match_rate = len(matches) / max(len(seqs), 1)
            w = mean_weights[proto]
            proto_match_data.append({
                'prototype': int(proto),
                'n_seqs': len(seqs),
                'n_matches': len(matches),
                'match_rate': round(match_rate, 4),
                'svm_weight': round(float(w), 6),
                'direction': 'Patient' if w > 0 else 'Control',
                'matched_seqs': list(matches)[:5],
            })
        match_df = pd.DataFrame(proto_match_data)
        print(f"  Mean match rate - Patient: {match_df[match_df['direction']=='Patient']['match_rate'].mean():.3f}", flush=True)
        print(f"  Mean match rate - Control: {match_df[match_df['direction']=='Control']['match_rate'].mean():.3f}", flush=True)
    else:
        print("  VDJdb data not found, skipping", flush=True)
        match_df = pd.DataFrame()
        pat_matches = set()

    # ============================================================
    # Save comprehensive results
    # ============================================================
    print("\n  Saving results...", flush=True)

    # Build top prototype annotation table (top 20 patient + 20 control)
    ann_list = []
    for proto in list(top_pat_50[:20]) + list(top_ctrl_50[:20]):
        seqs = proto_map.get(proto, [])
        vj = []
        for s in seqs:
            if s in seq_vj:
                vj.extend([(v, j) for v, j, _ in seq_vj[s]])
        v_counter = Counter([v for v, _ in vj])
        j_counter = Counter([j for _, j in vj])
        rep_seqs = seqs[:3] if len(seqs) >= 3 else seqs

        w = mean_weights[proto]
        ann_list.append({
            'prototype': int(proto),
            'svm_weight': round(float(w), 6),
            'direction': 'Patient' if w > 0 else 'Control',
            'n_sequences': len(seqs),
            'top_v_genes': v_counter.most_common(3),
            'top_j_genes': j_counter.most_common(3),
            'rep_cdr3': rep_seqs,
        })

    results_out = {
        'n_prototypes': n_proto,
        'n_for_50pct': int(n_for_50),
        'n_for_80pct': int(n_for_80),
        'n_for_90pct': int(n_for_90),
        'top_v_enrichment': v_enr_df.head(10).to_dict('records'),
        'bottom_v_enrichment': v_enr_df.tail(10).to_dict('records'),
        'top_motifs': motif_df.head(10).to_dict('records'),
        'bottom_motifs': motif_df.tail(10).to_dict('records'),
        'convergence': conv_df.to_dict('records'),
        'vdjdb_matches': match_df.to_dict('records') if len(match_df) > 0 else [],
        'prototype_annotations': ann_list,
    }
    results_path = os.path.join(OUTPUT_DIR, 'interpretability_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"  Results saved: {results_path}", flush=True)
    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
