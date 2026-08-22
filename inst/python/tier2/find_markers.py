#!/usr/bin/env python3
"""
FindMarkers: Identify prototypes that distinguish Control vs Patient
=====================================================================
Seurat's FindMarkers equivalent for TCR quantization space.

1. Differential abundance: Wilcoxon rank-sum + logFC + AUC per prototype
2. Volcano plot
3. Top markers heatmap (Control vs Patient)
4. Prototype annotation: representative CDR3 sequences + V/J gene enrichment
5. FeaturePlot on UMAP
"""
import os, sys, json, time, pickle, warnings, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ranksums
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

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


def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-10)


def main():
    print("=" * 70, flush=True)
    print("  FindMarkers: Differential Prototype Analysis", flush=True)
    print("=" * 70, flush=True)

    # Load data
    count = np.load(os.path.join(BASE, "tcr_reference_panel/ra_count_matrix_m10000.npy"))
    labels = np.load(os.path.join(BASE, "tcr_reference_panel/ra_labels_m10000.npy"))
    X = normalize(count.astype(np.float64), norm='l2', axis=1)
    n_proto = count.shape[1]
    ctrl_mask = labels == 0
    pat_mask = labels == 1
    n_ctrl, n_pat = ctrl_mask.sum(), pat_mask.sum()
    print(f"  Data: {count.shape}, Control={n_ctrl}, Patient={n_pat}", flush=True)

    # ============================================================
    # Step 1: Differential abundance analysis
    # ============================================================
    print("\n[1/5] Differential abundance analysis (10K prototypes)...", flush=True)
    t0 = time.time()

    ctrl_mean = count[ctrl_mask].mean(axis=0)
    pat_mean = count[pat_mask].mean(axis=0)
    ctrl_mean_safe = np.where(ctrl_mean == 0, 1e-6, ctrl_mean)
    pat_mean_safe = np.where(pat_mean == 0, 1e-6, pat_mean)
    logfc = np.log2(pat_mean_safe / ctrl_mean_safe)

    # Wilcoxon rank-sum test + AUC per prototype
    pvals = np.ones(n_proto)
    aucs = np.zeros(n_proto)
    effect_sizes = np.zeros(n_proto)

    # Only test prototypes expressed in >5% of samples (at least 27 samples)
    expressed = np.mean(count > 0, axis=0) >= 0.05
    n_tested = expressed.sum()
    print(f"  Testing {n_tested} prototypes (expressed in >5% samples)...", flush=True)

    for i in np.where(expressed)[0]:
        ctrl_vals = count[ctrl_mask, i]
        pat_vals = count[pat_mask, i]
        try:
            stat, p = mannwhitneyu(pat_vals, ctrl_vals, alternative='two-sided')
            pvals[i] = p
        except:
            pvals[i] = 1.0
        aucs[i] = roc_auc_score(labels, count[:, i])
        effect_sizes[i] = abs(aucs[i] - 0.5) * 2  # 0 to 1 scale

    # FDR correction (Benjamini-Hochberg)
    # Benjamini-Hochberg FDR (manual implementation)
    tested_idx = np.where(expressed)[0]
    tested_pvals = pvals[tested_idx]
    n_tested = len(tested_pvals)
    order = np.argsort(tested_pvals)
    ranked = tested_pvals[order]
    bh_fdr = ranked * n_tested / (np.arange(1, n_tested + 1))
    # Enforce monotonicity from the end
    bh_fdr = np.minimum.accumulate(bh_fdr[::-1])[::-1]
    bh_fdr = np.clip(bh_fdr, 0, 1)
    # Map back
    fdr_full = np.ones(n_proto)
    fdr_full[tested_idx[order]] = bh_fdr

    print(f"  Computed in {time.time()-t0:.1f}s", flush=True)
    sig = (fdr_full < 0.05) & (np.abs(aucs - 0.5) > 0.1)
    print(f"  Significant (FDR<0.05, |AUC-0.5|>0.1): {sig.sum()}", flush=True)

    # Sort by effect size
    markers_df = pd.DataFrame({
        'prototype': np.arange(n_proto),
        'logFC': logfc,
        'p_value': pvals,
        'FDR': fdr_full,
        'AUC': aucs,
        'effect_size': effect_sizes,
        'ctrl_mean': ctrl_mean,
        'pat_mean': pat_mean,
        'expressed_frac': np.mean(count > 0, axis=0),
    })

    # Top markers (up in patient = high AUC, up in control = low AUC)
    # Filter: only prototypes actually expressed in both groups (AUC != 0 and != 1)
    valid_auc = (markers_df['AUC'] > 0.01) & (markers_df['AUC'] < 0.99)
    markers_up_pat = markers_df[valid_auc & (markers_df['AUC'] > 0.5)].nlargest(25, 'AUC')
    markers_up_ctrl = markers_df[valid_auc & (markers_df['AUC'] < 0.5)].nsmallest(25, 'AUC')

    print(f"\n  Top 5 patient-enriched prototypes:", flush=True)
    for _, r in markers_up_pat.head(5).iterrows():
        print(f"    Proto {int(r['prototype'])}: AUC={r['AUC']:.3f}, "
              f"logFC={r['logFC']:.2f}, FDR={r['FDR']:.2e}", flush=True)
    print(f"\n  Top 5 control-enriched prototypes:", flush=True)
    for _, r in markers_up_ctrl.head(5).iterrows():
        print(f"    Proto {int(r['prototype'])}: AUC={r['AUC']:.3f}, "
              f"logFC={r['logFC']:.2f}, FDR={r['FDR']:.2e}", flush=True)

    # ============================================================
    # Step 2: Volcano plot
    # ============================================================
    print("\n[2/5] Volcano plot...", flush=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sig_up = (markers_df['AUC'] > 0.6) & (markers_df['FDR'] < 0.05)
    sig_down = (markers_df['AUC'] < 0.4) & (markers_df['FDR'] < 0.05)
    ns = ~(sig_up | sig_down)

    ax.scatter(markers_df.loc[ns, 'logFC'], -np.log10(markers_df.loc[ns, 'FDR'].clip(1e-300)),
              c='gray', s=5, alpha=0.3, label='Not significant')
    ax.scatter(markers_df.loc[sig_up, 'logFC'], -np.log10(markers_df.loc[sig_up, 'FDR'].clip(1e-300)),
              c=PAT, s=15, alpha=0.7, label=f'Patient-enriched (n={sig_up.sum()})')
    ax.scatter(markers_df.loc[sig_down, 'logFC'], -np.log10(markers_df.loc[sig_down, 'FDR'].clip(1e-300)),
              c=CTRL, s=15, alpha=0.7, label=f'Control-enriched (n={sig_down.sum()})')

    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('log2 Fold Change (Patient / Control)')
    ax.set_ylabel('-log10(FDR)')
    ax.set_title('Volcano Plot: Differential Prototype Abundance')
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_volcano.png'), bbox_inches='tight')
    plt.close()
    print("  Done", flush=True)

    # ============================================================
    # Step 3: Top markers heatmap
    # ============================================================
    print("[3/5] Top markers heatmap...", flush=True)
    top_n = 40
    top_idx = list(markers_up_pat.head(top_n // 2)['prototype'].values) + \
              list(markers_up_ctrl.head(top_n // 2)['prototype'].values)

    # Z-score normalize each prototype across samples
    top_data = X[:, top_idx]
    top_z = (top_data - top_data.mean(axis=0)) / (top_data.std(axis=0) + 1e-10)

    # Sort samples by label
    sort_idx = np.argsort(labels)
    top_z_sorted = top_z[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(top_z_sorted.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

    # Add label colorbar
    ax.axvline(x=n_ctrl, color='black', linewidth=2, linestyle='--')
    ax.text(n_ctrl * 0.5, -2.5, 'Control', ha='center', fontsize=10, fontweight='bold', color=CTRL)
    ax.text(n_ctrl + n_pat * 0.5, -2.5, 'Patient', ha='center', fontsize=10, fontweight='bold', color=PAT)

    ax.set_xlabel('Samples (sorted by label)')
    ax.set_ylabel('Prototypes')
    ax.set_title(f'Top {top_n} Differential Prototypes (Z-scored)')

    # Y-axis labels
    y_labels = []
    for i, p in enumerate(top_idx):
        auc = markers_df.iloc[p]['AUC']
        label = f'P{p} (AUC={auc:.2f})'
        y_labels.append(label)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xticks([])

    cbar = plt.colorbar(im, ax=ax, label='Z-score', fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("  Done", flush=True)

    # ============================================================
    # Step 4: Prototype annotation (representative sequences + V/J genes)
    # ============================================================
    print("[4/5] Prototype annotation...", flush=True)

    # Load reference panel to get sequences per prototype
    print("  Loading reference panel...", flush=True)
    with open(os.path.join(BASE, "tcr_reference_panel/reference_panel_m10000.pkl"), 'rb') as f:
        panel = pickle.load(f)
    centroids = panel['centroids']
    ref_seqs = panel['sequences']
    ref_emb = panel['embeddings']

    # Compute assignments for reference sequences
    print("  Computing prototype assignments...", flush=True)
    from sklearn.metrics.pairwise import paired_distances
    proto_map = {}
    batch_size = 5000
    for i in range(0, len(ref_seqs), batch_size):
        batch = ref_emb[i:i+batch_size]
        dists = np.dot(batch, centroids.T)  # cosine similarity (faster)
        assignments = np.argmax(dists, axis=1)
        for j, seq in enumerate(ref_seqs[i:i+batch_size]):
            p = int(assignments[j])
            if p not in proto_map:
                proto_map[p] = []
            proto_map[p].append(seq)

    # Load RA data to get V/J genes for sequences
    print("  Loading RA data for V/J gene lookup...", flush=True)
    sys.path.insert(0, BASE)
    import cross_disease_benchmark as cdb
    ra_samples = cdb.load_ra_dataset('TRB')

    # Build sequence -> V/J gene mapping
    seq_to_vgene = {}
    seq_to_jgene = {}
    for s in ra_samples:
        df = s['df']
        if 'v_call' in df.columns:
            for _, row in df.iterrows():
                seq = str(row.get('junction_aa', ''))
                v = str(row.get('v_call', ''))
                j = str(row.get('j_call', ''))
                if seq and v and v != 'nan' and v != 'None':
                    seq_to_vgene[seq] = v
                if seq and j and j != 'nan' and j != 'None':
                    seq_to_jgene[seq] = j

    print(f"  V/J gene mapping: {len(seq_to_vgene)} sequences", flush=True)

    # Annotate top markers
    annotations = []
    all_top = list(markers_up_pat.head(15)['prototype'].values) + \
              list(markers_up_ctrl.head(15)['prototype'].values)

    for proto_idx in all_top:
        seqs = proto_map.get(proto_idx, [])
        if not seqs:
            continue

        # Find V/J genes for sequences in this prototype
        v_genes = [seq_to_vgene[s] for s in seqs if s in seq_to_vgene]
        j_genes = [seq_to_jgene[s] for s in seqs if s in seq_to_jgene]

        from collections import Counter
        v_counter = Counter(v_genes)
        j_counter = Counter(j_genes)

        # Top V/J genes
        top_v = v_counter.most_common(3)
        top_j = j_counter.most_common(3)

        # Representative sequences (first few)
        rep_seqs = seqs[:5]

        # Sequence length stats
        seq_lens = [len(s) for s in seqs]

        # Physicochemical properties of representative sequences
        aa_props = {
            'hydrophobic': 'AVILMFWYC',
            'positive': 'KRH',
            'negative': 'DE',
            'aromatic': 'FWY',
            'glycine': 'G',
            'proline': 'P',
            'cysteine': 'C',
        }

        auc = markers_df.iloc[proto_idx]['AUC']
        logfc_val = markers_df.iloc[proto_idx]['logFC']
        fdr_val = markers_df.iloc[proto_idx]['FDR']

        ann = {
            'prototype': int(proto_idx),
            'auc': round(float(auc), 4),
            'logFC': round(float(logfc_val), 3),
            'FDR': float(fdr_val),
            'n_sequences': len(seqs),
            'mean_seq_len': round(float(np.mean(seq_lens)), 1) if seq_lens else 0,
            'top_v_genes': [(g, c) for g, c in top_v],
            'top_j_genes': [(g, c) for g, c in top_j],
            'rep_sequences': rep_seqs,
            'direction': 'Patient' if auc > 0.5 else 'Control',
        }
        annotations.append(ann)

        print(f"    Proto {proto_idx}: AUC={auc:.3f}, n_seqs={len(seqs)}, "
              f"top V={top_v[0][0] if top_v else 'N/A'}, "
              f"top J={top_j[0][0] if top_j else 'N/A'}", flush=True)

    # ============================================================
    # Step 5: FeaturePlot on UMAP
    # ============================================================
    print("[5/5] FeaturePlot on UMAP...", flush=True)
    coords = pd.read_csv(os.path.join(OUTPUT_DIR, 'unified_coordinates.csv'))

    # Select 6 top markers for FeaturePlot
    plot_protos = list(markers_up_pat.head(3)['prototype'].values) + \
                  list(markers_up_ctrl.head(3)['prototype'].values)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, proto_idx in enumerate(plot_protos):
        ax = axes[i // 3, i % 3]
        values = X[:, proto_idx]
        sc = ax.scatter(coords['UMAP1'], coords['UMAP2'], c=values, s=15, alpha=0.8,
                       edgecolors='white', linewidth=0.2, cmap='viridis')
        auc = markers_df.iloc[proto_idx]['AUC']
        direction = 'Patient' if auc > 0.5 else 'Control'
        ax.set_title(f'Proto {proto_idx} (AUC={auc:.2f}, {direction})', fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

    plt.suptitle('FeaturePlot: Top Differential Prototypes on UMAP', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_featureplot.png'), bbox_inches='tight')
    plt.close()
    print("  Done", flush=True)

    # ============================================================
    # Dot plot: V/J gene enrichment in top markers
    # ============================================================
    print("  Generating V/J gene dot plot...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Collect V/J gene data for all top markers
    v_gene_data = {}
    for ann in annotations:
        for gene, count in ann['top_v_genes']:
            if gene not in v_gene_data:
                v_gene_data[gene] = {}
            v_gene_data[gene][ann['prototype']] = count

    # Only show genes appearing in multiple prototypes
    v_gene_filtered = {g: d for g, d in v_gene_data.items() if len(d) >= 2}
    if v_gene_filtered:
        genes = sorted(v_gene_filtered.keys())[:15]
        protos = [a['prototype'] for a in annotations]

        dot_x = []
        dot_y = []
        dot_size = []
        dot_color = []
        for gi, gene in enumerate(genes):
            for pi, proto in enumerate(protos):
                count = v_gene_filtered.get(gene, {}).get(proto, 0)
                if count > 0:
                    dot_x.append(pi)
                    dot_y.append(gi)
                    dot_size.append(count * 3)
                    auc = markers_df.iloc[proto]['AUC']
                    dot_color.append(auc)

        sc = ax.scatter(dot_x, dot_y, c=dot_color, s=dot_size, alpha=0.7, cmap='RdBu_r',
                       edgecolors='white', linewidth=0.3, vmin=0.3, vmax=0.7)
        ax.set_xticks(range(len(protos)))
        ax.set_xticklabels([f'P{p}' for p in protos], fontsize=7, rotation=45)
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes, fontsize=8)
        ax.set_xlabel('Prototypes')
        ax.set_ylabel('V Gene')
        ax.set_title('V Gene Enrichment in Top Marker Prototypes')
        cbar = plt.colorbar(sc, ax=ax, label='AUC (Patient→1, Control→0)', fraction=0.03)
    else:
        ax.text(0.5, 0.5, 'Insufficient V/J gene overlap', transform=ax.transAxes, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig_dotplot_vgene.png'), bbox_inches='tight')
    plt.close()
    print("  Done", flush=True)

    # ============================================================
    # Save results
    # ============================================================
    markers_path = os.path.join(OUTPUT_DIR, 'find_markers_results.json')
    with open(markers_path, 'w') as f:
        json.dump({
            'n_total_prototypes': n_proto,
            'n_tested': int(n_tested),
            'n_significant': int(sig.sum()),
            'n_patient_enriched': int(sig_up.sum()),
            'n_control_enriched': int(sig_down.sum()),
            'annotations': annotations,
            'top_patient_markers': markers_up_pat.head(10).to_dict('records'),
            'top_control_markers': markers_up_ctrl.head(10).to_dict('records'),
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {markers_path}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
