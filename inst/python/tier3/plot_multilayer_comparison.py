#!/usr/bin/env python3
"""
Multi-Layer Spectrum Performance Comparison Figure
===================================================
Generates a publication-quality figure comparing:
  - Supervised: L1-only vs multi-layer (AUC + Accuracy)
  - Unsupervised: L1-only vs layer combinations (5 methods)
  - Per-layer ablation (standalone vs stacked)
"""
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

BASE = os.path.expanduser("~/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8")
SUP_DIR = os.path.join(BASE, "tier3_e2e_results")
UNSUP_DIR = os.path.join(BASE, "tier3_unsup_results")
OUT_DIR = os.path.join(BASE, "tier3_e2e_results")

# Colors (matching paper-figure style)
BG = '#FFFFFF'
GRID = '#E8E8E8'
TEXT = '#1a1a1a'
ACCENT_BLUE = '#2563EB'
ACCENT_ORANGE = '#EA580C'
ACCENT_GREEN = '#059669'
ACCENT_RED = '#DC2626'
ACCENT_PURPLE = '#7C3AED'
ACCENT_GRAY = '#6B7280'
LIGHT_BLUE = '#DBEAFE'
LIGHT_ORANGE = '#FFEDD5'

plt.rcParams.update({
    'font.family': 'Helvetica Neue, Arial, sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'savefig.facecolor': BG,
    'savefig.dpi': 300,
    'figure.dpi': 150,
})


def load_results():
    with open(os.path.join(SUP_DIR, 'e2e_results.json')) as f:
        sup = json.load(f)
    with open(os.path.join(UNSUP_DIR, 'unsup_multilayer_results.json')) as f:
        unsup = json.load(f)
    return sup, unsup


def main():
    sup, unsup = load_results()

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.1, 1.0, 1.1],
                          hspace=0.38, wspace=0.35,
                          left=0.05, right=0.975, top=0.92, bottom=0.07)

    # ── Panel a: Supervised AUC comparison ──
    ax1 = fig.add_subplot(gs[0, 0])
    sup_r = sup['classification_results']
    configs_sup = ['L1_only (v2)', 'L2_only', 'L4_only', 'S_only',
                   'L1+L2', 'L1+L4', 'L1+S', 'Multi-layer (v3)']
    labels_sup = ['L1', 'L2', 'L4', 'S', 'L1+L2', 'L1+L4', 'L1+S', 'L1+L2+L4+S']
    aucs_sup = [sup_r[c]['auc_mean'] for c in configs_sup]
    auc_errs = [sup_r[c]['auc_std'] for c in configs_sup]

    colors_sup = [ACCENT_BLUE, ACCENT_GRAY, ACCENT_GRAY, ACCENT_GRAY,
                  ACCENT_GREEN, ACCENT_GREEN, ACCENT_GREEN, ACCENT_ORANGE]
    bars = ax1.barh(range(len(labels_sup)), aucs_sup, xerr=auc_errs,
                     color=colors_sup, edgecolor='white', height=0.65,
                     error_kw={'linewidth': 0.8, 'capsize': 2, 'alpha': 0.5})
    ax1.set_yticks(range(len(labels_sup)))
    ax1.set_yticklabels(labels_sup, fontsize=8.5)
    ax1.set_xlim(0.5, 1.0)
    ax1.set_xlabel('AUC (5-fold CV)', fontsize=9)
    ax1.set_title('a | Supervised classification', fontsize=10, weight='bold', loc='left')
    ax1.axvline(x=aucs_sup[0], color=ACCENT_BLUE, linestyle='--', linewidth=0.8, alpha=0.4)
    for i, (v, e) in enumerate(zip(aucs_sup, auc_errs)):
        ax1.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=7.5, color=TEXT)
    ax1.invert_yaxis()

    # ── Panel b: Supervised Accuracy comparison ──
    ax2 = fig.add_subplot(gs[0, 1])
    accs = [sup_r[c]['acc_mean'] for c in configs_sup]
    acc_errs = [sup_r[c]['acc_std'] for c in configs_sup]

    bars2 = ax2.barh(range(len(labels_sup)), accs, xerr=acc_errs,
                      color=colors_sup, edgecolor='white', height=0.65,
                      error_kw={'linewidth': 0.8, 'capsize': 2, 'alpha': 0.5})
    ax2.set_yticks(range(len(labels_sup)))
    ax2.set_yticklabels(labels_sup, fontsize=8.5)
    ax2.set_xlim(0.5, 0.9)
    ax2.set_xlabel('Accuracy (5-fold CV)', fontsize=9)
    ax2.set_title('b | Supervised accuracy', fontsize=10, weight='bold', loc='left')
    ax2.axvline(x=accs[0], color=ACCENT_BLUE, linestyle='--', linewidth=0.8, alpha=0.4)
    for i, (v, e) in enumerate(zip(accs, acc_errs)):
        ax2.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=7.5, color=TEXT)
    ax2.invert_yaxis()

    # ── Panel c: Unsupervised heatmap ──
    ax3 = fig.add_subplot(gs[0, 2])
    unsup_r = unsup['results']
    configs_unsup = ['L1_only (v2)', 'L2_only', 'L4_only', 'S_only',
                     'L1+L2', 'L1+L4', 'L1+S', 'L1+L2+L4+S (v3)']
    labels_unsup = ['L1', 'L2', 'L4', 'S', 'L1+L2', 'L1+L4', 'L1+S', 'L1+L2+L4+S']
    methods = ['deviation_semi_auc', 'ocsvm_auc', 'isoforest_auc', 'lof_auc', 'ensemble_auc']
    method_labels = ['Deviation', 'OCSVM', 'IsoForest', 'LOF', 'Ensemble']

    matrix = np.array([[unsup_r[c][m] for m in methods] for c in configs_unsup])

    im = ax3.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=1.0)
    ax3.set_yticks(range(len(labels_unsup)))
    ax3.set_yticklabels(labels_unsup, fontsize=8.5)
    ax3.set_xticks(range(len(method_labels)))
    ax3.set_xticklabels(method_labels, fontsize=8, rotation=30, ha='right')
    ax3.set_title('c | Unsupervised AUC heatmap', fontsize=10, weight='bold', loc='left')

    for i in range(len(labels_unsup)):
        for j in range(len(method_labels)):
            val = matrix[i, j]
            color = 'white' if val > 0.7 else TEXT
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=7.5, color=color, weight='bold')

    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('AUC', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Panel d: Unsupervised key comparison (bar chart) ──
    ax4 = fig.add_subplot(gs[1, 0])
    key_configs = ['L1_only (v2)', 'L1+L4', 'L1+L2+L4+S (v3)']
    key_labels = ['L1 only\n(v2)', 'L1+L4\n(best)', 'L1+L2+L4+S\n(full v3)']
    key_methods = ['deviation_semi_auc', 'ocsvm_auc', 'lof_auc', 'ensemble_auc']
    key_method_labels = ['Deviation', 'OCSVM', 'LOF', 'Ensemble']
    key_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE]

    x = np.arange(len(key_method_labels))
    width = 0.22
    for i, (cfg, label, color) in enumerate(zip(key_configs, key_labels, key_colors)):
        vals = [unsup_r[cfg][m] for m in key_methods]
        offset = (i - 1) * width
        bars = ax4.bar(x + offset, vals, width, label=label, color=color,
                       edgecolor='white', linewidth=0.5)

    ax4.set_xticks(x)
    ax4.set_xticklabels(key_method_labels, fontsize=8.5)
    ax4.set_ylabel('AUC', fontsize=9)
    ax4.set_ylim(0.3, 1.0)
    ax4.set_title('d | Unsupervised: L1 vs L1+L4 vs full', fontsize=10, weight='bold', loc='left')
    ax4.legend(fontsize=7.5, loc='lower right', frameon=False)
    ax4.axhline(y=0.5, color=ACCENT_GRAY, linestyle=':', linewidth=0.6, alpha=0.5)

    # ── Panel e: Per-layer standalone (unsupervised ensemble) ──
    ax5 = fig.add_subplot(gs[1, 1])
    standalone_configs = ['L1_only (v2)', 'L2_only', 'L4_only', 'S_only']
    standalone_labels = ['L1\n(10,000d)', 'L2\n(400d)', 'L4\n(10d)', 'S\n(45d)']
    standalone_colors = [ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_RED]

    ens_aucs = [unsup_r[c]['ensemble_auc'] for c in standalone_configs]
    dev_aucs = [unsup_r[c]['deviation_semi_auc'] for c in standalone_configs]
    ocsvm_aucs = [unsup_r[c]['ocsvm_auc'] for c in standalone_configs]

    x5 = np.arange(len(standalone_labels))
    width5 = 0.25
    ax5.bar(x5 - width5, dev_aucs, width5, label='Deviation', color=ACCENT_BLUE, alpha=0.7, edgecolor='white')
    ax5.bar(x5, ocsvm_aucs, width5, label='OCSVM', color=ACCENT_GREEN, alpha=0.7, edgecolor='white')
    ax5.bar(x5 + width5, ens_aucs, width5, label='Ensemble', color=ACCENT_ORANGE, alpha=0.7, edgecolor='white')

    ax5.set_xticks(x5)
    ax5.set_xticklabels(standalone_labels, fontsize=8.5)
    ax5.set_ylabel('AUC', fontsize=9)
    ax5.set_ylim(0.3, 1.0)
    ax5.set_title('e | Per-layer standalone (unsupervised)', fontsize=10, weight='bold', loc='left')
    ax5.legend(fontsize=7.5, loc='upper right', frameon=False)
    ax5.axhline(y=0.5, color=ACCENT_GRAY, linestyle=':', linewidth=0.6, alpha=0.5)

    # ── Panel f: Supervised vs Unsupervised summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    sup_l1 = sup_r['L1_only (v2)']['auc_mean']
    sup_multi = sup_r['Multi-layer (v3)']['auc_mean']
    sup_delta = sup_multi - sup_l1

    unsup_l1 = unsup_r['L1_only (v2)']['ocsvm_auc']
    unsup_best = unsup_r['L1+L4']['ocsvm_auc']
    unsup_delta = unsup_best - unsup_l1

    unsup_l1_ens = unsup_r['L1_only (v2)']['ensemble_auc']
    unsup_best_ens = unsup_r['L1+L4']['ensemble_auc']
    unsup_delta_ens = unsup_best_ens - unsup_l1_ens

    text = (
        "Summary\n\n"
        f"{'':<22s} {'L1 (v2)':>8s}  {'Best v3':>8s}  {'Δ':>7s}\n"
        f"{'─'*48}\n"
        f"{'Supervised AUC':<22s} {sup_l1:>8.4f}  {sup_multi:>8.4f}  {sup_delta:>+7.4f}\n"
        f"{'Supervised Acc':<22s} "
        f"{sup_r['L1_only (v2)']['acc_mean']:>8.4f}  "
        f"{sup_r['Multi-layer (v3)']['acc_mean']:>8.4f}  "
        f"{sup_r['Multi-layer (v3)']['acc_mean']-sup_r['L1_only (v2)']['acc_mean']:>+7.4f}\n"
        f"{'─'*48}\n"
        f"{'Unsup OCSVM':<22s} {unsup_l1:>8.4f}  {unsup_best:>8.4f}  {unsup_delta:>+7.4f}\n"
        f"{'Unsup Ensemble':<22s} {unsup_l1_ens:>8.4f}  {unsup_best_ens:>8.4f}  {unsup_delta_ens:>+7.4f}\n"
        f"{'─'*48}\n\n"
        "Key findings:\n"
        "• Supervised: L1+L2 best (motif adds signal)\n"
        "• Unsupervised: L1+L4 best (macro adds anomaly axis)\n"
        "• Full stack hurts unsupervised (curse of dimensionality)\n"
        "• L4/S alone cannot capture distributed RA signal"
    )

    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=8.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9FAFB', edgecolor=GRID))

    # Title
    fig.suptitle('Multi-Layer Spectrum: Supervised vs Unsupervised Performance on RA-TRA (n=545)',
                fontsize=12, weight='bold', y=0.985)

    # Save
    out_png = os.path.join(OUT_DIR, 'multilayer_comparison.png')
    out_pdf = os.path.join(OUT_DIR, 'multilayer_comparison.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
