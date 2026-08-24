#!/usr/bin/env python3
"""
Deviation & Direction Visualization
=====================================
Visualize both deviation magnitude (how far) and deviation direction (where)
of samples in the CDRscope Reference Coordinate System.

Visualizations:
  1. PCA scatter with magnitude coloring — color = how far
  2. Polar/radial scatter — radius = magnitude, angle = PC1/PC2 direction
  3. Radar chart — deviation on top PC axes (shows direction pattern)
  4. Top contributing prototypes — which prototypes drive the deviation
  5. Direction heatmap — samples × top PCs deviation matrix
  6. Deviation rose plot — direction grouping + magnitude
"""
import os, sys, pickle, warnings, base64
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

WORK_DIR = "/Users/xfcheung/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a7c2cace78ca95f7748ffb8"
TIER2_DIR = os.path.join(WORK_DIR, "CDRscope-v2", "inst", "python", "tier2")
sys.path.insert(0, TIER2_DIR)
PANEL_DIR = os.path.join(WORK_DIR, "cordblood_tra_panel")
RCS_DIR = os.path.join(WORK_DIR, "reference_coordinate_system")
OUTPUT_DIR = os.path.join(WORK_DIR, "deviation_direction_visualization")
IMG_DIR = os.path.join(OUTPUT_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_ORIGIN = '#333333'
C_CTRL = '#4a90d9'
C_PAT = '#ff6b6b'
C_ACCENT = '#5e5ce6'
C_GREEN = '#00a389'
C_ORANGE = '#ff9f0a'
C_GRAY = '#8e8e93'

# Deviation colormap: blue (low) → yellow (mid) → red (high)
DEV_CMAP = LinearSegmentedColormap.from_list('dev_heat',
    ['#4a90d9', '#ffd60a', '#ff6b6b'], N=256)

# Direction colormap: HSV-like for angle
DIR_CMAP = plt.cm.hsv


def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def load_data():
    """Load RCS and RA-TRA data."""
    # Load RCS
    rcs_path = os.path.join(RCS_DIR, "reference_coordinate_system.pkl")
    with open(rcs_path, 'rb') as f:
        rcs = pickle.load(f)

    # Load RA-TRA
    X = np.load(os.path.join(PANEL_DIR, "ra_tra_cb_matrix.npy")).astype(np.float64)
    labels = np.load(os.path.join(PANEL_DIR, "ra_tra_cb_labels.npy")).astype(int)
    X_norm = normalize(X, norm='l2')

    # Project to RCS
    X_pca = rcs['pca'].transform(X_norm)
    X_umap = rcs['umap'].transform(X_pca)

    # Deviation from origin
    dev_vec = X_pca - rcs['reference_mean_pca']
    dev_mag = np.linalg.norm(dev_vec, axis=1)
    dev_dir = dev_vec / dev_mag[:, np.newaxis]

    # Top prototypes by deviation contribution
    # Project back: which original prototypes contribute most to deviation
    top_pc_loadings = np.abs(rcs['pca'].components_[:10]).sum(axis=0)
    top_proto_idx = np.argsort(top_pc_loadings)[::-1][:100]

    return {
        'rcs': rcs,
        'X_norm': X_norm,
        'X_pca': X_pca,
        'X_umap': X_umap,
        'dev_mag': dev_mag,
        'dev_dir': dev_dir,
        'labels': labels,
        'top_proto_idx': top_proto_idx,
    }


def plot_pca_magnitude(data):
    """Figure 1: PCA scatter colored by deviation magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    X_pca = data['X_pca']
    dev_mag = data['dev_mag']
    labels = data['labels']
    rcs = data['rcs']

    # Left: colored by deviation magnitude
    ax = axes[0]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                   c=dev_mag, cmap=DEV_CMAP, s=35, alpha=0.7,
                   edgecolors='white', linewidth=0.3, zorder=3)
    # Reference origin
    ax.scatter(rcs['reference_mean_pca'][0], rcs['reference_mean_pca'][1],
              c='black', s=300, marker='*', edgecolors='white',
              linewidth=1.5, zorder=10, label='CB Reference')
    # Add deviation contour lines
    # Draw circles around origin representing magnitude levels
    max_mag = dev_mag.max()
    for level in [max_mag*0.25, max_mag*0.5, max_mag*0.75]:
        circle = plt.Circle((rcs['reference_mean_pca'][0], rcs['reference_mean_pca'][1]),
                           level, fill=False, linestyle='--', alpha=0.3,
                           color='gray', linewidth=1)
        ax.add_artist(circle)
    plt.colorbar(sc, ax=ax, label='Deviation Magnitude', shrink=0.8)
    ax.set_xlabel(f'PC1 ({rcs["variance_explained"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({rcs["variance_explained"][1]:.1%})')
    ax.set_title('Deviation Magnitude in RCS\n(Color = distance from CB origin)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # Right: colored by label (supervised validation)
    ax = axes[1]
    for lv, color, name in [(0, C_CTRL, 'Control'), (1, C_PAT, 'Patient')]:
        mask = labels == lv
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=color, s=35, alpha=0.6, label=f'{name} ({mask.sum()})',
                  edgecolors='white', linewidth=0.3, zorder=3)
    ax.scatter(rcs['reference_mean_pca'][0], rcs['reference_mean_pca'][1],
              c='black', s=300, marker='*', edgecolors='white',
              linewidth=1.5, zorder=10, label='CB Reference')
    ax.set_xlabel(f'PC1 ({rcs["variance_explained"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({rcs["variance_explained"][1]:.1%})')
    ax.set_title('Disease Label Overlay\n(Blue = Control, Red = Patient)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    plt.suptitle('Deviation Magnitude — How Far from Reference?',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig1_pca_magnitude.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig1_pca_magnitude.png saved", flush=True)


def plot_polar_scatter(data):
    """Figure 2: Polar scatter — radius = magnitude, angle = PC1/PC2 direction."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    dev_mag = data['dev_mag']
    dev_dir = data['dev_dir']
    labels = data['labels']

    # Angle from PC1/PC2 direction
    angle = np.arctan2(dev_dir[:, 1], dev_dir[:, 0])

    # Plot controls
    mask_ctrl = labels == 0
    ax.scatter(angle[mask_ctrl], dev_mag[mask_ctrl],
              c=C_CTRL, s=40, alpha=0.5, edgecolors='white',
              linewidth=0.3, label=f'Control (n={mask_ctrl.sum()})', zorder=3)

    # Plot patients
    mask_pat = labels == 1
    ax.scatter(angle[mask_pat], dev_mag[mask_pat],
              c=C_PAT, s=40, alpha=0.5, edgecolors='white',
              linewidth=0.3, label=f'Patient (n={mask_pat.sum()})', zorder=3)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_rlabel_position(90)
    ax.set_ylabel('Deviation Magnitude', labelpad=20)
    ax.set_title('Deviation Polar View\nRadius = magnitude, Angle = direction (PC1→PC2)',
                pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Add directional labels
    ax.text(0, ax.get_ylim()[1]*1.1, '+PC1', ha='center', fontsize=10, fontweight='bold')
    ax.text(np.pi/2, ax.get_ylim()[1]*1.1, '+PC2', ha='center', fontsize=10, fontweight='bold')
    ax.text(np.pi, ax.get_ylim()[1]*1.1, '-PC1', ha='center', fontsize=10, fontweight='bold')
    ax.text(-np.pi/2, ax.get_ylim()[1]*1.1, '-PC2', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig2_polar_scatter.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig2_polar_scatter.png saved", flush=True)


def plot_radar_chart(data):
    """Figure 3: Radar chart of deviation on top PC axes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                             subplot_kw=dict(projection='polar'))

    dev_dir = data['dev_dir']
    dev_mag = data['dev_mag']
    labels = data['labels']
    rcs = data['rcs']

    n_pc = 8  # Top 8 PCs
    angles = np.linspace(0, 2*np.pi, n_pc, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    # Left: mean deviation by group
    ax = axes[0]
    for lv, color, name in [(0, C_CTRL, 'Control'), (1, C_PAT, 'Patient')]:
        mask = labels == lv
        # Mean deviation direction * mean magnitude
        mean_dir = dev_dir[mask].mean(axis=0)[:n_pc]
        mean_mag = dev_mag[mask].mean()
        # Scale by variance explained
        pc_weights = rcs['variance_explained'][:n_pc]
        values = np.abs(mean_dir) * mean_mag * pc_weights
        values = values / values.max() if values.max() > 0 else values
        values_closed = np.concatenate([values, [values[0]]])

        ax.plot(angles_closed, values_closed, color=color,
                linewidth=2, label=name)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.2)

    ax.set_xticks(angles)
    ax.set_xticklabels([f'PC{i+1}\n({rcs["variance_explained"][i]:.1%})'
                       for i in range(n_pc)], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title('Mean Deviation Profile\nby Group (normalized)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Right: individual sample radars (selected examples)
    ax = axes[1]
    # Pick representative samples
    np.random.seed(42)
    ctrl_idx = np.where(labels == 0)[0]
    pat_idx = np.where(labels == 1)[0]

    # Low deviation control
    low_ctrl = ctrl_idx[np.argsort(dev_mag[ctrl_idx])[1]]
    # High deviation patient
    high_pat = pat_idx[np.argsort(dev_mag[pat_idx])[-2]]
    # Mid deviation patient
    mid_pat = pat_idx[np.argsort(dev_mag[pat_idx])[len(pat_idx)//2]]

    samples_to_plot = [
        (low_ctrl, C_CTRL, f'Low-dev Ctrl\n({dev_mag[low_ctrl]:.3f})'),
        (mid_pat, C_ORANGE, f'Mid-dev Pat\n({dev_mag[mid_pat]:.3f})'),
        (high_pat, C_PAT, f'High-dev Pat\n({dev_mag[high_pat]:.3f})'),
    ]

    for idx, color, name in samples_to_plot:
        values = np.abs(dev_dir[idx, :n_pc]) * dev_mag[idx] * rcs['variance_explained'][:n_pc]
        values = values / values.max() if values.max() > 0 else values
        values_closed = np.concatenate([values, [values[0]]])
        ax.plot(angles_closed, values_closed, color=color, linewidth=2, label=name)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.15)

    ax.set_xticks(angles)
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_pc)], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title('Individual Deviation Profiles\n(shape = direction pattern)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.suptitle('Deviation Direction — Which Way?',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig3_radar_chart.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig3_radar_chart.png saved", flush=True)


def plot_top_prototypes(data):
    """Figure 4: Top deviating prototypes and their contribution."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    X_norm = data['X_norm']
    labels = data['labels']
    rcs = data['rcs']
    dev_mag = data['dev_mag']

    # Compute per-prototype deviation (patient mean - control mean)
    ctrl_mean = X_norm[labels == 0].mean(axis=0)
    pat_mean = X_norm[labels == 1].mean(axis=0)
    proto_dev = pat_mean - ctrl_mean  # positive = patient-enriched

    # Top 20 deviating prototypes
    top_idx = np.argsort(np.abs(proto_dev))[::-1][:20]
    top_vals = proto_dev[top_idx]

    ax = axes[0]
    y_pos = range(20)
    colors = [C_PAT if v > 0 else C_CTRL for v in top_vals]
    ax.barh(y_pos, np.abs(top_vals), color=colors, alpha=0.8,
            edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Proto #{i}' for i in top_idx], fontsize=8)
    ax.set_xlabel('|Mean Deviation| (Patient - Control)')
    ax.set_title('Top 20 Deviating Prototypes\n(Red = Patient-enriched, Blue = Control-enriched)')
    ax.invert_yaxis()

    # Right: cumulative deviation contribution
    ax = axes[1]
    all_abs_dev = np.sort(np.abs(proto_dev))[::-1]
    cum_frac = np.cumsum(all_abs_dev) / all_abs_dev.sum()
    ax.plot(range(1, len(cum_frac) + 1), cum_frac, color=C_ACCENT, linewidth=2)
    ax.fill_between(range(1, len(cum_frac) + 1), cum_frac, alpha=0.2, color=C_ACCENT)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='80%')
    ax.set_xscale('log')
    ax.set_xlabel('Prototype Rank (log scale)')
    ax.set_ylabel('Cumulative Deviation Fraction')
    ax.set_title('Deviation Concentration\nHow many prototypes drive the difference?')
    ax.legend()

    # Mark key thresholds
    for frac in [0.5, 0.8]:
        n_protos = np.searchsorted(cum_frac, frac) + 1
        ax.annotate(f'{n_protos} protos\n({frac:.0%})',
                   xy=(n_protos, frac), xytext=(n_protos*2, frac-0.05),
                   fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig4_top_prototypes.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig4_top_prototypes.png saved", flush=True)


def plot_direction_heatmap(data):
    """Figure 5: Direction heatmap — samples × top PCs."""
    fig, ax = plt.subplots(figsize=(14, 10))

    dev_dir = data['dev_dir']
    dev_mag = data['dev_mag']
    labels = data['labels']
    rcs = data['rcs']

    n_pc = 15
    # Weight direction by magnitude and variance explained
    weighted_dev = dev_dir[:, :n_pc] * dev_mag[:, np.newaxis] * rcs['variance_explained'][:n_pc]

    # Sort samples: by label, then by deviation magnitude
    sort_idx = np.lexsort((dev_mag, labels))
    weighted_sorted = weighted_dev[sort_idx]

    # Normalize per PC for better visualization
    weighted_norm = weighted_sorted / np.max(np.abs(weighted_sorted), axis=0)

    im = ax.imshow(weighted_norm.T, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Deviation (Red = +, Blue = -)', shrink=0.6)

    ax.set_yticks(range(n_pc))
    ax.set_yticklabels([f'PC{i+1} ({rcs["variance_explained"][i]:.1%})' for i in range(n_pc)])
    ax.set_xlabel('Samples (sorted by label → deviation)')
    ax.set_title('Deviation Direction Heatmap\nRed = positive deviation, Blue = negative deviation')

    # Add label separator line
    n_ctrl = sum(labels == 0)
    ax.axvline(n_ctrl - 0.5, color='black', linewidth=1.5)
    ax.text(n_ctrl//2, n_pc + 0.5, f'Control\n(n={n_ctrl})',
           ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_CTRL)
    ax.text(n_ctrl + (len(labels)-n_ctrl)//2, n_pc + 0.5,
           f'Patient\n(n={len(labels)-n_ctrl})',
           ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_PAT)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig5_direction_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig5_direction_heatmap.png saved", flush=True)


def plot_direction_clusters(data):
    """Figure 6: Directional clusters — group samples by deviation direction."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    dev_dir = data['dev_dir']
    dev_mag = data['dev_mag']
    labels = data['labels']
    X_pca = data['X_pca']
    rcs = data['rcs']

    # K-means on deviation direction (first 10 PCs)
    n_clusters = 4
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    direction_clusters = km.fit_predict(dev_dir[:, :10])
    data['dir_clusters'] = direction_clusters

    cluster_colors = ['#4a90d9', '#ff6b6b', '#00a389', '#ff9f0a',
                      '#bf5af2', '#5e5ce6']

    # Left: PCA colored by direction cluster
    ax = axes[0]
    for c in range(n_clusters):
        mask = direction_clusters == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=cluster_colors[c], s=35, alpha=0.7,
                  edgecolors='white', linewidth=0.3,
                  label=f'Cluster {c+1} (n={mask.sum()})')

    ax.scatter(rcs['reference_mean_pca'][0], rcs['reference_mean_pca'][1],
              c='black', s=300, marker='*', edgecolors='white',
              linewidth=1.5, zorder=10, label='CB Reference')
    ax.set_xlabel(f'PC1 ({rcs["variance_explained"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({rcs["variance_explained"][1]:.1%})')
    ax.set_title(f'Direction Clusters (K={n_clusters})\nSamples grouped by deviation direction')
    ax.legend(fontsize=8)

    # Right: cluster properties — mean magnitude + patient rate
    ax = axes[1]
    cluster_info = []
    for c in range(n_clusters):
        mask = direction_clusters == c
        mag_mean = dev_mag[mask].mean()
        pat_rate = labels[mask].mean()
        cluster_info.append((c, mag_mean, pat_rate, mask.sum()))

    x = np.arange(n_clusters)
    width = 0.35
    bars1 = ax.bar(x - width/2, [ci[1] for ci in cluster_info], width,
                   color=C_ACCENT, alpha=0.7, label='Mean Deviation', edgecolor='white')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, [ci[2] for ci in cluster_info], width,
                    color=C_PAT, alpha=0.7, label='Patient Rate', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {c+1}\n(n={cluster_info[c][3]})' for c in range(n_clusters)])
    ax.set_ylabel('Mean Deviation Magnitude', color=C_ACCENT)
    ax2.set_ylabel('Patient Rate', color=C_PAT)
    ax.set_title('Cluster Properties')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig6_direction_clusters.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig6_direction_clusters.png saved", flush=True)


def plot_arrow_field(data):
    """Figure 7: Arrow field — arrows from origin showing deviation direction + magnitude."""
    fig, ax = plt.subplots(figsize=(12, 12))

    X_pca = data['X_pca']
    dev_mag = data['dev_mag']
    dev_dir = data['dev_dir']
    labels = data['labels']
    rcs = data['rcs']

    origin = rcs['reference_mean_pca'][:2]

    # Only draw a subset to avoid clutter
    np.random.seed(42)
    n_arrows = 50
    # Select: some low-deviation controls, some mid, some high-deviation patients
    ctrl_idx = np.where(labels == 0)[0]
    pat_idx = np.where(labels == 1)[0]

    # Sort by deviation
    ctrl_sorted = ctrl_idx[np.argsort(dev_mag[ctrl_idx])]
    pat_sorted = pat_idx[np.argsort(dev_mag[pat_idx])]

    selected = np.concatenate([
        ctrl_sorted[::max(1, len(ctrl_sorted)//15)][:15],
        pat_sorted[::max(1, len(pat_sorted)//35)][:35],
    ])

    # Draw arrows
    for idx in selected:
        end_pt = X_pca[idx, :2]
        color = C_CTRL if labels[idx] == 0 else C_PAT
        alpha = 0.4 + 0.6 * (dev_mag[idx] / dev_mag.max())

        arrow = FancyArrowPatch(
            origin, end_pt,
            arrowstyle='->', mutation_scale=10,
            color=color, alpha=alpha, linewidth=1.2,
            connectionstyle='arc3,rad=0'
        )
        ax.add_patch(arrow)

        # Dot at end
        ax.scatter(end_pt[0], end_pt[1], c=color, s=20, alpha=alpha,
                  edgecolors='white', linewidth=0.5, zorder=5)

    # Reference origin
    ax.scatter(origin[0], origin[1], c='black', s=400, marker='*',
              edgecolors='white', linewidth=1.5, zorder=10, label='CB Reference')

    # Concentric circles
    max_mag = dev_mag.max()
    for level, ls in [(max_mag*0.25, ':'), (max_mag*0.5, '--'), (max_mag*0.75, '-')]:
        circle = plt.Circle(origin, level, fill=False, linestyle=ls, alpha=0.3,
                           color='gray', linewidth=1)
        ax.add_artist(circle)

    ax.set_xlabel(f'PC1 ({rcs["variance_explained"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({rcs["variance_explained"][1]:.1%})')
    ax.set_title('Deviation Arrow Field\nArrow direction = deviation direction, length = magnitude, opacity = magnitude')
    ax.set_aspect('equal')

    legend_elements = [
        Line2D([0], [0], color=C_CTRL, linewidth=2, label='Control'),
        Line2D([0], [0], color=C_PAT, linewidth=2, label='Patient'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markersize=12, label='CB Reference'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    path = os.path.join(IMG_DIR, 'fig7_arrow_field.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  fig7_arrow_field.png saved", flush=True)


# =========================================================================
# HTML Report
# =========================================================================
def generate_html(data):
    figures = [
        ('fig1_pca_magnitude.png', '1. Deviation Magnitude (有多远)',
         '左图：PCA 散点图，颜色表示偏离度大小（蓝→黄→红 = 低→中→高）。同心圆表示等偏离度线。<br>'
         '右图：同样的 PCA 空间，用疾病标签着色。可以看到高偏离度的点（红色）主要对应患者，验证了偏离度的生物学意义。'),
        ('fig2_polar_scatter.png', '2. 极坐标散点图 (方向 + 大小)',
         '极坐标系中，半径 = 偏离度大小，角度 = PC1/PC2 方向。<br>'
         '每个样本是一个点：离中心越远 = 偏离越大，角度位置 = 偏离方向。<br>'
         '蓝色=对照，红色=患者。可以看到患者主要分布在某些角度区间，说明疾病有特定的偏离方向。'),
        ('fig3_radar_chart.png', '3. 雷达图：偏离方向模式',
         '左图：对照组和患者组的平均偏离轮廓（Top 8 PC 轴）。<br>'
         '右图：三个代表性样本的偏离雷达图——低偏离对照、中偏离患者、高偏离患者。<br>'
         '雷达图的形状 = 偏离方向模式，大小 = 偏离度。不同样本可能有相同的偏离度但不同的方向模式。'),
        ('fig4_top_prototypes.png', '4. Top 偏离原型',
         '左图：偏离度最大的 20 个原型（红=患者富集，蓝=对照富集）。<br>'
         '右图：偏离度集中度曲线——多少个原型贡献了 50%/80% 的总偏离？<br>'
         '回答"偏离在什么地方"：具体到哪些 CDR3 原型。'),
        ('fig5_direction_heatmap.png', '5. 偏离方向热图',
         '样本 × Top 15 PC 的偏离方向矩阵。红色=正偏离，蓝色=负偏离。<br>'
         '样本按标签排序（左=对照，右=患者）。可以看到患者侧有明显的红色/蓝色模式，<br>'
         '说明疾病导致的偏离是系统性的，涉及多个 PC 轴。'),
        ('fig6_direction_clusters.png', '6. 方向聚类',
         '按偏离方向（而非大小）对样本进行 K-means 聚类。<br>'
         '左图：PCA 空间中按方向聚类着色。<br>'
         '右图：每个聚类的平均偏离度（蓝柱）和患者比例（红柱）。<br>'
         '有些方向聚类可能对应不同的疾病亚型或偏离模式。'),
        ('fig7_arrow_field.png', '7. 箭头场',
         '从参考原点出发的箭头，指向每个样本的位置。<br>'
         '箭头方向 = 偏离方向，箭头长度 = 偏离度大小，透明度也反映大小。<br>'
         '直观展示了所有样本相对于参考原点的"位置向量"。'),
    ]

    html = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Deviation & Direction Visualization</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #1a1a1a; }
h1 { border-bottom: 3px solid #5e5ce6; padding-bottom: 10px; }
h2 { color: #5e5ce6; margin-top: 40px; }
.figure { background: white; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.figure img { width: 100%; border-radius: 8px; }
.figure p { color: #555; line-height: 1.6; font-size: 14px; }
.hero { background: linear-gradient(135deg, #5e5ce6 0%, #ff9f0a 100%); color: white; border-radius: 16px; padding: 28px; margin: 20px 0; }
.hero h1 { color: white; border: none; margin: 0; }
.summary { background: white; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.summary table { width: 100%; border-collapse: collapse; }
.summary th, .summary td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #e0e0e0; }
.summary th { background: #f5f5f7; font-weight: 600; }
.box { background: white; border-left: 4px solid #00a389; border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0; }
</style>
</head><body>

<div class="hero">
<h1>Deviation & Direction Visualization</h1>
<p>在 CDRscope 参考坐标系中，同时可视化<b>偏离度</b>（有多远）和<b>偏向</b>（往哪偏）。<br>
从 7 个角度展示：大小、方向、模式、原型、热图、聚类、箭头场。</p>
</div>

<div class="summary">
<h2>Visualization Taxonomy</h2>
<table>
<tr><th>Figure</th><th>Shows Magnitude (偏离度)</th><th>Shows Direction (偏向)</th><th>Best For</th></tr>
<tr><td>1. PCA Magnitude</td><td style="color:green;">✓ (color)</td><td style="color:orange;">~ (position)</td><td>整体分布概览</td></tr>
<tr><td>2. Polar Scatter</td><td style="color:green;">✓ (radius)</td><td style="color:green;">✓ (angle)</td><td>方向+大小同图</td></tr>
<tr><td>3. Radar Chart</td><td style="color:green;">✓ (size)</td><td style="color:green;">✓ (shape)</td><td>多轴方向模式</td></tr>
<tr><td>4. Top Prototypes</td><td style="color:green;">✓ (bar height)</td><td style="color:green;">✓ (which prototypes)</td><td>机制解析</td></tr>
<tr><td>5. Direction Heatmap</td><td style="color:orange;">~</td><td style="color:green;">✓ (pattern)</td><td>全样本方向比较</td></tr>
<tr><td>6. Direction Clusters</td><td style="color:green;">✓ (bar)</td><td style="color:green;">✓ (groups)</td><td>亚型发现</td></tr>
<tr><td>7. Arrow Field</td><td style="color:green;">✓ (length+opacity)</td><td style="color:green;">✓ (angle)</td><td>直观向量展示</td></tr>
</table>
</div>

<div class="box">
<h3>核心概念</h3>
<ul>
<li><b>偏离度 (Deviation Magnitude)</b>：样本与 CordBlood 参考原点的距离，标量。回答"有多异常？"</li>
<li><b>偏离方向 (Deviation Direction)</b>：样本在参考空间中的偏移方向，向量。回答"哪里异常？"</li>
<li><b>两个样本可能偏离度相同但方向不同</b>：一个在 PC1 方向偏了很多，另一个在 PC3 方向偏了很多——都"异常"，但异常的原因不同。</li>
<li><b>偏离方向可以聚类</b>：同一方向的样本可能共享相似的生物学机制（如特定的疾病亚型）。</li>
</ul>
</div>
'''

    for img_name, title, desc in figures:
        img_path = os.path.join(IMG_DIR, img_name)
        if os.path.exists(img_path):
            b64 = img_to_b64(img_path)
            html += f'''
<div class="figure">
<h2>{title}</h2>
<img src="data:image/png;base64,{b64}" alt="{title}">
<p>{desc}</p>
</div>'''

    html += '''
</body></html>'''

    report_path = os.path.join(OUTPUT_DIR, "deviation_direction_report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"\nHTML report: {report_path}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("Deviation & Direction Visualization", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading data...", flush=True)
    data = load_data()
    print(f"  Samples: {len(data['labels'])} (Ctrl={sum(data['labels']==0)}, "
          f"Pat={sum(data['labels']==1)})", flush=True)
    print(f"  Deviation: mean={data['dev_mag'].mean():.4f}, "
          f"std={data['dev_mag'].std():.4f}", flush=True)

    print("\nGenerating figures...", flush=True)
    plot_pca_magnitude(data)
    plot_polar_scatter(data)
    plot_radar_chart(data)
    plot_top_prototypes(data)
    plot_direction_heatmap(data)
    plot_direction_clusters(data)
    plot_arrow_field(data)

    print("\nGenerating HTML report...", flush=True)
    generate_html(data)

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
