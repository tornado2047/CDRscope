#!/usr/bin/env python3
"""
CDRscope Reference Map Visualizer
==================================
Generates publication-quality visualizations of the reference map
with property domain coloring, and demonstrates how new data projects
onto the same fixed coordinate space.

Usage:
  # Visualize reference map with property domains
  python visualize_reference_map.py --ref-dir reference_map/ --output ref_map_visualization.png

  # Compare reference map (fixed) vs per-project UMAP (changing)
  python visualize_reference_map.py --ref-dir reference_map/ \
    --project-csv project1_coords.csv --project-csv project2_coords.csv \
    --output comparison.png
"""
import os, json, argparse, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

# Font setup
cjk_fonts = ['Noto Sans CJK SC', 'PingFang SC', 'STHeiti', 'Heiti SC', 'Arial Unicode MS']
font_name = None
for f in cjk_fonts:
    matches = [m for m in fm.fontManager.ttflist if f in m.name]
    if matches:
        font_name = f
        break
if font_name:
    plt.rcParams['font.family'] = font_name
    print(f"Using font: {font_name}")
else:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Color palette
C_NAVY = '#1B3A5C'
C_STEEL = '#3D6A8C'
C_ACCENT = '#D97742'
C_GRAY = '#9A9A9A'
C_LTGRAY = '#E0E4E8'

# Property domain color maps
DOMAIN_COLORMAPS = {
    'chain': {'TRA': '#5B9279', 'TRB': '#3D6A8C'},
    'length_class': {'Short': '#2166AC', 'Medium': '#92C5DE', 'Long': '#D6604D'},
    'charge_class': {'Negative': '#2166AC', 'Neutral': '#92C5DE', 'Positive': '#D6604D'},
    'hydro_class': {'Hydrophilic': '#2166AC', 'Neutral': '#F0E442', 'Hydrophobic': '#D6604D'},
    'aromatic_class': {'Low': '#92C5DE', 'Medium': '#FDBF6F', 'High': '#D6604D'},
}


def plot_reference_map(ref_coords, ref_meta, output_path, domains=None,
                        sample_size=50000, title="CDRscope Reference Map"):
    """Plot the reference map with multiple property domain colorings."""
    if domains is None:
        domains = ['chain', 'length_class', 'charge_class',
                   'hydro_class', 'aromatic_class']

    n_domains = len(domains)
    ncols = 3
    nrows = (n_domains + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows), dpi=150)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Subsample for plotting speed
    n_total = len(ref_coords)
    if n_total > sample_size:
        np.random.seed(42)
        idx = np.random.choice(n_total, sample_size, replace=False)
    else:
        idx = np.arange(n_total)

    for i, domain in enumerate(domains):
        ax = axes[i // ncols, i % ncols]
        if domain not in ref_meta.columns:
            ax.text(0.5, 0.5, f"'{domain}' not in metadata", ha='center', va='center')
            ax.set_title(domain, fontsize=11)
            continue

        cmap = DOMAIN_COLORMAPS.get(domain, {})
        categories = ref_meta[domain].unique()

        for cat in categories:
            mask = (ref_meta[domain].iloc[idx] == cat).values
            color = cmap.get(cat, C_GRAY)
            ax.scatter(ref_coords[idx[mask], 0], ref_coords[idx[mask], 1],
                      c=color, s=0.5, alpha=0.5, label=str(cat), rasterized=True)

        ax.set_xlabel('UMAP1', fontsize=9)
        ax.set_ylabel('UMAP2', fontsize=9)
        ax.set_title(f'{domain}', fontsize=11, fontweight='bold')
        ax.legend(markerscale=4, fontsize=7, loc='best')
        ax.set_aspect('equal')

    # Hide unused axes
    for j in range(n_domains, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reference map visualization saved: {output_path}")


def plot_comparison(ref_coords, ref_meta, project_coords_list, project_names,
                     output_path, sample_size=30000):
    """
    Compare reference map (fixed) with multiple project UMAPs (changing).
    Left: Reference map (fixed coordinate space)
    Right: Each project's own UMAP (different coordinate space each time)
    """
    n_projects = len(project_coords_list)
    ncols = n_projects + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5), dpi=150)
    fig.suptitle("Reference Map vs Per-Project UMAP: Fixed vs Changing Coordinates",
                 fontsize=13, fontweight='bold')

    # Reference map (leftmost)
    ax = axes[0]
    n_ref = len(ref_coords)
    if n_ref > sample_size:
        np.random.seed(42)
        idx = np.random.choice(n_ref, sample_size, replace=False)
    else:
        idx = np.arange(n_ref)

    tra_mask = (ref_meta['chain'].iloc[idx] == 'TRA').values
    trb_mask = (ref_meta['chain'].iloc[idx] == 'TRB').values
    ax.scatter(ref_coords[idx[tra_mask], 0], ref_coords[idx[tra_mask], 1],
              c='#5B9279', s=0.3, alpha=0.4, label='TRA', rasterized=True)
    ax.scatter(ref_coords[idx[trb_mask], 0], ref_coords[idx[trb_mask], 1],
              c='#3D6A8C', s=0.3, alpha=0.4, label='TRB', rasterized=True)
    ax.set_title('Reference Map\n(FIXED coordinates)', fontsize=10, fontweight='bold', color=C_ACCENT)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend(markerscale=8, fontsize=7)
    ax.set_aspect('equal')

    # Per-project UMAPs
    for j, (coords, name) in enumerate(zip(project_coords_list, project_names)):
        ax = axes[j + 1]
        n = len(coords)
        if n > sample_size:
            np.random.seed(42)
            idx = np.random.choice(n, sample_size, replace=False)
        else:
            idx = np.arange(n)
        ax.scatter(coords[idx, 0], coords[idx, 1],
                  c=C_GRAY, s=0.3, alpha=0.4, rasterized=True)
        ax.set_title(f'{name}\n(different coordinates)', fontsize=10, color=C_GRAY)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {output_path}")


def plot_reference_with_overlay(ref_coords, ref_meta, new_coords, new_meta,
                                 output_path, color_by='chain',
                                 sample_ref=30000, title="Reference Map + New Data Overlay"):
    """Plot reference map (gray) with new data overlaid (colored)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)

    # Reference background
    n_ref = len(ref_coords)
    if n_ref > sample_ref:
        np.random.seed(42)
        idx = np.random.choice(n_ref, sample_ref, replace=False)
    else:
        idx = np.arange(n_ref)
    ax.scatter(ref_coords[idx, 0], ref_coords[idx, 1],
              c='lightgray', s=0.3, alpha=0.3, label='Reference (956K)', rasterized=True)

    # New data
    if color_by and color_by in new_meta.columns:
        categories = new_meta[color_by].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
        for cat, color in zip(categories, colors):
            mask = (new_meta[color_by] == cat).values
            ax.scatter(new_coords[mask, 0], new_coords[mask, 1],
                      c=[color], s=1, alpha=0.7, label=f'New: {cat}', rasterized=True)
    else:
        ax.scatter(new_coords[:, 0], new_coords[:, 1],
                  c=C_ACCENT, s=1, alpha=0.7, label='New data', rasterized=True)

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(markerscale=6, fontsize=9, loc='best')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overlay plot saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CDRscope reference map")
    parser.add_argument("--ref-dir", required=True, help="Reference map directory")
    parser.add_argument("--output", default="ref_map_viz.png", help="Output image path")
    parser.add_argument("--mode", choices=["domains", "comparison", "overlay"],
                        default="domains", help="Visualization mode")
    parser.add_argument("--project-csv", action='append', default=None,
                        help="Project coordinates CSV (for comparison mode, can repeat)")
    parser.add_argument("--project-name", action='append', default=None,
                        help="Project name (matches --project-csv order)")
    parser.add_argument("--new-csv", default=None, help="New data coords CSV (overlay mode)")
    parser.add_argument("--color-by", default="chain", help="Color column for overlay")
    args = parser.parse_args()

    # Load reference
    ref_coords = np.load(os.path.join(args.ref_dir, "ref_coords.npy"))
    _meta_path = os.path.join(args.ref_dir, "ref_metadata.csv")
    if not os.path.exists(_meta_path):
        _meta_path = os.path.join(args.ref_dir, "ref_metadata.csv.gz")
    ref_meta = pd.read_csv(_meta_path)
    print(f"Loaded reference: {len(ref_coords):,} sequences")

    if args.mode == "domains":
        plot_reference_map(ref_coords, ref_meta, args.output,
                          title="CDRscope Reference Map v1.0 — Property Domain Coloring (956K sequences)")

    elif args.mode == "comparison":
        project_coords = []
        project_names = args.project_name or []
        for i, csv_path in enumerate(args.project_csv or []):
            df = pd.read_csv(csv_path)
            umap_cols = [c for c in df.columns if 'umap' in c.lower() or 'UMAP' in c]
            if len(umap_cols) >= 2:
                project_coords.append(df[umap_cols[:2]].values)
            else:
                project_coords.append(df.iloc[:, :2].values)
            if i >= len(project_names):
                project_names.append(f"Project {i+1}")

        plot_comparison(ref_coords, ref_meta, project_coords, project_names, args.output)

    elif args.mode == "overlay":
        new_df = pd.read_csv(args.new_csv)
        umap_cols = [c for c in new_df.columns if 'umap' in c.lower() or 'UMAP' in c]
        if len(umap_cols) >= 2:
            new_coords = new_df[umap_cols[:2]].values
        else:
            new_coords = new_df.iloc[:, :2].values
        plot_reference_with_overlay(ref_coords, ref_meta, new_coords, new_df,
                                     args.output, color_by=args.color_by)
