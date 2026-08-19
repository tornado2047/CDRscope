#!/usr/bin/env python3
"""
CDRscope Reference Map Projector
================================
Projects new CDR3 sequences onto a pre-built reference UMAP map.

Given a reference map package (built by build_reference_map.py),
this script:
  1. Computes ESM-2 embeddings for new CDR3 sequences
  2. Loads the saved reference neural network (480→2)
  3. Projects new sequences into the SAME 2D coordinate space
  4. Optionally overlays new points on the reference map

This guarantees that different projects with different inputs
all map to the SAME coordinate space for unified comparison.

Usage:
  python project_to_reference.py \
    --ref-dir reference_map/ \
    --input-csv new_project_sequences.csv \
    --output-csv new_project_coords.csv \
    --overlay  # also generate overlay plot

Input CSV must have a 'junction_aa' column (or specify --seq-col).
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── Model definition (must match build_reference_map.py) ───
class UMAPMapper(nn.Module):
    def __init__(self, input_dim=480, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ─── Amino acid properties ───
KD = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,
      'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,
      'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
CHARGE = {'K':+1, 'R':+1, 'H':+0.5, 'D':-1, 'E':-1}
AROMATIC = set('FWY')

def compute_properties(seq):
    L = len(seq)
    charge = sum(CHARGE.get(aa, 0) for aa in seq)
    hydro = np.mean([KD.get(aa, 0) for aa in seq]) if seq else 0
    arom = sum(1 for aa in seq if aa in AROMATIC) / L if L else 0
    return L, charge, hydro, arom

def classify_length(L):
    if L <= 12: return 'Short (≤12)'
    elif L <= 16: return 'Medium (13-16)'
    else: return 'Long (≥17)'

def classify_charge(q):
    if q <= -1: return 'Negative'
    elif q <= 1: return 'Neutral'
    else: return 'Positive'

def classify_hydro(h):
    if h < -0.5: return 'Hydrophilic'
    elif h < 0.3: return 'Neutral'
    else: return 'Hydrophobic'

def classify_aromatic(frac):
    if frac < 0.10: return 'Low (<10%)'
    elif frac < 0.20: return 'Medium (10-20%)'
    else: return 'High (≥20%)'


def compute_esm2_embeddings(sequences, model_name="facebook/esm2_t12_35M_UR50D",
                             batch_size=256, device=None):
    """Compute ESM-2 mean-pooled embeddings for a list of CDR3 sequences."""
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    print(f"Loading ESM-2 model: {model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-2 embed"):
            batch = sequences[i:i+batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=64)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)


def project_to_reference(ref_dir, sequences, seq_metadata=None,
                          device=None, batch_size=256):
    """
    Project new CDR3 sequences onto a reference map.

    Args:
        ref_dir: Path to reference map directory
        sequences: List of CDR3 amino acid sequences
        seq_metadata: Optional DataFrame with metadata (chain, group, etc.)
        device: torch device
        batch_size: ESM-2 batch size

    Returns:
        DataFrame with UMAP1, UMAP2 coordinates + properties
    """
    # Load reference config
    with open(os.path.join(ref_dir, "ref_config.json")) as f:
        config = json.load(f)

    input_dim = config["embedding_dim"]
    esm2_model = config["esm2_model"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")

    # 1. Compute ESM-2 embeddings
    print(f"\nStep 1: Computing ESM-2 embeddings for {len(sequences):,} sequences", flush=True)
    embeddings = compute_esm2_embeddings(sequences, model_name=esm2_model,
                                           batch_size=batch_size, device=device)
    print(f"  Embeddings: {embeddings.shape}", flush=True)

    # 2. Load reference NN and project
    print(f"\nStep 2: Projecting onto reference map", flush=True)
    model = UMAPMapper(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(ref_dir, "ref_mapper.pt"),
                                      map_location=device))
    model.eval()

    coords = np.zeros((len(sequences), 2), dtype=np.float32)
    PROJ_BATCH = 4096
    with torch.no_grad():
        for i in range(0, len(sequences), PROJ_BATCH):
            end = min(i+PROJ_BATCH, len(sequences))
            x = torch.tensor(embeddings[i:end], dtype=torch.float32).to(device)
            coords[i:end] = model(x).cpu().numpy()

    print(f"  Projected to reference space", flush=True)
    print(f"  UMAP1 range: [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]", flush=True)
    print(f"  UMAP2 range: [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]", flush=True)

    # 3. Build output DataFrame with properties
    rows = []
    for i, seq in enumerate(sequences):
        L, charge, hydro, arom = compute_properties(seq)
        row = {
            'sequence': seq,
            'length': L,
            'net_charge': charge,
            'hydrophobicity': round(hydro, 3),
            'aromatic_frac': round(arom, 3),
            'length_class': classify_length(L),
            'charge_class': classify_charge(charge),
            'hydro_class': classify_hydro(hydro),
            'aromatic_class': classify_aromatic(arom),
            'umap1': float(coords[i, 0]),
            'umap2': float(coords[i, 1]),
        }
        if seq_metadata is not None and i < len(seq_metadata):
            for col in seq_metadata.columns:
                if col not in row:
                    row[col] = seq_metadata.iloc[i][col]
        rows.append(row)

    return pd.DataFrame(rows)


def overlay_plot(ref_coords, ref_meta, new_coords, new_meta, output_path,
                  color_by='chain', title="Reference Map + New Data Overlay"):
    """Generate an overlay scatter plot: reference (gray) + new data (colored)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Reference background (subsampled for speed)
    n_ref = len(ref_coords)
    sample_idx = np.random.choice(n_ref, min(n_ref, 50000), replace=False)
    ax.scatter(ref_coords[sample_idx, 0], ref_coords[sample_idx, 1],
              c='lightgray', s=0.3, alpha=0.3, label='Reference', rasterized=True)

    # New data
    if color_by in new_meta.columns:
        categories = new_meta[color_by].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
        for cat, color in zip(categories, colors):
            mask = new_meta[color_by] == cat
            ax.scatter(new_coords[mask, 0], new_coords[mask, 1],
                      c=[color], s=1, alpha=0.6, label=str(cat), rasterized=True)
    else:
        ax.scatter(new_coords[:, 0], new_coords[:, 1],
                  c='red', s=1, alpha=0.6, label='New data', rasterized=True)

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(markerscale=5, fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overlay plot saved: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project new CDR3 data onto reference map")
    parser.add_argument("--ref-dir", required=True, help="Reference map directory")
    parser.add_argument("--input-csv", required=True, help="Input CDR3 sequences CSV")
    parser.add_argument("--seq-col", default="junction_aa", help="Column name for CDR3 sequences")
    parser.add_argument("--output-csv", default="projected_coords.csv", help="Output coordinates CSV")
    parser.add_argument("--overlay", action="store_true", help="Generate overlay plot")
    parser.add_argument("--overlay-png", default="overlay_plot.png", help="Overlay plot path")
    parser.add_argument("--color-by", default="chain", help="Column to color new data by")
    parser.add_argument("--batch-size", type=int, default=256, help="ESM-2 batch size")
    args = parser.parse_args()

    # Load input
    df = pd.read_csv(args.input_csv)
    seq_col = args.seq_col if args.seq_col in df.columns else df.columns[0]
    sequences = df[seq_col].dropna().unique().tolist()
    print(f"Loaded {len(sequences):,} unique CDR3 sequences from {args.input_csv}", flush=True)

    # Project
    result = project_to_reference(args.ref_dir, sequences,
                                    seq_metadata=df[[c for c in df.columns if c != seq_col]],
                                    batch_size=args.batch_size)

    # Save
    result.to_csv(args.output_csv, index=False)
    print(f"\nProjected coordinates saved: {args.output_csv}", flush=True)
    print(f"  {len(result)} sequences projected", flush=True)

    # Overlay plot
    if args.overlay:
        ref_coords = np.load(os.path.join(args.ref_dir, "ref_coords.npy"))
        _meta = os.path.join(args.ref_dir, "ref_metadata.csv")
        if not os.path.exists(_meta):
            _meta = os.path.join(args.ref_dir, "ref_metadata.csv.gz")
        ref_meta = pd.read_csv(_meta)
        new_coords = result[['umap1', 'umap2']].values
        overlay_plot(ref_coords, ref_meta, new_coords, result,
                     args.overlay_png, color_by=args.color_by,
                     title=f"Reference Map + {os.path.basename(args.input_csv)}")
