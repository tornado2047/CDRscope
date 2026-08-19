#!/usr/bin/env python3
"""
CDRscope Reference Map Builder
================================
Builds a fixed, reusable 2D reference map from a reference CDR3 dataset.

The reference map is a trained neural network f: R^480 → R^2 that
approximates the UMAP manifold learned from a reference dataset.
Once built, any new CDR3 sequence can be projected into the SAME
coordinate space by:
  1. Computing its ESM-2 embedding (480-dim)
  2. Passing it through the saved network (forward pass)

This eliminates the need to re-fit UMAP for each new project.

Usage:
  # Build from existing embeddings (fast — reuses precomputed data)
  python build_reference_map.py \
    --mode joint \
    --tra-emb docs/tra_embeddings.npy \
    --trb-emb docs/phase2_embeddings.npy \
    --tra-seqs docs/tra_sampled_cdr3s.txt \
    --trb-seqs docs/phase2_sampled_cdr3s.txt \
    --output reference_map/

  # Build from raw CDR3 sequences (slow — computes ESM-2 from scratch)
  python build_reference_map.py \
    --mode joint \
    --tra-csv data/tra_sequences.csv \
    --trb-csv data/trb_sequences.csv \
    --output reference_map/

Reference map package structure:
  reference_map/
    ├── ref_mapper.pt          # NN weights (655 KB)
    ├── ref_coords.npy         # 2D coordinates for all ref sequences
    ├── ref_metadata.csv       # Sequence, chain, group, V/J, properties
    ├── ref_config.json        # Model + UMAP parameters
    └── ref_summary.json       # Statistics and provenance
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# ─── Model definition (must match parametric_umap.py) ───
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

# ─── Amino acid properties for metadata ───
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

# ─── ESM-2 embedding (lazy import to avoid heavy deps at parse time) ───
def compute_esm2_embeddings(sequences, model_name="facebook/esm2_t12_35M_UR50D",
                             batch_size=256, device=None):
    """Compute ESM-2 mean-pooled embeddings for a list of CDR3 sequences."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    print(f"Loading ESM-2 model: {model_name}", flush=True)
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

# ─── Main builder ───
def build_reference_map(args):
    t_start = time.time()
    os.makedirs(args.output, exist_ok=True)

    # ── 1. Load or compute embeddings ──
    all_emb = []
    all_seqs = []
    all_chains = []

    for chain_tag, emb_path, seqs_path, csv_path in [
        ("TRA", args.tra_emb, args.tra_seqs, args.tra_csv),
        ("TRB", args.trb_emb, args.trb_seqs, args.trb_csv),
    ]:
        if emb_path and os.path.exists(emb_path):
            print(f"\nLoading {chain_tag} embeddings: {emb_path}", flush=True)
            emb = np.load(emb_path)
            with open(seqs_path) as f:
                seqs = [l.strip() for l in f]
            all_emb.append(emb)
            all_seqs.extend(seqs)
            all_chains.extend([chain_tag] * len(seqs))
            print(f"  {chain_tag}: {emb.shape[0]:,} sequences, {emb.shape[1]} dim", flush=True)
        elif csv_path and os.path.exists(csv_path):
            print(f"\nComputing {chain_tag} ESM-2 embeddings from: {csv_path}", flush=True)
            df = pd.read_csv(csv_path)
            seq_col = 'junction_aa' if 'junction_aa' in df.columns else df.columns[0]
            seqs = df[seq_col].dropna().unique().tolist()
            emb = compute_esm2_embeddings(seqs, device=args.device)
            all_emb.append(emb)
            all_seqs.extend(seqs)
            all_chains.extend([chain_tag] * len(seqs))
        else:
            print(f"  Skipping {chain_tag} (no data provided)")

    if not all_emb:
        print("ERROR: No embedding data found. Provide --tra-emb/--trb-emb or --tra-csv/--trb-csv")
        sys.exit(1)

    embeddings = np.vstack(all_emb)
    all_chains = np.array(all_chains)
    n_total, dim = embeddings.shape
    print(f"\nTotal: {n_total:,} sequences ({dim}-dim embeddings)", flush=True)

    # ── 2. UMAP fitting on subsample ──
    FIT_SAMPLES = min(args.fit_samples, n_total)
    BATCH_SIZE = 2048
    N_EPOCHS = args.epochs
    LR = 1e-3

    np.random.seed(42)
    if n_total > FIT_SAMPLES:
        fit_idx = np.random.choice(n_total, FIT_SAMPLES, replace=False)
    else:
        fit_idx = np.arange(n_total)
    fit_emb = embeddings[fit_idx]
    print(f"\nUMAP fitting on {FIT_SAMPLES:,} samples...", flush=True)

    import umap
    mapper = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=42,
        verbose=True,
        low_memory=False
    )
    fit_2d = mapper.fit_transform(fit_emb)
    print(f"UMAP fitted in {(time.time()-t_start):.0f}s", flush=True)

    # ── 3. Train neural network projector ──
    print(f"\nTraining NN projector ({N_EPOCHS} epochs)...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")

    X_train = torch.tensor(fit_emb, dtype=torch.float32)
    y_train = torch.tensor(fit_2d, dtype=torch.float32)
    model = UMAPMapper(input_dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    n_batches = (FIT_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(FIT_SAMPLES)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            x_batch = X_train[idx].to(device)
            y_batch = y_train[idx].to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{N_EPOCHS} | Loss: {epoch_loss/n_batches:.6f}", flush=True)

    # ── 4. Project all sequences ──
    print("\nProjecting all sequences...", flush=True)
    model.eval()
    all_2d = np.zeros((n_total, 2), dtype=np.float32)
    PROJ_BATCH = 4096
    with torch.no_grad():
        for b in tqdm(range(0, n_total, PROJ_BATCH), desc="  Project"):
            end = min(b+PROJ_BATCH, n_total)
            x = torch.tensor(embeddings[b:end], dtype=torch.float32).to(device)
            all_2d[b:end] = model(x).cpu().numpy()

    # ── 5. Build metadata ──
    print("\nBuilding metadata...", flush=True)
    meta_rows = []
    for i, (seq, chain) in enumerate(zip(all_seqs, all_chains)):
        L, charge, hydro, arom = compute_properties(seq)
        meta_rows.append({
            'idx': i,
            'sequence': seq,
            'chain': chain,
            'length': L,
            'net_charge': charge,
            'hydrophobicity': round(hydro, 3),
            'aromatic_frac': round(arom, 3),
            'umap1': float(all_2d[i, 0]),
            'umap2': float(all_2d[i, 1]),
        })
    metadata = pd.DataFrame(meta_rows)

    # ── 6. Save reference map package ──
    print("\nSaving reference map...", flush=True)
    torch.save(model.state_dict(), os.path.join(args.output, "ref_mapper.pt"))
    np.save(os.path.join(args.output, "ref_coords.npy"), all_2d)
    metadata.to_csv(os.path.join(args.output, "ref_metadata.csv"), index=False)

    config = {
        "esm2_model": "facebook/esm2_t12_35M_UR50D",
        "embedding_dim": dim,
        "umap_params": {
            "n_components": 2,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "random_state": 42,
        },
        "nn_arch": {
            "input_dim": dim,
            "hidden_dims": [256, 128, 64],
            "output_dim": 2,
            "activation": "relu",
            "dropout": 0.1,
            "batch_norm": True,
        },
        "training": {
            "fit_samples": FIT_SAMPLES,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "loss": "MSE",
        },
        "n_sequences": n_total,
        "chains": list(set(all_chains)),
        "random_state": 42,
        "version": "1.0.0",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output, "ref_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    summary = {
        "n_sequences": n_total,
        "n_tra": int((all_chains == 'TRA').sum()),
        "n_trb": int((all_chains == 'TRB').sum()),
        "embedding_dim": dim,
        "umap1_range": [float(all_2d[:,0].min()), float(all_2d[:,0].max())],
        "umap2_range": [float(all_2d[:,1].min()), float(all_2d[:,1].max())],
        "build_time_sec": round(time.time() - t_start, 1),
        "package_files": {
            "ref_mapper.pt": "NN weights (480→2)",
            "ref_coords.npy": "2D coordinates for all ref sequences",
            "ref_metadata.csv": "Per-sequence metadata with properties",
            "ref_config.json": "Model + UMAP + NN configuration",
        },
    }
    with open(os.path.join(args.output, "ref_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Reference map built successfully!")
    print(f"  Sequences: {n_total:,}")
    print(f"  UMAP1 range: [{all_2d[:,0].min():.2f}, {all_2d[:,0].max():.2f}]")
    print(f"  UMAP2 range: [{all_2d[:,1].min():.2f}, {all_2d[:,1].max():.2f}]")
    print(f"  Output: {args.output}/")
    print(f"  Package size: ~{os.path.getsize(os.path.join(args.output, 'ref_mapper.pt'))/1024:.0f} KB (NN) + "
          f"{os.path.getsize(os.path.join(args.output, 'ref_coords.npy'))/1e6:.1f} MB (coords) + "
          f"{os.path.getsize(os.path.join(args.output, 'ref_metadata.csv'))/1e6:.1f} MB (metadata)")
    print(f"  Build time: {time.time()-t_start:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CDRscope reference UMAP map")
    parser.add_argument("--mode", choices=["tra", "trb", "joint"], default="joint",
                       help="Chain mode for reference map")
    parser.add_argument("--tra-emb", default=None, help="TRA embeddings .npy file")
    parser.add_argument("--trb-emb", default=None, help="TRB embeddings .npy file")
    parser.add_argument("--tra-seqs", default=None, help="TRA sequences .txt file")
    parser.add_argument("--trb-seqs", default=None, help="TRB sequences .txt file")
    parser.add_argument("--tra-csv", default=None, help="TRA sequences CSV (will compute ESM-2)")
    parser.add_argument("--trb-csv", default=None, help="TRB sequences CSV (will compute ESM-2)")
    parser.add_argument("--output", default="reference_map/", help="Output directory")
    parser.add_argument("--fit-samples", type=int, default=200000, help="UMAP fit subset size")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--metric", default="cosine", help="UMAP metric")
    parser.add_argument("--epochs", type=int, default=50, help="NN training epochs")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()
    build_reference_map(args)
