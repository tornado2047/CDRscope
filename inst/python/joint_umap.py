#!/usr/bin/env python3
"""
TRA+TRB 联合 Embedding + Parametric UMAP + 可视化数据准备
1. 合并 TRA 和 TRB embedding，做联合 UMAP
2. 保存 2D 投影，供 R 可视化
"""
import os, sys, gc, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle

OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)

FIT_SAMPLES = 200000
BATCH_SIZE = 2048
N_EPOCHS = 50
LR = 1e-3

print("=" * 60, flush=True)
print("  TRA+TRB Joint UMAP", flush=True)
print("=" * 60, flush=True)

# ============================================================
# Step 1: Load TRA and TRB embeddings
# ============================================================
print("\nStep 1: Loading embeddings", flush=True)
tra_emb = np.load(os.path.join(OUT_DIR, "tra_embeddings.npy"))
trb_emb = np.load(os.path.join(OUT_DIR, "phase2_embeddings.npy"))
print(f"TRA: {tra_emb.shape}", flush=True)
print(f"TRB: {trb_emb.shape}", flush=True)

# Load sequences
with open(os.path.join(OUT_DIR, "tra_sampled_cdr3s.txt")) as f:
    tra_seqs = [l.strip() for l in f]
with open(os.path.join(OUT_DIR, "phase2_sampled_cdr3s.txt")) as f:
    trb_seqs = [l.strip() for l in f]

# 合并
n_tra, n_trb = len(tra_seqs), len(trb_seqs)
all_emb = np.vstack([tra_emb, trb_emb])
all_labels = np.array(["TRA"] * n_tra + ["TRB"] * n_trb)
all_seqs = tra_seqs + trb_seqs
n_total = len(all_seqs)
print(f"Combined: {n_total:,} samples ({n_tra:,} TRA + {n_trb:,} TRB)", flush=True)

# ============================================================
# Step 2: UMAP fitting (sample 200K)
# ============================================================
print("\nStep 2: UMAP fitting", flush=True)
import umap

if n_total > FIT_SAMPLES:
    np.random.seed(42)
    fit_idx = np.random.choice(n_total, FIT_SAMPLES, replace=False)
    fit_emb = all_emb[fit_idx]
else:
    fit_idx = np.arange(n_total)
    fit_emb = all_emb

print(f"  Fitting on {FIT_SAMPLES:,} samples...", flush=True)
t0 = time.time()
mapper = umap.UMAP(
    n_components=2, n_neighbors=30, min_dist=0.1,
    metric='cosine', random_state=42, verbose=True, low_memory=False
)
fit_2d = mapper.fit_transform(fit_emb)
print(f"UMAP fitted in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# Step 3: Train neural network mapper
# ============================================================
print("\nStep 3: Training neural network mapper", flush=True)

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
    def forward(self, x): return self.net(x)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}", flush=True)

X_train = torch.tensor(fit_emb, dtype=torch.float32)
y_train = torch.tensor(fit_2d, dtype=torch.float32)

model = UMAPMapper().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

n_batches = (FIT_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
model.train()
for epoch in range(N_EPOCHS):
    perm = torch.randperm(FIT_SAMPLES)
    epoch_loss = 0.0
    for b in range(n_batches):
        i_start = b * BATCH_SIZE
        i_end = min(i_start + BATCH_SIZE, FIT_SAMPLES)
        idx = perm[i_start:i_end]
        xb, yb = X_train[idx].to(device), y_train[idx].to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{N_EPOCHS} | Loss: {epoch_loss/n_batches:.6f}", flush=True)

# ============================================================
# Step 4: Project all sequences
# ============================================================
print("\nStep 4: Projecting all sequences", flush=True)
model.eval()
all_2d = np.zeros((n_total, 2), dtype=np.float32)
PROJ_BATCH = 4096
n_proj = (n_total + PROJ_BATCH - 1) // PROJ_BATCH

with torch.no_grad():
    for b in tqdm(range(n_proj), desc="  Projecting"):
        i_start = b * PROJ_BATCH
        i_end = min(i_start + PROJ_BATCH, n_total)
        x = torch.tensor(all_emb[i_start:i_end], dtype=torch.float32).to(device)
        all_2d[i_start:i_end] = model(x).cpu().numpy()

# ============================================================
# Step 5: Save
# ============================================================
print("\nStep 5: Saving", flush=True)
np.save(os.path.join(OUT_DIR, "tra_trb_joint_umap_2d.npy"), all_2d)

# Save as CSV for R
header = "UMAP1,UMAP2,chain,sequence"
data_rows = [f"{all_2d[i,0]:.6f},{all_2d[i,1]:.6f},{all_labels[i]},{all_seqs[i]}"
             for i in range(n_total)]
with open(os.path.join(OUT_DIR, "tra_trb_joint_umap.csv"), "w") as f:
    f.write(header + "\n")
    f.write("\n".join(data_rows))

torch.save(model.state_dict(), os.path.join(OUT_DIR, "tra_trb_joint_mapper.pt"))
with open(os.path.join(OUT_DIR, "tra_trb_joint_umap.pkl"), "wb") as f:
    pickle.dump(mapper, f)

# Also save TRA-only and TRB-only 2D projections
np.save(os.path.join(OUT_DIR, "tra_joint_umap_2d.npy"), all_2d[:n_tra])
np.savetxt(os.path.join(OUT_DIR, "tra_umap_2d.csv"), all_2d[:n_tra],
           delimiter=',', fmt='%.6f', header='UMAP1,UMAP2', comments='')

print(f"UMAP1 range: [{all_2d[:,0].min():.2f}, {all_2d[:,0].max():.2f}]", flush=True)
print(f"UMAP2 range: [{all_2d[:,1].min():.2f}, {all_2d[:,1].max():.2f}]", flush=True)
print(f"TRA mean UMAP1: {all_2d[:n_tra,0].mean():.3f}, TRB: {all_2d[n_tra:,0].mean():.3f}", flush=True)
print(f"TRA mean UMAP2: {all_2d[:n_tra,1].mean():.3f}, TRB: {all_2d[n_tra:,1].mean():.3f}", flush=True)
print("\nDone!", flush=True)