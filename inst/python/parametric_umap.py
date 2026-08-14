#!/usr/bin/env python3
"""
Phase 2b: Parametric UMAP 降维
将 ESM-2 480-dim embeddings 投影到 2D，训练神经网络映射 f: 480→2
"""
import os, sys, gc, time, random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)

FIT_SAMPLES = 200000  # UMAP 拟合用
BATCH_SIZE = 2048
N_EPOCHS = 50
LEARNING_RATE = 1e-3

print("=" * 60, flush=True)
print("  Phase 2b: Parametric UMAP 480D → 2D", flush=True)
print(f"  Fit on {FIT_SAMPLES:,} samples, {N_EPOCHS} epochs", flush=True)
print("=" * 60, flush=True)

# ============================================================
# Step 1: Load embeddings
# ============================================================
print("\nStep 1: Loading embeddings", flush=True)
embeddings = np.load(os.path.join(OUT_DIR, "phase2_embeddings.npy"))
print(f"Loaded: {embeddings.shape}", flush=True)

n_total, dim = embeddings.shape

# 采样用于 UMAP 拟合
if n_total > FIT_SAMPLES:
    np.random.seed(42)
    fit_idx = np.random.choice(n_total, FIT_SAMPLES, replace=False)
    fit_emb = embeddings[fit_idx]
    print(f"Sampled {FIT_SAMPLES:,} for UMAP fitting", flush=True)
else:
    fit_idx = np.arange(n_total)
    fit_emb = embeddings

# ============================================================
# Step 2: UMAP 拟合
# ============================================================
print("\nStep 2: UMAP fitting", flush=True)
import umap

print("  Learning UMAP manifold...", flush=True)
t0 = time.time()
mapper = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    verbose=True,
    low_memory=False
)
fit_2d = mapper.fit_transform(fit_emb)
t_umap = time.time() - t0
print(f"UMAP fitted in {t_umap:.0f}s ({t_umap/60:.1f} min)", flush=True)

# ============================================================
# Step 3: 训练神经网络映射器 f: 480→2
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

    def forward(self, x):
        return self.net(x)

# 设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}", flush=True)

# 准备训练数据
X_train = torch.tensor(fit_emb, dtype=torch.float32)
y_train = torch.tensor(fit_2d, dtype=torch.float32)

model = UMAPMapper(input_dim=dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

n_batches = (FIT_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Training {N_EPOCHS} epochs, {n_batches} batches/epoch", flush=True)

model.train()
for epoch in range(N_EPOCHS):
    # Shuffle
    perm = torch.randperm(FIT_SAMPLES)
    epoch_loss = 0.0

    for b in range(n_batches):
        i_start = b * BATCH_SIZE
        i_end = min(i_start + BATCH_SIZE, FIT_SAMPLES)
        idx = perm[i_start:i_end]

        x_batch = X_train[idx].to(device)
        y_batch = y_train[idx].to(device)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{N_EPOCHS} | Loss: {epoch_loss/n_batches:.6f}", flush=True)

print(f"Training complete.", flush=True)

# ============================================================
# Step 4: 投影全部序列
# ============================================================
print("\nStep 4: Projecting all sequences", flush=True)

model.eval()
all_2d = np.zeros((n_total, 2), dtype=np.float32)
PROJ_BATCH = 4096
n_proj_batches = (n_total + PROJ_BATCH - 1) // PROJ_BATCH

with torch.no_grad():
    for b in tqdm(range(n_proj_batches), desc="  Projecting"):
        i_start = b * PROJ_BATCH
        i_end = min(i_start + PROJ_BATCH, n_total)
        x = torch.tensor(embeddings[i_start:i_end], dtype=torch.float32).to(device)
        all_2d[i_start:i_end] = model(x).cpu().numpy()

# ============================================================
# Step 5: 保存
# ============================================================
print("\nStep 5: Saving", flush=True)

# 保存 2D 投影
np.save(os.path.join(OUT_DIR, "phase2_umap_2d.npy"), all_2d)

# 保存神经网络映射器
torch.save(model.state_dict(), os.path.join(OUT_DIR, "phase2_mapper.pt"))

# 保存拟合用序列的索引
np.save(os.path.join(OUT_DIR, "phase2_fit_idx.npy"), fit_idx)

# 保存 UMAP mapper
import pickle
with open(os.path.join(OUT_DIR, "phase2_umap_mapper.pkl"), "wb") as f:
    pickle.dump(mapper, f)

print(f"Saved: phase2_umap_2d.npy ({n_total} × 2)", flush=True)
print(f"Saved: phase2_mapper.pt (neural network)", flush=True)
print(f"Saved: phase2_umap_mapper.pkl (UMAP object)", flush=True)

# 打印坐标范围
print(f"\nPC1 range: [{all_2d[:,0].min():.2f}, {all_2d[:,0].max():.2f}]", flush=True)
print(f"PC2 range: [{all_2d[:,1].min():.2f}, {all_2d[:,1].max():.2f}]", flush=True)
print("\nDone!", flush=True)