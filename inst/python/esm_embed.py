#!/usr/bin/env python3
"""
Phase 2: ESM-2 Embedding Extraction via HuggingFace Transformers
使用 transformers 库加载 ESM-2，提取 CDR3 序列的 embedding
"""
import os, sys, gc, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================
# Config
# ============================================================
DATA_DIR = "RA_data"
CONTROL_DIR = os.path.join(DATA_DIR, "RA_Control_Files")
PATIENT_DIR = os.path.join(DATA_DIR, "RA_Patient_Files")
OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
EMBED_DIM = 480
MAX_SAMPLES = 500000
BATCH_SIZE = 256

print("=" * 60, flush=True)
print(f"  Phase 2: ESM-2 Embedding (via Transformers)", flush=True)
print(f"  Model: {MODEL_NAME} ({EMBED_DIM}-dim)", flush=True)
print(f"  Max samples: {MAX_SAMPLES:,}", flush=True)
print("=" * 60, flush=True)

# ============================================================
# Step 1: Load unique CDR3 sequences
# ============================================================
print("\nStep 1: Loading unique CDR3 sequences", flush=True)

def load_all_cdr3s(data_dir, label):
    all_seqs = set()
    files = [f for f in os.listdir(data_dir) if f.endswith("_r__TRB.csv")]
    for f in tqdm(files, desc=f"  {label}"):
        df = pd.read_csv(os.path.join(data_dir, f), usecols=["junction_aa"])
        all_seqs.update(df["junction_aa"].dropna().unique())
    return list(all_seqs)

control_seqs = load_all_cdr3s(CONTROL_DIR, "Control")
patient_seqs = load_all_cdr3s(PATIENT_DIR, "Patient")

all_unique = list(set(control_seqs + patient_seqs))
n_all = len(all_unique)
print(f"Total unique CDR3: {n_all:,}", flush=True)

if n_all > MAX_SAMPLES:
    random.seed(42)
    sampled = random.sample(all_unique, MAX_SAMPLES)
else:
    sampled = all_unique

# 过滤非标准氨基酸
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
sampled = [s for s in sampled if all(c in valid_aa for c in s)]
n_final = len(sampled)
print(f"After filtering: {n_final:,} sequences", flush=True)

# 保存序列列表
with open(os.path.join(OUT_DIR, "phase2_sampled_cdr3s.txt"), "w") as f:
    for s in sampled:
        f.write(s + "\n")

# ============================================================
# Step 2: Load ESM-2 tokenizer & model
# ============================================================
print("\nStep 2: Loading ESM-2 model via Transformers", flush=True)

# 检测设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}", flush=True)

print(f"Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model...", flush=True)
model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()
print("Model loaded.", flush=True)

# ============================================================
# Step 3: Extract embeddings
# ============================================================
print("\nStep 3: Extracting embeddings", flush=True)

embeddings = np.zeros((n_final, EMBED_DIM), dtype=np.float32)
n_batches = (n_final + BATCH_SIZE - 1) // BATCH_SIZE

start_time = time.time()
for b in range(n_batches):
    i_start = b * BATCH_SIZE
    i_end = min(i_start + BATCH_SIZE, n_final)
    batch_seqs = sampled[i_start:i_end]

    # 将序列转为空格分隔（transformers ESM tokenizer 需要）
    spaced_seqs = [" ".join(list(seq)) for seq in batch_seqs]

    inputs = tokenizer(spaced_seqs, return_tensors="pt", padding=True,
                       truncation=True, max_length=50)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 取 last_hidden_state，对每个序列取非 padding token 的均值
    hidden = outputs.last_hidden_state  # (batch, seq_len, dim)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)

    masked_hidden = hidden * attention_mask
    sum_hidden = masked_hidden.sum(dim=1)  # (batch, dim)
    count = attention_mask.sum(dim=1)       # (batch, 1)
    seq_embeddings = (sum_hidden / count).cpu().numpy()

    embeddings[i_start:i_end] = seq_embeddings

    if (b + 1) % 25 == 0 or b == n_batches - 1:
        elapsed = time.time() - start_time
        rate = (i_end) / elapsed if elapsed > 0 else 0
        eta = (n_final - i_end) / rate if rate > 0 else 0
        print(f"  Batch {b+1}/{n_batches} | {i_end:,}/{n_final:,} seqs "
              f"| {rate:.0f} seq/s | ETA {eta:.0f}s", flush=True)

    # 定期保存
    if (b + 1) % 500 == 0:
        np.save(os.path.join(OUT_DIR, "phase2_embeddings.npy"), embeddings[:i_end])
        print(f"  [checkpoint] Saved {i_end:,} embeddings", flush=True)

    gc.collect()

total_time = time.time() - start_time
print(f"\nDone: {n_final:,} sequences in {total_time:.0f}s "
      f"({n_final/total_time:.0f} seq/s)", flush=True)

# ============================================================
# Step 4: Save
# ============================================================
print("\nStep 4: Saving", flush=True)
np.save(os.path.join(OUT_DIR, "phase2_embeddings.npy"), embeddings)
print(f"Saved: phase2_embeddings.npy ({n_final} × {EMBED_DIM})", flush=True)
print("Done!", flush=True)