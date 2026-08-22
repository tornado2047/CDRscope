#!/usr/bin/env python3
"""
TCR Reference Quantization Module
===================================
Builds a fixed-dimensional "TCR transcriptome" from a reference pool of CDR3 sequences.

Core idea:
  1. Collect all unique CDR3s from multiple datasets → reference pool (size n)
  2. Embed each sequence with ESM-2 → n × k matrix (k = embedding dim)
  3. Cluster with k-means to find m prototype centroids (m << n)
  4. m becomes the "standard TCR gene count" (analogous to ~30k genes in RNA-seq)
  5. Any sample is projected to m dimensions: entry i = sum of counts of all
     sequences whose nearest centroid is prototype i

This solves the fundamental problem: different samples have wildly different
CDR3 sequences and sequence counts, making direct comparison impossible.

Tier 1 → Tier 2 integration:
  Tier 1 trains on per-sample data, outputs model checkpoint
  Tier 2 uses the reference panel to convert pool-only data to m-dim vectors,
  then applies the Tier 1 model for disease scoring.

Saturation hypothesis:
  As the reference pool grows, the optimal m (at fixed variance explained)
  grows but eventually saturates — the TCR sequence space is finite and structured.
"""
import os, sys, json, glob, pickle, time, warnings, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "tcr_reference_panel")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"
EMBED_DIM = 480

# =========================================================================
# Step 1: Build reference sequence pool
# =========================================================================

def load_sequences_from_file(filepath, seq_col, count_col=None):
    """Load unique CDR3 sequences and total counts from a single file."""
    try:
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, sep='\t' if '.tsv' in filepath or '.txt' in filepath else ',',
                             compression='gzip')
        elif filepath.endswith('.tsv') or filepath.endswith('.txt'):
            df = pd.read_csv(filepath, sep='\t')
        else:
            df = pd.read_csv(filepath)

        if seq_col not in df.columns:
            return Counter()

        seqs = df[seq_col].dropna().values
        if count_col and count_col in df.columns:
            counts = df[count_col].fillna(1).values
            counts = np.maximum(counts, 1).astype(int)
        else:
            counts = np.ones(len(seqs), dtype=int)

        seq_counter = Counter()
        for s, c in zip(seqs, counts):
            s = str(s).strip()
            if len(s) >= 8 and all(aa in STANDARD_AA for aa in s):
                seq_counter[s] += int(c)
        return seq_counter
    except Exception as e:
        print(f"  Warning: failed to load {filepath}: {e}", file=sys.stderr)
        return Counter()


def build_reference_pool(max_seqs=None, datasets=None):
    """Build the reference sequence pool from all available datasets.

    Args:
        max_seqs: Maximum number of unique sequences to keep (subsample if exceeded)
        datasets: List of dataset names to include, or None for all

    Returns:
        Counter of {sequence: total_count}
    """
    all_seqs = Counter()
    dataset_sources = {
        'RA': {
            'dirs': [
                os.path.join(BASE, 'CDRscope-analysis/RA_data/RA_Control_Files'),
                os.path.join(BASE, 'CDRscope-analysis/RA_data/RA_Patient_Files'),
            ],
            'pattern': '*_TRB.csv',
            'seq_col': 'junction_aa',
            'count_col': 'duplicate_count',
        },
        'CMV': {
            'dirs': [
                os.path.join(BASE, 'emerson_cmv_data/PreprocessedDataset/Train'),
                os.path.join(BASE, 'emerson_cmv_data/PreprocessedDataset/Test'),
            ],
            'pattern': '*.csv',
            'seq_col': 'amino_acid',
            'count_col': None,
        },
        'MS': {
            'dirs': [os.path.join(BASE, 'ms_tcr_data')],
            'pattern': '*.txt.gz',
            'seq_col': 'CDR3 amino acid sequence',
            'count_col': 'Count',
        },
        'SLE': {
            'dirs': [os.path.join(BASE, 'pird_sle_tcr_data')],
            'pattern': '*.csv',
            'seq_col': 'CDR3AA',
            'count_col': None,
        },
        'VDJdb': {
            'dirs': [os.path.join(BASE, 'vdjdb_data/disease_specific')],
            'pattern': '*.tsv',
            'seq_col': 'cdr3',
            'count_col': None,
        },
    }

    if datasets is None:
        datasets = list(dataset_sources.keys())

    for ds_name in datasets:
        if ds_name not in dataset_sources:
            continue
        src = dataset_sources[ds_name]
        ds_counter = Counter()
        n_files = 0

        for d in src['dirs']:
            if not os.path.exists(d):
                continue
            files = sorted(glob.glob(os.path.join(d, src['pattern'])))
            for f in files:
                fc = load_sequences_from_file(f, src['seq_col'], src['count_col'])
                ds_counter.update(fc)
                n_files += 1
                if n_files % 50 == 0:
                    print(f"  {ds_name}: {n_files} files, {len(ds_counter):,} unique seqs so far")

        print(f"  {ds_name}: {len(ds_counter):,} unique sequences ({n_files} files)")
        all_seqs.update(ds_counter)

    total = len(all_seqs)
    print(f"\n  Total reference pool: {total:,} unique CDR3 sequences")

    if max_seqs and total > max_seqs:
        # Subsample: keep the most frequent sequences
        top = all_seqs.most_common(max_seqs)
        all_seqs = Counter(dict(top))
        print(f"  Subsampled to top {max_seqs:,} by frequency")

    return all_seqs


# =========================================================================
# Step 2: ESM-2 Embedding
# =========================================================================

def compute_esm2_embeddings(sequences, model_name=ESM2_MODEL, device='auto', batch_size=256):
    """Compute ESM-2 embeddings for a list of sequences.

    Args:
        sequences: list of CDR3 amino acid strings
        model_name: HuggingFace model name
        device: 'auto', 'cpu', 'cuda', 'mps'
        batch_size: inference batch size

    Returns:
        numpy array of shape (len(sequences), EMBED_DIM)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    print(f"  Loading ESM-2 model ({model_name}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    n = len(sequences)
    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
    n_batches = (n + batch_size - 1) // batch_size

    start = time.time()
    for b in range(n_batches):
        i0 = b * batch_size
        i1 = min(i0 + batch_size, n)
        batch_seqs = sequences[i0:i1]
        spaced = [" ".join(list(s)) for s in batch_seqs]

        inputs = tokenizer(spaced, return_tensors='pt', padding=True,
                           truncation=True, max_length=50)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        masked = hidden * mask
        summed = masked.sum(dim=1).cpu().numpy()
        counts = mask.sum(dim=1).cpu().numpy()
        embeddings[i0:i1] = summed / counts

        if (b + 1) % 50 == 0 or b == n_batches - 1:
            elapsed = time.time() - start
            rate = i1 / elapsed if elapsed > 0 else 0
            eta = (n - i1) / rate if rate > 0 else 0
            print(f"    Batch {b+1}/{n_batches} | {i1:,}/{n:,} | {rate:.0f} seq/s | ETA {eta:.0f}s")

    print(f"  Embedding complete: {n:,} sequences in {time.time()-start:.0f}s")
    return embeddings


# =========================================================================
# Step 3: Reference Panel Training (k-means quantization)
# =========================================================================

def train_reference_panel(embeddings, n_prototypes, random_state=42):
    """Train a reference panel using MiniBatchKMeans.

    Finds m = n_prototypes prototype vectors in ESM-2 embedding space
    that minimize within-cluster variance.

    Args:
        embeddings: (n, k) numpy array of ESM-2 embeddings
        n_prototypes: number of prototype vectors (m)
        random_state: random seed

    Returns:
        dict with centroids, labels, inertia, variance_explained
    """
    n = embeddings.shape[0]
    print(f"  Training reference panel: {n:,} sequences → {n_prototypes} prototypes")

    kmeans = MiniBatchKMeans(
        n_clusters=n_prototypes,
        batch_size=min(10000, n),
        n_init=3,
        max_iter=100,
        random_state=random_state,
        verbose=0
    )
    labels = kmeans.fit_predict(embeddings)
    inertia = kmeans.inertia_

    # Total variance (for variance explained calculation)
    total_var = np.sum(np.var(embeddings, axis=0)) * n
    variance_explained = 1 - inertia / total_var

    # Cluster sizes
    cluster_sizes = np.bincount(labels, minlength=n_prototypes)

    print(f"  Inertia: {inertia:,.0f}")
    print(f"  Variance explained: {variance_explained:.4f} ({variance_explained*100:.1f}%)")
    print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"median={np.median(cluster_sizes):.0f}")

    return {
        'centroids': kmeans.cluster_centers_.astype(np.float32),
        'labels': labels,
        'inertia': float(inertia),
        'variance_explained': float(variance_explained),
        'cluster_sizes': cluster_sizes.tolist(),
        'n_prototypes': n_prototypes,
        'n_sequences': n,
        'kmeans': kmeans,
    }


def scan_prototype_count(embeddings, m_values, random_state=42):
    """Train panels at multiple m values and measure variance explained.

    Used to find the elbow / optimal m value.

    Args:
        embeddings: (n, k) array
        m_values: list of m values to test
        random_state: random seed

    Returns:
        list of dicts with m, variance_explained, inertia for each m
    """
    results = []
    total_var = np.sum(np.var(embeddings, axis=0)) * embeddings.shape[0]

    for m in m_values:
        print(f"\n  m = {m}")
        panel = train_reference_panel(embeddings, m, random_state=random_state)
        results.append({
            'm': m,
            'variance_explained': panel['variance_explained'],
            'inertia': panel['inertia'],
        })

    return results


# =========================================================================
# Step 4: Saturation Analysis
# =========================================================================

def saturation_analysis(embeddings, m_target_var_explained=0.80,
                         subsample_fracs=None, random_state=42):
    """Test if m (at fixed variance explained) saturates as pool grows.

    Subsamples the reference pool at different sizes, finds the m needed
    to reach the target variance explained, and plots m vs pool size.

    If m saturates (grows sub-linearly and plateaus), the reference panel
    approach is theoretically sound.

    Args:
        embeddings: full (n, k) embedding matrix
        m_target_var_explained: target variance explained (e.g., 0.80)
        subsample_fracs: list of fractions to subsample at
        random_state: random seed

    Returns:
        list of dicts with pool_size, m_needed, var_explained_by_m
    """
    n_total = embeddings.shape[0]
    if subsample_fracs is None:
        if n_total >= 200000:
            subsample_fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
        else:
            subsample_fracs = [0.0625, 0.125, 0.25, 0.5, 1.0]

    # Test m values roughly proportional to pool size
    m_candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    results = []

    for frac in subsample_fracs:
        pool_size = int(n_total * frac)
        print(f"\n{'='*50}")
        print(f"  Saturation test: pool_size = {pool_size:,} (frac={frac})")

        # Subsample
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_total, size=pool_size, replace=False)
        sub_emb = embeddings[idx]

        # Scan m values
        scan_results = []
        # Select m candidates appropriate for this pool size
        max_m = min(10000, max(50, pool_size // 10))
        test_ms = [m for m in m_candidates if m <= max_m]
        if pool_size >= 10000 and max_m not in test_ms:
            test_ms.append(max_m)

        for m in test_ms:
            panel = train_reference_panel(sub_emb, m, random_state=random_state)
            scan_results.append({
                'm': m,
                'variance_explained': panel['variance_explained'],
            })

        # Interpolate to find m needed for target variance explained
        ms = np.array([r['m'] for r in scan_results])
        ves = np.array([r['variance_explained'] for r in scan_results])

        if np.max(ves) >= m_target_var_explained:
            # Linear interpolation to find m at target
            idx_above = np.where(ves >= m_target_var_explained)[0][0]
            if idx_above > 0:
                # Interpolate between the points below and above target
                m_lo, ve_lo = ms[idx_above - 1], ves[idx_above - 1]
                m_hi, ve_hi = ms[idx_above], ves[idx_above]
                frac_interp = (m_target_var_explained - ve_lo) / (ve_hi - ve_lo)
                m_needed = m_lo + frac_interp * (m_hi - m_lo)
            else:
                m_needed = float(ms[0])
        else:
            m_needed = float(ms[-1])  # can't reach target with tested m values

        results.append({
            'pool_size': pool_size,
            'fraction': frac,
            'm_needed': round(m_needed),
            'target_var_explained': m_target_var_explained,
            'scan_results': scan_results,
        })

        print(f"  m needed for {m_target_var_explained*100:.0f}% var explained: ~{round(m_needed)}")

    return results


# =========================================================================
# Step 5: Sample Projection
# =========================================================================

def project_sequence_to_panel(seq_embedding, centroids):
    """Project a single sequence embedding to its nearest prototype.

    Args:
        seq_embedding: (k,) array
        centroids: (m, k) array

    Returns:
        index of nearest prototype
    """
    dists = np.sum((centroids - seq_embedding) ** 2, axis=1)
    return np.argmin(dists)


def project_sample_to_panel(sequences, counts, centroids):
    """Project a full sample (list of sequences + counts) to m-dimensional vector.

    This is the core operation that converts variable-length CDR3 repertoires
    into fixed-dimensional "TCR gene expression" profiles.

    Args:
        sequences: list of CDR3 strings
        counts: list/array of read counts (same length as sequences)
        centroids: (m, k) array of prototype vectors

    Returns:
        vector: (m,) array where entry i = total count of sequences
                assigned to prototype i
        n_mapped: number of sequences successfully assigned
    """
    m = centroids.shape[0]
    vector = np.zeros(m, dtype=np.float32)

    if len(sequences) == 0:
        return vector, 0

    # Embed all sequences in the sample
    # For efficiency, we batch this
    embeddings = compute_esm2_embeddings(sequences)

    # Assign each to nearest centroid
    # Vectorized: compute all pairwise distances at once
    # (n_seq, 1, k) - (1, m, k) → (n_seq, m, k) → sum over k → (n_seq, m)
    diff = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)  # (n_seq, m)
    assignments = np.argmin(dists, axis=1)  # (n_seq,)

    # Aggregate counts
    counts_arr = np.array(counts, dtype=np.float32)
    for i in range(len(assignments)):
        vector[assignments[i]] += counts_arr[i]

    return vector, len(sequences)


def assign_to_centroids(embeddings, centroids, batch_size=10000):
    """Assign sequences to nearest centroid in batches (memory-efficient).

    Args:
        embeddings: (n_seq, k) array
        centroids: (m, k) array
        batch_size: batch size for distance computation

    Returns:
        assignments: (n_seq,) array of centroid indices
    """
    n = embeddings.shape[0]
    m = centroids.shape[0]
    assignments = np.zeros(n, dtype=np.int32)

    for i in range(0, n, batch_size):
        batch = embeddings[i:i+batch_size]
        # (batch, 1, k) - (1, m, k) → (batch, m, k) → sum → (batch, m)
        diff = batch[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists = np.sum(diff ** 2, axis=2)
        assignments[i:i+batch_size] = np.argmin(dists, axis=1)

    return assignments


def project_sample_fast(embeddings, counts, centroids):
    """Fast projection when embeddings are already computed.

    Args:
        embeddings: (n_seq, k) array
        counts: (n_seq,) array of counts
        centroids: (m, k) array

    Returns:
        (m,) count vector
    """
    m = centroids.shape[0]
    assignments = assign_to_centroids(embeddings, centroids)

    vector = np.zeros(m, dtype=np.float32)
    counts_arr = np.array(counts, dtype=np.float32)
    for i in range(len(assignments)):
        vector[assignments[i]] += counts_arr[i]

    return vector


# =========================================================================
# Step 6: Reference Panel Checkpoint I/O
# =========================================================================

def save_reference_panel(panel, sequences, embeddings, filename='reference_panel.pkl'):
    """Save reference panel to disk for Tier 2 use.

    Args:
        panel: dict from train_reference_panel()
        sequences: list of reference CDR3 sequences
        embeddings: (n, k) array
        filename: output filename
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    ckpt = {
        'centroids': panel['centroids'],
        'n_prototypes': panel['n_prototypes'],
        'n_sequences': panel['n_sequences'],
        'variance_explained': panel['variance_explained'],
        'cluster_sizes': panel['cluster_sizes'],
        'sequences': sequences,
        'embeddings': embeddings,
        'esm2_model': ESM2_MODEL,
        'embed_dim': EMBED_DIM,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(ckpt, f)
    print(f"  Reference panel saved: {filepath}")
    print(f"    Prototypes: {panel['n_prototypes']}")
    print(f"    Reference sequences: {panel['n_sequences']:,}")
    print(f"    Variance explained: {panel['variance_explained']:.4f}")
    return filepath


def load_reference_panel(filepath):
    """Load reference panel from disk.

    Returns:
        dict with centroids, n_prototypes, etc.
    """
    with open(filepath, 'rb') as f:
        ckpt = pickle.load(f)
    return ckpt


# =========================================================================
# Step 7: Classification Validation
# =========================================================================

def validate_with_ra(samples, centroids, n_folds=5):
    """Validate reference panel features on RA per-sample data.

    Projects each sample to m-dimensional TCR gene expression vector,
    then runs 5-fold CV Random Forest classification.
    Compares with the original CDRscope 65-dim feature set.

    Args:
        samples: list of sample dicts (with 'df', 'label', 'sample_id')
        centroids: (m, k) reference panel centroids
        n_folds: cross-validation folds

    Returns:
        dict with AUC and comparison results
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    print(f"\n  Validating on RA-TRB ({len(samples)} samples)")
    print(f"  Reference panel: {centroids.shape[0]} prototypes × {centroids.shape[1]} dims")

    m = centroids.shape[0]
    n = len(samples)

    # Step 1: Extract all unique sequences across all samples
    all_seqs = set()
    for s in samples:
        all_seqs.update(s['df']['junction_aa'].values)
    all_seqs = list(all_seqs)
    print(f"  Total unique sequences across samples: {len(all_seqs):,}")

    # Step 2: Embed all unique sequences
    print(f"  Embedding all unique sequences...")
    seq_embeddings = compute_esm2_embeddings(all_seqs)
    seq_to_idx = {s: i for i, s in enumerate(all_seqs)}

    # Step 3: Assign each unique sequence to nearest centroid (batched)
    print(f"  Assigning sequences to prototypes...")
    seq_assignments = assign_to_centroids(seq_embeddings, centroids)

    # Step 4: Project each sample to m-dimensional vector
    X_quant = np.zeros((n, m), dtype=np.float32)
    y = np.zeros(n, dtype=int)
    sample_ids = []

    for i, s in enumerate(samples):
        df = s['df']
        seqs = df['junction_aa'].values
        counts = df['duplicate_count'].values if 'duplicate_count' in df.columns else np.ones(len(seqs))
        y[i] = s['label']
        sample_ids.append(s['sample_id'])

        for seq, cnt in zip(seqs, counts):
            idx = seq_to_idx.get(seq)
            if idx is not None:
                proto = seq_assignments[idx]
                X_quant[i, proto] += cnt

    # Normalize to frequencies (like TPM in RNA-seq)
    row_sums = X_quant.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_quant_norm = X_quant / row_sums

    print(f"  Quantized feature matrix: {X_quant_norm.shape}")
    print(f"  Sparsity: {np.mean(X_quant_norm == 0):.1%} zeros")

    # Step 5: Run 5-fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_probs = []
    all_true = []

    for train_idx, test_idx in skf.split(X_quant_norm, y):
        rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(X_quant_norm[train_idx], y[train_idx])
        prob = rf.predict_proba(X_quant_norm[test_idx])[:, 1]
        all_probs.extend(prob)
        all_true.extend(y[test_idx].tolist())

    auc_quant = roc_auc_score(all_true, all_probs)
    print(f"  Quantized features AUC: {auc_quant:.4f} (m={m})")

    return {
        'auc': auc_quant,
        'm': m,
        'n_samples': n,
        'sparsity': float(np.mean(X_quant_norm == 0)),
        'sample_ids': sample_ids,
        'predictions': list(zip(sample_ids, all_true, all_probs)),
        'X_quant': X_quant_norm,
        'y': y,
    }


# =========================================================================
# Main: End-to-end reference panel construction
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TCR Reference Quantization')
    parser.add_argument('--mode', choices=['build', 'saturate', 'validate', 'all'],
                        default='all', help='Operation mode')
    parser.add_argument('--max-seqs', type=int, default=500000,
                        help='Max unique sequences in reference pool')
    parser.add_argument('--m-prototypes', type=int, default=500,
                        help='Number of prototype vectors (m)')
    parser.add_argument('--target-var', type=float, default=0.80,
                        help='Target variance explained for saturation analysis')
    parser.add_argument('--datasets', type=str, default='RA,CMV,MS,SLE,VDJdb',
                        help='Comma-separated dataset names')
    parser.add_argument('--device', type=str, default='auto',
                        help='Compute device for ESM-2')
    args = parser.parse_args()

    datasets = args.datasets.split(',') if args.datasets else None

    print("=" * 70)
    print("  TCR Reference Quantization — Building TCR Transcriptome")
    print("=" * 70)

    # Step 1: Build reference pool
    if args.mode in ('build', 'all', 'saturate', 'validate'):
        print(f"\n[1/4] Building reference pool (max {args.max_seqs:,} seqs)...")
        ref_pool = build_reference_pool(max_seqs=args.max_seqs, datasets=datasets)
        ref_sequences = list(ref_pool.keys())
        print(f"  Reference pool: {len(ref_sequences):,} unique sequences")

    # Step 2: ESM-2 embedding
    if args.mode in ('build', 'all', 'saturate', 'validate'):
        print(f"\n[2/4] Computing ESM-2 embeddings...")
        embeddings = compute_esm2_embeddings(ref_sequences, device=args.device)
        print(f"  Embedding matrix: {embeddings.shape}")

    # Step 3: Train reference panel
    if args.mode in ('build', 'all'):
        print(f"\n[3/4] Training reference panel (m={args.m_prototypes})...")
        panel = train_reference_panel(embeddings, args.m_prototypes)
        panel_path = save_reference_panel(
            panel, ref_sequences, embeddings,
            f'reference_panel_m{args.m_prototypes}.pkl'
        )

    # Step 4: Saturation analysis
    if args.mode in ('saturate', 'all'):
        print(f"\n[3.5/4] Saturation analysis (target={args.target_var*100:.0f}% var)...")
        sat_results = saturation_analysis(
            embeddings,
            m_target_var_explained=args.target_var,
        )
        sat_path = os.path.join(OUTPUT_DIR, 'saturation_analysis_full.json')
        with open(sat_path, 'w') as f:
            json.dump(sat_results, f, indent=2, default=str)
        print(f"  Saturation results saved: {sat_path}")

    # Step 5: Validate on RA data
    if args.mode in ('validate', 'all'):
        print(f"\n[4/4] Validating on RA-TRB...")
        import cross_disease_benchmark as cdb
        ra_samples = cdb.load_ra_dataset('TRB')

        # Load or use current panel
        if args.mode == 'validate':
            panel_file = os.path.join(OUTPUT_DIR, f'reference_panel_m{args.m_prototypes}.pkl')
            if os.path.exists(panel_file):
                ckpt = load_reference_panel(panel_file)
                centroids = ckpt['centroids']
            else:
                print(f"  Panel not found: {panel_file}")
                sys.exit(1)
        else:
            centroids = panel['centroids']

        val_result = validate_with_ra(ra_samples, centroids)
        print(f"\n  Validation AUC: {val_result['auc']:.4f}")
        print(f"  Sparsity: {val_result['sparsity']:.1%}")

    print("\nDone!")


if __name__ == '__main__':
    main()
