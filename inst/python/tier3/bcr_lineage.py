#!/usr/bin/env python3
"""
BCR Lineage Tree Module (Direction 2)
=====================================
SHM-aware sequence evolution spectrum for BCR repertoires.

B cell adaptation is written in sequence (SHM lineage trees), NOT in frequency.
This module extracts:
  B1: Lineage tree topology features (15-d)
  B2: Mutation spectrum (20-d)

Key distinction from TCR track:
  TCR → frequency spectrum (clone expansion/contraction)
  BCR → sequence evolution spectrum (SHM tree topology + mutation patterns)

Lineage tree construction:
  1. Group sequences by V gene + J gene + CDR3 length (same clonotype)
  2. Within each group, build minimum spanning tree from pairwise edit distances
  3. Root = germline-inferred sequence (shortest or most-germline-like)
  4. Extract topology features

Mutation spectrum:
  1. For each clonotype group, compute position-wise mutation rate
  2. Classify mutations: synonymous/conservative/replacement
  3. R/S ratio as affinity maturation proxy
  4. Hotspot motif (RGYW/WRCY) enrichment
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

# Codon table for synonymous/non-synonymous classification (simplified)
# R/S ratio: replacement / synonymous substitutions
AA_CODON_GROUPS = {
    # Conservative groups (substitutions within same group = conservative)
    'positive': set('KHR'),
    'negative': set('DE'),
    'polar': set('STNQ'),
    'hydrophobic': set('AVILM'),
    'aromatic': set('FWY'),
    'special': set('CGP'),
}

# SHM hotspot motifs
HOTSPOT_MOTIFS = {'RGYW', 'WRCY', 'WA', 'TW', 'AGY', 'SRC'}


def _edit_distance_matrix(seqs):
    """Compute pairwise edit distance matrix for a list of sequences."""
    n = len(seqs)
    if n <= 1:
        return np.zeros((n, n))
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = sum(1 for a, b in zip(seqs[i], seqs[j]) if a != b) + abs(len(seqs[i]) - len(seqs[j]))
            mat[i, j] = d
            mat[j, i] = d
    return mat


def group_clonotypes(sequences, v_genes=None, j_genes=None):
    """Group BCR sequences into clonotypes by V-J + CDR3 length.

    Args:
        sequences: list of CDR3 amino acid strings
        v_genes: list of V gene names (optional)
        j_genes: list of J gene names (optional)

    Returns:
        dict: {clonotype_key: [indices into sequences]}
    """
    groups = defaultdict(list)
    for i, seq in enumerate(sequences):
        v = v_genes[i] if v_genes else 'unknown'
        j = j_genes[i] if j_genes else 'unknown'
        key = f"{v}_{j}_{len(seq)}"
        groups[key].append(i)
    return dict(groups)


def build_lineage_tree(sequences, counts=None):
    """Build a lineage tree from a group of related BCR sequences.

    Uses minimum spanning tree of pairwise edit distances.
    Root = sequence most similar to germline (shortest or lowest mutation count).

    Args:
        sequences: list of CDR3 strings (same clonotype)
        counts: optional clone sizes

    Returns:
        dict with tree topology features
    """
    n = len(sequences)
    if n <= 1:
        return _empty_tree_features()

    # Compute distance matrix
    dist_mat = _edit_distance_matrix(sequences)

    # Build minimum spanning tree
    if n <= 50:
        # For small groups, use direct MST
        sparse_dist = csr_matrix(dist_mat)
        mst = minimum_spanning_tree(sparse_dist)
        mst_array = mst.toarray()
    else:
        # For large groups, subsample
        indices = np.random.choice(n, min(50, n), replace=False)
        sub_seqs = [sequences[i] for i in indices]
        sub_dist = _edit_distance_matrix(sub_seqs)
        sparse_dist = csr_matrix(sub_dist)
        mst = minimum_spanning_tree(sparse_dist)
        mst_array = mst.toarray()
        n = len(sub_seqs)

    # Tree topology
    n_edges = int(np.sum(mst_array > 0))
    total_branch_length = float(np.sum(mst_array))

    # Identify root (most central or shortest sequence)
    root_idx = np.argmin([len(s) for s in (sequences if n <= 50 else sub_seqs)])

    # Compute tree depth (longest path from root)
    adjacency = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if mst_array[i, j] > 0:
                adjacency[i].append((j, mst_array[i, j]))
                adjacency[j].append((i, mst_array[i, j]))

    max_depth = _compute_max_depth(adjacency, root_idx, n)

    # Trunk length (root to nearest branching point)
    trunk_length = _compute_trunk_length(adjacency, root_idx, n)

    # Branching ratio (number of internal branches / total edges)
    branching_nodes = sum(1 for node, edges in adjacency.items()
                          if len(edges) > 2)
    branching_ratio = branching_nodes / max(n - 1, 1)

    # Trunk-to-tip ratio
    tip_distances = _compute_avg_tip_distance(adjacency, root_idx, n)
    avg_tip_distance = float(np.mean(tip_distances)) if tip_distances else 0.0
    trunk_to_tip = trunk_length / max(avg_tip_distance, 1e-6)

    # Internal branch lengths
    branch_lengths = [mst_array[i, j] for i in range(n) for j in range(i+1, n)
                      if mst_array[i, j] > 0]

    features = {
        'n_sequences': n,
        'n_edges': n_edges,
        'total_branch_length': total_branch_length,
        'avg_branch_length': float(np.mean(branch_lengths)) if branch_lengths else 0.0,
        'max_depth': float(max_depth),
        'trunk_length': float(trunk_length),
        'trunk_to_tip_ratio': float(trunk_to_tip),
        'branching_ratio': float(branching_ratio),
        'n_internal_branches': branching_nodes,
        'max_branch_length': float(max(branch_lengths)) if branch_lengths else 0.0,
        'branch_length_cv': float(np.std(branch_lengths) / max(np.mean(branch_lengths), 1e-6))
        if branch_lengths else 0.0,
        'avg_diversity': float(np.mean(dist_mat[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
        'n_unique_seqs': len(set(sequences)),
        'clonal_diversification_rate': float(n / max(total_branch_length, 1e-6)),
    }

    return features


def _empty_tree_features():
    return {k: 0.0 for k in [
        'n_sequences', 'n_edges', 'total_branch_length', 'avg_branch_length',
        'max_depth', 'trunk_length', 'trunk_to_tip_ratio', 'branching_ratio',
        'n_internal_branches', 'max_branch_length', 'branch_length_cv',
        'avg_diversity', 'n_unique_seqs', 'clonal_diversification_rate',
    ]}


def _compute_max_depth(adjacency, root, n, visited=None):
    """Compute maximum depth from root via DFS."""
    if visited is None:
        visited = set()
    if root in visited:
        return 0
    visited.add(root)
    max_d = 0
    for child, weight in adjacency.get(root, []):
        if child not in visited:
            d = weight + _compute_max_depth(adjacency, child, n, visited)
            max_d = max(max_d, d)
    return max_d


def _compute_trunk_length(adjacency, root, n, visited=None):
    """Compute trunk length (root to first branching point)."""
    if visited is None:
        visited = set()
    visited.add(root)
    children = [(c, w) for c, w in adjacency.get(root, []) if c not in visited]
    if len(children) <= 1:
        if not children:
            return 0.0
        return children[0][1] + _compute_trunk_length(adjacency, children[0][0], n, visited)
    return 0.0  # branching point reached


def _compute_avg_tip_distance(adjacency, root, n, visited=None, depth=0):
    """Compute average distance from root to all leaves."""
    if visited is None:
        visited = set()
    if root in visited:
        return [0.0]
    visited.add(root)
    children = [(c, w) for c, w in adjacency.get(root, []) if c not in visited]
    if not children:
        return [depth]
    distances = []
    for child, weight in children:
        distances.extend(_compute_avg_tip_distance(adjacency, child, n, visited, depth + weight))
    return distances


def compute_mutation_spectrum(sequences, germline=None):
    """Compute position-wise mutation rate and substitution spectrum.

    Args:
        sequences: list of CDR3 strings (same clonotype)
        germline: inferred germline sequence (optional)

    Returns:
        dict with mutation spectrum features
    """
    n = len(sequences)
    if n <= 1:
        return _empty_mutation_features()

    # Align (all same length within clonotype by grouping)
    seq_len = len(sequences[0])
    if any(len(s) != seq_len for s in sequences):
        # Use the most common length
        seq_len = Counter(len(s) for s in sequences).most_common(1)[0][0]
        sequences = [s for s in sequences if len(s) == seq_len]
        n = len(sequences)
        if n <= 1:
            return _empty_mutation_features()

    # Infer germline as consensus sequence
    if germline is None:
        germline = ''
        for pos in range(seq_len):
            col = [s[pos] for s in sequences]
            germline += Counter(col).most_common(1)[0][0]

    # Position-wise mutation rate
    pos_mutation_rate = np.zeros(seq_len)
    for pos in range(seq_len):
        n_mut = sum(1 for s in sequences if s[pos] != germline[pos])
        pos_mutation_rate[pos] = n_mut / n

    # Substitution classification
    n_synonymous = 0
    n_conservative = 0
    n_replacement = 0

    for seq in sequences:
        for pos in range(seq_len):
            if seq[pos] != germline[pos]:
                orig_aa = germline[pos]
                new_aa = seq[pos]
                # Check if substitution is within same AA group
                orig_group = _get_aa_group(orig_aa)
                new_group = _get_aa_group(new_aa)
                if orig_group == new_group:
                    n_conservative += 1
                else:
                    n_replacement += 1

    total_mutations = n_synonymous + n_conservative + n_replacement
    if total_mutations == 0:
        rs_ratio = 0.0
    else:
        rs_ratio = n_replacement / max(n_conservative + n_synonymous, 1)

    # Hotspot enrichment
    hotspot_count = 0
    total_motifs = 0
    for seq in sequences:
        for k in [4]:
            for i in range(len(seq) - k + 1):
                motif = seq[i:i+k]
                total_motifs += 1
                if motif in HOTSPOT_MOTIFS:
                    hotspot_count += 1
    hotspot_enrichment = hotspot_count / max(total_motifs, 1)

    # Top mutated positions
    top_mut_pos = np.argsort(pos_mutation_rate)[::-1][:5]
    top_mut_rates = pos_mutation_rate[top_mut_pos]

    features = {
        'mean_mutation_rate': float(np.mean(pos_mutation_rate)),
        'max_mutation_rate': float(np.max(pos_mutation_rate)),
        'std_mutation_rate': float(np.std(pos_mutation_rate)),
        'n_total_mutations': total_mutations,
        'n_replacement': n_replacement,
        'n_conservative': n_conservative,
        'rs_ratio': float(rs_ratio),
        'hotspot_enrichment': float(hotspot_enrichment),
        'top5_mut_positions': top_mut_pos.tolist(),
        'top5_mut_rates': top_mut_rates.tolist(),
        'mutation_entropy': float(-np.sum(
            pos_mutation_rate[pos_mutation_rate > 0] *
            np.log2(pos_mutation_rate[pos_mutation_rate > 0] + 1e-10)
        )),
        'seq_len': seq_len,
        'n_sequences': n,
        'avg_diversity': float(np.mean([
            sum(1 for a, b in zip(s1, s2) if a != b)
            for s1, s2 in combinations(sequences, 2)
        ])) / max(seq_len, 1) if n > 1 else 0.0,
    }

    return features


def _get_aa_group(aa):
    """Get physicochemical group of an amino acid."""
    for group_name, aas in AA_CODON_GROUPS.items():
        if aa in aas:
            return group_name
    return 'other'


def _empty_mutation_features():
    return {k: 0.0 for k in [
        'mean_mutation_rate', 'max_mutation_rate', 'std_mutation_rate',
        'n_total_mutations', 'n_replacement', 'n_conservative', 'rs_ratio',
        'hotspot_enrichment', 'mutation_entropy', 'seq_len', 'n_sequences',
        'avg_diversity',
    ]}


class BCRLineageExtractor:
    """Unified BCR lineage tree + mutation spectrum extractor.

    B1: Lineage tree topology (15-d)
    B2: Mutation spectrum (20-d)
    """

    def __init__(self, min_clonotype_size=2, max_clonotypes=100):
        self.min_clonotype_size = min_clonotype_size
        self.max_clonotypes = max_clonotypes

    def transform(self, sequences, v_genes=None, j_genes=None, counts=None):
        """Extract BCR lineage features from a sample.

        Args:
            sequences: list of CDR3 strings
            v_genes: list of V gene names
            j_genes: list of J gene names
            counts: optional clone counts

        Returns:
            dict with 'topology' (15-d) and 'mutation' (20-d) feature vectors
        """
        # Group into clonotypes
        groups = group_clonotypes(sequences, v_genes, j_genes)

        # Filter small groups and subsample if too many
        valid_groups = {k: v for k, v in groups.items()
                        if len(v) >= self.min_clonotype_size}

        if not valid_groups:
            return {
                'topology': np.zeros(15, dtype=np.float32),
                'mutation': np.zeros(20, dtype=np.float32),
                'n_clonotypes': 0,
            }

        # Subsample if too many clonotypes
        if len(valid_groups) > self.max_clonotypes:
            keys = list(valid_groups.keys())
            selected = np.random.choice(keys, self.max_clonotypes, replace=False)
            valid_groups = {k: valid_groups[k] for k in selected}

        # Extract features from each clonotype
        topology_features = []
        mutation_features = []

        for key, indices in valid_groups.items():
            group_seqs = [sequences[i] for i in indices]
            tree = build_lineage_tree(group_seqs)
            mut = compute_mutation_spectrum(group_seqs)
            topology_features.append(tree)
            mutation_features.append(mut)

        # Aggregate across clonotypes
        topo_vec = self._aggregate_topology(topology_features)
        mut_vec = self._aggregate_mutation(mutation_features)

        return {
            'topology': topo_vec,
            'mutation': mut_vec,
            'n_clonotypes': len(valid_groups),
        }

    def _aggregate_topology(self, features_list):
        """Aggregate tree topology features across clonotypes → 15-d."""
        keys = [
            'n_sequences', 'n_edges', 'total_branch_length', 'avg_branch_length',
            'max_depth', 'trunk_length', 'trunk_to_tip_ratio', 'branching_ratio',
            'n_internal_branches', 'max_branch_length', 'branch_length_cv',
            'avg_diversity', 'n_unique_seqs', 'clonal_diversification_rate',
        ]
        vec = np.zeros(15, dtype=np.float32)
        for i, key in enumerate(keys):
            if i >= 15:
                break
            values = [f.get(key, 0) for f in features_list]
            vec[i] = float(np.median(values))
        # 15th: number of clonotypes
        vec[14] = len(features_list)
        return vec

    def _aggregate_mutation(self, features_list):
        """Aggregate mutation features across clonotypes → 20-d."""
        scalar_keys = [
            'mean_mutation_rate', 'max_mutation_rate', 'std_mutation_rate',
            'n_total_mutations', 'n_replacement', 'n_conservative', 'rs_ratio',
            'hotspot_enrichment', 'mutation_entropy', 'seq_len', 'n_sequences',
            'avg_diversity',
        ]
        vec = np.zeros(20, dtype=np.float32)
        for i, key in enumerate(scalar_keys):
            if i >= 20:
                break
            values = [f.get(key, 0) for f in features_list]
            vec[i] = float(np.median(values))

        # Additional aggregate features
        vec[12] = float(np.mean([f.get('rs_ratio', 0) for f in features_list]))
        vec[13] = float(np.mean([f.get('hotspot_enrichment', 0) for f in features_list]))
        vec[14] = float(np.sum([f.get('n_total_mutations', 0) for f in features_list]))
        vec[15] = float(np.sum([f.get('n_replacement', 0) for f in features_list]))
        vec[16] = float(np.sum([f.get('n_conservative', 0) for f in features_list]))
        vec[17] = float(np.mean([f.get('mean_mutation_rate', 0) for f in features_list]))
        vec[18] = float(np.std([f.get('mean_mutation_rate', 0) for f in features_list]))
        vec[19] = float(len(features_list))

        return vec


if __name__ == '__main__':
    # Smoke test with simulated BCR lineage
    np.random.seed(42)

    # Simulate a BCR clonotype: germline + SHM variants
    germline = 'ARHDYYGSSYFDV'
    mutations = [
        germline,
        germline[:2] + 'K' + germline[3:],     # R→K (conservative, positive group)
        germline[:5] + 'F' + germline[6:],     # G→F (replacement)
        germline[:2] + 'R' + germline[3:5] + 'F' + germline[6:],  # 2 mutations
        germline[:8] + 'A' + germline[9:],     # S→A (conservative, polar)
        germline[:5] + 'Y' + germline[6:8] + 'T' + germline[9:],  # 2 mutations
        germline[:3] + 'D' + germline[4:],     # D→D (no change, but included)
        germline[:1] + 'R' + germline[2:4] + 'H' + germline[5:],  # 2 mutations
    ]

    v_genes = ['IGHV3-23'] * len(mutations)
    j_genes = ['IGHJ4'] * len(mutations)

    ext = BCRLineageExtractor(min_clonotype_size=2)
    result = ext.transform(mutations, v_genes, j_genes)

    print("=== BCR Lineage Smoke Test ===\n")
    print(f"Input: {len(mutations)} sequences, {result['n_clonotypes']} clonotype(s)")
    print(f"Topology vector ({len(result['topology'])}-d):")
    print(f"  n_sequences: {result['topology'][0]:.0f}")
    print(f"  n_edges:     {result['topology'][1]:.0f}")
    print(f"  trunk/tip:   {result['topology'][6]:.3f}")
    print(f"  branching:   {result['topology'][7]:.3f}")
    print(f"  n_clonotypes:{result['topology'][14]:.0f}")

    print(f"\nMutation vector ({len(result['mutation'])}-d):")
    print(f"  mean_mut_rate: {result['mutation'][0]:.4f}")
    print(f"  max_mut_rate:  {result['mutation'][1]:.4f}")
    print(f"  n_replacement: {result['mutation'][4]:.0f}")
    print(f"  n_conservative:{result['mutation'][5]:.0f}")
    print(f"  R/S ratio:     {result['mutation'][6]:.3f}")
    print(f"  hotspot_enr:   {result['mutation'][7]:.4f}")

    # Multi-clonotype test
    seqs2 = mutations + ['AQYLSGSTYFDV', 'AQYLSGSSYFDV', 'AQYLSGSTYYFDV']
    v2 = v_genes + ['IGHV1-2'] * 3
    j2 = j_genes + ['IGHJ3'] * 3
    result2 = ext.transform(seqs2, v2, j2)
    print(f"\nMulti-clonotype: {result2['n_clonotypes']} clonotypes")
    print(f"Topology: {result2['topology'][:5]}")
    print(f"Mutation: {result2['mutation'][:5]}")

    print("\nSmoke test passed.")
