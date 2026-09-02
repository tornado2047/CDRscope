#!/usr/bin/env python3
"""
Motif Spectrum Layer (Direction 3, L2)
=======================================
Fixed motif dictionary → k-mer frequency spectrum.
Captures short positional patterns lost in the prototype spectrum.

Design:
  - Build motif dictionary from reference pool: top-k 3-mers and 4-mers
  - Each sample → count of each motif → L2-normalized 500-d vector
  - Motif channels are fixed across samples (like prototype channels)
  - Orthogonal to L1: L1 captures global sequence identity via embedding,
    L2 captures local compositional patterns
"""
import numpy as np
from collections import Counter
from itertools import product

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def _generate_kmers(k):
    """Generate all possible k-mers of standard amino acids."""
    return [''.join(p) for p in product(sorted(STANDARD_AA), repeat=k)]


def extract_kmers(sequence, k):
    """Extract all k-mers from a CDR3 sequence."""
    if len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def build_motif_dictionary(reference_sequences, top_k_3mer=250, top_k_4mer=250):
    """Build a fixed motif dictionary from reference pool.

    Args:
        reference_sequences: list of CDR3 amino acid strings
        top_k_3mer: number of top 3-mers to keep
        top_k_4mer: number of top 4-mers to keep

    Returns:
        list of motif strings (fixed dictionary, order = channel order)
    """
    # Count 3-mers
    kmer3_counter = Counter()
    for seq in reference_sequences:
        seq = str(seq).strip()
        if len(seq) >= 3 and all(aa in STANDARD_AA for aa in seq):
            kmer3_counter.update(extract_kmers(seq, 3))

    # Count 4-mers
    kmer4_counter = Counter()
    for seq in reference_sequences:
        seq = str(seq).strip()
        if len(seq) >= 4 and all(aa in STANDARD_AA for aa in seq):
            kmer4_counter.update(extract_kmers(seq, 4))

    top_3 = [m for m, _ in kmer3_counter.most_common(top_k_3mer)]
    top_4 = [m for m, _ in kmer4_counter.most_common(top_k_4mer)]

    dictionary = top_3 + top_4
    print(f"  Motif dictionary: {len(top_3)} 3-mers + {len(top_4)} 4-mers = {len(dictionary)} motifs")
    print(f"    3-mer coverage: {sum(c for _, c in kmer3_counter.most_common(top_k_3mer)) / max(sum(kmer3_counter.values()), 1):.1%}")
    print(f"    4-mer coverage: {sum(c for _, c in kmer4_counter.most_common(top_k_4mer)) / max(sum(kmer4_counter.values()), 1):.1%}")

    return dictionary


class MotifSpectrumExtractor:
    """L2: Motif frequency spectrum layer."""

    def __init__(self, dictionary=None, top_k_3mer=250, top_k_4mer=250):
        """Args:
            dictionary: pre-built motif list, or None to build from reference
            top_k_3mer / top_k_4mer: params for dictionary building
        """
        self.dictionary = dictionary
        self.top_k_3mer = top_k_3mer
        self.top_k_4mer = top_k_4mer
        self._motif_index = None
        if dictionary is not None:
            self._build_index()

    def _build_index(self):
        """Build lookup index for fast counting."""
        self._motif_index = {m: i for i, m in enumerate(self.dictionary)}

    def fit(self, reference_sequences):
        """Build motif dictionary from reference pool.

        Args:
            reference_sequences: list of CDR3 strings from reference pool
        """
        self.dictionary = build_motif_dictionary(
            reference_sequences, self.top_k_3mer, self.top_k_4mer
        )
        self._build_index()
        print(f"  MotifSpectrumExtractor: dictionary built ({len(self.dictionary)} motifs)")

    def transform(self, sequences, counts=None):
        """Extract motif spectrum for one sample.

        Args:
            sequences: list of CDR3 strings
            counts: optional clone counts (if None, all 1)

        Returns:
            (len(dictionary),) L2-normalized count vector
        """
        if self.dictionary is None or self._motif_index is None:
            raise RuntimeError("Must fit() or provide dictionary before transform()")

        if counts is None:
            counts = np.ones(len(sequences))
        else:
            counts = np.array(counts, dtype=float)

        spec = np.zeros(len(self.dictionary), dtype=np.float32)
        for seq, c in zip(sequences, counts):
            seq = str(seq).strip()
            if len(seq) < 3 or not all(aa in STANDARD_AA for aa in seq):
                continue
            c_int = int(c)
            for k in [3, 4]:
                for km in extract_kmers(seq, k):
                    idx = self._motif_index.get(km)
                    if idx is not None:
                        spec[idx] += c_int

        norm = np.linalg.norm(spec)
        if norm > 0:
            spec /= norm
        return spec

    def transform_many(self, samples):
        """Extract motif spectra for multiple samples.

        Args:
            samples: list of (sequences, counts) tuples

        Returns:
            (n_samples, len(dictionary)) array
        """
        results = []
        for seqs, counts in samples:
            results.append(self.transform(seqs, counts))
        return np.array(results, dtype=np.float32)

    def get_dim(self):
        return len(self.dictionary) if self.dictionary else 0


if __name__ == '__main__':
    # Smoke test
    np.random.seed(42)

    # Generate synthetic reference sequences
    aa_list = sorted(STANDARD_AA)
    ref_seqs = []
    for _ in range(5000):
        length = np.random.randint(8, 20)
        ref_seqs.append(''.join(np.random.choice(list(aa_list), length)))

    # Build extractor
    ext = MotifSpectrumExtractor(top_k_3mer=100, top_k_4mer=100)
    ext.fit(ref_seqs)

    # Test sample
    test_seqs = ref_seqs[:100]
    test_counts = np.random.randint(1, 50, 100)
    spectrum = ext.transform(test_seqs, test_counts)

    print(f"\n=== Motif Spectrum Smoke Test ===")
    print(f"Dictionary size: {ext.get_dim()}")
    print(f"Spectrum shape:  {spectrum.shape}")
    print(f"L2 norm:         {np.linalg.norm(spectrum):.6f}")
    print(f"Nonzero channels: {np.sum(spectrum > 0)}")
    print(f"Max channel:     {spectrum.max():.4f}")

    # Test batch
    samples = [(ref_seqs[100:150], None), (ref_seqs[150:200], None)]
    batch = ext.transform_many(samples)
    print(f"\nBatch transform: {batch.shape}")
    assert batch.shape == (2, ext.get_dim())
    print("Smoke test passed.")
