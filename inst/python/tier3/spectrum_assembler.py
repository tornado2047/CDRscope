#!/usr/bin/env python3
"""
Multi-Layer Spectrum Assembler
===============================
Integrates all representation layers into a unified multi-layer spectrum.

TCR track (frequency-based):
  L1: Prototype spectrum (10,000-d)      [from tier2]
  L2: Motif spectrum (500-d)             [motif_layer]
  L4: Macro indices (10-d)                [macro_indices]
  S1-S4: Selection imprints (45-d)       [selection_imprints]
  T1-T3: Temporal dynamics (optional)    [temporal_dynamics]

BCR track (sequence-evolution-based):
  L1: Prototype spectrum (10,000-d)      [from tier2, separate panel]
  L2: Motif spectrum (500-d)             [motif_layer]
  L4: Macro indices (10-d)                [macro_indices]
  B1: Lineage tree topology (15-d)       [bcr_lineage]
  B2: Mutation spectrum (20-d)           [bcr_lineage]
  S1-S4: Selection imprints (45-d)       [selection_imprints]

Assembled spectrum = [L1 ⊕ L2 ⊕ L4 ⊕ S-axes ⊕ (TCR: T-axes | BCR: B-axes)]

Each layer is independently L2-normalized before concatenation to prevent
high-dimensional layers (L1: 10,000-d) from dominating low-dimensional ones.
"""
import numpy as np
from .macro_indices import MacroIndexExtractor
from .selection_imprints import SelectionImprintExtractor
from .motif_layer import MotifSpectrumExtractor
from .track_separator import TrackSeparator
from .bcr_lineage import BCRLineageExtractor
from .temporal_dynamics import TemporalDynamicsExtractor


class SpectrumAssembler:
    """Assembles multi-layer spectrum from raw sample data.

    Usage:
        assembler = SpectrumAssembler()
        assembler.fit(reference_samples, reference_sequences, chain='TRA')

        spectrum = assembler.transform(sample, chain='TRA')
        # spectrum.shape = (10,555,) for TCR
        # spectrum.shape = (10,590,) for BCR
    """

    def __init__(self, motif_dict_size=500, top_k_3mer=250, top_k_4mer=250):
        self.macro_ext = MacroIndexExtractor()
        self.imprint_ext = SelectionImprintExtractor()
        self.motif_ext = MotifSpectrumExtractor(
            top_k_3mer=top_k_3mer, top_k_4mer=top_k_4mer
        )
        self.track_sep = TrackSeparator()
        self.bcr_ext = BCRLineageExtractor()
        self.temporal_ext = TemporalDynamicsExtractor()

        self._fitted = False
        self._track = None
        self._chain = None

        self.layer_dims = {}
        self.layer_names = {}

    def fit(self, reference_samples, reference_sequences, chain='TRA',
            prototype_spectrum_dim=10000):
        """Fit all extractors on reference (CordBlood) data.

        Args:
            reference_samples: list of dicts with 'sequences', 'counts',
                               'v_genes', 'j_genes'
            reference_sequences: flat list of CDR3 strings for motif dictionary
            chain: chain type for track routing
            prototype_spectrum_dim: dimension of L1 (from tier2 K-means)
        """
        # Route to track
        track_info = self.track_sep.route_sample(
            reference_sequences[:100], chain=chain
        )
        self._track = track_info['track']
        self._chain = chain

        # Fit selection imprints
        print(f"  Fitting selection imprints...")
        self.imprint_ext.fit(reference_samples)

        # Fit motif dictionary
        print(f"  Fitting motif dictionary...")
        self.motif_ext.fit(reference_sequences)

        # Record layer dimensions
        self.layer_dims = {
            'L1_prototype': prototype_spectrum_dim,
            'L2_motif': self.motif_ext.get_dim(),
            'L4_macro': 10,
            'S_imprints': 45,
        }

        if self._track == 'TCR':
            self.layer_dims['T_temporal'] = 10015  # T1(10000) + T2(10) + T3(5)
            self.layer_names = list(self.layer_dims.keys())
        else:
            self.layer_dims['B_lineage'] = 35  # B1(15) + B2(20)
            self.layer_names = list(self.layer_dims.keys())

        total = sum(self.layer_dims.values())
        print(f"  Track: {self._track} ({chain})")
        print(f"  Total spectrum dimension: {total}")
        for name, dim in self.layer_dims.items():
            print(f"    {name}: {dim}")

        self._fitted = True

    def transform(self, sample, prototype_spectrum=None, chain=None,
                  temporal_data=None):
        """Assemble multi-layer spectrum for one sample.

        Args:
            sample: dict with 'sequences', 'counts', 'v_genes', 'j_genes'
            prototype_spectrum: (m,) L2-normalized prototype count vector from L1
            chain: override chain type
            temporal_data: optional dict with 'pair' for temporal extraction

        Returns:
            dict with 'spectrum' (concatenated), 'layers' (per-layer dict)
        """
        if not self._fitted and prototype_spectrum is None:
            raise RuntimeError("Must fit() before transform(), or provide prototype_spectrum")

        chain = chain or self._chain or 'TRA'
        track_info = self.track_sep.route_sample(
            sample.get('sequences', []),
            sample.get('v_genes'),
            sample.get('j_genes'),
            chain=chain,
        )
        track = track_info['track']

        layers = {}

        # L1: Prototype spectrum (from tier2)
        if prototype_spectrum is not None:
            l1 = np.array(prototype_spectrum, dtype=np.float32)
            layers['L1_prototype'] = l1

        # L2: Motif spectrum
        l2 = self.motif_ext.transform(
            sample.get('sequences', []),
            sample.get('counts'),
        )
        layers['L2_motif'] = l2

        # L4: Macro indices
        counts = sample.get('counts')
        if counts is None:
            counts = np.ones(len(sample.get('sequences', [])))
        l4 = self.macro_ext.transform(counts)
        layers['L4_macro'] = l4

        # S-axes: Selection imprints
        s_axes = self.imprint_ext.transform(sample)
        layers['S_imprints'] = s_axes

        # Track-specific layers
        if track == 'TCR':
            # T-axes: Temporal dynamics (optional)
            if temporal_data is not None:
                temporal_result = self.temporal_ext.transform_pair(
                    temporal_data.get('t1', {}),
                    temporal_data.get('t2', {}),
                    temporal_data.get('delta_days'),
                )
                t_vec = np.concatenate([
                    temporal_result.get('delta_frequency', np.zeros(10000)),
                    temporal_result.get('delta_diversity', np.zeros(10)),
                    np.array([
                        temporal_result.get('velocity', {}).get('magnitude', 0),
                        temporal_result.get('velocity', {}).get('speed', 0) or 0,
                        temporal_result.get('velocity', {}).get('direction_cosine', 0),
                    ]),
                ])
                layers['T_temporal'] = t_vec.astype(np.float32)

        elif track == 'BCR':
            # B-axes: Lineage + mutation
            bcr_result = self.bcr_ext.transform(
                sample.get('sequences', []),
                sample.get('v_genes'),
                sample.get('j_genes'),
            )
            b_vec = np.concatenate([
                bcr_result['topology'],
                bcr_result['mutation'],
            ])
            layers['B_lineage'] = b_vec

        # Per-layer L2 normalization (prevent high-dim layers from dominating)
        normalized_layers = {}
        for name, vec in layers.items():
            norm = np.linalg.norm(vec)
            if norm > 0:
                normalized_layers[name] = vec / norm
            else:
                normalized_layers[name] = vec

        # Concatenate
        spectrum = np.concatenate(list(normalized_layers.values()))

        return {
            'spectrum': spectrum,
            'layers': normalized_layers,
            'layer_dims': {k: len(v) for k, v in normalized_layers.items()},
            'track': track,
            'track_info': track_info,
        }

    def get_layer_description(self):
        """Return human-readable description of the assembled spectrum."""
        if not self._fitted:
            return "Not fitted yet"

        desc = f"CDRscope Multi-Layer Spectrum ({self._track} track, {self._chain})\n"
        desc += f"Total dimension: {sum(self.layer_dims.values())}\n"
        desc += "Layers:\n"
        for name, dim in self.layer_dims.items():
            desc += f"  {name}: {dim} dimensions\n"
        return desc


if __name__ == '__main__':
    # Smoke test
    np.random.seed(42)

    # Simulate reference data
    aa_list = sorted('ACDEFGHIKLMNPQRSTVWY')
    ref_sequences = []
    ref_samples = []
    for _ in range(20):
        n_seqs = np.random.randint(50, 200)
        seqs = []
        for _ in range(n_seqs):
            length = np.random.randint(8, 20)
            seqs.append(''.join(np.random.choice(list(aa_list), length)))
        ref_sequences.extend(seqs)
        ref_samples.append({
            'sequences': seqs,
            'counts': np.random.randint(1, 50, len(seqs)).tolist(),
            'v_genes': [f'TRAV{np.random.randint(1,10)}-1'] * len(seqs),
            'j_genes': [f'TRAJ{np.random.randint(1,10)}'] * len(seqs),
        })

    # Fit assembler
    assembler = SpectrumAssembler(top_k_3mer=100, top_k_4mer=100)
    assembler.fit(ref_samples, ref_sequences, chain='TRA', prototype_spectrum_dim=10000)

    # Transform a sample
    test_seqs = ref_sequences[:100]
    test_counts = np.random.randint(1, 50, 100)
    test_sample = {
        'sequences': test_seqs,
        'counts': test_counts.tolist(),
        'v_genes': [f'TRAV{np.random.randint(1,10)}-1'] * 100,
        'j_genes': [f'TRAJ{np.random.randint(1,10)}'] * 100,
    }
    # Mock prototype spectrum (L1 from tier2)
    mock_l1 = np.random.dirichlet(np.ones(10000) * 0.01)

    result = assembler.transform(test_sample, prototype_spectrum=mock_l1, chain='TRA')

    print("\n=== Spectrum Assembler Smoke Test ===\n")
    print(assembler.get_layer_description())
    print(f"Assembled spectrum shape: {result['spectrum'].shape}")
    print(f"Track: {result['track']}")
    print(f"Layer dims:")
    for name, dim in result['layer_dims'].items():
        vec = result['layers'][name]
        print(f"  {name}: {dim}-d, L2 norm={np.linalg.norm(vec):.4f}, "
              f"nonzero={np.sum(vec > 0)}")

    # BCR track test
    bcr_seqs = ['ARHDYYGSSYFDV', 'ARKDYYGSSYFDV', 'ARHDYYFSSYFDV',
                'AQYLQSGTYFDV', 'AQYLQSGAYFDV']
    bcr_sample = {
        'sequences': bcr_seqs,
        'counts': [10, 5, 3, 8, 2],
        'v_genes': ['IGHV3-23', 'IGHV3-23', 'IGHV3-23', 'IGHV1-2', 'IGHV1-2'],
        'j_genes': ['IGHJ4', 'IGHJ4', 'IGHJ4', 'IGHJ3', 'IGHJ3'],
    }
    mock_l1_bcr = np.random.dirichlet(np.ones(10000) * 0.01)
    bcr_result = assembler.transform(bcr_sample, prototype_spectrum=mock_l1_bcr, chain='IGH')

    print(f"\nBCR spectrum shape: {bcr_result['spectrum'].shape}")
    print(f"Track: {bcr_result['track']}")
    for name, dim in bcr_result['layer_dims'].items():
        vec = bcr_result['layers'][name]
        print(f"  {name}: {dim}-d, L2 norm={np.linalg.norm(vec):.4f}")

    print("\nSmoke test passed.")
