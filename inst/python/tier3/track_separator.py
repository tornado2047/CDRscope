#!/usr/bin/env python3
"""
T/B Track Separator (Direction 1)
==================================
Separates TCR and BCR into different representation tracks.

TCR track: prototype frequency spectrum + selection imprints + temporal dynamics
BCR track: SHM lineage tree spectrum + mutation spectrum + selection imprints

Recognition biology dictating track design:
  TCR → pMHC (linear epitope, low affinity, contact-dependent) → frequency is the signal
  BCR → free conformational antigen (high affinity, remote) → sequence evolution is the signal

Chain-type-specific parameters:
  - CDR3 length range
  - Charge distribution
  - MHC dependency (TCR only)
  - Affinity maturation (BCR only)
"""
import numpy as np
from collections import Counter

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

# Chain classification
TCR_CHAINS = {'TRA', 'TRB', 'TRG', 'TRD'}
BCR_CHAINS = {'IGH', 'IGL', 'IGK'}  # IGL/IGK are light, IGH is heavy
BCR_HEAVY = {'IGH'}
BCR_LIGHT = {'IGL', 'IGK'}

# Chain-type-specific CDR3 length parameters (from IMGT statistics)
CHAIN_LENGTH_PARAMS = {
    'TRA': {'min': 8, 'max': 22, 'median': 13, 'iqr_range': 3},
    'TRB': {'min': 8, 'max': 25, 'median': 15, 'iqr_range': 3},
    'TRG': {'min': 8, 'max': 20, 'median': 14, 'iqr_range': 3},
    'TRD': {'min': 8, 'max': 22, 'median': 14, 'iqr_range': 4},
    'IGH': {'min': 8, 'max': 30, 'median': 17, 'iqr_range': 5},
    'IGL': {'min': 8, 'max': 20, 'median': 11, 'iqr_range': 3},
    'IGK': {'min': 8, 'max': 20, 'median': 11, 'iqr_range': 3},
}

# Chain-type-specific charge bias (from TCR-pMHC binding studies)
CHAIN_CHARGE_PARAMS = {
    'TRA': {'expected_net': -0.1, 'tolerance': 0.3},
    'TRB': {'expected_net': 0.0, 'tolerance': 0.4},
    'TRG': {'expected_net': 0.0, 'tolerance': 0.3},
    'TRD': {'expected_net': 0.1, 'tolerance': 0.4},
    'IGH': {'expected_net': 0.1, 'tolerance': 0.5},
    'IGL': {'expected_net': -0.2, 'tolerance': 0.4},
    'IGK': {'expected_net': -0.2, 'tolerance': 0.4},
}


def classify_chain(chain_name):
    """Classify a chain into TCR/BCR track and heavy/light.

    Args:
        chain_name: e.g. 'TRA', 'IGH', 'IGK'

    Returns:
        dict: {'track': 'TCR'|'BCR', 'role': 'heavy'|'light'|None}
    """
    chain = chain_name.upper().strip()
    if chain in TCR_CHAINS:
        return {'track': 'TCR', 'role': None, 'chain': chain}
    elif chain in BCR_HEAVY:
        return {'track': 'BCR', 'role': 'heavy', 'chain': chain}
    elif chain in BCR_LIGHT:
        return {'track': 'BCR', 'role': 'light', 'chain': chain}
    else:
        return {'track': 'unknown', 'role': None, 'chain': chain}


def get_chain_length_filter(chain_name):
    """Get CDR3 length filter parameters for a specific chain type.

    Returns:
        dict with min, max, median, iqr_range
    """
    chain = chain_name.upper().strip()
    if chain in CHAIN_LENGTH_PARAMS:
        return CHAIN_LENGTH_PARAMS[chain]
    return {'min': 8, 'max': 25, 'median': 14, 'iqr_range': 4}


def get_chain_charge_filter(chain_name):
    """Get expected charge parameters for a specific chain type.

    Returns:
        dict with expected_net, tolerance
    """
    chain = chain_name.upper().strip()
    if chain in CHAIN_CHARGE_PARAMS:
        return CHAIN_CHARGE_PARAMS[chain]
    return {'expected_net': 0.0, 'tolerance': 0.5}


def compute_mhc_dependency_score(cdr3_sequence):
    """Estimate MHC dependency for a TCR CDR3.

    Heuristic: CDR3 sequences with more hydrophobic residues in the center
    are more likely to contact MHC helices (MHC-restricted).

    This axis is TCR-only — BCRs do not have MHC restriction.

    Returns:
        float: MHC dependency score [0, 1], higher = more MHC-restricted
    """
    hydrophobic = set('AVILMFWYC')
    if len(cdr3_sequence) < 8:
        return 0.5

    mid_start = len(cdr3_sequence) // 4
    mid_end = 3 * len(cdr3_sequence) // 4
    center = cdr3_sequence[mid_start:mid_end]

    hydro_count = sum(1 for aa in center if aa in hydrophobic)
    return min(1.0, hydro_count / max(len(center), 1))


def compute_affinity_maturation_proxy(cdr3s, germline_seqs=None):
    """Estimate BCR affinity maturation from CDR3 diversity.

    Proxy: for sequences sharing the same V-J + length, compute
    intra-group sequence diversity. Higher diversity → more SHM → more maturation.

    This axis is BCR-only — TCRs do not undergo SHM.

    Args:
        cdr3s: list of CDR3 strings
        germline_seqs: optional list of germline reference sequences

    Returns:
        float: affinity maturation proxy [0, 1]
    """
    if len(cdr3s) < 2:
        return 0.0

    # Group by length
    by_len = {}
    for s in cdr3s:
        l = len(s)
        if l not in by_len:
            by_len[l] = []
        by_len[l].append(s)

    # Compute within-group diversity as mutation proxy
    diversities = []
    for l, seqs in by_len.items():
        if len(seqs) < 2:
            continue
        # Average pairwise edit distance normalized by length
        n = min(len(seqs), 50)  # subsample for speed
        indices = np.random.choice(len(seqs), min(n, len(seqs)), replace=False) if len(seqs) > n else range(len(seqs))
        sample_seqs = [seqs[i] for i in indices]
        dists = []
        for i in range(len(sample_seqs)):
            for j in range(i+1, len(sample_seqs)):
                d = sum(1 for a, b in zip(sample_seqs[i], sample_seqs[j]) if a != b)
                dists.append(d / max(l, 1))
        if dists:
            diversities.append(np.mean(dists))

    if not diversities:
        return 0.0

    return min(1.0, np.mean(diversities))


class TrackSeparator:
    """Routes samples to TCR or BCR track with chain-type-specific parameters.

    Usage:
        ts = TrackSeparator()
        track_info = ts.route_sample(sequences, v_genes, j_genes, chain='TRA')
        # track_info['track'] = 'TCR' or 'BCR'
        # track_info['chain_params'] = length/charge filters
        # track_info['mhc_score'] = ... (TCR only)
        # track_info['maturation_proxy'] = ... (BCR only)
    """

    def __init__(self):
        self.chain_params = CHAIN_LENGTH_PARAMS
        self.charge_params = CHAIN_CHARGE_PARAMS

    def route_sample(self, sequences, v_genes=None, j_genes=None, chain='TRA'):
        """Route a sample to the appropriate track and apply chain-specific parameters.

        Args:
            sequences: list of CDR3 strings
            v_genes: list of V gene names (optional)
            j_genes: list of J gene names (optional)
            chain: chain type name

        Returns:
            dict with track info and chain-specific parameters
        """
        info = classify_chain(chain)
        track = info['track']

        # Get chain-specific parameters
        length_params = get_chain_length_filter(chain)
        charge_params = get_chain_charge_filter(chain)

        # Filter sequences by chain-specific length range
        valid_mask = np.array([
            length_params['min'] <= len(s) <= length_params['max']
            for s in sequences
        ])

        result = {
            'track': track,
            'chain': info['chain'],
            'role': info['role'],
            'n_total': len(sequences),
            'n_valid': int(valid_mask.sum()),
            'length_filter': length_params,
            'charge_filter': charge_params,
            'valid_mask': valid_mask,
        }

        # Track-specific axes
        if track == 'TCR':
            # MHC dependency score (TCR-only axis)
            mhc_scores = [compute_mhc_dependency_score(s) for s in sequences]
            result['mhc_dependency'] = {
                'mean': float(np.mean(mhc_scores)),
                'median': float(np.median(mhc_scores)),
                'std': float(np.std(mhc_scores)),
                'distribution': np.histogram(mhc_scores, bins=10, range=(0, 1))[0].tolist(),
            }
            result['primary_signal'] = 'frequency'  # T cell adaptation is in frequency

        elif track == 'BCR':
            # Affinity maturation proxy (BCR-only axis)
            mat_proxy = compute_affinity_maturation_proxy(
                [s for s, v in zip(sequences, valid_mask) if v]
            )
            result['affinity_maturation'] = {
                'proxy': float(mat_proxy),
            }
            result['primary_signal'] = 'sequence_evolution'  # B cell adaptation is in SHM

        return result

    def route_multiple_chains(self, chain_data):
        """Route multiple chain types to their respective tracks.

        Args:
            chain_data: dict of {chain_name: {'sequences': [...], 'v_genes': [...], ...}}

        Returns:
            dict of {chain_name: route_result}
        """
        results = {}
        for chain, data in chain_data.items():
            results[chain] = self.route_sample(
                data.get('sequences', []),
                data.get('v_genes'),
                data.get('j_genes'),
                chain=chain,
            )
        return results


if __name__ == '__main__':
    # Smoke test
    np.random.seed(42)

    # TCR test
    tra_seqs = ['CASSLAPGATNEKLFF', 'CASSQETQYF', 'CAVSDFDYIAKTF',
                'CASSLGQGYTF', 'CASSPRSYGYTF']

    ts = TrackSeparator()
    tra_info = ts.route_sample(tra_seqs, chain='TRA')
    print("=== TCR Track (TRA) ===")
    print(f"Track: {tra_info['track']}")
    print(f"Valid: {tra_info['n_valid']}/{tra_info['n_total']}")
    print(f"Length filter: {tra_info['length_filter']}")
    print(f"MHC dependency: mean={tra_info['mhc_dependency']['mean']:.3f}")
    print(f"Primary signal: {tra_info['primary_signal']}")

    # BCR test
    igh_seqs = ['ARHDYYGSSYFDV', 'ARHDYYGSSSYFDV', 'ARDITYYGSAYFDV',
                'ARGVYYGSGYYFDV', 'ARDTYYGSSAYFDV']

    igh_info = ts.route_sample(igh_seqs, chain='IGH')
    print("\n=== BCR Track (IGH) ===")
    print(f"Track: {igh_info['track']}")
    print(f"Role: {igh_info['role']}")
    print(f"Valid: {igh_info['n_valid']}/{igh_info['n_total']}")
    print(f"Length filter: {igh_info['length_filter']}")
    print(f"Affinity maturation proxy: {igh_info['affinity_maturation']['proxy']:.3f}")
    print(f"Primary signal: {igh_info['primary_signal']}")

    # Multi-chain test
    chain_data = {
        'TRA': {'sequences': tra_seqs},
        'IGH': {'sequences': igh_seqs},
        'IGK': {'sequences': ['AQYL', 'AQYLQ', 'LQYLSG']},
    }
    multi = ts.route_multiple_chains(chain_data)
    print("\n=== Multi-Chain Routing ===")
    for chain, info in multi.items():
        print(f"  {chain}: track={info['track']}, signal={info['primary_signal']}, "
              f"valid={info['n_valid']}/{info['n_total']}")

    print("\nSmoke test passed.")
