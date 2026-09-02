#!/usr/bin/env python3
"""
Temporal Dynamics Module (Direction 4)
=======================================
Fast/slow separation of repertoire dynamics over time.

T cells: no SHM → diversity changes slowly, frequency changes fast.
Time series analysis uses frequency as primary signal, diversity as secondary.

Components:
  T1: Δ-Frequency Spectrum — prototype count changes between timepoints
  T2: Δ-Diversity Vector — macro index changes (slow signal)
  T3: Velocity in RCS — position change rate in reference coordinate system
  QC: Sampling interval validation against response timescales
"""
import numpy as np
from collections import Counter
from .macro_indices import MacroIndexExtractor

# Minimum sampling intervals (days) for each response type
RESPONSE_TIMESCALES = {
    'acute': {'min_interval': 1, 'max_interval': 7,
              'description': 'Infection/vaccine response'},
    'chronic': {'min_interval': 7, 'max_interval': 90,
                'description': 'Chronic inflammation (RA, SLE)'},
    'aging': {'min_interval': 90, 'max_interval': 1825,
              'description': 'Aging/steady-state drift'},
}

# Sequencing noise floor (detectable clone frequency)
NOISE_FLOOR = 1e-5  # ~10^-5 detection limit


def compute_delta_frequency_spectrum(spec_t1, spec_t2, normalize=True):
    """T1: Δ-Frequency Spectrum — change in prototype counts between timepoints.

    Args:
        spec_t1: (m,) L2-normalized spectrum at time t1
        spec_t2: (m,) L2-normalized spectrum at time t2
        normalize: if True, return relative change; else absolute

    Returns:
        (m,) delta spectrum: expansion (+) / contraction (−) per prototype
    """
    delta = spec_t2 - spec_t1
    if normalize:
        denom = (spec_t1 + spec_t2) / 2 + 1e-10
        delta = delta / np.maximum(denom, 1e-10)
    return delta.astype(np.float32)


def compute_delta_diversity(counts_t1, counts_t2):
    """T2: Δ-Diversity Vector — macro index changes (slow signal).

    Args:
        counts_t1: clone count vector at time t1
        counts_t2: clone count vector at time t2

    Returns:
        (10,) delta diversity vector
    """
    ext = MacroIndexExtractor()
    m1 = ext.transform(counts_t1)
    m2 = ext.transform(counts_t2)
    return (m2 - m1).astype(np.float32)


def compute_rcs_velocity(coords_t1, coords_t2, delta_days=None):
    """T3: Velocity in RCS — displacement vector and speed.

    Args:
        coords_t1: (d,) position in RCS at time t1 (PCA or UMAP coords)
        coords_t2: (d,) position in RCS at time t2
        delta_days: time interval in days (for speed normalization)

    Returns:
        dict with:
          'displacement': (d,) displacement vector
          'magnitude': float — Euclidean distance
          'speed': float — magnitude / days (if delta_days given)
          'direction_cosine': float — cos angle from t1 to t2
    """
    disp = np.array(coords_t2, dtype=np.float32) - np.array(coords_t1, dtype=np.float32)
    mag = np.linalg.norm(disp)
    speed = mag / delta_days if delta_days and delta_days > 0 else None

    cos_sim = 0.0
    if mag > 1e-10:
        norm_disp = disp / mag
        # Direction relative to reference origin
        origin_dir = np.array(coords_t1, dtype=np.float32)
        origin_norm = np.linalg.norm(origin_dir)
        if origin_norm > 1e-10:
            cos_sim = float(np.dot(norm_disp, origin_dir) / (mag * origin_norm))

    return {
        'displacement': disp,
        'magnitude': float(mag),
        'speed': float(speed) if speed is not None else None,
        'direction_cosine': cos_sim,
    }


def validate_sampling_interval(delta_days, response_type='auto'):
    """QC: Check if sampling interval matches expected response timescale.

    Args:
        delta_days: sampling interval in days
        response_type: 'acute', 'chronic', 'aging', or 'auto'

    Returns:
        dict with validation result
    """
    if response_type == 'auto':
        if delta_days <= 7:
            response_type = 'acute'
        elif delta_days <= 90:
            response_type = 'chronic'
        else:
            response_type = 'aging'

    scale = RESPONSE_TIMESCALES.get(response_type, RESPONSE_TIMESCALES['chronic'])

    if delta_days < scale['min_interval']:
        return {
            'valid': False,
            'warning': f"Interval {delta_days}d shorter than {response_type} minimum "
                       f"({scale['min_interval']}d) — signal likely noise-dominated",
            'response_type': response_type,
            'noise_floor': NOISE_FLOOR,
        }

    if delta_days > scale['max_interval']:
        return {
            'valid': True,
            'warning': f"Interval {delta_days}d longer than {response_type} typical "
                       f"({scale['max_interval']}d) — may miss transient dynamics",
            'response_type': response_type,
            'noise_floor': NOISE_FLOOR,
        }

    return {
        'valid': True,
        'warning': None,
        'response_type': response_type,
        'noise_floor': NOISE_FLOOR,
    }


def compute_noise_floor(n_cells):
    """Estimate the detection noise floor for a given cell count.

    At n_cells, the smallest detectable clone has frequency ~1/n_cells.
    Clones below this threshold are sampling noise.

    Returns:
        (noise_floor, n_detectable_clones)
    """
    noise = 1.0 / max(n_cells, 1)
    n_detectable = int(1.0 / noise) if noise > 0 else 0
    return noise, n_detectable


class TemporalDynamicsExtractor:
    """Unified temporal dynamics extractor for longitudinal samples."""

    def __init__(self, response_type='auto'):
        self.response_type = response_type
        self.macro_ext = MacroIndexExtractor()

    def transform_pair(self, sample_t1, sample_t2, delta_days=None):
        """Extract temporal dynamics between two timepoints.

        Args:
            sample_t1: dict with 'spectrum' (m-d), 'counts', 'rcs_coords' (optional)
            sample_t2: same structure for time t2
            delta_days: time interval in days

        Returns:
            dict with T1, T2, T3, and QC results
        """
        result = {}

        # QC
        if delta_days is not None:
            qc = validate_sampling_interval(delta_days, self.response_type)
            result['qc'] = qc
            if not qc['valid']:
                print(f"  WARNING: {qc['warning']}")

        # T1: Δ-Frequency Spectrum
        if 'spectrum' in sample_t1 and 'spectrum' in sample_t2:
            result['delta_frequency'] = compute_delta_frequency_spectrum(
                sample_t1['spectrum'], sample_t2['spectrum']
            )
            # Separate expansion and contraction events
            delta = result['delta_frequency']
            result['expansion_events'] = np.sum(delta > 0.1).astype(int)
            result['contraction_events'] = np.sum(delta < -0.1).astype(int)
            result['stable_channels'] = np.sum(np.abs(delta) <= 0.1).astype(int)

        # T2: Δ-Diversity Vector
        if 'counts' in sample_t1 and 'counts' in sample_t2:
            result['delta_diversity'] = compute_delta_diversity(
                sample_t1['counts'], sample_t2['counts']
            )

        # T3: RCS Velocity
        if 'rcs_coords' in sample_t1 and 'rcs_coords' in sample_t2:
            vel = compute_rcs_velocity(
                sample_t1['rcs_coords'], sample_t2['rcs_coords'], delta_days
            )
            result['velocity'] = vel

        # Noise floor estimation
        if 'n_cells' in sample_t1:
            nf, nd = compute_noise_floor(sample_t1['n_cells'])
            result['noise_floor_t1'] = nf
            result['n_detectable_t1'] = nd
        if 'n_cells' in sample_t2:
            nf, nd = compute_noise_floor(sample_t2['n_cells'])
            result['noise_floor_t2'] = nf
            result['n_detectable_t2'] = nd

        return result

    def transform_series(self, samples, days=None):
        """Extract temporal dynamics across a time series.

        Args:
            samples: list of sample dicts, ordered by time
            days: list of day points (same length as samples)

        Returns:
            list of pairwise results (len = len(samples) - 1)
        """
        results = []
        for i in range(len(samples) - 1):
            delta_days = None
            if days is not None:
                delta_days = days[i+1] - days[i]
            r = self.transform_pair(samples[i], samples[i+1], delta_days)
            r['pair_index'] = i
            r['days_t1'] = days[i] if days is not None else i
            r['days_t2'] = days[i+1] if days is not None else i+1
            results.append(r)
        return results


if __name__ == '__main__':
    # Smoke test
    np.random.seed(42)

    # Simulate two timepoints
    m = 10000
    n_cells_t1, n_cells_t2 = 50000, 55000

    # t1: baseline spectrum
    spec_t1 = np.random.dirichlet(np.ones(m) * 0.01)
    counts_t1 = np.random.negative_binomial(2, 0.1, size=m)
    counts_t1[counts_t1 < 0] = 0

    # t2: some clones expand, others contract
    spec_t2 = spec_t1.copy()
    spec_t2[:5] *= 3  # expansion
    spec_t2[5:10] *= 0.3  # contraction
    spec_t2 /= spec_t2.sum()  # re-normalize

    counts_t2 = counts_t1.copy()
    counts_t2[:5] *= 3
    counts_t2[5:10] = (counts_t2[5:10] * 0.3).astype(int)

    # RCS coords
    coords_t1 = np.random.randn(2) * 0.05
    coords_t2 = coords_t1 + np.array([0.02, 0.01])

    sample_t1 = {
        'spectrum': spec_t1, 'counts': counts_t1, 'rcs_coords': coords_t1,
        'n_cells': n_cells_t1,
    }
    sample_t2 = {
        'spectrum': spec_t2, 'counts': counts_t2, 'rcs_coords': coords_t2,
        'n_cells': n_cells_t2,
    }

    ext = TemporalDynamicsExtractor()
    result = ext.transform_pair(sample_t1, sample_t2, delta_days=30)

    print("=== Temporal Dynamics Smoke Test ===\n")
    print(f"QC: {result.get('qc', 'N/A')}")
    print(f"Δ-frequency: shape={result['delta_frequency'].shape}, "
          f"expansion={result['expansion_events']}, "
          f"contraction={result['contraction_events']}, "
          f"stable={result['stable_channels']}")
    print(f"Δ-diversity: {result['delta_diversity']}")
    print(f"Velocity:    mag={result['velocity']['magnitude']:.6f}, "
          f"speed={result['velocity']['speed']}, "
          f"cos={result['velocity']['direction_cosine']:.4f}")
    print(f"Noise floor: t1={result['noise_floor_t1']:.2e} ({result['n_detectable_t1']} clones), "
          f"t2={result['noise_floor_t2']:.2e} ({result['n_detectable_t2']} clones)")

    # Test time series
    samples = [sample_t1, sample_t2, sample_t2]
    days = [0, 30, 60]
    series = ext.transform_series(samples, days)
    print(f"\nSeries: {len(series)} pairs")
    for s in series:
        print(f"  Pair {s['pair_index']}: t={s['days_t1']}→{s['days_t2']}d, "
              f"expansion={s['expansion_events']}, velocity={s['velocity']['magnitude']:.6f}")
    print("\nSmoke test passed.")
