"""Canonical, leakage-aware building blocks for CDRscope v2.4+."""
from .io_qc import AIRRSchema, QCConfig, validate_airr, infer_donor_groups
from .spectra import SpectrumTransformer, multiscale_spectra
from .validation import NestedGroupEvaluator
from .anomaly import HealthyReferenceDetector
from .panel import PanelManifest, verify_panel_independence
from .provenance import RunTracker
from .cache import EmbeddingCache
from .diagnostics import benjamini_hochberg, rarefaction_stability, label_permutation_test, leave_one_cohort_out

__all__ = [
    "AIRRSchema", "QCConfig", "validate_airr", "infer_donor_groups",
    "SpectrumTransformer", "multiscale_spectra", "NestedGroupEvaluator",
    "HealthyReferenceDetector", "PanelManifest", "verify_panel_independence",
    "RunTracker", "EmbeddingCache", "benjamini_hochberg",
    "rarefaction_stability", "label_permutation_test", "leave_one_cohort_out",
]
