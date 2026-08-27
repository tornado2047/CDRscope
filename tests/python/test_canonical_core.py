import numpy as np
import pandas as pd
import pytest
from cdrscope_core.io_qc import QCConfig, infer_donor_groups, validate_airr
from cdrscope_core.spectra import SpectrumTransformer, multiscale_spectra, transform_counts
from cdrscope_core.validation import NestedGroupEvaluator
from cdrscope_core.anomaly import HealthyReferenceDetector
from cdrscope_core.diagnostics import benjamini_hochberg, rarefaction_stability


def test_replicates_share_group():
    ids = ["P01", "P01_r", "P01_r2", "P02-rep1", "P02"]
    assert infer_donor_groups(ids).tolist() == ["P01", "P01", "P01", "P02", "P02"]


def test_airr_qc_filters_invalid_and_flags_depth():
    df = pd.DataFrame({
        "sample_id": ["S1", "S1", "S2"], "cdr3_aa": ["CASSL", "BAD*", "CAVVV"],
        "chain": ["TRB", "TRB", "TRA"], "count": [1200, 3, 5],
        "productive": [True, True, True],
    })
    clean, qc = validate_airr(df, config=QCConfig(min_unique_cdr3=1, min_total_count=10))
    assert clean.cdr3_aa.tolist() == ["CASSL", "CAVVV"]
    assert bool(qc.set_index("sample_id").loc["S1", "qc_pass"])
    assert not bool(qc.set_index("sample_id").loc["S2", "qc_pass"])


def test_soft_assignment_conserves_mass():
    centroids = np.array([[0., 0.], [1., 0.], [0., 1.]])
    emb = np.array([[0.1, 0.1], [0.9, 0.1]])
    t = SpectrumTransformer(centroids, assignment="soft", top_k=2, normalization="relative")
    weights = t.assign(emb)
    assert np.allclose(weights.sum(axis=1), 1)
    spectrum = t.sample_spectrum(emb, np.array([2., 3.]))
    assert np.isclose(spectrum.sum(), 1)


def test_compositional_transforms_and_multiscale():
    x = np.array([[1., 3., 0., 0.], [0., 1., 1., 2.]])
    assert np.allclose(np.sum(transform_counts(x, "relative"), axis=1), 1)
    scales = multiscale_spectra(x, {"coarse": np.array([0, 0, 1, 1])})
    assert scales["coarse"].shape == (2, 2)


def test_grouped_nested_cv_has_no_overlap():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    groups = np.repeat([f"D{i}" for i in range(20)], 2)
    y = np.repeat([0] * 10 + [1] * 10, 2)
    X = rng.normal(size=(40, 12)); X[:, 0] += y * 1.5
    evaluator = NestedGroupEvaluator(outer_splits=4, inner_splits=3, n_bootstrap=20,
                                     c_grid=(0.1,), l1_ratios=(0.5,))
    result = evaluator.evaluate(X, y, groups)
    assert result.leakage_checks["group_overlap_max"] == 0
    assert result.leakage_checks["all_samples_out_of_fold"]
    assert len(result.predictions) == len(y)


def test_diagnostics_fdr_and_rarefaction():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.9])
    assert np.all((q >= 0) & (q <= 1))
    result = rarefaction_stability(np.array([[50, 50], [80, 20]]), depths=(20, 50), repeats=4)
    assert result[1]["median_cosine_similarity"] >= result[0]["median_cosine_similarity"] - 0.1


def test_anomaly_detector_fits_reference_only():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(40, 8))
    test = np.vstack([rng.normal(size=(4, 8)), rng.normal(loc=5, size=(4, 8))])
    detector = HealthyReferenceDetector(n_components=5).fit(ref)
    result = detector.score_samples(test)
    assert result["score"][4:].mean() > result["score"][:4].mean()
    assert np.all((result["conformal_p"] > 0) & (result["conformal_p"] <= 1))
