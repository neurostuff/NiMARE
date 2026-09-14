"""Tests for nimare.meta.cbma.effectsize (coordinate-based effect-size meta-analysis)."""

import copy
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nimare.correct import FDRCorrector, FWECorrector
from nimare.generate import create_effect_size_coordinate_studyset
from nimare.meta.cbma.effectsize import (
    _NULL_Z_STEP,
    CBES,
    NULL_METHODS,
    _local_dersimonian_laird,
    _null_bin_edges,
    null_effect_variance,
    peak_stat_to_hedges_g,
)
from nimare.transforms import d_to_g, t_to_d
from nimare.utils import mm2vox

TRUTH = (0, 0, 0)


@pytest.fixture(scope="module")
def small_mask():
    """Build a small 4mm box around the origin, so the fits in these tests stay quick."""
    shape = (21, 21, 21)
    affine = np.array([[4.0, 0, 0, -40.0], [0, 4.0, 0, -40.0], [0, 0, 4.0, -40.0], [0, 0, 0, 1.0]])
    return nib.Nifti1Image(np.ones(shape, dtype=np.int32), affine)


TRUE_G = 0.5


@pytest.fixture(scope="module")
def studyset():
    """30 studies with a true g of 0.5 at the origin, thresholded at p < .001."""
    return create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=TRUE_G,
        n_studies=30,
        sample_size=(20, 40),
        tau=0.1,
        seed=7,
        n_noise_foci=1,
        noise_extent=30.0,
        spatial_sd=5.0,
    )


@pytest.fixture(scope="module")
def mixed_studyset():
    """Half the studies genuinely have no effect at the focus."""
    return create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=0.8,
        n_studies=30,
        sample_size=(20, 40),
        tau=0.1,
        prevalence=0.5,
        seed=4,
        n_noise_foci=1,
        noise_extent=30.0,
        spatial_sd=5.0,
    )


@pytest.fixture(scope="module")
def permutation_fit(studyset, small_mask):
    """One permutation fit at ``n_iters=20``, shared by the tests that all wanted the same one."""
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="permute-magnitudes", n_iters=20)
    result = estimator.fit(studyset)
    return estimator, result


def value_at(result, name, xyz=TRUTH):
    """Read one map value at an xyz (mm) location."""
    img = result.get_map(name)
    ijk = mm2vox(np.array([xyz]), img.affine)[0]
    return float(img.get_fdata()[tuple(ijk)])


def test_peak_stat_to_hedges_g_matches_transforms():
    """The one-sample path is exactly the existing t -> d -> g conversion."""
    t_values = np.array([3.5, -4.2, 6.0])
    sample_sizes = np.array([20.0, 35.0, 50.0])

    g, var_g = peak_stat_to_hedges_g(t_values, sample_sizes, stat_type="t")
    expected_g, expected_var = d_to_g(
        t_to_d(t_values, sample_sizes), sample_sizes, return_variance=True
    )

    assert np.allclose(g, expected_g)
    assert np.allclose(var_g, expected_var)
    assert g[1] < 0  # sign is carried through

    # The same t means a smaller effect in a bigger study -- the point of the conversion.
    same_t, _ = peak_stat_to_hedges_g([4.0, 4.0], [20.0, 80.0], stat_type="t")
    assert same_t[1] < same_t[0]


def test_peak_stat_to_hedges_g_z_and_t_agree_in_large_samples():
    """A z and the t it corresponds to give the same effect size."""
    from nimare.transforms import t_to_z

    sample_sizes = np.full(3, 500.0)
    t_values = np.array([3.5, 4.5, 5.5])
    z_values = t_to_z(t_values, sample_sizes - 1)

    g_from_t, _ = peak_stat_to_hedges_g(t_values, sample_sizes, stat_type="t")
    g_from_z, _ = peak_stat_to_hedges_g(z_values, sample_sizes, stat_type="z")
    assert np.allclose(g_from_t, g_from_z, atol=1e-6)


def test_peak_stat_to_hedges_g_rejects_tiny_studies():
    """A study too small for the Hedges correction is refused rather than silently converted."""
    with pytest.raises(ValueError, match="at least 4 subjects"):
        peak_stat_to_hedges_g([3.0], [3], stat_type="t")


def test_null_effect_variance_shrinks_with_sample_size():
    """A silent study still carries precision, and a larger one carries more."""
    variances = null_effect_variance(np.array([20.0, 80.0]))
    assert variances[0] > variances[1]
    assert np.allclose(variances, 1.0 / np.array([20.0, 80.0]), rtol=0.15)


def test_local_dl_reduces_to_dersimonian_laird():
    """With unit kernel weights the local estimator must be the textbook DL estimator."""
    from pymare.estimators import DerSimonianLaird

    rng = np.random.default_rng(0)
    g = rng.normal(0.5, 0.3, size=12)
    var_g = rng.uniform(0.02, 0.08, size=12)

    reference = DerSimonianLaird()
    reference.fit(y=g[:, None], v=var_g[:, None], X=np.ones((len(g), 1)))
    expected = float(np.asarray(reference.params_["tau2"]).ravel()[0])

    a = 1.0 / var_g
    actual = _local_dersimonian_laird(
        sum_w=np.array([float(len(g))]),
        sum_a=np.array([a.sum()]),
        sum_a2=np.array([(a**2).sum()]),
        sum_ag=np.array([(a * g).sum()]),
        sum_ag2=np.array([(a * g**2).sum()]),
        sum_w2_over_s2=np.array([a.sum()]),
        n_studies=np.array([float(len(g))]),
    )
    assert np.isclose(actual[0], expected)


def test_pooling_reduces_to_weighted_least_squares(studyset, small_mask):
    """With unit kernel weights the pooled estimate is ordinary inverse-variance weighting."""
    from pymare.stats import weighted_least_squares

    estimator = CBES(fwhm=8.0, mask=small_mask, null_method="none", selection_model="none")
    estimator.fit(studyset)
    fit = estimator._pool(estimator._focus_table_)

    # Rebuild one covered voxel as a dense one-voxel dataset and pool it with PyMARE.
    voxel = int(np.argmax(fit["n_studies"]))
    g, v, w = [], [], []
    for _, cols, weights, g_k, var_k in fit["contributions"]:
        hit = np.flatnonzero(cols == voxel)
        if hit.size:
            g.append(g_k[hit[0]])
            v.append(var_k[hit[0]])
            w.append(weights[hit[0]])
    g, v = np.asarray(g), np.asarray(v)
    assert len(g) >= 3

    tau2 = float(fit["tau2"][voxel])
    expected, cov = weighted_least_squares(
        g[:, None], v[:, None], np.ones((len(g), 1)), tau2=tau2, return_cov=True
    )
    # PyMARE weights purely by inverse variance, so the comparison holds at unit kernel weight.
    unit = 1.0 / (v + tau2)
    mine = float(np.sum(unit * g) / np.sum(unit))
    assert np.isclose(mine, float(np.asarray(expected).ravel()[0]))
    assert np.isclose(1.0 / np.sqrt(np.sum(unit)), float(np.sqrt(np.asarray(cov).ravel()[0])))

    # And the estimator's own pooled value is that same weighted mean once the kernel weights
    # it actually used are put back in.
    w = np.asarray(w)
    pooling = w / (v + tau2)
    assert np.isclose(fit["g"][voxel], float(np.sum(pooling * g) / np.sum(pooling)))
    assert np.isclose(
        fit["se"][voxel], float(np.sqrt(np.sum(w**2 / (v + tau2))) / np.sum(pooling))
    )


def test_local_dl_is_zero_without_two_studies():
    """Heterogeneity is not estimable from a single study, so it is reported as zero."""
    zeros = np.zeros(1)
    tau2 = _local_dersimonian_laird(
        zeros, np.ones(1), np.ones(1), np.ones(1), np.ones(1), np.ones(1), np.ones(1)
    )
    assert tau2[0] == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"design": "three-sample"}, "design must be"),
        ({"tau2_method": "reml"}, "tau2_method must be"),
        ({"selection_model": "heckman"}, "selection_model must be"),
        ({"null_method": "bootstrap"}, "null_method must be"),
        ({"threshold": object()}, "threshold must be"),
        ({"peak_bias": "shrink"}, "peak_bias must be"),
        ({"peak_bias": 1.5}, "peak_bias must be"),
    ],
)
def test_cbes_rejects_bad_parameters(kwargs, match):
    """Every option is validated at construction, before any fitting is paid for."""
    with pytest.raises(ValueError, match=match):
        CBES(**kwargs)


def test_cbes_requires_a_reported_statistic(small_mask):
    """Coordinates without statistics get a message pointing at the convergence estimators."""
    from nimare.generate import create_coordinate_studyset

    _, plain = create_coordinate_studyset(foci=1, n_studies=5, sample_size=20, seed=1)
    with pytest.raises(ValueError, match="no usable 'z_stat' or 't_stat'"):
        CBES(mask=small_mask, selection_model="none", null_method="none").fit(plain)


def test_cbes_produces_expected_maps(studyset, small_mask):
    """Every advertised map is present and in range, and the effect lands where simulated."""
    result = CBES(fwhm=12.0, mask=small_mask, null_method="none").fit(studyset)

    expected = {"g", "se", "z", "p", "logp", "tau2", "n_studies", "n_eff", "prevalence"}
    assert expected <= set(result.maps)
    assert "g_marginal" in result.maps

    p_values = result.get_map("p", return_type="array")
    assert np.all((p_values >= 0) & (p_values <= 1))

    prevalence = result.get_map("prevalence", return_type="array")
    assert np.all((prevalence >= 0) & (prevalence <= 1))

    # The effect is where it was simulated, not somewhere else. Localization is judged on z
    # rather than g: a voxel reached by a single noise focus can have a large g with no
    # precision behind it, so the raw effect-size map is not a detection statistic.
    z_map = result.get_map("z", return_type="array")
    assert value_at(result, "z") > np.percentile(z_map, 99)
    assert value_at(result, "n_studies") > 1

    # And the estimate at the truth is close to the simulated value.
    assert abs(value_at(result, "g") - TRUE_G) < 0.15

    # g_marginal is the population-average effect: prevalence times the conditional effect.
    assert np.allclose(
        result.get_map("g_marginal", return_type="array"),
        result.get_map("g", return_type="array") * prevalence,
        atol=1e-6,
    )


def test_cbes_description_mentions_the_model(studyset, small_mask):
    """The generated description names the model actually fitted."""
    result = CBES(fwhm=12.0, mask=small_mask, null_method="none").fit(studyset)
    assert "Hedges" in result.description_
    assert "censor" in result.description_.lower()

    quiet = CBES(fwhm=12.0, mask=small_mask, generate_description=False, null_method="none").fit(
        studyset
    )
    assert quiet.description_ == ""


def test_selection_model_reduces_the_winners_curse(studyset, small_mask):
    """Pooling reported peaks alone overestimates; modelling the silence pulls it back."""
    naive = CBES(fwhm=12.0, mask=small_mask, selection_model="none", null_method="none").fit(
        studyset
    )
    corrected = CBES(
        fwhm=12.0, mask=small_mask, selection_model="zero-inflated", null_method="none"
    ).fit(studyset)

    naive_g = value_at(naive, "g")
    corrected_g = value_at(corrected, "g")

    # The naive estimate is biased away from zero, in the direction theory predicts.
    assert naive_g > TRUE_G + 0.05
    assert corrected_g < naive_g
    assert abs(corrected_g - TRUE_G) < abs(naive_g - TRUE_G)


def test_zero_component_keeps_silence_from_reading_as_a_small_common_effect(
    mixed_studyset, small_mask
):
    """Half the studies are genuinely null at the focus, and the model must be able to say so."""
    result = CBES(
        fwhm=12.0, mask=small_mask, selection_model="zero-inflated", null_method="none"
    ).fit(mixed_studyset)

    assert 0.0 < value_at(result, "prevalence") < 1.0  # some studies null, some not
    assert value_at(result, "g") > 0.0  # and the effect among the rest is positive
    # The marginal is the product, and is what a convergence method would be approximating.
    assert value_at(result, "g_marginal") < value_at(result, "g")


def test_prevalence_tracks_the_simulated_fraction(small_mask):
    """Halving the fraction of studies with a real effect halves the estimated prevalence."""
    estimates = {}
    for prevalence in (1.0, 0.4):
        studyset = create_effect_size_coordinate_studyset(
            [TRUTH],
            effect_sizes=0.9,
            n_studies=30,
            sample_size=(25, 45),
            prevalence=prevalence,
            seed=11,
            n_noise_foci=2,
            noise_extent=30.0,
            spatial_sd=4.0,
        )
        result = CBES(fwhm=12.0, mask=small_mask, null_method="none").fit(studyset)
        estimates[prevalence] = value_at(result, "prevalence")

    assert estimates[1.0] > estimates[0.4]


def test_fixed_effects_option_zeroes_tau2(studyset, small_mask):
    """``tau2_method='none'`` is a fixed-effects fit, so heterogeneity is identically zero."""
    result = CBES(fwhm=12.0, mask=small_mask, tau2_method="none", null_method="none").fit(studyset)
    assert np.all(result.get_map("tau2", return_type="array") == 0)


def test_correct_fwe_montecarlo(studyset, small_mask):
    """Correcting for the family can only make a p-value larger, never smaller."""
    estimator = CBES(
        fwhm=12.0,
        mask=small_mask,
        selection_model="none",
        null_method="permute-magnitudes",
        n_iters=20,
        seed=0,
    )
    result = estimator.fit(studyset)
    maps, tables, description = estimator.correct_fwe_montecarlo(
        result, n_iters=20, seed=0, vfwe_only=True
    )

    assert tables == {}
    assert "permutation" in description
    p_corrected = 10.0 ** -maps["logp_level-voxel"]
    assert np.all((p_corrected > 0) & (p_corrected <= 1))
    assert np.all(p_corrected >= result.get_map("p", return_type="array") - 1e-6)


def test_correct_fwe_montecarlo_needs_a_fit(small_mask):
    """Correcting before fitting is an error rather than an empty result."""
    with pytest.raises(ValueError, match="requires a fitted estimator"):
        CBES(mask=small_mask, null_method="none").correct_fwe_montecarlo(None, n_iters=2)


def test_simulator_respects_the_reporting_threshold():
    """Nothing below the threshold is ever reported -- that is the censoring being simulated."""
    studyset = create_effect_size_coordinate_studyset(
        [TRUTH], effect_sizes=0.5, n_studies=20, threshold_z=3.0, seed=5, n_noise_foci=1
    )
    z_stats = studyset.coordinates["z_stat"].astype(float).values
    assert np.all(np.abs(z_stats) >= 3.0)


def test_simulator_prevalence_reduces_reporting():
    """Fewer studies with a real effect means fewer reported peaks."""
    counts = []
    for prevalence in (1.0, 0.3):
        studyset = create_effect_size_coordinate_studyset(
            [TRUTH], effect_sizes=1.0, n_studies=40, prevalence=prevalence, seed=2
        )
        counts.append(len(studyset.coordinates))
    assert counts[0] > counts[1]


# ---------------------------------------------------------------------------- inference


@pytest.fixture(scope="module")
def null_studyset():
    """30 studies reporting nothing but noise: no effect exists anywhere."""
    return create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=0.0,
        n_studies=30,
        sample_size=(20, 40),
        prevalence=0.0,
        n_noise_foci=8,
        noise_extent=30.0,
        seed=21,
    )


def test_no_null_reports_no_p_values(studyset, small_mask):
    """Absence of inference must not be mistakable for inference."""
    result = CBES(fwhm=8.0, mask=small_mask, null_method="none").fit(studyset)

    assert np.all(result.get_map("p", return_type="array") == 1.0)
    # The estimates themselves are still produced.
    assert np.any(result.get_map("g", return_type="array") != 0)
    assert "No null distribution" in result.description_


@pytest.mark.parametrize(
    "corrector,map_name",
    [
        (FDRCorrector(method="indep"), "p_corr-FDR_method-indep"),
        (FWECorrector(method="bonferroni"), "p_corr-FWE_method-bonferroni"),
    ],
)
def test_stock_correctors_work(permutation_fit, corrector, map_name):
    """The generic correctors need only a p map, which CBES provides."""
    _, result = permutation_fit
    corrected = corrector.transform(result)

    p_corr = corrected.get_map(map_name, return_type="array")
    assert np.all((p_corr > 0) & (p_corr <= 1))
    # Correction can only make p-values larger.
    assert np.all(p_corr >= result.get_map("p", return_type="array") - 1e-6)


def test_fwe_montecarlo_reports_voxel_and_cluster_levels(studyset, small_mask):
    """Voxel-level, cluster-size and cluster-mass corrections all come from one permutation."""
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="permute-magnitudes", n_iters=25)
    result = estimator.fit(studyset)
    maps, tables, description = estimator.correct_fwe_montecarlo(result, voxel_thresh=0.01)

    assert tables == {}
    assert set(maps) == {
        "logp_level-voxel",
        "z_level-voxel",
        "logp_desc-size_level-cluster",
        "z_desc-size_level-cluster",
        "logp_desc-mass_level-cluster",
        "z_desc-mass_level-cluster",
    }
    for name, values in maps.items():
        assert np.all(np.isfinite(values)), name
        if name.startswith("logp"):
            assert np.all(values >= 0)  # -log10(p) of a p in (0, 1]

    for key in (
        "values_desc-size_level-cluster_corr-fwe_method-montecarlo",
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo",
    ):
        assert len(estimator.null_distributions_[key]) == 25
    assert "corresponds to |z|" in description


def test_fwe_montecarlo_vfwe_only_returns_only_voxel_maps(permutation_fit):
    """``vfwe_only`` skips the cluster measures and says so in its description."""
    estimator, result = copy.deepcopy(permutation_fit)
    maps, _, description = estimator.correct_fwe_montecarlo(result, vfwe_only=True)

    assert set(maps) == {"logp_level-voxel", "z_level-voxel"}
    assert "voxel-level" in description


def test_cluster_null_is_built_during_fit(permutation_fit):
    """The permutations fit() runs already record cluster measures, so correcting is free."""
    estimator, _ = permutation_fit

    assert "cluster_forming_stat" in estimator.null_distributions_
    for key in (
        "values_desc-size_level-cluster_corr-fwe_method-montecarlo",
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo",
    ):
        assert len(estimator.null_distributions_[key]) == 20


def test_cluster_threshold_none_skips_the_cluster_null(studyset, small_mask):
    """Opting out of cluster inference avoids recording the cluster nulls at all."""
    estimator = CBES(
        fwhm=12.0,
        mask=small_mask,
        null_method="permute-magnitudes",
        n_iters=20,
        cluster_threshold=None,
    )
    estimator.fit(studyset)

    assert (
        "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
        not in estimator.null_distributions_
    )


def test_stat_from_histogram_finds_the_threshold_for_a_target_p():
    """The cluster-forming threshold is read off the null rather than assumed."""
    from nimare.meta.cbma.effectsize import _stat_from_histogram

    rng = np.random.default_rng(3)
    draws = np.clip(np.abs(rng.standard_normal(500_000)), 0, 50.0)
    histogram, _ = np.histogram(draws, bins=_null_bin_edges())
    histogram = histogram.astype(float)

    previous = 0.0
    for target in (0.05, 0.01, 0.001):
        stat = _stat_from_histogram(target, histogram)
        # No more than the target share of the null sits at or above it ...
        assert np.mean(draws >= stat) <= target
        # ... and it is not needlessly high: one bin lower overshoots.
        assert np.mean(draws >= stat - _NULL_Z_STEP) > target
        assert stat > previous  # a smaller p demands a larger statistic
        previous = stat

    # An empty null cannot name a threshold, and says so rather than guessing.
    assert not np.isfinite(_stat_from_histogram(0.05, np.zeros_like(histogram)))


def test_fwe_montecarlo_reuses_the_null_from_fit(permutation_fit):
    """Fitting with the Monte Carlo null already paid for the max-statistic distribution."""
    estimator, result = copy.deepcopy(permutation_fit)
    assert "values_level-voxel_corr-fwe_method-montecarlo" in estimator.null_distributions_

    cached = estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
    maps, _, _ = estimator.correct_fwe_montecarlo(result, n_iters=20, vfwe_only=True)
    assert np.array_equal(
        cached, estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
    )
    assert maps["logp_level-voxel"].shape == result.get_map("p", return_type="array").shape


def test_null_is_built_from_the_selected_statistic(studyset, small_mask):
    """The permutation refits under the same selection model the observed map used."""
    estimator = CBES(
        fwhm=12.0,
        mask=small_mask,
        selection_model="zero-inflated",
        null_method="permute-magnitudes",
        n_iters=10,
    )
    estimator.fit(studyset)
    histogram = estimator.null_distributions_["histweights_corr-none_method-montecarlo"]
    assert histogram.sum() > 0


def test_kernel_truncation_bounds_the_support(studyset, small_mask):
    """A tighter truncation lets each focus reach fewer voxels."""
    wide = CBES(fwhm=12.0, mask=small_mask, kernel_min_weight=1e-6, null_method="none").fit(
        studyset
    )
    narrow = CBES(fwhm=12.0, mask=small_mask, kernel_min_weight=0.25, null_method="none").fit(
        studyset
    )

    reached_wide = np.sum(wide.get_map("n_studies", return_type="array") > 0)
    reached_narrow = np.sum(narrow.get_map("n_studies", return_type="array") > 0)
    assert reached_narrow < reached_wide


def test_fit_chunk_ignores_studies_that_say_nothing_here():
    """The EM visits only weighted (study, voxel) pairs; padding must not change the answer."""
    rng = np.random.default_rng(0)
    n_studies, n_voxels = 6, 40
    weights = np.where(
        rng.random((n_studies, n_voxels)) < 0.5, rng.random((n_studies, n_voxels)), 0.0
    )
    kwargs = dict(
        weights=weights,
        g_obs=np.where(weights > 0, rng.normal(0.5, 0.3, weights.shape), 0.0),
        var_obs=rng.uniform(0.02, 0.1, weights.shape),
        covered=rng.random(weights.shape) < 0.7,
        tau2=rng.uniform(0.0, 0.04, n_voxels),
        null_var=rng.uniform(0.02, 0.08, (n_studies, 1)),
        cutoffs=rng.uniform(0.3, 0.8, (n_studies, 1)),
        start=rng.normal(0.5, 0.2, n_voxels),
    )
    estimator = CBES(max_iter=8, null_method="none")
    baseline = estimator._fit_chunk(**kwargs)

    # Three extra studies that are covered everywhere and reach no voxel.
    padded = dict(kwargs)
    pad = np.zeros((3, n_voxels))
    padded["weights"] = np.vstack([kwargs["weights"], pad])
    padded["g_obs"] = np.vstack([kwargs["g_obs"], pad])
    padded["var_obs"] = np.vstack([kwargs["var_obs"], np.ones((3, n_voxels))])
    padded["covered"] = np.vstack([kwargs["covered"], np.ones((3, n_voxels), dtype=bool)])
    padded["null_var"] = np.vstack([kwargs["null_var"], np.full((3, 1), 0.05)])
    padded["cutoffs"] = np.vstack([kwargs["cutoffs"], np.full((3, 1), 0.5)])
    padded_result = estimator._fit_chunk(**padded)

    for name, before, after in zip(("mu", "prevalence", "se"), baseline, padded_result):
        finite = np.isfinite(before)
        assert np.array_equal(finite, np.isfinite(after)), name
        assert np.allclose(before[finite], after[finite], rtol=1e-12), name


# ------------------------------------------------------------- images alongside coordinates


@pytest.fixture(scope="module")
def image_studyset(tmp_path_factory):
    """Ten studies supplying both g images and the peaks thresholded out of them."""
    directory = tmp_path_factory.mktemp("cbes_images")
    shape = (10, 10, 10)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -18.0

    grid = np.indices(shape).astype(float)
    centre = (np.array(shape) - 1) / 2.0
    truth = 0.8 * np.exp(-sum((grid[i] - centre[i]) ** 2 for i in range(3)) / 8.0)

    rng = np.random.default_rng(0)
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), directory / "mask.nii.gz")
    studies = []
    for k in range(10):
        n_subjects = int(rng.integers(20, 40))
        observed = truth + rng.normal(0, 1 / np.sqrt(n_subjects), shape)
        nib.save(nib.Nifti1Image(observed.astype(np.float32), affine), directory / f"{k}_g.nii.gz")
        nib.save(
            nib.Nifti1Image(np.full(shape, 1.0 / n_subjects, np.float32), affine),
            directory / f"{k}_var.nii.gz",
        )
        # One reported peak at the true centre, with the statistic implied by the image.
        value = float(observed[tuple(int(c) for c in centre)])
        bias = 1.0 - 3.0 / (4.0 * (n_subjects - 1) - 1)
        from nimare.transforms import t_to_z

        z = float(t_to_z(np.array([value / bias * np.sqrt(n_subjects)]), n_subjects - 1)[0])
        studies.append(
            {
                "id": f"s{k}",
                "name": f"s{k}",
                "metadata": {"sample_sizes": [n_subjects]},
                "analyses": [
                    {
                        "id": f"s{k}-1",
                        "name": "1",
                        "metadata": {"sample_sizes": [n_subjects]},
                        "points": [
                            {
                                "space": "MNI",
                                "coordinates": [0.0, 0.0, 0.0],
                                "values": [{"kind": "Z", "value": z}],
                            }
                        ],
                        "images": [
                            {
                                "url": str(directory / f"{k}_g.nii.gz"),
                                "filename": f"{k}_g.nii.gz",
                                "space": "MNI",
                                "value_type": "g",
                            },
                            {
                                "url": str(directory / f"{k}_var.nii.gz"),
                                "filename": f"{k}_var.nii.gz",
                                "space": "MNI",
                                "value_type": "g_var",
                            },
                        ],
                    }
                ],
            }
        )

    from nimare.studyset import Studyset

    studyset = Studyset(
        {"id": "img", "name": "img", "studies": studies},
        target=None,
        mask=str(directory / "mask.nii.gz"),
    )
    return studyset, truth


@pytest.fixture(scope="module")
def mixed_image_studyset(tmp_path_factory):
    """Eight studies, of which the first three supply g images; the rest are coordinates only.

    The image calibration needs both kinds in one collection -- it compares a coordinate-only
    fit against each donor's own fit -- so a collection where every study has an image cannot
    exercise it.
    """
    directory = tmp_path_factory.mktemp("cbes_mixed_images")
    shape = (10, 10, 10)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -18.0

    grid = np.indices(shape).astype(float)
    centre = (np.array(shape) - 1) / 2.0
    truth = 0.8 * np.exp(-sum((grid[i] - centre[i]) ** 2 for i in range(3)) / 8.0)

    from nimare.studyset import Studyset
    from nimare.transforms import t_to_z

    rng = np.random.default_rng(3)
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), directory / "mask.nii.gz")
    studies = []
    for k in range(8):
        n_subjects = int(rng.integers(20, 40))
        observed = truth + rng.normal(0, 1 / np.sqrt(n_subjects), shape)
        value = float(observed[tuple(int(c) for c in centre)])
        bias = 1.0 - 3.0 / (4.0 * (n_subjects - 1) - 1)
        z = float(t_to_z(np.array([value / bias * np.sqrt(n_subjects)]), n_subjects - 1)[0])
        analysis = {
            "id": f"m{k}-1",
            "name": "1",
            "metadata": {"sample_sizes": [n_subjects]},
            "points": [
                {
                    "space": "MNI",
                    "coordinates": [float(c * 4.0 - 18.0) for c in centre],
                    "values": [{"kind": "Z", "value": z}],
                }
            ],
            "images": [],
        }
        if k < 3:
            nib.save(
                nib.Nifti1Image(observed.astype(np.float32), affine), directory / f"m{k}_g.nii.gz"
            )
            nib.save(
                nib.Nifti1Image(np.full(shape, 1.0 / n_subjects, np.float32), affine),
                directory / f"m{k}_var.nii.gz",
            )
            analysis["images"] = [
                {
                    "url": str(directory / f"m{k}_g.nii.gz"),
                    "filename": f"m{k}_g.nii.gz",
                    "space": "MNI",
                    "value_type": "g",
                },
                {
                    "url": str(directory / f"m{k}_var.nii.gz"),
                    "filename": f"m{k}_var.nii.gz",
                    "space": "MNI",
                    "value_type": "g_var",
                },
            ]
        studies.append(
            {
                "id": f"m{k}",
                "name": f"m{k}",
                "metadata": {"sample_sizes": [n_subjects]},
                "analyses": [analysis],
            }
        )

    return Studyset(
        {"id": "mixed", "name": "mixed", "studies": studies},
        target=None,
        mask=str(directory / "mask.nii.gz"),
    )


def test_images_are_used_in_place_of_coordinates(image_studyset):
    """An image supersedes that study's own peaks: it says more, with no selection."""
    studyset, _ = image_studyset
    estimator = CBES(fwhm=8.0, null_method="none", use_images=True)
    estimator.fit(studyset)

    assert len(estimator._image_studies_) == 10
    # Every study had an image, so no coordinate survives into the focus table.
    assert len(estimator._focus_table_) == 0


def test_images_recover_the_truth_better_than_coordinates(image_studyset):
    """The point of admitting images: they are unbiased where reported peaks are not."""
    studyset, truth = image_studyset
    truth_vector = studyset.masker.transform(
        nib.Nifti1Image(truth.astype(np.float32), studyset.masker.mask_img.affine)
    ).ravel()

    from_images = (
        CBES(fwhm=8.0, null_method="none", use_images=True)
        .fit(studyset)
        .get_map("g", return_type="array")
        .ravel()
    )
    from_coords = (
        CBES(fwhm=8.0, null_method="none", use_images=False)
        .fit(studyset)
        .get_map("g", return_type="array")
        .ravel()
    )
    hot = truth_vector > 0.2
    assert abs(from_images[hot].mean() - truth_vector[hot].mean()) < 0.15

    # Compare only where both produced an estimate: a voxel no kernel reaches is reported as
    # zero by the coordinate fit, which would drag its mean down for reasons unrelated to bias.
    both = hot & (from_coords != 0) & (from_images != 0)
    assert both.sum() > 10
    # Reported peaks are local maxima, so they overstate the same effect.
    assert from_coords[both].mean() > from_images[both].mean()


def test_use_images_false_ignores_them(image_studyset):
    """``use_images=False`` falls back to the coordinates even when images are available."""
    studyset, _ = image_studyset
    estimator = CBES(fwhm=8.0, null_method="none", use_images=False)
    estimator.fit(studyset)
    assert estimator._image_studies_ == {}
    assert len(estimator._focus_table_) == 10


def test_peak_bias_rescales_the_estimate_exactly(image_studyset):
    """Rho rescales g, its variance and the threshold together, so the fit scales with it."""
    studyset, _ = image_studyset
    plain = (
        CBES(fwhm=8.0, null_method="none", use_images=False)
        .fit(studyset)
        .get_map("g", return_type="array")
        .ravel()
    )
    scaled = (
        CBES(fwhm=8.0, null_method="none", use_images=False, peak_bias=0.4)
        .fit(studyset)
        .get_map("g", return_type="array")
        .ravel()
    )
    covered = plain != 0
    assert np.allclose(scaled[covered], 0.4 * plain[covered], rtol=1e-6)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_peak_bias_rejects_out_of_range_values(bad):
    """A rho outside (0, 1] would inflate rather than discount the reported peaks."""
    with pytest.raises(ValueError, match="peak_bias must be None"):
        CBES(peak_bias=bad)


def test_null_peak_overshoot_matches_the_rft_expectation():
    """A pure-noise peak sits about 1/u above the threshold."""
    from nimare.meta.cbma.effectsize import null_peak_overshoot

    for u in (2.5, 3.2905, 4.0):
        mean_height = null_peak_overshoot(u)
        assert u < mean_height < u + 1.0
        assert abs((mean_height - u) - 1.0 / u) < 0.15


def test_peak_information_separates_signal_from_the_null_floor():
    """Peaks carrying real effect sit above the height pure noise reaches at the same threshold."""
    from nimare.meta.cbma.effectsize import peak_information

    u = 3.2905
    # Heights indistinguishable from null peaks carry no effect-size information.
    observed, expected, excess = peak_information(np.full(500, 3.63), u)
    assert abs(excess) < 0.1
    # Heights well above the null floor do.
    _, _, excess_signal = peak_information(np.full(500, 5.5), u)
    assert excess_signal > 1.5


def test_uninformative_peaks_are_flagged(small_mask, caplog):
    """The estimator says so when the reported magnitudes cannot identify the effect size."""
    studyset = create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=0.0,
        n_studies=20,
        sample_size=(20, 40),
        prevalence=0.0,
        n_noise_foci=6,
        noise_extent=30.0,
        seed=5,
    )
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="none")
    with caplog.at_level("WARNING"):
        estimator.fit(studyset)

    assert "uninformative" in caplog.text or "close to uninformative" in caplog.text
    assert estimator.peak_information_["excess_z"] < 0.25


def test_null_peak_mean_g_grows_with_threshold_and_shrinks_with_n():
    """The artefact in a reported effect size is a function of (u, N), and a strong one."""
    from nimare.meta.cbma.effectsize import null_peak_mean_g

    at_n20 = [null_peak_mean_g(u, 20) for u in (2.3, 3.29, 4.3)]
    assert at_n20[0] < at_n20[1] < at_n20[2]

    at_u33 = [null_peak_mean_g(3.29, n) for n in (15, 30, 60)]
    assert at_u33[0] > at_u33[1] > at_u33[2]


def test_infer_threshold_from_minimum_recovers_a_known_threshold():
    """Simulate the RFT peak-height null, take the minimum of m draws, recover u."""
    from nimare.meta.cbma.effectsize import infer_threshold_from_minimum

    rng = np.random.default_rng(0)
    for true_u in (2.3, 3.2905, 4.0):
        for n_peaks in (3, 8, 20):
            # Inverse-transform sampling from S(z|u) = (z^2-1)exp(-z^2/2) / (u^2-1)exp(-u^2/2).
            grid = np.linspace(true_u, true_u + 8.0, 4000)
            survival = (grid**2 - 1) * np.exp(-0.5 * grid**2)
            survival = survival / survival[0]
            minima = [
                np.interp(rng.random(n_peaks), survival[::-1], grid[::-1]).min()
                for _ in range(400)
            ]
            raw = float(np.mean(minima))
            fixed = float(np.mean([infer_threshold_from_minimum(m, n_peaks) for m in minima]))
            assert raw > true_u  # the minimum of m peaks always overshoots
            assert abs(fixed - true_u) < abs(raw - true_u)
            assert abs(fixed - true_u) < 0.15


def test_study_min_undoes_the_order_statistic(studyset, small_mask):
    """``"study-min"`` infers each study's threshold rather than believing the minimum."""
    estimator = CBES(fwhm=8.0, null_method="none", threshold="study-min", mask=small_mask)
    estimator.fit(studyset)

    table = estimator._focus_table_
    raw = (
        pd.Series(estimator._reported_z(table), index=np.asarray(table["id"].values, dtype=object))
        .groupby(level=0)
        .min()
    )
    inferred = estimator._cutoffs_z_.reindex(raw.index)

    assert (inferred <= raw + 1e-8).all()
    assert (inferred < raw - 1e-3).any()


def test_threshold_can_name_a_metadata_field(small_mask):
    """Papers that state their threshold should not be put through an inference."""
    studyset = create_effect_size_coordinate_studyset(
        [TRUTH], effect_sizes=0.9, n_studies=20, sample_size=25, threshold_z=3.0, seed=11
    )
    from_metadata = CBES(
        fwhm=8.0, null_method="none", threshold="reporting_threshold", mask=small_mask
    ).fit(studyset)
    from_float = CBES(fwhm=8.0, null_method="none", threshold=3.0, mask=small_mask).fit(studyset)

    assert np.allclose(
        from_metadata.get_map("g", return_type="array"),
        from_float.get_map("g", return_type="array"),
    )


def test_threshold_metadata_field_must_exist(studyset, small_mask):
    """Naming a missing metadata field fails loudly instead of falling back to a default."""
    estimator = CBES(fwhm=8.0, null_method="none", threshold="nope", mask=small_mask)
    with pytest.raises(ValueError, match="metadata field"):
        estimator.fit(studyset)


def test_per_study_peak_bias_discounts_strict_thresholds_and_small_samples():
    """rho_k is the inverse of the artefact, so it falls as u rises and as N falls."""
    import pandas as pd

    estimator = CBES(peak_bias="per-study")
    ids = ["a", "b", "c"]

    sizes = pd.Series([20.0, 20.0, 20.0], index=ids)
    by_threshold = estimator._peak_bias_factors(pd.Series([2.3, 3.29, 4.3], index=ids), sizes, ids)
    assert by_threshold["a"] > by_threshold["b"] > by_threshold["c"]
    assert by_threshold["b"] == pytest.approx(1.0)  # the median study anchors the scale

    cutoffs = pd.Series([3.29, 3.29, 3.29], index=ids)
    by_size = estimator._peak_bias_factors(cutoffs, pd.Series([15.0, 30.0, 60.0], index=ids), ids)
    assert by_size["a"] < by_size["b"] < by_size["c"]


def test_per_study_peak_bias_reduces_to_the_scalar_when_studies_agree(small_mask):
    """With one threshold and one sample size there is nothing between studies to correct."""
    studyset = create_effect_size_coordinate_studyset(
        [TRUTH], effect_sizes=0.9, n_studies=20, sample_size=25, threshold_z=3.0, seed=12
    )
    common = dict(fwhm=8.0, null_method="none", threshold=3.0, mask=small_mask)
    per_study = CBES(peak_bias="per-study", peak_bias_scale=0.4, **common).fit(studyset)
    scalar = CBES(peak_bias=0.4, **common).fit(studyset)

    assert np.allclose(
        per_study.get_map("g", return_type="array"),
        scalar.get_map("g", return_type="array"),
    )


def test_per_study_peak_bias_equalizes_a_mixed_threshold_collection(small_mask):
    """The point of the correction: studies that thresholded differently stop disagreeing."""
    from nimare.meta.cbma.effectsize import null_peak_mean_g

    def gap(peak_bias):
        estimator = CBES(
            fwhm=8.0,
            null_method="none",
            threshold="reporting_threshold",
            peak_bias=peak_bias,
            mask=small_mask,
        )
        estimator.fit(studyset)
        table = estimator._focus_table_
        cutoffs = estimator._cutoffs_z_.reindex(table["id"].values).values
        strict = np.abs(table["g"].values[cutoffs > 3.5])
        lenient = np.abs(table["g"].values[cutoffs < 3.5])
        return strict.mean() / lenient.mean()

    studyset = create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=0.6,
        n_studies=40,
        sample_size=25,
        threshold_z=[2.3263, 4.2649],
        seed=13,
        n_noise_foci=2,
        noise_extent=30.0,
    )
    assert null_peak_mean_g(4.2649, 25) > null_peak_mean_g(2.3263, 25)
    uncorrected, corrected = gap(None), gap("per-study")
    assert uncorrected > 1.2
    assert abs(corrected - 1.0) < abs(uncorrected - 1.0)


@pytest.fixture(scope="module")
def half_image_studyset(image_studyset):
    """Build ten studies where only the first five supply images."""
    from nimare.studyset import Studyset

    studyset, truth = image_studyset
    paths = {
        str(row.id): (row.g, row.g_var)
        for row in studyset.images.itertuples()
        if row.g is not None
    }
    coords = studyset.coordinates
    sizes = dict(zip([str(i) for i in studyset.ids], studyset.sample_sizes()))

    studies = []
    for position, (analysis_id, sub) in enumerate(coords.groupby("id")):
        analysis_id = str(analysis_id)
        study_id = analysis_id.split("-")[0]
        meta = {"sample_sizes": [int(sizes[analysis_id])]}
        analysis = {
            "id": analysis_id,
            "name": "1",
            "metadata": meta,
            "points": [
                {
                    "space": "MNI",
                    "coordinates": [float(row.x), float(row.y), float(row.z)],
                    "values": [{"kind": "Z", "value": float(row.z_stat)}],
                }
                for row in sub.itertuples()
            ],
        }
        if position < 5 and analysis_id in paths:
            g_path, var_path = paths[analysis_id]
            analysis["images"] = [
                {"url": str(g_path), "filename": "g", "space": "MNI", "value_type": "g"},
                {
                    "url": str(var_path),
                    "filename": "g_var",
                    "space": "MNI",
                    "value_type": "g_var",
                },
            ]
        studies.append(
            {"id": study_id, "name": study_id, "metadata": meta, "analyses": [analysis]}
        )

    mixed = Studyset(
        {"id": "half", "name": "half", "studies": studies},
        target=None,
        mask=studyset.masker.mask_img,
    )
    return mixed, truth


def test_mixing_images_with_an_uncalibrated_scale_warns(half_image_studyset, caplog):
    """The default scale is the wrong one as soon as images are in the fit, so say so."""
    studyset, _ = half_image_studyset
    estimator = CBES(fwhm=8.0, null_method="none", peak_bias="per-study", peak_bias_scale=1.0)
    with caplog.at_level("WARNING"):
        estimator.fit(studyset)
    assert "peak_bias_scale" in caplog.text


def test_auto_peak_bias_scale_puts_coordinates_on_the_images_scale(half_image_studyset):
    """'auto' reads the constant off the images, and it has to be the ratio it corrects."""
    studyset, _ = half_image_studyset
    estimator = CBES(fwhm=8.0, null_method="none", peak_bias="per-study", peak_bias_scale="auto")
    estimator.fit(studyset)
    scale = estimator._peak_bias_scale_

    assert 0.0 < scale < 1.0  # a reported peak overstates the field around it

    uncalibrated = CBES(fwhm=8.0, null_method="none", peak_bias="per-study", peak_bias_scale=1.0)
    uncalibrated.fit(studyset)
    assert np.allclose(
        estimator._peak_bias_.values, scale * uncalibrated._peak_bias_.values, rtol=1e-8
    )


def test_auto_peak_bias_scale_falls_back_without_images(studyset, small_mask, caplog):
    """Nothing to calibrate against is a warning and a relative map, not a failure."""
    estimator = CBES(
        fwhm=8.0,
        null_method="none",
        peak_bias="per-study",
        peak_bias_scale="auto",
        mask=small_mask,
    )
    with caplog.at_level("WARNING"):
        estimator.fit(studyset)
    assert "needs images" in caplog.text
    assert estimator._peak_bias_scale_ == 1.0


@pytest.mark.parametrize("bad", ["biggest", 0.0, -1.0])
def test_peak_bias_scale_rejects_bad_values(bad):
    """Only the documented keywords and positive floats set the scale."""
    with pytest.raises(ValueError, match="peak_bias_scale must be"):
        CBES(peak_bias_scale=bad)


def test_unknown_null_methods_are_rejected():
    """An unrecognised null method is refused at construction."""
    with pytest.raises(ValueError, match="null_method must be"):
        CBES(null_method="factorised")
    # The relocation nulls were removed, not renamed; asking for one is an error.
    for removed in ("montecarlo", "approximate"):
        with pytest.raises(ValueError, match="null_method must be"):
            CBES(null_method=removed)


@pytest.fixture(scope="module")
def images_only_studyset(tmp_path_factory):
    """Six studies supplying g images and no coordinates at all."""
    from nimare.studyset import Studyset

    directory = tmp_path_factory.mktemp("cbes_images_only")
    shape = (8, 8, 8)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -14.0
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), directory / "mask.nii.gz")

    rng = np.random.default_rng(0)
    studies = []
    for k in range(6):
        n = int(rng.integers(20, 40))
        g = (0.5 + rng.normal(0, 1 / np.sqrt(n), shape)).astype(np.float32)
        nib.save(nib.Nifti1Image(g, affine), directory / f"{k}_g.nii.gz")
        nib.save(
            nib.Nifti1Image(np.full(shape, 1.0 / n, np.float32), affine),
            directory / f"{k}_var.nii.gz",
        )
        studies.append(
            {
                "id": f"s{k}",
                "name": f"s{k}",
                "metadata": {"sample_sizes": [n]},
                "analyses": [
                    {
                        "id": f"s{k}-1",
                        "name": "1",
                        "metadata": {"sample_sizes": [n]},
                        "points": [],
                        "images": [
                            {
                                "url": str(directory / f"{k}_g.nii.gz"),
                                "filename": f"{k}_g.nii.gz",
                                "space": "MNI",
                                "value_type": "g",
                            },
                            {
                                "url": str(directory / f"{k}_var.nii.gz"),
                                "filename": f"{k}_var.nii.gz",
                                "space": "MNI",
                                "value_type": "g_var",
                            },
                        ],
                    }
                ],
            }
        )
    source = directory / "studyset.json"
    source.write_text(json.dumps({"id": "imgs", "name": "imgs", "studies": studies}))
    return Studyset(str(source)), str(directory / "mask.nii.gz")


def test_the_null_sign_flips_images_rather_than_holding_them_fixed():
    """Coordinates and images are exchangeable in different ways, so the null must move both."""
    estimator = CBES()
    estimator._image_studies_ = {
        "a": (np.array([1.0, -2.0, 3.0]), np.array([0.1, 0.1, 0.1]), np.ones(3, bool)),
        "b": (np.array([4.0, 5.0, -6.0]), np.array([0.2, 0.2, 0.2]), np.ones(3, bool)),
    }
    seen = set()
    for seed in range(40):
        flipped = estimator._flip_image_signs(np.random.default_rng(seed))
        for name, (g, var_g, usable) in flipped.items():
            original = estimator._image_studies_[name]
            # Either the map or its negation, never anything else, and variances untouched.
            assert np.allclose(g, original[0]) or np.allclose(g, -original[0])
            assert np.array_equal(var_g, original[1])
            assert np.array_equal(usable, original[2])
            seen.add((name, bool(np.allclose(g, -original[0]))))
    # Both signs must actually occur, or this is not a permutation.
    assert seen == {("a", True), ("a", False), ("b", True), ("b", False)}

    # A coordinate-only fit has nothing to flip and must be left exactly alone.
    estimator._image_studies_ = {}
    assert not estimator._flip_image_signs(np.random.default_rng(0))


def test_images_only_collection_is_redirected_to_an_image_estimator(images_only_studyset):
    """A collection with images and no coordinates is an error, not a quiet reduction."""
    studyset, mask = images_only_studyset
    for null_method in ("permute-magnitudes", "none"):
        with pytest.raises(ValueError, match="coordinate-based estimator") as raised:
            CBES(fwhm=10.0, mask=mask, use_images=True, null_method=null_method).fit(studyset)
        assert "nimare.meta.ibma" in str(raised.value)


def test_a_collection_with_neither_coordinates_nor_images_still_raises(tmp_path):
    """The images-only path must not swallow the genuinely empty case."""
    from nimare.studyset import Studyset

    source = tmp_path / "empty.json"
    source.write_text(
        json.dumps(
            {
                "id": "empty",
                "name": "empty",
                "studies": [
                    {
                        "id": "s0",
                        "name": "s0",
                        "metadata": {"sample_sizes": [20]},
                        "analyses": [
                            {
                                "id": "s0-1",
                                "name": "1",
                                "metadata": {"sample_sizes": [20]},
                                "points": [],
                                "images": [],
                            }
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="no data for 'coordinates'"):
        CBES(null_method="none").fit(Studyset(str(source)))


def test_the_only_null_is_the_permutation_one_and_removed_options_fail_loudly():
    """CBES estimates effect size, so its null randomizes magnitudes, not positions."""
    assert CBES().null_method == "permute-magnitudes"
    assert set(NULL_METHODS) == {"permute-magnitudes", "none"}

    for removed in ("montecarlo", "approximate"):
        with pytest.raises(ValueError, match="null_method must be"):
            CBES(null_method=removed)
    for gone in ("_null_iteration", "_compute_montecarlo_null", "_approximate_null"):
        assert not hasattr(CBES, gone)

    with pytest.raises(TypeError):
        CBES(censoring="rft")
    with pytest.raises(TypeError):
        CBES(smoothness_fwhm=12.0)


def test_auto_scale_falls_back_to_a_relative_map_without_images(studyset, small_mask, caplog):
    """``"auto"`` has nothing to calibrate against here, and must say so rather than guess."""
    estimator = CBES(
        fwhm=8.0,
        mask=small_mask,
        null_method="none",
        peak_bias="per-study",
        peak_bias_scale="auto",
    )
    with caplog.at_level("WARNING"):
        result = estimator.fit(studyset)

    assert estimator._peak_bias_scale_ == 1.0
    assert estimator.scale_source_ == "unset"
    assert "g_absolute" not in result.maps
    assert "g_relative" in result.maps
    assert "needs images" in caplog.text


def test_the_external_corpus_prior_is_gone_and_fails_loudly():
    """Borrowing the scale from another corpus assumed this collection resembled it."""
    from nimare.meta.cbma import effectsize as module

    assert module.PEAK_BIAS_SCALE_KEYWORDS == ("auto", "images")
    with pytest.raises(ValueError):
        CBES(peak_bias="per-study", peak_bias_scale="reference")
    for gone in ("reference_magnitude", "REFERENCE_MAGNITUDE_BY_N", "REFERENCE_MAGNITUDE_LOG_SD"):
        assert not hasattr(module, gone)
    assert not hasattr(CBES, "_calibrate_scale_from_reference")


def test_permuting_magnitudes_leaves_the_spatial_design_untouched(small_mask):
    """The invariant the null depends on, tested where it is easiest to break."""
    rng = np.random.default_rng(11)
    studies = []
    for k in range(12):
        n_subjects = int(rng.integers(20, 40))
        anchor = rng.uniform(-20, 20, 3)
        points = [
            {
                "space": "MNI",
                # deliberately tight: within one kernel width of each other
                "coordinates": [float(v) for v in anchor + rng.normal(0, 3, 3)],
                "values": [{"kind": "Z", "value": float(rng.uniform(3.3, 5.0))}],
            }
            for _ in range(4)
        ]
        studies.append(
            {
                "id": f"s{k}",
                "name": f"s{k}",
                "metadata": {"sample_sizes": [n_subjects]},
                "analyses": [
                    {
                        "id": f"s{k}-1",
                        "name": "1",
                        "metadata": {"sample_sizes": [n_subjects]},
                        "points": points,
                        "images": [],
                    }
                ],
            }
        )

    from nimare.studyset import Studyset

    clustered = Studyset({"id": "clus", "name": "clus", "studies": studies})
    estimator = CBES(fwhm=10.0, mask=small_mask, null_method="none")
    estimator.fit(clustered)
    table = estimator._focus_table_
    args = (estimator._sample_sizes_, estimator._thresholds_, estimator._image_studies_)
    observed_fit, _ = estimator._statistic(table, *args)

    for seed in range(5):
        permuted = estimator._permute_magnitudes(np.random.default_rng(seed))

        # The spatial design is untouched, column by column rather than in aggregate.
        for column in ("i", "j", "k", "id", "sample_size"):
            assert np.array_equal(permuted[column].values, table[column].values), column
        # The values moved, and only by reordering.
        assert not np.array_equal(permuted["g"].values, table["g"].values)
        assert sorted(permuted["g"].values) == sorted(table["g"].values)
        # A value keeps its own variance, or the pooling weights would be nonsense.
        pairs = {(g, v) for g, v in zip(table["g"].values, table["var_g"].values)}
        assert {(g, v) for g, v in zip(permuted["g"].values, permuted["var_g"].values)} == pairs

        # The consequence, and the reason for all of the above.
        permuted_fit, _ = estimator._statistic(permuted, *args)
        assert np.array_equal(observed_fit["covered"], permuted_fit["covered"])
        assert np.array_equal(observed_fit["n_studies"], permuted_fit["n_studies"])

    # The arrangement really does put several foci of one study within reach of one voxel,
    # so the guard above is testing something.
    assert observed_fit["n_studies"].max() < len(table)


def test_convergence_alone_does_not_make_a_voxel_significant(small_mask):
    """The property that decides the null was worth changing for."""
    rng = np.random.default_rng(5)
    studies = []
    for k in range(30):
        n_subjects = int(rng.integers(20, 40))
        # one peak on the convergence site, three scattered; all heights from one distribution
        positions = [(0.0, 0.0, 0.0)] + [
            tuple(float(v) for v in rng.uniform(-30, 30, 3)) for _ in range(3)
        ]
        points = [
            {
                "space": "MNI",
                "coordinates": list(position),
                "values": [{"kind": "Z", "value": float(rng.uniform(3.4, 4.2))}],
            }
            for position in positions
        ]
        studies.append(
            {
                "id": f"s{k}",
                "name": f"s{k}",
                "metadata": {"sample_sizes": [n_subjects]},
                "analyses": [
                    {
                        "id": f"s{k}-1",
                        "name": "1",
                        "metadata": {"sample_sizes": [n_subjects]},
                        "points": points,
                        "images": [],
                    }
                ],
            }
        )

    from nimare.studyset import Studyset

    convergent = Studyset({"id": "conv", "name": "conv", "studies": studies})

    estimator = CBES(
        fwhm=12.0,
        mask=small_mask,
        selection_model="none",
        null_method="permute-magnitudes",
        n_iters=100,
        seed=0,
    )
    result = estimator.fit(convergent)

    centre = int(np.ravel_multi_index((10, 10, 10), small_mask.shape))
    p_values = result.get_map("p", return_type="array")
    z_values = result.get_map("z", return_type="array")

    # The site really is the one every study reported at, and the statistic really is large
    # there -- thirty studies make the standard error small. The p-value is still not small.
    assert result.get_map("n_studies", return_type="array")[centre] == 30
    assert abs(z_values[centre]) > 3
    assert p_values[centre] > 0.05


def test_permutation_null_calibrates_uncorrected_p(null_studyset, small_mask):
    """Under a global null the permutation null must also return roughly the nominal rate."""
    permutation = CBES(
        fwhm=12.0,
        mask=small_mask,
        selection_model="none",
        null_method="permute-magnitudes",
        n_iters=50,
    ).fit(null_studyset)

    assert np.mean(permutation.get_map("p", return_type="array") < 0.05) < 0.15


def test_fwe_correction_permutes_even_without_a_null_from_fit(null_studyset, small_mask):
    """A maximum statistic has to come from somewhere, so the correction permutes on demand."""
    estimator = CBES(
        fwhm=12.0,
        mask=small_mask,
        selection_model="none",
        null_method="none",
        n_iters=30,
    )
    result = estimator.fit(null_studyset)
    assert "values_level-voxel_corr-fwe_method-montecarlo" not in estimator.null_distributions_

    maps, _, description = estimator.correct_fwe_montecarlo(result, vfwe_only=True)

    assert np.all(np.isfinite(maps["logp_level-voxel"]))
    assert "permutation" in description
    assert (
        len(estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]) == 30
    )


def test_scale_is_reported_as_an_interval_or_not_at_all(
    studyset, small_mask, mixed_image_studyset
):
    """A partially identified scale must not be handed over as a bare point estimate."""
    coordinates_only = CBES(fwhm=8.0, mask=small_mask, null_method="none", peak_bias="per-study")
    coordinates_only.fit(studyset)
    assert coordinates_only.scale_interval_ is None

    from_images = CBES(
        fwhm=8.0, null_method="none", peak_bias="per-study", peak_bias_scale="images"
    )
    result = from_images.fit(mixed_image_studyset)
    interval = from_images.scale_interval_
    assert interval is not None
    low, high = interval
    # The bounds are the spread of the donors' own estimates, so the pooled median sits inside.
    assert 0 < low <= from_images._peak_bias_scale_ <= high
    assert from_images.scale_source_ == "images"
    assert "order of scale" in result.description_


def test_the_description_reports_what_the_peak_heights_carry(studyset, small_mask):
    """The magnitude caveat belongs in the methods text, not only in the log."""
    estimator = CBES(fwhm=8.0, mask=small_mask, null_method="none", peak_bias="per-study")
    result = estimator.fit(studyset)

    assert set(estimator.peak_information_) == {
        "observed_mean_z",
        "null_peak_mean_z",
        "excess_z",
    }
    description = result.description_
    assert "peaks of pure noise" in description
    informative = estimator.peak_information_["excess_z"] >= 0.25
    if informative:
        assert "carry information about the size of the effect" in description
    else:
        assert "relative map only" in description


def test_the_relative_map_cancels_the_scale_the_coordinates_cannot_identify(studyset, small_mask):
    """Two fits differing only by the scale constant must give the same relative map."""
    one = CBES(fwhm=8.0, mask=small_mask, null_method="none", peak_bias="per-study")
    other = CBES(
        fwhm=8.0,
        mask=small_mask,
        null_method="none",
        peak_bias="per-study",
        peak_bias_scale=0.5,
    )
    first = one.fit(studyset)
    second = other.fit(studyset)

    g_one = first.get_map("g", return_type="array").ravel()
    g_two = second.get_map("g", return_type="array").ravel()
    relative_one = first.get_map("g_relative", return_type="array").ravel()
    relative_two = second.get_map("g_relative", return_type="array").ravel()

    # The absolute maps differ by the constant...
    moved = np.isfinite(g_one) & np.isfinite(g_two) & (np.abs(g_one) > 1e-6)
    assert moved.any()
    assert not np.allclose(g_one[moved], g_two[moved])
    # ...and the relative maps agree to the accuracy the fit itself has. The cancellation is
    # exact in the pooling step but only approximate through the EM, which is truncated at
    # max_iter rather than converged (see ``max_iter``): starting from a rescaled g, the
    # iteration stops at a slightly different point on the same plateau. Measured, 8 voxels in
    # 9261 move at all and the largest moves by 3e-5 -- the relative figure looks worse only
    # because those voxels sit near zero. Real scale leakage would move the map by order 1.
    np.testing.assert_allclose(relative_one, relative_two, rtol=5e-3, atol=1e-4)
    # A high percentile of the magnitude is the unit, so the map reaches about 1 and not much
    # more, and it carries sign.
    covered = np.abs(relative_one) > 0
    assert 0.9 <= np.percentile(np.abs(relative_one[covered]), 95) <= 1.1


def test_an_absolute_map_appears_only_when_something_pins_the_scale(studyset, small_mask):
    """``g_absolute`` is a claim about units, so it must be absent when the units are unknown."""
    shared = dict(fwhm=8.0, mask=small_mask, null_method="none", peak_bias="per-study")

    coordinates_only = CBES(**shared).fit(studyset)
    assert "g_relative" in coordinates_only.maps
    assert "g_absolute" not in coordinates_only.maps

    supplied = CBES(**shared, peak_bias_scale=0.6)
    result = supplied.fit(studyset)
    assert supplied.scale_source_ == "supplied"
    assert "g_absolute" in result.maps
    # Same numbers as "g"; the separate name is what carries the claim.
    np.testing.assert_array_equal(
        result.get_map("g_absolute", return_type="array"),
        result.get_map("g", return_type="array"),
    )


def test_hartung_knapp_replaces_the_se_without_touching_the_estimate(studyset, small_mask):
    """HKSJ is a different variance, not a different fit."""
    shared = dict(
        fwhm=8.0,
        mask=small_mask,
        null_method="none",
        peak_bias="per-study",
        selection_model="none",
    )
    model = CBES(**shared, se_method="model").fit(studyset)
    hksj = CBES(**shared, se_method="hksj").fit(studyset)

    np.testing.assert_allclose(
        model.get_map("g", return_type="array"),
        hksj.get_map("g", return_type="array"),
        rtol=1e-10,
    )
    se_model = model.get_map("se", return_type="array").ravel()
    se_hksj = hksj.get_map("se", return_type="array").ravel()
    n_eff = model.get_map("n_eff", return_type="array").ravel()

    assert np.all(np.isfinite(se_hksj))
    # Where there is spread to measure the two disagree; where there is not, they agree.
    spread = n_eff > 1.0
    assert spread.any()
    assert not np.allclose(se_model[spread], se_hksj[spread])
    flat = (n_eff > 0) & (n_eff <= 1.0)
    if flat.any():
        np.testing.assert_allclose(se_model[flat], se_hksj[flat], rtol=1e-10)


def test_hartung_knapp_uses_the_effective_study_count_not_the_weight_total():
    """The degrees of freedom must be Kish's ``n_eff``, which small kernel weights cannot break."""
    from nimare.meta.cbma.effectsize import _hartung_knapp_se

    # Three studies, all weights 0.1: sum(w) = 0.3 but n_eff = 3.
    weights = np.full(3, 0.1)
    n_eff = np.array([weights.sum() ** 2 / (weights**2).sum()])
    assert np.isclose(n_eff[0], 3.0)

    g = np.array([0.2, 0.5, 0.8])
    var = np.full(3, 0.1)
    a = weights / var
    g_hat = np.array([(a * g).sum() / a.sum()])
    se = _hartung_knapp_se(
        g_hat=g_hat,
        sum_a=np.array([a.sum()]),
        sum_a_g2=np.array([(a * g * g).sum()]),
        n_eff=n_eff,
        covered=np.array([True]),
        fallback=np.array([np.inf]),
    )
    expected = np.sqrt((a * (g - g_hat[0]) ** 2).sum() / ((3.0 - 1.0) * a.sum()))
    np.testing.assert_allclose(se[0], expected)
    # Rescaling every weight leaves it untouched, which sum(w) would not.
    rescaled = weights * 17.0
    a2 = rescaled / var
    se_rescaled = _hartung_knapp_se(
        g_hat=g_hat,
        sum_a=np.array([a2.sum()]),
        sum_a_g2=np.array([(a2 * g * g).sum()]),
        n_eff=np.array([rescaled.sum() ** 2 / (rescaled**2).sum()]),
        covered=np.array([True]),
        fallback=np.array([np.inf]),
    )
    np.testing.assert_allclose(se[0], se_rescaled[0], rtol=1e-10)


def test_hksj_is_refused_rather_than_ignored_under_the_selection_model():
    """The combination that would silently do nothing has to fail instead."""
    with pytest.raises(ValueError, match="hksj.*selection_model"):
        CBES(se_method="hksj", selection_model="zero-inflated")
    # And the supported combination constructs.
    CBES(se_method="hksj", selection_model="none")


def test_images_pin_the_scale_and_produce_an_absolute_map(mixed_image_studyset):
    """The only remaining route to an absolute map, now that the corpus prior is gone."""
    estimator = CBES(fwhm=8.0, null_method="none", peak_bias="per-study", peak_bias_scale="images")
    result = estimator.fit(mixed_image_studyset)

    assert estimator.scale_source_ == "images"
    assert estimator.n_scale_donors_ >= 2
    assert "g_absolute" in result.maps
    assert "g_relative" in result.maps

    # One donor is below the threshold for claiming an absolute scale, so the map is withheld
    # even though a scale was still calibrated and applied.
    one_donor = CBES(
        fwhm=8.0,
        null_method="none",
        peak_bias="per-study",
        peak_bias_scale="images",
        use_images=True,
    )
    from nimare.meta.cbma.effectsize import _MIN_SCALE_DONORS

    assert _MIN_SCALE_DONORS == 2
    original = CBES._load_image_studies

    def only_first(self, dataset):
        studies = original(self, dataset)
        return dict(list(studies.items())[:1])

    CBES._load_image_studies = only_first
    try:
        sparse = one_donor.fit(mixed_image_studyset)
    finally:
        CBES._load_image_studies = original
    assert one_donor.n_scale_donors_ == 1
    assert one_donor.scale_interval_ is None
    assert "g_absolute" not in sparse.maps


def test_the_censored_mixture_em_finds_the_same_optimum_as_a_brute_force_search():
    """No reference implementation exists, so the EM is checked against the likelihood itself.

    ``_fit_chunk`` is an EM over a zero-inflated censored likelihood, and everything downstream
    -- ``g``, ``prevalence``, ``se`` -- is whatever it returns. Writing the likelihood out a
    second time, independently, and maximizing it on a grid is the only available oracle: if the
    two disagree, either the EM is not climbing the likelihood the model describes or the model
    is not the one documented.
    """
    from scipy import stats

    from nimare.meta.cbma.effectsize import _PROBABILITY_FLOOR, null_effect_variance

    n_studies, width = 6, 1
    sample_sizes = np.array([20.0, 24.0, 30.0, 36.0, 40.0, 28.0])
    null_var = null_effect_variance(sample_sizes, design="one-sample")[:, None]
    cutoff = 0.55
    cutoffs = np.full((n_studies, 1), cutoff)
    tau2 = np.zeros(width)

    # Three studies reported here, three were silent: the configuration the mixture is for.
    weights = np.zeros((n_studies, width))
    g_obs = np.zeros((n_studies, width))
    var_obs = np.ones((n_studies, width))
    covered = np.zeros((n_studies, width), dtype=bool)
    reported = {0: 0.72, 1: 0.61, 2: 0.95}
    for study, value in reported.items():
        weights[study, 0] = 1.0
        g_obs[study, 0] = value
        var_obs[study, 0] = float(null_var[study, 0])
        covered[study, 0] = True  # reported, so not silent

    estimator = CBES(fwhm=8.0, null_method="none", max_iter=400)
    mu, pi, _ = estimator._fit_chunk(
        weights=weights,
        g_obs=g_obs,
        var_obs=var_obs,
        covered=covered,
        tau2=tau2,
        null_var=null_var,
        cutoffs=cutoffs,
        start=np.array([float(np.mean(list(reported.values())))]),
    )

    def log_likelihood(mu_value, pi_value):
        """Evaluate the documented model, written out again from scratch.

        With probability pi a study has a real effect of size mu here, otherwise none. A
        reporting study contributes the density of what it reported under that mixture; a silent
        study contributes the probability that it would have stayed below its cutoff.
        """
        total = 0.0
        for study, value in reported.items():
            sd = np.sqrt(float(null_var[study, 0]))
            present = np.exp(-0.5 * ((value - mu_value) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            absent = np.exp(-0.5 * (value / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            total += np.log(max(pi_value * present + (1 - pi_value) * absent, 1e-300))
        # Silent studies enter at the average reporting weight, which is 1.0 here.
        for study in range(n_studies):
            if study in reported:
                continue
            sd = np.sqrt(float(null_var[study, 0]))
            silent_present = max(
                stats.norm.cdf((cutoff - mu_value) / sd)
                - stats.norm.cdf((-cutoff - mu_value) / sd),
                _PROBABILITY_FLOOR,
            )
            silent_absent = max(
                stats.norm.cdf(cutoff / sd) - stats.norm.cdf(-cutoff / sd), _PROBABILITY_FLOOR
            )
            total += np.log(
                max(pi_value * silent_present + (1 - pi_value) * silent_absent, 1e-300)
            )
        return total

    # Grid spacing of 0.01 in each parameter, which is finer than the tolerances asserted
    # below and keeps this to a couple of seconds rather than a couple of minutes.
    mu_grid = np.linspace(0.0, 1.6, 161)
    pi_grid = np.linspace(0.01, 0.99, 99)
    surface = np.array([[log_likelihood(m, p) for p in pi_grid] for m in mu_grid])
    best = np.unravel_index(int(np.argmax(surface)), surface.shape)
    brute_mu, brute_pi = mu_grid[best[0]], pi_grid[best[1]]

    # The EM must not be beaten by the grid: its optimum is at least as good, to grid accuracy.
    assert log_likelihood(mu[0], pi[0]) >= surface[best] - 1e-3
    assert abs(mu[0] - brute_mu) < 0.05, (mu[0], brute_mu)
    assert abs(pi[0] - brute_pi) < 0.08, (pi[0], brute_pi)


def test_the_censored_likelihood_reads_silence_as_evidence_against_a_large_effect():
    """The model's defining behaviour, checked on the likelihood rather than on a fitted map.

    Adding silent studies must pull the optimum down: silence is improbable when the effect is
    large, so the more studies stay quiet the smaller the effect that best explains the data.
    This is the whole reason the censoring term exists, and it is worth pinning separately from
    the optimizer that exploits it.
    """
    from nimare.meta.cbma.effectsize import null_effect_variance

    fitted = []
    for n_silent in (0, 3, 9):
        n_studies = 3 + n_silent
        sample_sizes = np.full(n_studies, 30.0)
        null_var = null_effect_variance(sample_sizes, design="one-sample")[:, None]
        weights = np.zeros((n_studies, 1))
        g_obs = np.zeros((n_studies, 1))
        var_obs = np.ones((n_studies, 1))
        covered = np.zeros((n_studies, 1), dtype=bool)
        for study, value in enumerate((0.8, 0.7, 0.9)):
            weights[study, 0] = 1.0
            g_obs[study, 0] = value
            var_obs[study, 0] = float(null_var[study, 0])
            covered[study, 0] = True

        estimator = CBES(fwhm=8.0, null_method="none", max_iter=400)
        mu, _, _ = estimator._fit_chunk(
            weights=weights,
            g_obs=g_obs,
            var_obs=var_obs,
            covered=covered,
            tau2=np.zeros(1),
            null_var=null_var,
            cutoffs=np.full((n_studies, 1), 0.55),
            start=np.array([0.8]),
        )
        fitted.append(float(mu[0]))

    assert fitted[0] > fitted[1] > fitted[2], fitted


@pytest.fixture(scope="module")
def roi_studyset(tmp_path_factory):
    """Twelve studies; four examined only a slab, and are silent everywhere else."""
    directory = tmp_path_factory.mktemp("cbes_roi")
    shape = (12, 12, 12)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -22.0
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), directory / "mask.nii.gz")

    # The slab the partial-coverage studies examined: one end of the volume, away from the
    # focus every study reports at, so their silence at the focus is uninformative.
    slab = np.zeros(shape, np.int32)
    slab[:3] = 1
    nib.save(nib.Nifti1Image(slab, affine), directory / "slab.nii.gz")

    from nimare.studyset import Studyset

    studies = []
    for k in range(12):
        partial = k >= 8
        analysis = {
            "id": f"r{k}-1",
            "name": "1",
            "metadata": {"sample_sizes": [30]},
            # Whole-brain studies report at the focus; the partial ones report inside their slab.
            # The partial studies report on the last plane they examined (voxel i = 2), so
            # their kernel reaches i = 3 just outside it -- which is where a value must not
            # leak. The whole-brain studies report at the centre.
            "points": [
                {
                    "space": "MNI",
                    "coordinates": [-14.0, -18.0, -18.0] if partial else [0.0, 0.0, 0.0],
                    "values": [{"kind": "Z", "value": 4.0}],
                }
            ],
            "images": [],
        }
        if partial:
            analysis["images"] = [
                {
                    "url": str(directory / "slab.nii.gz"),
                    "filename": "slab.nii.gz",
                    "space": "MNI",
                    "value_type": "analysis_mask",
                }
            ]
        studies.append(
            {
                "id": f"r{k}",
                "name": f"r{k}",
                "metadata": {"sample_sizes": [30]},
                "analyses": [analysis],
            }
        )

    return Studyset(
        {"id": "roi", "name": "roi", "studies": studies},
        target=None,
        mask=str(directory / "mask.nii.gz"),
    )


def test_a_declared_analysis_mask_stops_silence_being_read_where_nobody_looked(roi_studyset):
    """An ROI study never looked outside its region, so its silence there is not evidence."""
    from nimare.utils import mm2vox

    masker = roi_studyset.masker
    ijk = mm2vox(np.array([[0.0, 0.0, 0.0]]), masker.mask_img.affine)[0]
    mask = np.asarray(masker.mask_img.dataobj).astype(bool)
    lookup = np.full(mask.shape, -1, dtype=np.int64)
    lookup[mask] = np.arange(mask.sum())
    focus = int(lookup[tuple(ijk)])
    assert focus >= 0

    shared = dict(fwhm=8.0, null_method="none", peak_bias=None)
    ignored = CBES(**shared).fit(roi_studyset)
    honoured = CBES(**shared, analysis_mask="analysis_mask").fit(roi_studyset)

    # Reading the slab studies' silence as evidence drags the focus down; honouring the mask
    # removes four spurious censoring terms, so the estimate there rises.
    g_ignored = ignored.get_map("g", return_type="array").ravel()[focus]
    g_honoured = honoured.get_map("g", return_type="array").ravel()[focus]
    assert g_honoured > g_ignored

    # Same for prevalence: silence that was never observed should not lower it.
    pi_ignored = ignored.get_map("prevalence", return_type="array").ravel()[focus]
    pi_honoured = honoured.get_map("prevalence", return_type="array").ravel()[focus]
    assert pi_honoured > pi_ignored

    # The studies that examined the whole volume are untouched either way.
    assert honoured.get_map("n_studies", return_type="array").ravel()[focus] == (
        ignored.get_map("n_studies", return_type="array").ravel()[focus]
    )


def test_an_absent_analysis_mask_changes_nothing_but_says_so(
    roi_studyset, studyset, small_mask, caplog
):
    """Inert unless a study declares a mask -- but not silently, since that is a trap.

    Requesting a value type that is not there leaves every study's silence read as evidence,
    which is the behaviour the caller asked to switch off. A typo does it, and so does a loader
    skipping the value type because it is not one NiMARE recognises.
    """
    shared = dict(fwhm=8.0, mask=small_mask, null_method="none", peak_bias=None)
    without = CBES(**shared).fit(studyset)
    # This collection carries no images at all, so naming a value type finds nothing.
    with caplog.at_level("WARNING"):
        with_name = CBES(**shared, analysis_mask="analysis_mask").fit(studyset)
    assert "matches no image value type" in caplog.text
    np.testing.assert_allclose(
        without.get_map("g", return_type="array"),
        with_name.get_map("g", return_type="array"),
        rtol=1e-10,
    )
    assert CBES(fwhm=8.0, null_method="none")._load_analysis_masks(roi_studyset) == {}


def test_dof_is_emitted_so_se_can_be_referred_to_a_t(studyset, small_mask):
    """``se`` is observed information on few studies, so it needs a t reference, not a normal.

    Simulated against a known effect, a normal interval on this ``se`` covers 85-94% of nominal
    95%, where a t on ``dof`` covers 91-97%. The map is emitted so a caller can do that; the
    p-values are unaffected, coming from the permutation null rather than from any reference
    distribution.
    """
    result = CBES(fwhm=8.0, mask=small_mask, null_method="none").fit(studyset)
    dof = result.get_map("dof", return_type="array").ravel()
    n_eff = result.get_map("n_eff", return_type="array").ravel()

    assert np.all(dof >= 0.0)
    covered = n_eff > 0
    np.testing.assert_allclose(dof[covered], np.clip(n_eff[covered] - 1.0, 0.0, None), rtol=1e-6)
    # Somewhere has enough studies for a t interval to be usable at all.
    assert np.any(dof > 1.0)


def test_an_implausibly_high_inferred_threshold_is_called_out(studyset, small_mask, caplog):
    """Cluster-extent reporting is indistinguishable from strict height thresholding.

    The inference overshoots by about 1 z when reporting was by extent, which leaves g intact
    but saturates prevalence. It cannot tell the two apart, so the honest move is to say when
    the answer lands where extent reporting would put it, and name the output at risk.
    """
    from nimare.meta.cbma.effectsize import _SUSPICIOUS_INFERRED_THRESHOLD_Z

    quiet = CBES(fwhm=8.0, mask=small_mask, null_method="none", threshold="study-min")
    with caplog.at_level("WARNING"):
        quiet.fit(studyset)
    assert "above the usual range" not in caplog.text
    assert np.median(quiet._cutoffs_z_.values) <= _SUSPICIOUS_INFERRED_THRESHOLD_Z

    # A collection whose reported peaks are all far above any plausible height cut, which is
    # what an extent-thresholded table looks like to this inference.
    strict = studyset.copy()
    coords = strict.coordinates
    caplog.clear()
    loud = CBES(fwhm=8.0, mask=small_mask, null_method="none", threshold="study-min")
    with caplog.at_level("WARNING"):
        # Supplied directly rather than simulated: the point under test is the warning, not the
        # inference that feeds it.
        loud._warn_if_threshold_implausible(np.full(len(coords["id"].unique()), 4.6))
    assert "above the usual range" in caplog.text
    assert "prevalence" in caplog.text

    # No threshold at all must not warn, and must not raise.
    caplog.clear()
    loud._warn_if_threshold_implausible(np.array([np.nan, np.nan]))
    assert caplog.text == ""


def test_the_analysis_mask_is_keyed_per_contrast_not_per_study(roi_studyset):
    """The key is the analysis id, which is what the censoring roster is indexed by.

    Per-contrast subsumes per-study, but it means a paper contributing several contrasts has to
    declare the mask on each one it applies to. If the two were keyed differently the lookup
    would silently miss and the feature would never fire, so the agreement is worth pinning.
    """
    estimator = CBES(fwhm=8.0, null_method="none", analysis_mask="analysis_mask")
    estimator.fit(roi_studyset)

    masks = estimator._analysis_masks_
    roster = set(estimator._sample_sizes_.index)
    # Four studies declare a slab; whole-brain ones are skipped rather than stored.
    assert len(masks) == 4
    # Whatever was found must be addressable by the roster the coverage pass iterates over.
    assert set(masks).issubset(roster)
    # The ids are analysis-level, carrying a contrast suffix rather than a bare study id.
    assert all(key.rsplit("-", 1)[-1].isdigit() for key in masks), sorted(masks)


@pytest.fixture(scope="module")
def two_group_studyset(tmp_path_factory):
    """Eight analyses whose metadata declares two groups of 30, i.e. sixty subjects each."""
    directory = tmp_path_factory.mktemp("cbes_two_group")
    shape = (8, 8, 8)
    affine = np.diag([6.0, 6.0, 6.0, 1.0])
    affine[:3, 3] = -21.0
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), directory / "mask.nii.gz")

    from nimare.studyset import Studyset

    studies = [
        {
            "id": f"t{k}",
            "name": f"t{k}",
            "metadata": {"sample_sizes": [30, 30]},
            "analyses": [
                {
                    "id": f"t{k}-1",
                    "name": "1",
                    "metadata": {"sample_sizes": [30, 30]},
                    "points": [
                        {
                            "space": "MNI",
                            "coordinates": [0.0, 0.0, 0.0],
                            "values": [{"kind": "T", "value": 3.0}],
                        }
                    ],
                    "images": [],
                }
            ],
        }
        for k in range(8)
    ]
    return Studyset(
        {"id": "two_group", "name": "two_group", "studies": studies},
        target=None,
        mask=str(directory / "mask.nii.gz"),
    )


def test_a_two_sample_design_gets_the_total_sample_size(two_group_studyset):
    """``design="two-sample"`` splits its argument into equal groups, so it needs the total.

    Metadata of ``[30, 30]`` means sixty subjects. Reducing it by mean handed the converter
    thirty, which it read as two groups of fifteen -- inflating ``g`` by 39% at ``t = 3``
    (1.066 against 0.765) and carrying the same error into the sampling variances, the cutoff
    conversion and the null variances.
    """
    two_sample = CBES(fwhm=10.0, null_method="none", design="two-sample", peak_bias="per-study")
    two_sample.fit(two_group_studyset)
    assert set(two_sample._sample_sizes_.values) == {60.0}
    assert set(two_sample._focus_table_["sample_size"]) == {60}

    # One-sample keeps the mean, which is what a single or repeated value means there.
    one_sample = CBES(fwhm=10.0, null_method="none", design="one-sample", peak_bias="per-study")
    one_sample.fit(two_group_studyset)
    assert set(one_sample._sample_sizes_.values) == {30.0}

    # And the resulting g is the textbook value for two balanced groups of thirty.
    expected, _ = peak_stat_to_hedges_g([3.0], [60.0], stat_type="t", design="two-sample")
    got = two_sample._focus_table_["g"].abs().max()
    assert np.isclose(got, expected[0], rtol=1e-6), (got, expected[0])


def test_a_declared_analysis_mask_also_stops_the_value_leaking_outside_it(roi_studyset):
    """Suppressing silence outside a region but still pooling the value there is the worst case.

    A kernel reaches about 13 mm for a 10 mm FWHM, so a peak just inside a declared region
    spills outside it. Honouring the mask for censoring while ignoring it for contributions
    would hand an unexamined voxel a number from a study that never looked there -- measured at
    g = 0.31 with two contributing studies where the correct answer was 0.20 from one.
    """
    masker = roi_studyset.masker
    mask = np.asarray(masker.mask_img.dataobj).astype(bool)
    lookup = np.full(mask.shape, -1, dtype=np.int64)
    lookup[mask] = np.arange(mask.sum())
    # The partial studies report at voxel (2, 1, 1) and examined only i < 3, so (3, 1, 1) is
    # one voxel away -- inside the kernel's support -- and outside the slab. That is exactly
    # where the value used to leak.
    outside = int(lookup[3, 1, 1])
    assert outside >= 0

    shared = dict(fwhm=8.0, null_method="none", peak_bias=None, selection_model="none")
    ignored = CBES(**shared).fit(roi_studyset)
    honoured = CBES(**shared, analysis_mask="analysis_mask").fit(roi_studyset)

    n_ignored = ignored.get_map("n_studies", return_type="array").ravel()[outside]
    n_honoured = honoured.get_map("n_studies", return_type="array").ravel()[outside]
    # The slab studies' kernels reach this voxel, and must stop counting once the mask is read.
    assert n_ignored > n_honoured, (n_ignored, n_honoured)


def test_an_analysis_mask_covering_nothing_means_nothing_rather_than_everything(
    roi_studyset, tmp_path, caplog
):
    """An empty declared mask must not fall back to whole-brain, which inverts its meaning.

    Dropping it used to do exactly that: a study saying "I examined nothing in this volume"
    would be restored to contributing everywhere, instead of contributing neither a value nor
    a silence.
    """
    from nimare.studyset import Studyset

    affine = roi_studyset.masker.mask_img.affine
    shape = roi_studyset.masker.mask_img.shape[:3]
    empty = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, np.int32), affine), empty)
    brain = tmp_path / "brain.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), brain)

    studies = [
        {
            "id": f"e{k}",
            "name": f"e{k}",
            "metadata": {"sample_sizes": [30]},
            "analyses": [
                {
                    "id": f"e{k}-1",
                    "name": "1",
                    "metadata": {"sample_sizes": [30]},
                    "points": [
                        {
                            "space": "MNI",
                            "coordinates": [0.0, 0.0, 0.0],
                            "values": [{"kind": "Z", "value": 4.0}],
                        }
                    ],
                    "images": [
                        {
                            "url": str(empty if k == 0 else brain),
                            "filename": "m.nii.gz",
                            "space": "MNI",
                            "value_type": "analysis_mask",
                        }
                    ],
                }
            ],
        }
        for k in range(6)
    ]
    studyset = Studyset(
        {"id": "empty_mask", "name": "empty_mask", "studies": studies},
        target=None,
        mask=str(roi_studyset.masker.mask_img.get_filename() or brain),
    )

    estimator = CBES(fwhm=8.0, null_method="none", analysis_mask="analysis_mask")
    with caplog.at_level("WARNING"):
        estimator.fit(studyset)
    assert "covering no in-mask voxel" in caplog.text
    # Kept as an all-False mask rather than discarded, so the study contributes nothing.
    empty_masks = [m for m in estimator._analysis_masks_.values() if not m.any()]
    assert len(empty_masks) == 1


def test_the_coverage_cache_rebuilds_when_the_configuration_changes(mixed_image_studyset):
    """A cache keyed on shape alone hands one fit's censoring matrix to a different fit.

    Calibration fits the coordinates alone and then each image donor alone. Those share a
    roster and an active extent but not a focus table, so a key of (extent, analysis count)
    matched and the donor fits silently reused the coordinate fit's coverage.
    """
    from nimare.meta.cbma import effectsize as module

    plain = module.CBES._coverage_entries
    calls = {"n": 0}

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return plain(self, *args, **kwargs)

    module.CBES._coverage_entries = counted
    try:
        estimator = CBES(
            fwhm=8.0, null_method="none", peak_bias="per-study", peak_bias_scale="images"
        )
        estimator.fit(mixed_image_studyset)
    finally:
        module.CBES._coverage_entries = plain

    # Coordinates alone, each donor alone, then the fit itself: more than one configuration.
    assert calls["n"] > 1, calls["n"]
    # And the key names what decides coverage, not merely its shape.
    key = estimator._coverage_[0]
    assert len(key) >= 8, key
