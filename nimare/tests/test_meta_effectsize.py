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
    """30 studies with a true g of 0.5 at the origin, thresholded at p < .001.

    Deliberately a moderate effect: at this power only about a third of the studies clear the
    threshold, which is the regime where the selection bias is large enough to be worth
    correcting. With a large effect nearly every study reports and the naive estimate is fine.
    """
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
    """One permutation fit at ``n_iters=20``, shared by the tests that all wanted the same one.

    Five tests fitted this identical configuration at ~8.6 s each, which was a fifth of the
    suite's runtime spent recomputing the same permutations. Tests that touch the estimator take
    a deepcopy, so they stay order-independent even though ``correct_fwe_montecarlo`` can write
    back into ``null_distributions_`` when its cache does not match.
    """
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
    """With unit kernel weights the local estimator must be the textbook DL estimator.

    Checked against PyMARE's :class:`~pymare.estimators.DerSimonianLaird`, which is what
    :mod:`nimare.meta.ibma` uses, rather than against a formula written out again here -- a
    hand-copied reference can be wrong in the same way the implementation is. The kernel-weighted
    form cannot be delegated to PyMARE (``fit`` takes no per-observation weight, and folding the
    weight into the variance would scale tau-squared with it), so this pins the generalization at
    the point where the two must agree.
    """
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
    """With unit kernel weights the pooled estimate is ordinary inverse-variance weighting.

    The second check that CBES is a weighting scheme over a standard random-effects model rather
    than a separate algorithm, again against PyMARE rather than a restatement of the formula.
    CBES cannot call :func:`~pymare.stats.weighted_least_squares` itself: it returns the
    model-based ``(X'WX)^-1``, where a kernel weight is a design weight distinct from the
    inverse-variance one and needs the sandwich form, and it wants dense ``(studies, voxels)``
    arrays where the fit is sparse. At unit weight the two coincide, which is what is tested.
    """
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
    """Half the studies are genuinely null at the focus, and the model must be able to say so.

    Without a zero component the only way to explain their silence is a small shared effect,
    which drags the estimate below the truth -- measured at -0.16 to -0.34 before the mixture
    was added, and the reason the plain Tobit is not offered. With it, the silence goes to the
    zero component and the effect among the studies that have one is recovered.
    """
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
    """Correcting for the family can only make a p-value larger, never smaller.

    Needs a real uncorrected null to compare against: with ``null_method="none"`` the
    uncorrected p is 1 everywhere by construction and the comparison says nothing.
    """
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
    """Absence of inference must not be mistakable for inference.

    ``null_method="none"`` exists for inspecting the estimates cheaply. It replaced a parametric
    option that referred ``g / se`` to a normal distribution, which was anticonservative by a
    factor of eight under a global null. Returning 1 everywhere makes the absence explicit,
    where a plausible-looking p-value would not.
    """
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
    """The permutations fit() runs already record cluster measures, so correcting is free.

    The forming threshold comes from a pilot run rather than from a second pass over the
    permutations, which is what it would otherwise cost.
    """
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
    """The cluster-forming threshold is read off the null rather than assumed.

    CBES's ``z`` is not standard normal, so a nominal 3.29 is not a p of .001 and the cutoff has
    to come from the null actually observed. This is the one place a histogram pooled over
    voxels is still used -- a cluster-forming threshold has to be a single number -- so it is
    checked directly rather than through the p-values, which are per voxel.
    """
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
    """The EM visits only weighted (study, voxel) pairs; padding must not change the answer.

    A study that covers the region but whose peak is outside this voxel's kernel informs
    neither the density term nor the censoring term. Adding such studies is the case the
    sparse-pair EM has to get right, since they are exactly the entries it skips.
    """
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
    """Rho rescales g, its variance and the threshold together, so the fit scales with it.

    That exactness is what makes rho calibratable: a single ratio of summaries recovers it.
    """
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
    """The artefact in a reported effect size is a function of (u, N), and a strong one.

    A study that only reports above z = 4.3, on 15 subjects, would report g near 1 from pure
    noise; one reporting above z = 2.3 on 60 subjects would report a third of that. Pooling the
    two without correction compares numbers that are not on the same scale.
    """
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
    """``"study-min"`` infers each study's threshold, it does not take the minimum at face value.

    The smallest of a study's reported peaks sits above its threshold by an amount that grows as
    the study reports fewer of them, so the inferred cutoff must land below the raw minimum --
    never above it, and by more when there are fewer peaks to draw from.
    """
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
    """With one threshold and one sample size there is nothing between studies to correct.

    The per-study factor then has to collapse to exactly ``peak_bias_scale``, which is what
    makes the two options comparable: ``"per-study"`` only ever redistributes around it.
    """
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
    """The point of the correction: studies that thresholded differently stop disagreeing.

    Two halves of one collection, identical truth, differing only in how strictly they
    thresholded. Uncorrected, the strict half reports much larger effect sizes than the lenient
    half. The per-study factor should shrink that gap.
    """
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
    """Build ten studies where only the first five supply images.

    A collection that actually mixes the two kinds of evidence, which is what makes the
    overall scale identifiable: the coordinate studies have to be put on the images' scale.
    """
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
    """'auto' reads the constant off the images, and it has to be the ratio it corrects.

    The fit is exactly linear in the scale, so the calibrated coordinate-only estimate must
    land on the image-only one -- that is the whole content of the calibration.
    """
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
    """Coordinates and images are exchangeable in different ways, so the null must move both.

    Relocating a focus destroys its position, which is what the coordinate null says is
    arbitrary. An image has no position to destroy; what the null says about it is that its sign
    is arbitrary. Holding images fixed instead carries their signal into the null: with the
    effect size held constant and only its extent varied, power fell from 1.00 at 3% of the
    volume to 0.67 at 100%, and sign-flipping restored it to 1.00 throughout.
    """
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
    """A collection with images and no coordinates is an error, not a quiet reduction.

    With no peaks there is nothing for the kernel to spread or the selection model to explain,
    and nothing for the null to permute -- an empty focus table is invariant under every
    permutation, so the p-values would look perfectly calibrated and mean nothing. The fit
    would silently become a random-effects meta-analysis of the images, which
    ``nimare.meta.ibma`` does directly and with valid inference, so the error says so.
    """
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
    """CBES estimates effect size, so its null randomizes magnitudes, not positions.

    A relocation null -- move every focus to a random in-mask voxel, keep its effect size --
    asks whether foci pile up at a voxel. That is the question ALE and MKDA exist to answer and
    it is not this one, so it was removed rather than offered alongside: leaving it in would
    have let a caller obtain a convergence result from an estimator whose output is an effect
    size. The RFT regional censoring term went earlier, for cost and calibration.

    Removed options fail at construction rather than being silently ignored, so a script written
    against an earlier version stops instead of quietly answering a different question.
    """
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


def test_reference_magnitude_falls_with_sample_size():
    """Studies powered for subtle effects are large, so the reference has to fall with N.

    Measured across 258 NeuroVault group maps: corr(log N, log magnitude) = -0.395. The table
    is binned rather than a fitted line because the relationship is not log-linear.
    """
    from nimare.meta.cbma.effectsize import reference_magnitude

    small = reference_magnitude([16] * 10)
    medium = reference_magnitude([50] * 10)
    large = reference_magnitude([200] * 10)
    assert small > medium > large
    assert 0.5 < small < 1.2
    assert 0.1 < large < 0.4

    # Mixed collections interpolate rather than taking an extreme.
    mixed = reference_magnitude([16, 200])
    assert large < mixed < small
    assert reference_magnitude([]) is None
    assert reference_magnitude([np.nan, -5]) is None


def test_reference_scale_is_opt_in_and_not_chosen_by_auto(studyset, small_mask, caplog):
    """It made a known truth worse by up to 2x, so it must never be selected implicitly."""
    estimator = CBES(
        fwhm=8.0,
        mask=small_mask,
        null_method="none",
        peak_bias="per-study",
        peak_bias_scale="auto",
    )
    with caplog.at_level("WARNING"):
        estimator.fit(studyset)
    # No images here, so "auto" falls back to a relative map rather than reaching for the
    # reference corpus.
    assert estimator._peak_bias_scale_ == 1.0
    assert "reference" in caplog.text  # but it does point at the option

    explicit = CBES(
        fwhm=8.0,
        mask=small_mask,
        null_method="none",
        peak_bias="per-study",
        peak_bias_scale="reference",
    )
    explicit.fit(studyset)
    assert explicit._peak_bias_scale_ != 1.0


def test_permuting_magnitudes_leaves_the_spatial_design_untouched(small_mask):
    """The invariant the null depends on, tested where it is easiest to break.

    Only the reported value and its variance move; position, study and sample size stay put. So
    the set of covered voxels and the number of studies reaching each one must be bit-identical
    to the observed fit, in every iteration. If they are not, the null's standard errors differ
    from the observed map's for reasons that have nothing to do with effect size, and voxels
    become significant on how many studies happen to reach them.

    The collection here gives every study several foci close together, which is the arrangement
    that breaks a permutation that also moves the study label: a voxel keeps one observation per
    study, so relabelling can put two foci of one study on the same voxel and lose a count.
    Moving values alone cannot, however the foci are arranged.
    """
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
    """The property that decides the null was worth changing for.

    Every study reports a peak at the same place, and every peak in the collection -- there and
    in the scatter around it -- is drawn from one distribution. So the site has overwhelming
    spatial convergence and an entirely unremarkable magnitude. An estimator that reports effect
    size should be unmoved by it: nothing about those thirty studies says the effect there is
    any larger than the effects reported elsewhere, only that more studies happened to report
    there. A relocation null would call this significant, which is why it is not the null here.

    The guard is against regressing to a null pooled over voxels. Pooling would refer this
    voxel's thirty studies to a distribution made mostly of voxels carrying two; the standard
    error falls with the number of contributing studies, so the site would come out significant
    on study count alone even though the magnitudes are permuted.
    """
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
    """Under a global null the permutation null must also return roughly the nominal rate.

    Its immunity to the multiplicity mismatch is worth nothing if the rate is wrong for some
    other reason, so it is held to the same standard as the relocation null beside it.
    """
    permutation = CBES(
        fwhm=12.0,
        mask=small_mask,
        selection_model="none",
        null_method="permute-magnitudes",
        n_iters=50,
    ).fit(null_studyset)

    assert np.mean(permutation.get_map("p", return_type="array") < 0.05) < 0.15


def test_fwe_correction_permutes_even_without_a_null_from_fit(null_studyset, small_mask):
    """A maximum statistic has to come from somewhere, so the correction permutes on demand.

    Fitting with ``null_method="none"`` skips the null to save the refits, which leaves nothing
    cached for familywise correction to reuse. It must then build one rather than fail or, worse,
    correct against an empty distribution.
    """
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


def test_scale_is_reported_as_an_interval_or_not_at_all(studyset, small_mask):
    """A partially identified scale must not be handed over as a bare point estimate.

    Rescaling every coordinate study by one constant leaves the coordinate-only likelihood
    unchanged, so a coordinate-only fit identifies the pattern and not the scale. Saying so in
    ``scale_interval_`` is the difference between a caller who knows the magnitudes are
    relative and one who reads them as Hedges' g.
    """
    coordinates_only = CBES(fwhm=8.0, mask=small_mask, null_method="none", peak_bias="per-study")
    coordinates_only.fit(studyset)
    assert coordinates_only.scale_interval_ is None

    from_reference = CBES(
        fwhm=8.0,
        mask=small_mask,
        null_method="none",
        peak_bias="per-study",
        peak_bias_scale="reference",
    )
    result = from_reference.fit(studyset)
    interval = from_reference.scale_interval_
    assert interval is not None
    low, high = interval
    assert 0 < low < from_reference._peak_bias_scale_ < high
    # The corpus the reference was fitted on is good to about a factor of two either way.
    assert 3.0 < high / low < 5.0
    assert "order of scale" in result.description_


def test_the_description_reports_what_the_peak_heights_carry(studyset, small_mask):
    """The magnitude caveat belongs in the methods text, not only in the log.

    A collection whose reported heights are indistinguishable from noise peaks cannot support
    a magnitude -- measured overestimating by a factor of 10.7 on one collection -- and the
    estimator detected that and said so among a dozen other log lines, where nobody read it.
    """
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
    """Two fits differing only by the scale constant must give the same relative map.

    Rescaling every reported effect size leaves the coordinate-only likelihood unchanged, so
    ``g`` moves and nothing that matters does. ``g_relative`` is the map that says so: it is
    invariant to the constant by construction, which is the property that makes it comparable
    across collections where ``g`` is not.
    """
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
    # iteration stops at a slightly different point on the same plateau. One voxel in 9261
    # moved, by 0.001 relative.
    np.testing.assert_allclose(relative_one, relative_two, rtol=2e-3, atol=1e-6)
    # A high percentile of the magnitude is the unit, so the map reaches about 1 and not much
    # more, and it carries sign.
    covered = np.abs(relative_one) > 0
    assert 0.9 <= np.percentile(np.abs(relative_one[covered]), 95) <= 1.1


def test_an_absolute_map_appears_only_when_something_pins_the_scale(studyset, small_mask):
    """``g_absolute`` is a claim about units, so it must be absent when the units are unknown.

    A coordinate-only fit identifies the pattern and not the scale, and borrowing the scale
    from another corpus is an assumption about this collection rather than a measurement of it.
    Emitting the map anyway would leave a reader unable to tell Hedges' g from Hedges' g times
    an unknown constant.
    """
    shared = dict(fwhm=8.0, mask=small_mask, null_method="none", peak_bias="per-study")

    coordinates_only = CBES(**shared).fit(studyset)
    assert "g_relative" in coordinates_only.maps
    assert "g_absolute" not in coordinates_only.maps

    from_reference = CBES(**shared, peak_bias_scale="reference").fit(studyset)
    assert "g_absolute" not in from_reference.maps

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
    """HKSJ is a different variance, not a different fit.

    It also has to survive the voxels it cannot apply to: with one effective study there is no
    spread about the pooled value to measure, and the degrees of freedom would be zero or less.
    Those voxels keep the model-based value rather than producing an infinity.
    """
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
    """The degrees of freedom must be Kish's ``n_eff``, which small kernel weights cannot break.

    ``sum(w)`` and Kish's ``(sum w)^2 / sum w^2`` agree when every weight is one, but only the
    latter is invariant to rescaling. At a voxel reached only by distant foci the weights sum to
    well under one, and using that as a study count sends the degrees of freedom to zero and the
    interval to absurdity -- measured as 99.3% coverage of a nominal 95% interval before this was
    fixed.
    """
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
    """The combination that would silently do nothing has to fail instead.

    The zero-inflated model overwrites the pooled inverse-variance SE with the curvature of the
    censored likelihood, so an HKSJ correction applied during pooling is discarded before it
    reaches the caller. Accepting the combination would hand back an uncorrected SE while
    reporting that a correction was requested.
    """
    with pytest.raises(ValueError, match="hksj.*selection_model"):
        CBES(se_method="hksj", selection_model="zero-inflated")
    # And the supported combination constructs.
    CBES(se_method="hksj", selection_model="none")
