"""Tests for nimare.meta.cbma.effectsize (coordinate-based effect-size meta-analysis)."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nimare.correct import FDRCorrector, FWECorrector

from nimare.generate import create_effect_size_coordinate_studyset
from nimare.meta.cbma.effectsize import (
    CBES,
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
    """A small 4mm box around the origin, so the fits in these tests stay quick."""
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
    with pytest.raises(ValueError, match="at least 4 subjects"):
        peak_stat_to_hedges_g([3.0], [3], stat_type="t")


def test_null_effect_variance_shrinks_with_sample_size():
    variances = null_effect_variance(np.array([20.0, 80.0]))
    assert variances[0] > variances[1]
    assert np.allclose(variances, 1.0 / np.array([20.0, 80.0]), rtol=0.15)


def test_local_dl_reduces_to_dersimonian_laird():
    """With unit kernel weights the local estimator is the textbook DL estimator."""
    rng = np.random.default_rng(0)
    g = rng.normal(0.5, 0.3, size=12)
    var_g = rng.uniform(0.02, 0.08, size=12)

    weights = 1.0 / var_g
    mean = np.sum(weights * g) / np.sum(weights)
    q_stat = np.sum(weights * (g - mean) ** 2)
    scale = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    expected = max(0.0, (q_stat - (len(g) - 1)) / scale)

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


def test_local_dl_is_zero_without_two_studies():
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
    with pytest.raises(ValueError, match=match):
        CBES(**kwargs)


def test_cbes_requires_a_reported_statistic(small_mask):
    """Coordinates without statistics get a message pointing at the convergence estimators."""
    from nimare.generate import create_coordinate_studyset

    _, plain = create_coordinate_studyset(foci=1, n_studies=5, sample_size=20, seed=1)
    with pytest.raises(ValueError, match="no usable 'z_stat' or 't_stat'"):
        CBES(mask=small_mask, selection_model="none", null_method="none").fit(plain)


def test_cbes_produces_expected_maps(studyset, small_mask):
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
        null_method="montecarlo",
        n_iters=20,
        seed=0,
    )
    result = estimator.fit(studyset)
    maps, tables, description = estimator.correct_fwe_montecarlo(
        result, n_iters=20, seed=0, vfwe_only=True
    )

    assert tables == {}
    assert "Monte Carlo" in description
    p_corrected = 10.0 ** -maps["logp_level-voxel"]
    assert np.all((p_corrected > 0) & (p_corrected <= 1))
    assert np.all(p_corrected >= result.get_map("p", return_type="array") - 1e-6)


def test_correct_fwe_montecarlo_needs_a_fit(small_mask):
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


def test_montecarlo_null_calibrates_uncorrected_p(null_studyset, small_mask):
    """Under a global null the spatial null returns roughly the nominal rate.

    This is the reason ``null_method`` defaults to ``"montecarlo"``: ``g / se`` is not a
    null-referenced statistic, since the standard error treats tau-squared as known and ignores
    that the peaks being pooled were selected for being large.
    """
    montecarlo = CBES(
        fwhm=12.0, mask=small_mask, selection_model="none", null_method="montecarlo", n_iters=50
    ).fit(null_studyset)

    assert np.mean(montecarlo.get_map("p", return_type="array") < 0.05) < 0.15


@pytest.mark.parametrize(
    "corrector,map_name",
    [
        (FDRCorrector(method="indep"), "p_corr-FDR_method-indep"),
        (FWECorrector(method="bonferroni"), "p_corr-FWE_method-bonferroni"),
    ],
)
def test_stock_correctors_work(studyset, small_mask, corrector, map_name):
    """The generic correctors need only a p map, which CBES provides."""
    result = CBES(fwhm=12.0, mask=small_mask, null_method="montecarlo", n_iters=20).fit(studyset)
    corrected = corrector.transform(result)

    p_corr = corrected.get_map(map_name, return_type="array")
    assert np.all((p_corr > 0) & (p_corr <= 1))
    # Correction can only make p-values larger.
    assert np.all(p_corr >= result.get_map("p", return_type="array") - 1e-6)


def test_fwe_montecarlo_reports_voxel_and_cluster_levels(studyset, small_mask):
    """Voxel-level, cluster-size and cluster-mass corrections all come from one permutation."""
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="montecarlo", n_iters=25)
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


def test_fwe_montecarlo_vfwe_only_returns_only_voxel_maps(studyset, small_mask):
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="montecarlo", n_iters=20)
    result = estimator.fit(studyset)
    maps, _, description = estimator.correct_fwe_montecarlo(result, vfwe_only=True)

    assert set(maps) == {"logp_level-voxel", "z_level-voxel"}
    assert "voxel-level" in description


def test_cluster_null_is_built_during_fit(studyset, small_mask):
    """The permutations fit() runs already record cluster measures, so correcting is free.

    The forming threshold comes from a pilot run rather than from a second pass over the
    permutations, which is what it would otherwise cost.
    """
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="montecarlo", n_iters=20)
    estimator.fit(studyset)

    assert "cluster_forming_stat" in estimator.null_distributions_
    for key in (
        "values_desc-size_level-cluster_corr-fwe_method-montecarlo",
        "values_desc-mass_level-cluster_corr-fwe_method-montecarlo",
    ):
        assert len(estimator.null_distributions_[key]) == 20


def test_cluster_threshold_none_skips_the_cluster_null(studyset, small_mask):
    estimator = CBES(
        fwhm=12.0,
        mask=small_mask,
        null_method="montecarlo",
        n_iters=20,
        cluster_threshold=None,
    )
    estimator.fit(studyset)

    assert (
        "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
        not in estimator.null_distributions_
    )


def test_stat_from_histogram_inverts_p_from_histogram():
    """The cluster-forming threshold is read off the same null the p-values come from."""
    from nimare.meta.cbma.effectsize import _p_from_histogram, _stat_from_histogram

    rng = np.random.default_rng(3)
    histogram, _ = np.histogram(
        np.clip(np.abs(rng.standard_normal(500_000)), 0, 50.0), bins=_null_bin_edges()
    )
    histogram = histogram.astype(float)

    for target in (0.05, 0.01, 0.001):
        stat = _stat_from_histogram(target, histogram)
        assert _p_from_histogram(np.array([stat]), histogram)[0] <= target
        # And the bin just below it does not reach the threshold.
        below = _p_from_histogram(np.array([stat - 0.01]), histogram)[0]
        assert below > target


def test_fwe_montecarlo_reuses_the_null_from_fit(studyset, small_mask):
    """Fitting with the Monte Carlo null already paid for the max-statistic distribution."""
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="montecarlo", n_iters=20)
    result = estimator.fit(studyset)
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
        null_method="montecarlo",
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
    studyset, _ = image_studyset
    estimator = CBES(fwhm=8.0, null_method="none", use_images=False)
    estimator.fit(studyset)
    assert estimator._image_studies_ == {}
    assert len(estimator._focus_table_) == 10


def test_peak_bias_rescales_the_estimate_exactly(image_studyset):
    """rho rescales g, its variance and the threshold together, so the fit scales with it.

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
    """The same ten studies, but only the first five supply images.

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
    with pytest.raises(ValueError, match="peak_bias_scale must be"):
        CBES(peak_bias_scale=bad)


def test_only_the_magnitude_depends_on_the_uncalibrated_scale(small_mask):
    """What "relative map" costs, and what it does not.

    The common scale is exactly non-identified from coordinates: rescaling g, its variance and
    the censoring threshold together leaves the likelihood unchanged. So ``g`` and ``se`` move
    with it and are readable only up to a constant -- but ``z = g/se`` divides it out, and the
    prevalence is a probability that cancels from the mixture responsibilities. Inference and
    prevalence are therefore on an absolute scale even when the magnitude is not, which is what
    makes an uncalibrated coordinate-only fit worth reporting at all.
    """
    studyset = create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=0.7,
        n_studies=30,
        sample_size=25,
        threshold_z=[2.3263, 3.2905, 4.2649],
        seed=5,
        n_noise_foci=1,
        noise_extent=30.0,
    )

    maps = {}
    for scale in (1.0, 0.4):
        result = CBES(
            fwhm=8.0,
            mask=small_mask,
            null_method="none",
            threshold="study-min",
            peak_bias="per-study",
            peak_bias_scale=scale,
        ).fit(studyset)
        maps[scale] = {
            name: result.get_map(name, return_type="array").ravel()
            for name in ("g", "se", "z", "p", "prevalence", "g_marginal")
        }

    full, scaled = maps[1.0], maps[0.4]
    covered = full["g"] != 0
    assert covered.any()

    for name in ("g", "se", "g_marginal"):
        assert np.allclose(scaled[name][covered], 0.4 * full[name][covered], rtol=1e-3)
    for name in ("z", "p", "prevalence"):
        assert np.allclose(scaled[name][covered], full[name][covered], atol=1e-3)


def test_approximate_null_agrees_with_the_relocation_null(small_mask):
    """The factorised null has to reproduce the null it replaces, in the tail especially.

    Relocation moves every focus and refits the brain; the approximate null samples each
    study's local configuration and fits one voxel at a time. They are only interchangeable if
    the p-values agree, so that is what is checked -- not merely that both run.
    """
    studyset = create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=0.5,
        n_studies=30,
        sample_size=(15, 45),
        seed=3,
        n_noise_foci=4,
        noise_extent=30.0,
        threshold_z=[2.3263, 3.0902, 3.2905, 4.2649],
    )
    common = dict(fwhm=10.0, mask=small_mask, threshold="study-min", peak_bias="per-study", seed=0)
    exact = CBES(null_method="montecarlo", n_iters=150, **common).fit(studyset)
    approximate = CBES(null_method="approximate", n_iters=300, **common).fit(studyset)

    z_values = exact.get_map("z", return_type="array").ravel()
    covered = z_values != 0
    assert covered.sum() > 100

    p_exact = exact.get_map("p", return_type="array").ravel()[covered]
    p_approx = approximate.get_map("p", return_type="array").ravel()[covered]
    assert np.all((p_approx >= 0) & (p_approx <= 1))
    assert np.corrcoef(p_exact, p_approx)[0, 1] > 0.97
    for alpha in (0.05, 0.01):
        assert abs(np.mean(p_exact < alpha) - np.mean(p_approx < alpha)) < 0.02


def test_approximate_null_is_recorded_and_does_not_depend_on_mask_size(small_mask):
    """Its cost is n_draws x n_studies, with no dependence on the number of voxels.

    That independence is the whole point -- it is what turns a null that scales with the brain
    into one that does not -- so the same draws must serve a mask of any size.
    """
    import nibabel as nib

    studyset = create_effect_size_coordinate_studyset(
        [TRUTH], effect_sizes=0.6, n_studies=20, sample_size=25, seed=4, n_noise_foci=2
    )
    shape = small_mask.shape
    bigger = nib.Nifti1Image(np.ones(tuple(s + 6 for s in shape), np.int32), small_mask.affine)

    histograms = []
    for mask in (small_mask, bigger):
        estimator = CBES(fwhm=8.0, mask=mask, null_method="approximate", n_iters=200, seed=0)
        estimator.fit(studyset)
        histograms.append(
            estimator.null_distributions_["histweights_corr-none_method-approximate"]
        )

    for histogram in histograms:
        assert histogram.sum() > 0
    # Same studies and the same draws, so the sampled configurations cost the same either way.
    assert histograms[0].sum() == histograms[1].sum()


def test_approximate_null_rejects_unknown_methods():
    with pytest.raises(ValueError, match="null_method must be"):
        CBES(null_method="factorised")


def test_expected_ec_derivatives_are_exact():
    """The Newton step in the EM uses these, so they have to be the real derivatives."""
    from nimare.meta.cbma.effectsize import _coverage_resels, _expected_ec

    resels = _coverage_resels(20.0, 8.0)
    z = np.linspace(1.2, 6.0, 25)
    step = 1e-6
    _, first, second = _expected_ec(z, resels)

    numeric_first = (_expected_ec(z + step, resels)[0] - _expected_ec(z - step, resels)[0]) / (
        2 * step
    )
    numeric_second = (_expected_ec(z + step, resels)[1] - _expected_ec(z - step, resels)[1]) / (
        2 * step
    )
    assert np.abs(first - numeric_first).max() < 1e-7
    assert np.abs(second - numeric_second).max() < 1e-7

    # Above its turning point the expansion is in range: positive and falling, as an expected
    # Euler characteristic must be. Below it the approximation is simply invalid -- it counts
    # handles and holes rather than clusters -- which is why the censoring term clamps there
    # rather than trusting it.
    from nimare.meta.cbma.effectsize import _ec_peak

    peak = _ec_peak(resels)
    above = np.linspace(peak, 8.0, 300)
    values = _expected_ec(above, resels)[0]
    assert (values > 0).all()
    assert np.all(np.diff(values) <= 1e-12)


def test_rft_censoring_score_matches_the_log_probability():
    """score must be d log P / d mu, or the EM optimizes a function it is not evaluating.

    Clipping the Euler characteristic at zero -- the obvious way to keep it positive -- breaks
    exactly this: it flattens the numerical derivative while leaving the analytic score
    untouched, so the Newton step walks a gradient belonging to no function.
    """
    from nimare.meta.cbma.effectsize import _coverage_resels, _rft_censoring_terms

    resels = _coverage_resels(20.0, 8.0)
    mu = np.linspace(0.0, 1.2, 25)
    cutoff = np.full(mu.shape, 3.2905)
    sqrt_n = np.full(mu.shape, 4.0)
    step = 1e-6

    terms = _rft_censoring_terms(mu, cutoff, sqrt_n, resels)

    def log_prob(value):
        return np.log(_rft_censoring_terms(value, cutoff, sqrt_n, resels)["prob"])

    numeric = (log_prob(mu + step) - log_prob(mu - step)) / (2 * step)
    assert np.abs(terms["score"] - numeric).max() < 1e-5


def test_rft_censoring_reaches_rates_the_pointwise_form_cannot():
    """Why the term exists: silence is a maximum over a region, not a single voxel.

    A 20 mm sphere holds thousands of voxels, so the chance of clearing a threshold *somewhere*
    inside it far exceeds the chance at any one voxel. The pointwise form therefore predicts
    far less reporting than really happens, which is harmless for a relative map and fatal for
    anything that needs absolute reporting rates.
    """
    from nimare.meta.cbma.effectsize import (
        _censoring_terms,
        _coverage_resels,
        _rft_censoring_terms,
    )

    resels = _coverage_resels(20.0, 8.0)
    sample_size, cutoff_z = 16.0, 3.2905
    effect = np.array([0.3])
    sqrt_n = np.array([np.sqrt(sample_size)])

    regional = _rft_censoring_terms(effect, np.array([cutoff_z]), sqrt_n, resels)["prob"][0]

    cutoff_g = np.array([cutoff_z / np.sqrt(sample_size)])
    sigma = np.array([np.sqrt(1.0 / sample_size)])
    inverse = 1.0 / sigma
    pointwise = _censoring_terms(
        effect, cutoff_g * inverse, 2.0 * cutoff_g * inverse, inverse, inverse**2
    )["prob"][0]

    assert regional < pointwise  # reporting somewhere beats reporting here
    assert regional < 0.25 < pointwise

    # The measurement behind this: on the 21 NIDM pain studies the observed coverage rate is
    # 0.677, and the pointwise form cannot reach it at any effect size -- it gives 0.44 even at
    # g = 0.8 -- which is why matching reporting rates ran to the top of its search range.

    # And it has to be monotone in the effect: a larger effect cannot make silence likelier.
    grid = np.linspace(0.0, 1.0, 20)
    probs = _rft_censoring_terms(
        grid, np.full(20, cutoff_z), np.full(20, np.sqrt(sample_size)), resels
    )["prob"]
    assert np.all(np.diff(probs) <= 1e-12)


def test_rft_quadrature_softens_silence_and_keeps_exact_derivatives():
    """A study's own effect is drawn around mu, not equal to it, and that shape matters.

    Treating the noncentrality as exactly ``mu * sqrt(N)`` makes P(silent) collapse so steeply
    that silence becomes near-proof of a null effect, which drives the fitted effect to zero.
    The pointwise term this replaced carried the spread through ``sqrt(1/N + tau^2)``; dropping
    it was a regression introduced while fixing a different error. Integrating it back has to
    soften the curve without costing the exactness the Newton step depends on.
    """
    from nimare.meta.cbma.effectsize import (
        _coverage_resels,
        _ec_peak,
        _rft_censoring_terms,
    )

    resels = _coverage_resels(20.0, 12.0)
    peak = _ec_peak(resels)
    grid = np.linspace(0.0, 1.0, 15)
    cutoff = np.full(grid.shape, 3.2905)
    sqrt_n = np.full(grid.shape, 4.0)
    tau = np.full(grid.shape, 0.15)

    sharp = _rft_censoring_terms(grid, cutoff, sqrt_n, resels, peak)["prob"]
    smooth = _rft_censoring_terms(grid, cutoff, sqrt_n, resels, peak, tau=tau)["prob"]

    # Softer means silence stays more plausible as the effect grows.
    assert smooth[4] > sharp[4]
    assert (sharp[0] / sharp[4]) > 3.0 * (smooth[0] / smooth[4])

    for spread in (None, tau):
        terms = _rft_censoring_terms(grid, cutoff, sqrt_n, resels, peak, tau=spread)
        step = 1e-6

        def prob_at(value, spread=spread):
            return _rft_censoring_terms(value, cutoff, sqrt_n, resels, peak, tau=spread)["prob"]

        numeric = (prob_at(grid + step) - prob_at(grid - step)) / (2 * step)
        # score is P'/P, so compare P' itself against the finite difference.
        assert np.abs(terms["score"] * terms["prob"] - numeric).max() < 1e-6
        assert np.all(np.diff(terms["prob"]) <= 1e-12)
