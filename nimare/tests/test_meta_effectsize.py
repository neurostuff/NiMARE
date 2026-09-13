"""Tests for nimare.meta.cbma.effectsize (coordinate-based effect-size meta-analysis)."""

import nibabel as nib
import numpy as np
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
        ({"threshold": "global-min"}, "threshold must be"),
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
        CBES(mask=small_mask, selection_model="none", null_method="parametric").fit(plain)


def test_cbes_produces_expected_maps(studyset, small_mask):
    result = CBES(fwhm=12.0, mask=small_mask, null_method="parametric").fit(studyset)

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
    result = CBES(fwhm=12.0, mask=small_mask, null_method="parametric").fit(studyset)
    assert "Hedges" in result.description_
    assert "censor" in result.description_.lower()

    quiet = CBES(
        fwhm=12.0, mask=small_mask, generate_description=False, null_method="parametric"
    ).fit(studyset)
    assert quiet.description_ == ""


def test_selection_model_reduces_the_winners_curse(studyset, small_mask):
    """Pooling reported peaks alone overestimates; modelling the silence pulls it back."""
    naive = CBES(fwhm=12.0, mask=small_mask, selection_model="none", null_method="parametric").fit(
        studyset
    )
    corrected = CBES(
        fwhm=12.0, mask=small_mask, selection_model="zero-inflated", null_method="parametric"
    ).fit(studyset)

    naive_g = value_at(naive, "g")
    corrected_g = value_at(corrected, "g")

    # The naive estimate is biased away from zero, in the direction theory predicts.
    assert naive_g > TRUE_G + 0.05
    assert corrected_g < naive_g
    assert abs(corrected_g - TRUE_G) < abs(naive_g - TRUE_G)


def test_tobit_undercorrects_when_studies_genuinely_have_no_effect(mixed_studyset, small_mask):
    """The zero component is what keeps silence from being read as a small common effect.

    With half the studies genuinely null at the focus, a plain Tobit has to explain their
    silence as a small shared effect and drags the estimate down; the zero-inflated model can
    attribute it to the zero component instead.
    """
    tobit = CBES(
        fwhm=12.0, mask=small_mask, selection_model="tobit", null_method="parametric"
    ).fit(mixed_studyset)
    zero_inflated = CBES(
        fwhm=12.0, mask=small_mask, selection_model="zero-inflated", null_method="parametric"
    ).fit(mixed_studyset)
    assert value_at(zero_inflated, "g") > value_at(tobit, "g")
    assert value_at(zero_inflated, "prevalence") < 1.0


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
        result = CBES(fwhm=12.0, mask=small_mask, null_method="parametric").fit(studyset)
        estimates[prevalence] = value_at(result, "prevalence")

    assert estimates[1.0] > estimates[0.4]


def test_fixed_effects_option_zeroes_tau2(studyset, small_mask):
    result = CBES(fwhm=12.0, mask=small_mask, tau2_method="none", null_method="parametric").fit(
        studyset
    )
    assert np.all(result.get_map("tau2", return_type="array") == 0)


def test_correct_fwe_montecarlo(studyset, small_mask):
    estimator = CBES(fwhm=12.0, mask=small_mask, selection_model="none", null_method="parametric")
    result = estimator.fit(studyset)
    maps, tables, description = estimator.correct_fwe_montecarlo(
        result, n_iters=5, seed=0, vfwe_only=True
    )

    assert tables == {}
    assert "Monte Carlo" in description
    logp = maps["logp_level-voxel"]
    p_corrected = 10.0**-logp
    assert np.all((p_corrected > 0) & (p_corrected <= 1))
    # Correction can only make p-values larger.
    assert np.all(p_corrected >= result.get_map("p", return_type="array") - 1e-6)


def test_correct_fwe_montecarlo_needs_a_fit(small_mask):
    with pytest.raises(ValueError, match="requires a fitted estimator"):
        CBES(mask=small_mask, null_method="parametric").correct_fwe_montecarlo(None, n_iters=2)


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


def test_parametric_null_warns_that_it_is_anticonservative(small_mask, caplog):
    with caplog.at_level("WARNING"):
        CBES(mask=small_mask, null_method="parametric")
    assert "anticonservative" in caplog.text


def test_montecarlo_null_calibrates_uncorrected_p(null_studyset, small_mask):
    """Under a global null the parametric p-values are far too liberal; the spatial null is not.

    This is the reason ``null_method`` defaults to ``"montecarlo"``. The parametric standard
    error treats tau-squared as known and ignores that the peaks being pooled were selected for
    being large, so ``g / se`` is not a null-referenced statistic at all.
    """
    parametric = CBES(
        fwhm=12.0, mask=small_mask, selection_model="none", null_method="parametric"
    ).fit(null_studyset)
    montecarlo = CBES(
        fwhm=12.0, mask=small_mask, selection_model="none", null_method="montecarlo", n_iters=50
    ).fit(null_studyset)

    parametric_rate = np.mean(parametric.get_map("p", return_type="array") < 0.05)
    montecarlo_rate = np.mean(montecarlo.get_map("p", return_type="array") < 0.05)

    assert parametric_rate > 0.20  # measured around 0.40
    assert montecarlo_rate < 0.15
    assert montecarlo_rate < parametric_rate


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
    wide = CBES(fwhm=12.0, mask=small_mask, kernel_min_weight=1e-6, null_method="parametric").fit(
        studyset
    )
    narrow = CBES(
        fwhm=12.0, mask=small_mask, kernel_min_weight=0.25, null_method="parametric"
    ).fit(studyset)

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
    estimator = CBES(max_iter=8, null_method="parametric")
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
    estimator = CBES(fwhm=8.0, null_method="parametric", use_images=True)
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
        CBES(fwhm=8.0, null_method="parametric", use_images=True)
        .fit(studyset)
        .get_map("g", return_type="array")
        .ravel()
    )
    from_coords = (
        CBES(fwhm=8.0, null_method="parametric", use_images=False)
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
    estimator = CBES(fwhm=8.0, null_method="parametric", use_images=False)
    estimator.fit(studyset)
    assert estimator._image_studies_ == {}
    assert len(estimator._focus_table_) == 10


def test_peak_bias_rescales_the_estimate_exactly(image_studyset):
    """rho rescales g, its variance and the threshold together, so the fit scales with it.

    That exactness is what makes rho calibratable: a single ratio of summaries recovers it.
    """
    studyset, _ = image_studyset
    plain = (
        CBES(fwhm=8.0, null_method="parametric", use_images=False)
        .fit(studyset)
        .get_map("g", return_type="array")
        .ravel()
    )
    scaled = (
        CBES(fwhm=8.0, null_method="parametric", use_images=False, peak_bias=0.4)
        .fit(studyset)
        .get_map("g", return_type="array")
        .ravel()
    )
    covered = plain != 0
    assert np.allclose(scaled[covered], 0.4 * plain[covered], rtol=1e-6)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_peak_bias_rejects_out_of_range_values(bad):
    with pytest.raises(ValueError, match="peak_bias must be None or in"):
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
    estimator = CBES(fwhm=12.0, mask=small_mask, null_method="parametric")
    with caplog.at_level("WARNING"):
        estimator.fit(studyset)

    assert "uninformative" in caplog.text or "close to uninformative" in caplog.text
    assert estimator.peak_information_["excess_z"] < 0.25
