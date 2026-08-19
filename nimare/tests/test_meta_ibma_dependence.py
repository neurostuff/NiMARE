"""Tests for dependence handling in IBMA estimators."""

import copy

import numpy as np
import pymare
import pytest

from nimare.meta import ibma
from nimare.meta._permutation import _permuted_ols

# Estimators backed by a PyMARE meta-regression estimator, which take weight_scheme/rho and
# report Satterthwaite degrees of freedom.
REGRESSION_ESTIMATORS = [
    ibma.WeightedLeastSquares,
    ibma.DerSimonianLaird,
    ibma.Hedges,
    ibma.SampleSizeBasedLikelihood,
    ibma.VarianceBasedLikelihood,
    ibma.FixedEffectsHedges,
]

# Estimators backed by a PyMARE combination test, which need the null correlation matrix.
COMBINATION_ESTIMATORS = [ibma.Fishers, ibma.Stouffers]

PYMARE_ESTIMATORS = [*COMBINATION_ESTIMATORS, *REGRESSION_ESTIMATORS]

ALL_ESTIMATORS = [*PYMARE_ESTIMATORS, ibma.PermutedOLS]

# One estimator per family. Behaviour that lives entirely in IBMAEstimator cannot vary by
# subclass, so sweeping all nine would re-run the same base-class lines nine times; these
# three still catch a subclass that forgets to forward its kwargs.
REPRESENTATIVE_ESTIMATORS = [ibma.Fishers, ibma.DerSimonianLaird, ibma.PermutedOLS]


def _explicit_cr2_weighted_t(contributions, weights):
    """Return an intercept-only WLS t with singleton-cluster CR2 variance."""
    weights = np.asarray(weights, dtype=float)
    total_weight = weights.sum()
    mean = (weights[:, None] * contributions).sum(axis=0) / total_weight
    leverage = weights / total_weight
    residuals = contributions - mean
    meat = (np.square(weights)[:, None] * np.square(residuals) / (1.0 - leverage)[:, None]).sum(
        axis=0
    )
    return mean / (np.sqrt(meat) / total_weight)


@pytest.fixture(scope="module")
def dependent_dataset(testdata_ibma_multiple_contrasts):
    """Return repeated images whose study groups share one participant count."""
    dataset = copy.deepcopy(testdata_ibma_multiple_contrasts)
    for _, indices in dataset.metadata.groupby("study_id").groups.items():
        indices = list(indices)
        value = copy.deepcopy(dataset.metadata.at[indices[0], "sample_sizes"])
        for index in indices:
            dataset.metadata.at[index, "sample_sizes"] = copy.deepcopy(value)
    return dataset


# Parameters


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_groupby_defaults_to_study_id(estimator):
    """The default must group by study, which is what None means."""
    assert estimator().groupby is None


@pytest.mark.parametrize("estimator", REGRESSION_ESTIMATORS)
def test_weighting_parameters_are_validated(estimator):
    """Invalid PyMARE weighting arguments should fail at construction."""
    with pytest.raises(ValueError, match="Invalid weight_scheme"):
        estimator(weight_scheme="nonsense")
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        estimator(rho=1.5)
    with pytest.raises(ValueError, match="must be a number"):
        estimator(rho="0.8")


@pytest.mark.parametrize("estimator", REGRESSION_ESTIMATORS)
def test_weighting_parameter_defaults(estimator):
    """The default rescale/0.8 is the correlated-effects working model of robumeta."""
    instance = estimator()
    assert instance.weight_scheme == "rescale"
    assert instance.rho == 0.8


@pytest.mark.parametrize(
    "weight_scheme,codes,expected",
    [
        ("collapse", [0, 0, 1], {"weight_scheme": "collapse", "rho": 0.8}),
        ("rescale", [0, 0, 1], {"weight_scheme": "rescale", "rho": 0.8}),
        # 'individual' models no within-group correlation, and PyMARE warns if handed rho.
        ("individual", [0, 0, 1], {"weight_scheme": "individual"}),
        # No image is repeated, so there is no within-group correlation to assume.
        ("rescale", [0, 1, 2], {"weight_scheme": "rescale"}),
    ],
)
def test_pymare_weighting_kwargs(weight_scheme, codes, expected):
    """Pass PyMARE its own parameters, and only the ones that apply."""
    estimator = ibma.WeightedLeastSquares(weight_scheme=weight_scheme)
    estimator.inputs_ = {"contrast_names": np.array(codes)}

    assert estimator._pymare_weighting_kwargs(np.arange(len(codes))) == expected


def test_stouffers_rejects_the_removed_normalization_parameter():
    """The removed parameter should explain itself rather than raise a bare TypeError."""
    with pytest.raises(TypeError, match="normalize_contrast_weights was removed"):
        ibma.Stouffers(normalize_contrast_weights=True)


# groupby


@pytest.mark.parametrize("estimator", REPRESENTATIVE_ESTIMATORS)
def test_no_dependence_when_unrepeated_or_opted_out(estimator, testdata_ibma, dependent_dataset):
    """`labels` is None both when no study repeats and when the caller opts out."""
    unrepeated = estimator()
    unrepeated.fit(testdata_ibma)
    assert unrepeated._dependence().labels is None
    assert unrepeated.inputs_["corr_matrix"] is None

    opted_out = estimator(groupby=False)
    opted_out.fit(dependent_dataset)
    assert opted_out._dependence().labels is None


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_grouping_recorded_and_correlation_gated(estimator, dependent_dataset):
    """Every estimator records the grouping; only the combination tests estimate a correlation.

    Parametrized over all nine because `_requires_corr_matrix` genuinely varies by class.
    """
    meta = estimator()
    meta.fit(dependent_dataset)

    n_images = len(meta.inputs_["id"])
    groups = meta._dependence().labels
    assert groups is not None
    assert len(groups) == n_images
    assert np.unique(groups).size < n_images

    corr = meta.inputs_["corr_matrix"]
    if meta._requires_corr_matrix:
        # Brown's method and Stouffer's inflation term both need it.
        assert corr is not None
        assert corr.shape == (n_images, n_images)
    else:
        # Cluster-robust inference is distribution-free, so nothing would read it.
        assert corr is None


def test_groupby_accepts_a_metadata_field(dependent_dataset):
    """A metadata column can define the grouping instead of study_id."""
    meta = ibma.Stouffers(groupby="sample_sizes")
    meta.fit(dependent_dataset)

    labels = meta.inputs_["dependence_groups"]
    assert len(labels) == len(meta.inputs_["id"])
    # The integer codes must follow that field, not study_id.
    codes = meta.inputs_["contrast_names"]
    for left in range(len(codes)):
        for right in range(len(codes)):
            assert (codes[left] == codes[right]) == (labels[left] == labels[right])


def test_groupby_accepts_an_explicit_label_array(dependent_dataset):
    """Users can supply one label per image directly."""
    n_images = len(ibma.Stouffers().fit(dependent_dataset).estimator.inputs_["id"])
    labels = np.zeros(n_images, dtype=int)

    meta = ibma.Stouffers(groupby=labels)
    meta.fit(dependent_dataset)

    assert np.unique(meta.inputs_["contrast_names"]).size == 1


def test_groupby_rejects_a_mismatched_label_array(dependent_dataset):
    """A label array that does not cover every image is a user error, not a silent one."""
    with pytest.raises(ValueError, match="one label per image"):
        ibma.Stouffers(groupby=np.array([0, 1])).fit(dependent_dataset)


def test_groupby_can_split_a_study_back_into_independent_samples(dependent_dataset):
    """Splitting a study apart must reproduce the ungrouped result.

    The NeuroVault case where one paper uploads patients and controls separately.
    """
    reference = ibma.DerSimonianLaird(groupby=False).fit(dependent_dataset)
    n_images = len(reference.estimator.inputs_["id"])

    split = ibma.DerSimonianLaird(groupby=np.arange(n_images)).fit(dependent_dataset)

    for name in ("z", "p", "est", "se", "dof"):
        valid = np.isfinite(reference.maps[name]) & np.isfinite(split.maps[name])
        assert valid.any()
        assert np.allclose(reference.maps[name][valid], split.maps[name][valid])


# Inference


@pytest.mark.parametrize("estimator", PYMARE_ESTIMATORS)
def test_dependence_changes_inference(estimator, dependent_dataset):
    """Correcting for dependence should move both the estimates and the p-values.

    Widening standard errors alone would leave image multiplicity changing the point
    estimate; the default weighting gives every group the same total weight first.
    """
    corrected = estimator().fit(dependent_dataset)
    naive = estimator(groupby=False).fit(dependent_dataset)

    corrected_p = corrected.maps["p"]
    naive_p = naive.maps["p"]

    valid = np.isfinite(corrected_p) & np.isfinite(naive_p)
    assert valid.any()
    assert not np.allclose(corrected_p[valid], naive_p[valid])

    # The estimand changes too, because the weights do.
    if "est" in corrected.maps and "est" in naive.maps:
        est_valid = np.isfinite(corrected.maps["est"]) & np.isfinite(naive.maps["est"])
        assert not np.allclose(
            corrected.maps["est"][est_valid],
            naive.maps["est"][est_valid],
        )


@pytest.mark.parametrize("estimator", COMBINATION_ESTIMATORS)
def test_combination_dof_counts_groups_not_images(estimator, dependent_dataset):
    """A study contributing several images must not buy degrees of freedom."""
    meta = estimator()
    results = meta.fit(dependent_dataset)

    n_images = len(meta.inputs_["id"])
    n_groups = np.unique(meta.inputs_["contrast_names"]).size
    assert n_groups < n_images

    dof = results.maps["dof"]
    assert set(dof[dof > 0].tolist()) == {n_groups - 1}


@pytest.mark.parametrize("estimator", REGRESSION_ESTIMATORS)
def test_regression_dof_is_pymare_satterthwaite(estimator, dependent_dataset):
    """The dof map must be PyMARE's `fe_dof`, not a recomputation or a count of images."""
    meta = estimator()
    results = meta.fit(dependent_dataset)

    n_images = len(meta.inputs_["id"])
    group_count_dof = meta._dependence().dof
    reported = results.maps["dof"]
    reported = reported[np.isfinite(reported)]
    assert reported.size
    # Satterthwaite dof are non-integer and cannot exceed what the images support.
    assert np.all(reported < n_images - 1)
    assert not np.allclose(reported, np.round(reported))
    assert not np.allclose(reported, group_count_dof)

    # And they are literally what PyMARE reported, not something derived alongside it.
    voxel_mask = meta.inputs_["aggressive_mask"]
    captured = {}
    # summary() resolves this name from the estimators module, which imported it directly,
    # so patching pymare.results would not be seen.
    real = pymare.estimators.estimators.MetaRegressionResults

    class _Spy(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["fe_dof"] = self.fe_dof

    pymare.estimators.estimators.MetaRegressionResults = _Spy
    try:
        maps = meta._fit_model(
            *[
                meta.inputs_[name][:, voxel_mask]
                for name, (type_, _) in meta._required_inputs.items()
                if type_ == "image"
            ]
        )
    finally:
        pymare.estimators.estimators.MetaRegressionResults = real

    assert captured["fe_dof"] is not None
    assert np.allclose(maps[-1], np.ravel(captured["fe_dof"]))


@pytest.mark.parametrize("aggressive_mask", [True, False])
@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_fit_produces_well_formed_maps(estimator, aggressive_mask, dependent_dataset):
    """Every estimator returns usable maps under both masking strategies.

    Well-formedness only. Robust standard errors are not guaranteed to be larger than
    model-based ones -- RVE estimates between-group variability rather than trusting the
    reported sampling variances, so an overstated varcope can legitimately shrink them --
    which leaves no inequality to assert here.
    """
    meta = estimator(aggressive_mask=aggressive_mask)
    results = meta.fit(dependent_dataset)

    assert np.isfinite(results.maps["z"]).any()

    if "p" in results.maps:
        finite_p = results.maps["p"][np.isfinite(results.maps["p"])]
        assert np.all((finite_p >= 0) & (finite_p <= 1))

    # Float, so voxels outside the mask read as NaN rather than INT_MIN.
    dof = results.maps["dof"]
    assert dof.dtype.kind == "f"
    assert np.isfinite(dof).any()
    if aggressive_mask:
        outside = ~meta.inputs_["aggressive_mask"]
        if outside.any():
            assert np.isnan(dof[outside]).all()


@pytest.mark.parametrize("weight_scheme", ["individual", "rescale", "collapse"])
def test_every_weight_scheme_runs(weight_scheme, dependent_dataset):
    """All three PyMARE schemes must be reachable from NiMARE."""
    results = ibma.DerSimonianLaird(weight_scheme=weight_scheme).fit(dependent_dataset)

    assert np.isfinite(results.maps["z"]).any()


def test_rho_barely_moves_the_result(dependent_dataset):
    """Sweeping rho barely moves the result, because it enters only through tau^2.

    This is what makes 0.8 a safe default and a sensitivity sweep cheap.
    """
    low = ibma.DerSimonianLaird(rho=0.0).fit(dependent_dataset).maps["z"]
    high = ibma.DerSimonianLaird(rho=1.0).fit(dependent_dataset).maps["z"]

    valid = np.isfinite(low) & np.isfinite(high)
    assert valid.any()

    # Judged on the bulk of the map rather than its worst voxel. The published claim is
    # about aggregate behaviour -- error rates moving well under a percentage point -- not
    # a supremum bound, and a single voxel can move appreciably more than the rest. Here
    # the median shift is a few thousandths of the range the map covers while the single
    # worst voxel moves ~7% of it.
    shift = np.abs(low[valid] - high[valid])
    spread = np.ptp(low[valid])
    assert np.median(shift) < 0.01 * spread
    assert np.percentile(shift, 95) < 0.05 * spread


def test_correlation_matrix_is_not_measuring_shared_signal(dependent_dataset):
    """Studies that are independent by construction must look independent.

    Correlating the raw maps conflates dependence with agreement, which would spread the
    variance correction across the whole analysis instead of the repeated study.
    """
    meta = ibma.Stouffers()
    meta.fit(dependent_dataset)

    corr = meta.inputs_["corr_matrix"]
    groups = meta.inputs_["contrast_names"]
    same_study = groups[:, None] == groups[None, :]
    off_diagonal = ~np.eye(corr.shape[0], dtype=bool)

    between = np.nanmean(corr[~same_study])
    within = np.nanmean(corr[same_study & off_diagonal])

    assert abs(between) < 0.1  # different studies are independent
    assert within > between  # the repeated study is where dependence lives


# Combination tests


def test_stouffers_weights_each_group_by_sqrt_n():
    """Weight a group by sqrt(n), repeated per image and not divided by map count."""
    estimator = ibma.Stouffers(use_sample_size=True)
    estimator.inputs_ = {
        "contrast_names": np.array([0, 0, 0, 1]),
        "num_contrasts": np.array([3, 3, 3, 1]),
        "corr_matrix": np.eye(4),
        "sample_sizes": np.array([[25.0], [25.0], [25.0], [100.0]]),
        "id": ["a", "b", "c", "d"],
    }

    captured = {}
    real_dataset = ibma.pymare.Dataset

    def _spy(**kwargs):
        captured.update(kwargs)
        return real_dataset(**kwargs)

    ibma.pymare.Dataset = _spy
    try:
        estimator._fit_model(np.ones((4, 5)), study_mask=np.arange(4), corr=np.eye(4))
    finally:
        ibma.pymare.Dataset = real_dataset

    assert np.allclose(np.asarray(captured["n"]).ravel(), [5.0, 5.0, 5.0, 10.0])


def test_stouffers_aggregates_groups_whenever_they_exist():
    """Aggregate whenever groups exist, so a prolific study cannot outvote its peers."""
    estimator = ibma.Stouffers()
    estimator.inputs_ = {
        "contrast_names": np.array([0, 0, 0, 1]),
        "num_contrasts": np.array([3, 3, 3, 1]),
        "corr_matrix": np.eye(4),
        "id": ["a", "b", "c", "d"],
    }

    captured = {}
    real_estimator = ibma.pymare.estimators.StoufferCombinationTest

    def _spy(**kwargs):
        captured.update(kwargs)
        return real_estimator(**kwargs)

    ibma.pymare.estimators.StoufferCombinationTest = _spy
    try:
        estimator._fit_model(np.ones((4, 5)), study_mask=np.arange(4), corr=np.eye(4))
    finally:
        ibma.pymare.estimators.StoufferCombinationTest = real_estimator

    assert captured["group_level"] is True


def test_fishers_passes_one_sample_size_coefficient_per_study_to_pymare():
    """Weighted Fisher receives repeated group weights, not image-count weights."""
    estimator = ibma.Fishers(use_sample_size=True)
    estimator.inputs_ = {
        "contrast_names": np.array([0, 0, 0, 1]),
        "num_contrasts": np.array([3, 3, 3, 1]),
        "corr_matrix": np.eye(4),
        "sample_sizes": np.array([[25.0], [25.0], [25.0], [100.0]]),
        "id": ["a", "b", "c", "d"],
    }

    captured = {}
    real_dataset = ibma.pymare.Dataset

    def _spy(**kwargs):
        captured.update(kwargs)
        return real_dataset(**kwargs)

    ibma.pymare.Dataset = _spy
    try:
        estimator._fit_model(np.ones((4, 5)), study_mask=np.arange(4), corr=np.eye(4))
    finally:
        ibma.pymare.Dataset = real_dataset

    assert np.array_equal(np.asarray(captured["n"]).ravel(), [25.0, 25.0, 25.0, 100.0])


def test_fishers_preserves_two_sided_positional_argument():
    """Adding sample-size weighting must not reinterpret released positional calls."""
    estimator = ibma.Fishers(False)

    assert estimator.two_sided is False
    assert estimator.use_sample_size is False


# PermutedOLS


def test_permuted_ols_matches_nilearn_when_ungrouped():
    """The ungrouped statistic must be Nilearn's, not an approximation of it."""
    from nilearn.mass_univariate import permuted_ols

    rng = np.random.default_rng(3)
    maps = rng.normal(0.4, 1.0, size=(12, 40))

    nilearn_result = permuted_ols(
        np.ones((12, 1)),
        maps,
        confounding_vars=None,
        model_intercept=False,
        n_perm=0,
        two_sided_test=True,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    nilearn_t = nilearn_result["t"] if isinstance(nilearn_result, dict) else nilearn_result[1]
    nimare_t = _permuted_ols(maps, exchangeability_blocks=np.arange(12))["t"]

    assert np.allclose(np.ravel(nilearn_t), np.ravel(nimare_t))


@pytest.mark.parametrize("groupby,grouped", [(None, True), (False, False)])
def test_permuted_ols_blocks_follow_groupby(groupby, grouped, dependent_dataset):
    """Grouping collapses a study's images into one block; opting out makes it the identity."""
    meta = ibma.PermutedOLS(groupby=groupby)
    meta.fit(dependent_dataset)

    n_images = len(meta.inputs_["id"])
    blocks, weights = meta._blocks_and_weights(np.arange(n_images))

    assert weights is None
    if grouped:
        assert np.unique(blocks).size < n_images
    else:
        assert np.array_equal(blocks, np.arange(n_images))


def test_permuted_ols_dof_counts_groups(dependent_dataset):
    """Grouping must cost degrees of freedom, not preserve the image count."""
    grouped = ibma.PermutedOLS().fit(dependent_dataset)
    ungrouped = ibma.PermutedOLS(groupby=False).fit(dependent_dataset)

    n_images = len(ungrouped.estimator.inputs_["id"])
    n_groups = np.unique(grouped.estimator.inputs_["contrast_names"]).size

    # Outside the mask the map is NaN, as every other float map already is.
    ungrouped_dof = ungrouped.maps["dof"]
    grouped_dof = grouped.maps["dof"]
    assert set(ungrouped_dof[np.isfinite(ungrouped_dof)].tolist()) == {float(n_images - 1)}
    assert set(grouped_dof[np.isfinite(grouped_dof)].tolist()) == {float(n_groups - 1)}


def test_permuted_ols_sample_size_weights_shrink_the_dof(dependent_dataset):
    """Unequal weights make Satterthwaite dof fall below the group count."""
    equal = ibma.PermutedOLS().fit(dependent_dataset)
    weighted = ibma.PermutedOLS(use_sample_size=True).fit(dependent_dataset)

    valid = np.isfinite(weighted.maps["dof"]) & np.isfinite(equal.maps["dof"])
    assert valid.any()
    assert np.all(weighted.maps["dof"][valid] <= equal.maps["dof"][valid])
    assert np.isfinite(weighted.maps["z"]).any()


def test_permuted_ols_weights_each_group_once(dependent_dataset):
    """A study's sample size must not be counted once per map it uploaded."""
    meta = ibma.PermutedOLS(use_sample_size=True)
    meta.fit(dependent_dataset)

    n_images = len(meta.inputs_["id"])
    _, weights = meta._blocks_and_weights(np.arange(n_images))
    group_order = meta._dependence().group_order

    assert weights.shape == (group_order.size,)
    assert group_order.size < n_images


def test_permuted_ols_fwe_shares_one_null_across_bags(dependent_dataset):
    """The max-statistic null must describe the whole brain, not one bag of it."""
    from nimare.correct import FWECorrector

    meta = ibma.PermutedOLS(aggressive_mask=False)
    corrector = FWECorrector(method="montecarlo", n_iters=20)
    corrected = corrector.transform(meta.fit(dependent_dataset))

    # FWECorrector operates on a copy of the estimator and hands it back on the result.
    null = corrected.estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
    assert null.shape == (20,)
    assert np.all(np.isfinite(null))

    logp = corrected.maps["logp_level-voxel_corr-FWE_method-montecarlo"]
    finite = logp[np.isfinite(logp)]
    assert finite.size
    assert np.all(finite >= 0)


def test_permuted_ols_fwe_smoke(dependent_dataset):
    """FWE correction runs on the aggressive-mask path too."""
    from nimare.correct import FWECorrector

    meta = ibma.PermutedOLS(use_sample_size=True)
    corrected = FWECorrector(method="montecarlo", n_iters=20).transform(
        meta.fit(dependent_dataset)
    )

    p_map = corrected.maps["p_level-voxel_corr-FWE_method-montecarlo"]
    finite = p_map[np.isfinite(p_map)]
    assert finite.size
    assert np.all(finite > 0)
    assert np.all(finite <= 1)


# The permutation module itself


def test_permutation_collapses_blocks_to_their_means():
    """The statistic is over group means, so within-block spread is irrelevant."""
    beta_maps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
    groups = np.array([0, 0, 1])

    result = _permuted_ols(beta_maps, exchangeability_blocks=groups)

    contributions = np.array([beta_maps[:2].mean(axis=0), beta_maps[2]])
    expected = contributions.mean(axis=0) / (contributions.std(axis=0, ddof=1) / np.sqrt(2))
    assert np.allclose(result["t"].squeeze(), expected)
    assert result["dof"] == 1


def test_permutation_ignores_within_block_dispersion():
    """Changing within-block spread at fixed block means must not alter inference."""
    groups = np.array([0, 0, 1, 1, 2, 2])
    block_means = np.array([[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]])
    concentrated = np.repeat(block_means, 2, axis=0)
    dispersed = concentrated.copy()
    dispersed[::2] -= 10
    dispersed[1::2] += 10

    assert np.allclose(
        _permuted_ols(concentrated, exchangeability_blocks=groups)["t"],
        _permuted_ols(dispersed, exchangeability_blocks=groups)["t"],
    )


def test_permutation_uses_one_weight_per_block():
    """Map multiplicity must not multiply a block's sample-size weight."""
    beta_maps = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 8.0]])
    groups = np.array([0, 0, 1])
    weights = np.array([20.0, 80.0])

    result = _permuted_ols(
        beta_maps,
        exchangeability_blocks=groups,
        group_weights=weights,
    )

    contributions = np.vstack([beta_maps[:2].mean(axis=0), beta_maps[2]])
    assert np.allclose(result["t"].squeeze(), _explicit_cr2_weighted_t(contributions, weights))


def test_permutation_unit_weights_reduce_to_the_ordinary_one_sample_t():
    """CR2 WLS with equal weights must equal the ordinary t, including flips."""
    contributions = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [4.0, -1.0]])
    sign_flips = np.array([[1.0, 1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    kwargs = {
        "exchangeability_blocks": np.arange(4),
        "n_perm": 2,
        "sign_flips": sign_flips,
    }

    ordinary = _permuted_ols(contributions, **kwargs)
    weighted = _permuted_ols(contributions, group_weights=np.ones(4), **kwargs)

    for key in ("t", "h0_max_t", "logp_max_t", "dof"):
        assert np.allclose(weighted[key], ordinary[key])


@pytest.mark.filterwarnings("ignore:Cluster-robust variance")
def test_permutation_statistic_matches_pymare_cr2_intercept_model():
    """Match generic PyMARE CR2 WLS with the vectorized statistic."""
    contributions = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [4.0, -1.0]])
    weights = np.array([20.0, 40.0, 80.0, 25.0])

    observed = _permuted_ols(
        contributions,
        exchangeability_blocks=np.arange(4),
        group_weights=weights,
    )["t"].squeeze()

    pymare_estimator = pymare.estimators.WeightedLeastSquares().fit(
        y=contributions,
        v=np.broadcast_to(1.0 / weights[:, None], contributions.shape),
        X=np.ones((4, 1)),
        g=np.arange(4),
    )
    beta = pymare_estimator.params_["fe_params"].squeeze()
    covariance = pymare_estimator.params_["inv_cov"].squeeze()

    assert np.allclose(observed, beta / np.sqrt(covariance))


def test_permutation_recomputes_weighted_studentization_after_flips():
    """Every null statistic must use the weighted mean and its CR2 residuals."""
    beta_maps = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    groups = np.array([0, 0, 1])
    weights = np.array([20.0, 80.0])
    sign_flips = np.array([[1.0, -1.0], [-1.0, 1.0]])

    result = _permuted_ols(
        beta_maps,
        exchangeability_blocks=groups,
        group_weights=weights,
        n_perm=2,
        sign_flips=sign_flips,
    )

    contributions = np.vstack([beta_maps[:2].mean(axis=0), beta_maps[2]])
    expected = np.array(
        [
            np.max(np.abs(_explicit_cr2_weighted_t(contributions * signs[:, None], weights)))
            for signs in sign_flips
        ]
    )
    assert np.allclose(result["h0_max_t"], expected)


def test_permutation_flips_blocks_as_units():
    """A supplied sign schedule should apply one sign to every map in a block."""
    beta_maps = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    groups = np.array([0, 0, 1])
    sign_flips = np.array([[1.0, -1.0], [-1.0, 1.0]])

    result = _permuted_ols(
        beta_maps,
        exchangeability_blocks=groups,
        n_perm=2,
        sign_flips=sign_flips,
    )

    contributions = np.array([0.5 * beta_maps[0] + 0.5 * beta_maps[1], beta_maps[2]])
    sum_squares = np.square(contributions).sum(axis=0)
    permuted_sums = sign_flips @ contributions
    permuted_t = permuted_sums / np.sqrt(2 * sum_squares - np.square(permuted_sums))
    assert np.allclose(result["h0_max_t"], np.max(np.abs(permuted_t), axis=1))


def test_permutation_parallel_null_is_reproducible():
    """The same seed should produce the same null regardless of worker count."""
    beta_maps = np.arange(24, dtype=float).reshape(6, 4) - 10
    groups = np.repeat(np.arange(3), 2)
    kwargs = {
        "exchangeability_blocks": groups,
        "n_perm": 8,
        "random_state": 7,
    }

    single_job = _permuted_ols(beta_maps, n_jobs=1, **kwargs)
    two_jobs = _permuted_ols(beta_maps, n_jobs=2, **kwargs)

    assert np.array_equal(single_job["h0_max_t"], two_jobs["h0_max_t"])
    assert np.array_equal(single_job["logp_max_t"], two_jobs["logp_max_t"])
