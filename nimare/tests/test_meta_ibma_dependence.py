"""Tests for dependence handling in IBMA estimators."""

import inspect
import logging

import numpy as np
import pymare
import pytest
from nilearn.maskers import NiftiMasker

from nimare.meta import ibma
from nimare.meta._dependence import DependenceModel
from nimare.meta._permutation import _permuted_ols
from nimare.meta.ibma import _null_correlation
from nimare.meta.utils import _apply_liberal_mask
from nimare.transforms import z_to_p


class _FakeDataset:
    """Stand in for a Dataset when only its masker is read.

    ``_fit`` is exercised directly here so that hand-built bags can pin an exact coverage
    pattern, which a real Dataset of NIfTI files cannot.
    """

    def __init__(self, masker):
        self.masker = masker


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


def _capture_kwargs(monkeypatch, target, name):
    """Patch ``target.name`` with a pass-through spy, and return the dict it records into.

    The four tests that check what NiMARE hands PyMARE differ only in which callable they
    watch and which argument they assert on.
    """
    captured = {}
    real = getattr(target, name)

    def _spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(target, name, _spy)
    return captured


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
    """Return a dataset in which some studies contributed several images.

    Its sample sizes are left as they are, which means a study's contrasts disagree about
    them -- ``pain.nidm`` reports 25, 20, 9 and 12. That is what real data looks like, so
    every estimator here has to cope with it.
    """
    metadata = testdata_ibma_multiple_contrasts.metadata
    # sample_sizes holds lists, which are unhashable, so reduce before counting.
    sizes = metadata["sample_sizes"].map(lambda value: float(np.mean(value)))
    assert (
        sizes.groupby(metadata["study_id"]).nunique().max() > 1
    ), "fixture must vary sample size within a study"
    return testdata_ibma_multiple_contrasts


@pytest.fixture(scope="module")
def fitted(dependent_dataset):
    """Fit an estimator on ``dependent_dataset`` once per configuration, then reuse it.

    A fit is by far the slowest thing in this file, and several tests read different things
    off the same one -- the maps, the dof map, the resolved grouping. Keyword values must be
    hashable; a test that passes an array calls ``fit`` itself.
    """
    cache = {}

    def _fit(estimator, **kwargs):
        key = (estimator, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = estimator(**kwargs).fit(dependent_dataset)
        return cache[key]

    return _fit


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
def test_small_sample_correction_is_validated(estimator):
    """A misspelled correction should fail at construction, not reach PyMARE."""
    with pytest.raises(ValueError, match="Invalid small_sample_correction"):
        estimator(small_sample_correction="knapp_hartung")


@pytest.mark.parametrize("estimator", REGRESSION_ESTIMATORS)
def test_small_sample_correction_defers_to_pymare_by_default(estimator):
    """None must leave the argument out, so each PyMARE estimator keeps its own default.

    They differ: the estimators that estimate tau^2 correct by default, while
    WeightedLeastSquares does not, since its tau^2 is supplied rather than estimated.
    """
    meta = estimator()
    assert meta.small_sample_correction is None
    meta.inputs_ = {"contrast_names": np.array([0, 1, 2])}

    est = meta._pymare_estimator(np.arange(3))
    expected = inspect.signature(type(est).__init__).parameters["small_sample_correction"].default
    assert est.small_sample_correction == expected


@pytest.mark.parametrize("estimator", REGRESSION_ESTIMATORS)
@pytest.mark.parametrize("correction", ["knapp-hartung", "knapp-hartung-conservative", "wald"])
def test_small_sample_correction_is_forwarded(estimator, correction):
    """An explicit choice must reach the PyMARE estimator and change the inference."""
    meta = estimator(small_sample_correction=correction)
    meta.inputs_ = {"contrast_names": np.array([0, 1, 2])}

    assert meta._pymare_estimator(np.arange(3)).small_sample_correction == correction


def test_small_sample_correction_changes_the_model_based_standard_errors(fitted):
    """On the model-based path the correction must move the p-values, and only those."""
    corrected = fitted(
        ibma.DerSimonianLaird, groupby=False, small_sample_correction="knapp-hartung"
    )
    uncorrected = fitted(ibma.DerSimonianLaird, groupby=False, small_sample_correction="wald")

    valid = np.isfinite(corrected.maps["p"]) & np.isfinite(uncorrected.maps["p"])
    assert valid.any()
    assert not np.allclose(corrected.maps["p"][valid], uncorrected.maps["p"][valid])
    # Only the reference distribution changes; the point estimate is untouched.
    assert np.allclose(corrected.maps["est"][valid], uncorrected.maps["est"][valid])


def test_small_sample_correction_is_ignored_once_images_are_grouped(fitted):
    """Group labels bring CR2 and Satterthwaite degrees of freedom, which replace it.

    So the parameter is inert on the grouped path -- which is why PyMARE spells the
    uncorrected option 'wald' rather than 'none'.
    """
    corrected = fitted(ibma.DerSimonianLaird, small_sample_correction="knapp-hartung")
    uncorrected = fitted(ibma.DerSimonianLaird, small_sample_correction="wald")

    assert np.unique(corrected.estimator.inputs_["contrast_names"]).size < len(
        corrected.estimator.inputs_["id"]
    ), "fixture must actually group something"
    valid = np.isfinite(corrected.maps["p"]) & np.isfinite(uncorrected.maps["p"])
    assert valid.any()
    assert np.allclose(corrected.maps["p"][valid], uncorrected.maps["p"][valid])


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


@pytest.mark.parametrize("estimator", REPRESENTATIVE_ESTIMATORS)
@pytest.mark.parametrize("parameter", ["resample", "memory_limit"])
def test_parameters_removed_in_0_2_0_say_so(estimator, parameter):
    """A parameter dropped in 0.2.0 must explain itself, not point at 'resample__'.

    Both were silently swallowed by ``**kwargs`` until unknown arguments started raising, so
    calls still passing them are out there -- two of NiMARE's own examples were.
    """
    with pytest.raises(TypeError, match=f"{parameter} was removed in 0.2.0"):
        estimator(**{parameter: True})


@pytest.mark.parametrize("estimator", REPRESENTATIVE_ESTIMATORS)
def test_unknown_kwargs_point_at_the_resampling_prefix(estimator):
    """Anything else unrecognized should still name itself and the prefix that does work."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        estimator(not_a_parameter=1)

    # And a genuine resampling argument is accepted.
    assert estimator(resample__interpolation="nearest")._resample_kwargs["interpolation"] == (
        "nearest"
    )


def test_stouffers_rejects_the_removed_normalization_parameter():
    """The removed parameter should explain itself rather than raise a bare TypeError."""
    with pytest.raises(TypeError, match="normalize_contrast_weights was removed"):
        ibma.Stouffers(normalize_contrast_weights=True)


# groupby


@pytest.mark.parametrize("estimator", REPRESENTATIVE_ESTIMATORS)
def test_no_dependence_when_unrepeated_or_opted_out(estimator, testdata_ibma, fitted):
    """`labels` is None both when no study repeats and when the caller opts out."""
    unrepeated = estimator()
    unrepeated.fit(testdata_ibma)
    assert unrepeated._dependence().labels is None
    assert unrepeated.inputs_["corr_matrix"] is None

    opted_out = fitted(estimator, groupby=False).estimator
    assert opted_out._dependence().labels is None


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_grouping_recorded_and_correlation_gated(estimator, fitted):
    """Every estimator records the grouping; only the combination tests estimate a correlation.

    Parametrized over all nine because `_requires_corr_matrix` genuinely varies by class.
    """
    meta = fitted(estimator).estimator

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


def test_groupby_accepts_a_metadata_field(fitted):
    """A metadata column can define the grouping instead of study_id."""
    meta = fitted(ibma.Stouffers, groupby="sample_sizes").estimator

    labels = meta.inputs_["dependence_groups"]
    assert len(labels) == len(meta.inputs_["id"])
    # The integer codes must follow that field, not study_id.
    codes = meta.inputs_["contrast_names"]
    for left in range(len(codes)):
        for right in range(len(codes)):
            assert (codes[left] == codes[right]) == (labels[left] == labels[right])


def test_groupby_accepts_an_explicit_label_array(dependent_dataset, fitted):
    """Users can supply one label per image directly."""
    n_images = len(fitted(ibma.Stouffers).estimator.inputs_["id"])
    labels = np.zeros(n_images, dtype=int)

    meta = ibma.Stouffers(groupby=labels)
    meta.fit(dependent_dataset)

    assert np.unique(meta.inputs_["contrast_names"]).size == 1


def test_groupby_rejects_a_mismatched_label_array(dependent_dataset):
    """A label array that does not cover every image is a user error, not a silent one."""
    with pytest.raises(ValueError, match="one label per image"):
        ibma.Stouffers(groupby=np.array([0, 1])).fit(dependent_dataset)


def test_groupby_can_split_a_study_back_into_independent_samples(dependent_dataset, fitted):
    """Splitting a study apart must reproduce the ungrouped result.

    The NeuroVault case where one paper uploads patients and controls separately.
    """
    reference = fitted(ibma.DerSimonianLaird, groupby=False)
    n_images = len(reference.estimator.inputs_["id"])

    split = ibma.DerSimonianLaird(groupby=np.arange(n_images)).fit(dependent_dataset)

    for name in ("z", "p", "est", "se", "dof"):
        valid = np.isfinite(reference.maps[name]) & np.isfinite(split.maps[name])
        assert valid.any()
        assert np.allclose(reference.maps[name][valid], split.maps[name][valid])


# Inference


@pytest.mark.parametrize("estimator", PYMARE_ESTIMATORS)
def test_dependence_changes_inference(estimator, fitted):
    """Correcting for dependence should move both the estimates and the p-values.

    Widening standard errors alone would leave image multiplicity changing the point
    estimate; the default weighting gives every group the same total weight first.
    """
    corrected = fitted(estimator)
    naive = fitted(estimator, groupby=False)

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
def test_combination_dof_counts_groups_not_images(estimator, fitted):
    """A study contributing several images must not buy degrees of freedom."""
    results = fitted(estimator)
    meta = results.estimator

    n_images = len(meta.inputs_["id"])
    n_groups = np.unique(meta.inputs_["contrast_names"]).size
    assert n_groups < n_images

    dof = results.maps["dof"]
    assert set(dof[dof > 0].tolist()) == {n_groups - 1}


@pytest.mark.parametrize("estimator", REGRESSION_ESTIMATORS)
def test_regression_dof_is_pymare_satterthwaite(estimator, fitted, monkeypatch):
    """The dof map must be PyMARE's `fe_dof`, not a recomputation or a count of images."""
    # Aggressive masking, so that one model covers the whole map and its `fe_dof` can be
    # compared against every in-mask voxel. The liberal path fits one model per bag, each
    # with its own dof; that is covered by test_liberal_mask_dof_is_per_bag.
    results = fitted(estimator, aggressive_mask=True)
    meta = results.estimator

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

    monkeypatch.setattr(pymare.estimators.estimators, "MetaRegressionResults", _Spy)
    maps = meta._fit_model(
        *[meta.inputs_[name][:, voxel_mask] for name in meta._image_inputs],
        study_mask=np.arange(n_images),
    )

    assert captured["fe_dof"] is not None
    assert np.allclose(maps[-1], np.ravel(captured["fe_dof"]))


@pytest.mark.parametrize("aggressive_mask", [True, False])
@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_fit_produces_well_formed_maps(estimator, aggressive_mask, fitted):
    """Every estimator returns usable maps under both masking strategies.

    Well-formedness only. Robust standard errors are not guaranteed to be larger than
    model-based ones -- RVE estimates between-group variability rather than trusting the
    reported sampling variances, so an overstated varcope can legitimately shrink them --
    which leaves no inequality to assert here.
    """
    results = fitted(estimator, aggressive_mask=aggressive_mask)
    meta = results.estimator

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
def test_every_weight_scheme_runs(weight_scheme, fitted):
    """All three PyMARE schemes must be reachable from NiMARE."""
    results = fitted(ibma.DerSimonianLaird, weight_scheme=weight_scheme)

    assert np.isfinite(results.maps["z"]).any()


def test_rho_barely_moves_the_result(fitted):
    """Sweeping rho barely moves the result, because it enters only through tau^2.

    This is what makes 0.8 a safe default and a sensitivity sweep cheap.
    """
    low = fitted(ibma.DerSimonianLaird, rho=0.0).maps["z"]
    high = fitted(ibma.DerSimonianLaird, rho=1.0).maps["z"]

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


def test_correlation_matrix_is_not_measuring_shared_signal(fitted):
    """Studies that are independent by construction must look independent.

    Correlating the raw maps conflates dependence with agreement, which would spread the
    variance correction across the whole analysis instead of the repeated study.
    """
    meta = fitted(ibma.Stouffers).estimator

    corr = meta.inputs_["corr_matrix"]
    groups = meta.inputs_["contrast_names"]
    same_study = groups[:, None] == groups[None, :]
    off_diagonal = ~np.eye(corr.shape[0], dtype=bool)

    between = np.nanmean(corr[~same_study])
    within = np.nanmean(corr[same_study & off_diagonal])

    assert abs(between) < 0.1  # different studies are independent
    assert within > between  # the repeated study is where dependence lives


# Liberal-mask bags


def _bagged_estimator(cls, arrays, codes, **kwargs):
    """Return ``cls`` with hand-built liberal-mask bags, bypassing image loading.

    ``arrays`` maps each ``_required_inputs`` image name to a (K x V) array whose NaNs define
    the coverage pattern the bags are cut from.
    """
    meta = cls(**kwargs)
    meta.aggressive_mask = False
    meta.masker = NiftiMasker()
    n_images, _ = next(iter(arrays.values())).shape

    keys = ["values", "voxel_mask", "study_mask"]
    meta.inputs_ = {
        "id": [f"img{i}" for i in range(n_images)],
        "contrast_names": np.asarray(codes),
        "corr_matrix": None,
        "data_bags": {
            name: [dict(zip(keys, bag)) for bag in zip(*_apply_liberal_mask(values))]
            for name, values in arrays.items()
        },
        **arrays,
    }
    return meta


def _single_group_bag_arrays(names, seed=0):
    """Return arrays whose last two voxels are covered only by images 0 and 1.

    Images 0 and 1 share group 0, so that bag holds two images but only one group.
    """
    rng = np.random.RandomState(seed)
    arrays = {}
    for i_name, name in enumerate(names):
        values = np.abs(rng.normal(size=(4, 8))) + 0.5
        if name != "varcope_maps":
            values = values + 0.5 * (-1) ** i_name
        values[2, 6:] = np.nan
        values[3, 6:] = np.nan
        arrays[name] = values
    return arrays


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_single_group_bags_are_skipped_not_fatal(estimator, caplog):
    """A bag whose images all share one group has no independent replication.

    PyMARE and the sign-flip null both reject a single block outright, so before this was
    handled such a bag aborted the whole fit -- and a study contributing several maps with
    coverage the others lack is ordinary.
    """
    image_names = [
        name for name, (type_, _) in estimator._required_inputs.items() if type_ == "image"
    ]
    arrays = _single_group_bag_arrays(image_names)
    meta = _bagged_estimator(estimator, arrays, codes=[0, 0, 1, 1])
    if "sample_sizes" in estimator._required_inputs:
        meta.inputs_["sample_sizes"] = np.array([[25.0], [25.0], [30.0], [30.0]])
    meta.generate_description = False

    with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
        maps, _, _ = meta._fit(_FakeDataset(meta.masker))

    assert "single group" in caplog.text
    # The well-covered voxels still get an answer; the single-group bag comes back NaN.
    assert np.isfinite(maps["z" if "z" in maps else "t"][:6]).all()
    assert np.isnan(maps["z" if "z" in maps else "t"][6:]).all()


def test_permuted_ols_fwe_handles_a_bag_with_one_image_per_group():
    """The shared sign-flip matrix is indexed by block label, so bags must agree on those.

    A bag holding one image from each of two studies has no within-bag dependence, but its
    blocks still have to be the dataset-wide group codes -- falling back to image indices
    would index past the end of the shared matrix.
    """
    rng = np.random.RandomState(0)
    betas = rng.normal(size=(4, 8)) + 0.5
    # Voxels 6-7 are covered only by images 1 and 3, which belong to different studies.
    betas[0, 6:] = np.nan
    betas[2, 6:] = np.nan

    meta = _bagged_estimator(ibma.PermutedOLS, {"beta_maps": betas}, codes=[0, 0, 1, 1])
    bags = meta.inputs_["data_bags"]["beta_maps"]
    assert [bag["study_mask"].size for bag in bags] == [4, 2], "fixture must produce a mixed bag"

    maps, _, _ = meta.correct_fwe_montecarlo(None, n_iters=20)

    p_map = maps["p_level-voxel"]
    assert np.isfinite(p_map).all()
    assert np.all((p_map > 0) & (p_map <= 1))


def test_permuted_ols_reports_an_uncorrected_p_map(fitted):
    """Without a ``p`` map, FDR and Bonferroni correction have nothing to read."""
    from nimare.correct import FDRCorrector

    results = fitted(ibma.PermutedOLS)

    p_map = results.maps["p"]
    finite = np.isfinite(p_map)
    assert finite.any()
    assert np.all((p_map[finite] > 0) & (p_map[finite] <= 1))
    # Voxels the model did not cover stay NaN, as in every other map.
    assert np.isnan(p_map[~np.isfinite(results.maps["t"])]).all()

    corrected = FDRCorrector(method="indep", alpha=0.05).transform(results)
    q_map = corrected.maps["p_corr-FDR_method-indep"]
    covered = np.isfinite(q_map)
    assert covered.any()
    # Correcting can only raise a p-value.
    assert np.all(q_map[covered] >= p_map[covered] - 1e-12)


@pytest.mark.parametrize("two_sided", [True, False])
def test_permuted_ols_p_map_agrees_with_its_z_map(two_sided):
    """Match the z map's own p-value, on the tail the test was run on.

    Deriving p from t independently would let the two maps disagree, since ``t_to_z`` floors
    the tail probability it converts through.
    """
    betas = np.array([[1.0, -1.0], [1.2, -1.2], [0.8, -0.8], [1.1, -1.1]])
    meta = _bagged_estimator(
        ibma.PermutedOLS, {"beta_maps": betas}, codes=np.arange(4), two_sided=two_sided
    )
    meta.generate_description = False
    maps, _, _ = meta._fit(_FakeDataset(meta.masker))

    p = maps["p"]
    assert np.allclose(p, z_to_p(maps["z"], tail="two" if two_sided else "one"))
    # Voxel 1 is the mirror of voxel 0. Two-sided cannot tell them apart; one-sided must.
    assert np.isclose(p[0], p[1]) == two_sided


def test_permuted_ols_fwe_is_not_saturated_by_a_small_bag():
    """A two-image bag must not set the max-statistic threshold for the whole brain.

    A t statistic carries the degrees of freedom of the bag it came from, so a two-image bag
    reaches |t| in the tens or hundreds where a well-covered bag reaches 8 -- and yet is the
    weaker result once both are referred to their own null. Maximizing over t therefore let a
    handful of thinly covered voxels dominate the null and nothing anywhere survived. z is
    pivotal, so the maximum compares like with like.
    """
    rng = np.random.RandomState(0)
    betas = rng.normal(0.0, 1.0, size=(8, 40)) + 3.0

    # Voxels 32+ are covered only by images 0 and 1, which nearly agree. One degree of
    # freedom and a tiny spread give an enormous t but only a middling z.
    betas[2:, 32:] = np.nan
    betas[0, 32:] = 1.0
    betas[1, 32:] = 1.05

    meta = _bagged_estimator(ibma.PermutedOLS, {"beta_maps": betas}, codes=np.arange(8))
    meta.generate_description = False
    fit_maps, _, _ = meta._fit(_FakeDataset(meta.masker))

    well_covered, thin = slice(0, 32), slice(32, 40)
    assert np.nanmax(np.abs(fit_maps["t"][thin])) > np.nanmax(np.abs(fit_maps["t"][well_covered]))
    assert np.nanmax(np.abs(fit_maps["z"][thin])) < np.nanmax(np.abs(fit_maps["z"][well_covered]))

    maps, _, _ = meta.correct_fwe_montecarlo(None, n_iters=200)

    p_map = maps["p_level-voxel"]
    assert np.isfinite(p_map).all()
    assert p_map[well_covered].min() < 0.05, "the thin bag swallowed the whole-brain threshold"


def test_blocks_keep_one_label_space_across_bags():
    """Restricting to a bag must not switch which label space `blocks` is drawn from."""
    full = DependenceModel(np.array([0, 0, 1, 1]))
    assert np.array_equal(full.blocks, [0, 0, 1, 1])

    # One image per group: no within-bag dependence, but the labels are still group codes.
    bag = full.for_images(np.array([1, 3]))
    assert bag.labels is None
    assert np.array_equal(bag.blocks, [0, 1])
    assert set(bag.group_order) <= set(np.unique(full.blocks))

    # With no dependence anywhere, blocks are dataset-wide image indices in both.
    independent = DependenceModel(np.array([0, 1, 2, 3]))
    assert np.array_equal(independent.for_images(np.array([1, 3])).blocks, [1, 3])


def test_single_group_model_reports_no_support():
    """One block cannot support the inference, whatever the image count."""
    assert not DependenceModel(np.array([0, 0, 0])).supports_inference
    assert DependenceModel(np.array([0, 0, 1])).supports_inference


@pytest.mark.parametrize(
    "estimator", [ibma.WeightedLeastSquares, ibma.DerSimonianLaird, ibma.VarianceBasedLikelihood]
)
def test_multi_input_bags_stay_aligned(estimator, testdata_ibma):
    """Every image input must be cut into the same bags, so the parallel lists zip.

    Varcope values that are non-positive or too small to square are blanked during
    preprocessing, which gives the varcopes a coverage pattern their betas do not have. Cut
    per input, the two bag lists then describe different voxels and ``zip`` pairs them up
    anyway -- or drops the tail outright. One shared grouping is also the cheaper half of
    the work, so the masks below are the same objects, not merely equal ones.
    """
    meta = estimator(aggressive_mask=False)
    meta.fit(testdata_ibma)

    beta_bags = meta.inputs_["data_bags"]["beta_maps"]
    varcope_bags = meta.inputs_["data_bags"]["varcope_maps"]

    assert len(beta_bags) == len(varcope_bags)
    for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
        assert beta_bag["voxel_mask"] is varcope_bag["voxel_mask"]
        assert beta_bag["study_mask"] is varcope_bag["study_mask"]
        assert beta_bag["values"].shape == varcope_bag["values"].shape


# Combination tests


@pytest.mark.parametrize(
    "estimator,expected",
    [
        # Fisher weights a study by its sample size; Stouffer by the square root of it.
        (ibma.Fishers, [25.0, 25.0, 25.0, 100.0]),
        (ibma.Stouffers, [5.0, 5.0, 5.0, 10.0]),
    ],
)
def test_combination_weights_are_constant_within_a_group(estimator, expected, monkeypatch):
    """One weight per group, repeated per image and not divided by map count.

    PyMARE rejects a group whose images carry different weights, and a study routinely
    reports a different sample size per contrast -- 20/25/30 below. The group's mean, 25, is
    what every one of its images must be weighted by.
    """
    meta = estimator(use_sample_size=True)
    meta.inputs_ = {
        "contrast_names": np.array([0, 0, 0, 1]),
        "corr_matrix": np.eye(4),
        "sample_sizes": np.array([[20.0], [25.0], [30.0], [100.0]]),
        "id": ["a", "b", "c", "d"],
    }

    captured = _capture_kwargs(monkeypatch, ibma.pymare, "Dataset")
    meta._fit_model(np.ones((4, 5)), study_mask=np.arange(4), corr=np.eye(4))

    assert np.allclose(np.asarray(captured["n"]).ravel(), expected)


@pytest.mark.parametrize("estimator", COMBINATION_ESTIMATORS)
def test_combination_weights_survive_heterogeneous_group_sample_sizes(estimator, fitted):
    """A study reporting a different sample size per contrast must still fit.

    Passing a study's per-image sample sizes straight through makes PyMARE reject the whole
    fit, because it requires one weight per group.
    """
    results = fitted(estimator, use_sample_size=True)

    assert np.isfinite(results.maps["z"]).any()


def test_stouffers_aggregates_groups_whenever_they_exist(monkeypatch):
    """Aggregate whenever groups exist, so a prolific study cannot outvote its peers."""
    meta = ibma.Stouffers()
    meta.inputs_ = {
        "contrast_names": np.array([0, 0, 0, 1]),
        "corr_matrix": np.eye(4),
        "id": ["a", "b", "c", "d"],
    }

    captured = _capture_kwargs(monkeypatch, ibma.pymare.estimators, "StoufferCombinationTest")
    meta._fit_model(np.ones((4, 5)), study_mask=np.arange(4), corr=np.eye(4))

    assert captured["group_level"] is True


def test_fishers_preserves_two_sided_positional_argument():
    """Adding sample-size weighting must not reinterpret released positional calls."""
    estimator = ibma.Fishers(False)

    assert estimator.two_sided is False
    assert estimator.use_sample_size is False


@pytest.mark.parametrize("estimator", COMBINATION_ESTIMATORS)
def test_null_correlation_ignores_voxels_missing_from_any_image(estimator):
    """A NaN anywhere must not wipe out the correlation it was called to measure.

    np.corrcoef propagates one NaN across a whole row, and estimate_null_correlation reads a
    non-finite entry as zero correlation -- so correlating the unmasked maps silently
    discarded the dependence for any image with a single missing voxel.
    """
    rng = np.random.RandomState(0)
    shared = rng.normal(size=(1, 400))
    # Images 0 and 1 are the same study, so they should come out strongly correlated.
    maps = np.vstack(
        [shared + 0.05 * rng.normal(size=(1, 400)) for _ in range(2)]
        + [rng.normal(size=(1, 400)) for _ in range(2)]
    )
    image_name = next(iter(estimator._required_inputs))

    def _corr(values):
        meta = estimator()
        meta.aggressive_mask = False
        meta.inputs_ = {
            "id": ["a", "b", "c", "d"],
            image_name: values,
            "corr_matrix": None,
        }
        meta._resolve_group_labels = lambda _: ["s0", "s0", "s1", "s2"]
        meta._preprocess_dependence(None)
        return meta.inputs_["corr_matrix"]

    clean = _corr(maps)
    with_gap = maps.copy()
    with_gap[0, 3] = np.nan
    dirty = _corr(with_gap)

    assert clean[0, 1] > 0.8
    assert np.allclose(dirty, clean, atol=0.05)


# Cancelling dependence groups


def _combination_estimator(cls, z_maps, codes, corr):
    """Return a combination test with hand-built bags and a supplied correlation matrix.

    ``corr`` stands in for the null correlation ``_preprocess_dependence`` would estimate, so
    a block can be pinned at exactly the value the verdict turns on.
    """
    meta = _bagged_estimator(cls, {"z_maps": np.asarray(z_maps, dtype=float)}, codes=codes)
    meta.inputs_["corr_matrix"] = np.asarray(corr, dtype=float)
    meta._group_labels = {int(code): f"study{int(code)}" for code in np.unique(codes)}
    meta.generate_description = False
    return meta


@pytest.mark.parametrize("estimator", COMBINATION_ESTIMATORS)
def test_cancelling_group_is_dropped_per_bag_not_fatally(estimator, caplog, monkeypatch):
    """A group whose images cancel is excluded from the bags it cancels in, and no others.

    PyMARE refuses to aggregate a group whose block sums to zero, which used to end the whole
    meta-analysis over one study's uploads. Both combination tests are run because the drop
    is not confined to the one that raised.
    """
    rng = np.random.RandomState(0)
    z_maps = rng.normal(size=(4, 8))
    z_maps[1] = -z_maps[0]
    # Only images 0, 2 and 3 cover voxels 6-7, so image 0's group holds two images in one bag
    # and one in the other.
    z_maps[1, 6:] = np.nan
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = -1.0
    meta = _combination_estimator(estimator, z_maps, [0, 0, 1, 2], corr)

    masks = []
    real = meta._fit_model

    def _record(*arrays, study_mask, **kwargs):
        masks.append(list(study_mask))
        return real(*arrays, study_mask=study_mask, **kwargs)

    monkeypatch.setattr(meta, "_fit_model", _record)

    with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
        maps, _, _ = meta._fit(_FakeDataset(meta.masker))

    assert masks == [[2, 3], [0, 2, 3]]
    assert np.isfinite(maps["z"]).all()
    # Degrees of freedom count the groups actually fitted, which differ between the bags.
    assert np.allclose(maps["dof"][:6], 1.0)
    assert np.allclose(maps["dof"][6:], 2.0)

    assert "study0" in caplog.text
    assert "excluded from 1 of 2 model(s)" in caplog.text
    # The group went, so it must not also be reported as a mirror pair that was kept.
    assert "correlate exactly -1" not in caplog.text


def test_three_way_cancellation_drops_the_whole_group(caplog):
    """A block can sum to nothing with no pair anywhere near -1, and all three must go.

    Complementary networks from one cohort do this. With three members there is no principled
    choice of which one to remove, so the group goes as a unit -- which here leaves a single
    group, the case the existing skip already handles.
    """
    rng = np.random.RandomState(1)
    corr = np.eye(4)
    # Sums to zero: 3 + 2 * (-0.5 - 0.5 - 0.5).
    corr[np.ix_([0, 1, 2], [0, 1, 2])] = [
        [1.0, -0.5, -0.5],
        [-0.5, 1.0, -0.5],
        [-0.5, -0.5, 1.0],
    ]
    meta = _combination_estimator(ibma.Stouffers, rng.normal(size=(4, 8)), [0, 0, 0, 1], corr)

    with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
        maps, _, _ = meta._fit(_FakeDataset(meta.masker))

    # Nothing was fitted, so every one of the three went: dropping any one or two of them
    # would have left two groups to fit.
    assert np.isnan(maps["z"]).all()
    assert "img0, img1, img2" in caplog.text
    assert "single group" in caplog.text
    assert "correlate exactly -1" not in caplog.text


def test_perfect_mirror_that_does_not_cancel_warns_but_keeps_the_data(caplog):
    """Two images that are exact negatives can still fit, and that is worth saying loudly.

    A third member keeps the group's block sum off zero. The estimated null correlation
    cannot see the duplicate, being pulled away from -1 by the signal the two maps share.
    """
    rng = np.random.RandomState(2)
    z_maps = rng.normal(size=(4, 8))
    z_maps[1] = -z_maps[0]
    corr = np.eye(4)
    corr[np.ix_([0, 1, 2], [0, 1, 2])] = [[1.0, -0.5, 0.4], [-0.5, 1.0, 0.6], [0.4, 0.6, 1.0]]
    meta = _combination_estimator(ibma.Stouffers, z_maps, [0, 0, 0, 1], corr)

    with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
        maps, _, _ = meta._fit(_FakeDataset(meta.masker))

    # Dropping the group would have left one image, and so no fit at all.
    assert np.isfinite(maps["z"]).all()
    assert "correlate exactly -1" in caplog.text
    assert "img0" in caplog.text and "img1" in caplog.text


def test_near_cancellation_is_left_alone_silently(caplog):
    """Strongly anti-correlated images that neither cancel nor mirror get no warning.

    Where to draw a line short of exact cancellation is a policy only the caller can set, and
    warning about every anti-correlated pair would bury the two cases that are worth
    reporting.
    """
    rng = np.random.RandomState(3)
    z_maps = rng.normal(size=(4, 8))
    z_maps[1] = -z_maps[0] + 0.5 * rng.normal(size=8)
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = -0.99
    meta = _combination_estimator(ibma.Stouffers, z_maps, [0, 0, 1, 2], corr)

    with caplog.at_level(logging.WARNING, logger="nimare.meta.ibma"):
        maps, _, _ = meta._fit(_FakeDataset(meta.masker))

    assert -1.0 < np.corrcoef(z_maps[:2])[0, 1] < -0.8, "fixture must be near, not exact"
    # Three groups, so nothing was dropped: a drop would leave two, and dof 1.
    assert np.allclose(maps["dof"], 2.0)
    assert np.isfinite(maps["z"]).all()
    assert caplog.text == ""


def test_cancellation_floor_still_matches_pymare():
    """Keep the floor pinned to PyMARE's, which is a function local and so drifts silently.

    Excluding a group here only pre-empts PyMARE while everything PyMARE refuses is already
    excluded. The value cannot be imported, so the two are pinned by behaviour instead.
    """
    rng = np.random.RandomState(5)
    z_maps, weights = rng.normal(size=(4, 6)), np.ones((4, 1))
    codes = np.array([0, 0, 1, 2])

    def pymare_refuses(variance):
        corr = np.eye(4)
        # (2 + 2 * rho) / 4 is the block variance of a two-image group.
        corr[0, 1] = corr[1, 0] = 2.0 * variance - 1.0
        try:
            pymare.estimators.StoufferCombinationTest(group_level=True).fit(
                z=z_maps, w=weights, g=codes, corr=corr
            )
        except ValueError as exc:
            return "cancel" in str(exc)
        return False

    assert pymare_refuses(ibma._CANCELLATION_FLOOR)
    # The direction that matters: whatever NiMARE keeps, PyMARE must still accept.
    assert not pymare_refuses(ibma._CANCELLATION_FLOOR * 1e3)


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
def test_permuted_ols_blocks_follow_groupby(groupby, grouped, fitted):
    """Grouping collapses a study's images into one block; opting out makes it the identity."""
    meta = fitted(ibma.PermutedOLS, groupby=groupby).estimator

    n_images = len(meta.inputs_["id"])
    blocks, weights = meta._blocks_and_weights(np.arange(n_images))

    assert weights is None
    if grouped:
        assert np.unique(blocks).size < n_images
    else:
        assert np.array_equal(blocks, np.arange(n_images))


@pytest.mark.parametrize("aggressive_mask", [True, False], ids=["aggressive", "liberal"])
def test_permuted_ols_dof_counts_blocks_not_images(aggressive_mask, fitted):
    """Each model's dof counts the blocks it was fitted over, not the images it saw.

    Under the aggressive mask one model covers the whole map, so the dof is a single number.
    Under the liberal mask each bag is its own model, so a bag covered by fewer images
    reports fewer degrees of freedom -- which is the point of fitting per bag rather than
    dropping the voxels.
    """
    grouped = fitted(ibma.PermutedOLS, aggressive_mask=aggressive_mask)
    ungrouped = fitted(ibma.PermutedOLS, aggressive_mask=aggressive_mask, groupby=False)

    n_images = len(ungrouped.estimator.inputs_["id"])
    # Outside the mask the map is NaN, as every other float map already is.
    ungrouped_dof = ungrouped.maps["dof"]
    grouped_dof = grouped.maps["dof"]

    # Ungrouped, every image is its own block, so a model's dof is its image count - 1.
    if aggressive_mask:
        expected = {float(n_images - 1)}
    else:
        bags = ungrouped.estimator.inputs_["data_bags"]["beta_maps"]
        assert len({bag["study_mask"].size for bag in bags}) > 1, "fixture must vary coverage"
        expected = {float(bag["study_mask"].size - 1) for bag in bags}
    assert set(ungrouped_dof[np.isfinite(ungrouped_dof)].tolist()) <= expected
    assert np.nanmax(ungrouped_dof) == float(n_images - 1)

    # Grouping collapses images to groups, so it can only cost degrees of freedom.
    valid = np.isfinite(grouped_dof) & np.isfinite(ungrouped_dof)
    assert valid.any()
    assert np.all(grouped_dof[valid] <= ungrouped_dof[valid])
    if aggressive_mask:
        n_groups = np.unique(grouped.estimator.inputs_["contrast_names"]).size
        assert set(grouped_dof[np.isfinite(grouped_dof)].tolist()) == {float(n_groups - 1)}


def test_permuted_ols_sample_size_weights_shrink_the_dof(fitted):
    """Unequal weights make Satterthwaite dof fall below the group count."""
    equal = fitted(ibma.PermutedOLS)
    weighted = fitted(ibma.PermutedOLS, use_sample_size=True)

    valid = np.isfinite(weighted.maps["dof"]) & np.isfinite(equal.maps["dof"])
    assert valid.any()
    assert np.all(weighted.maps["dof"][valid] <= equal.maps["dof"][valid])
    assert np.isfinite(weighted.maps["z"]).any()


def test_permuted_ols_weights_each_group_once(fitted):
    """A study's sample size must not be counted once per map it uploaded."""
    meta = fitted(ibma.PermutedOLS, use_sample_size=True).estimator

    n_images = len(meta.inputs_["id"])
    _, weights = meta._blocks_and_weights(np.arange(n_images))
    group_order = meta._dependence().group_order

    assert weights.shape == (group_order.size,)
    assert group_order.size < n_images


@pytest.mark.parametrize(
    "aggressive_mask,use_sample_size",
    [(False, False), (True, True)],
    ids=["liberal-unweighted", "aggressive-weighted"],
)
def test_permuted_ols_fwe_shares_one_null_across_bags(
    aggressive_mask, use_sample_size, dependent_dataset
):
    """The max-statistic null must describe the whole brain, not one bag of it."""
    from nimare.correct import FWECorrector

    meta = ibma.PermutedOLS(aggressive_mask=aggressive_mask, use_sample_size=use_sample_size)
    corrector = FWECorrector(method="montecarlo", n_iters=20)
    corrected = corrector.transform(meta.fit(dependent_dataset))

    # FWECorrector operates on a copy of the estimator and hands it back on the result.
    null = corrected.estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
    assert null.shape == (20,)
    assert np.all(np.isfinite(null))

    p_map = corrected.maps["p_level-voxel_corr-FWE_method-montecarlo"]
    finite_p = p_map[np.isfinite(p_map)]
    assert finite_p.size
    assert np.all((finite_p > 0) & (finite_p <= 1))

    logp = corrected.maps["logp_level-voxel_corr-FWE_method-montecarlo"]
    assert np.all(logp[np.isfinite(logp)] >= 0)


# The permutation module itself


def test_permutation_collapses_blocks_to_their_means():
    """The statistic is over block means, so within-block spread is irrelevant."""
    beta_maps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
    groups = np.array([0, 0, 1])

    result = _permuted_ols(beta_maps, exchangeability_blocks=groups)

    contributions = np.array([beta_maps[:2].mean(axis=0), beta_maps[2]])
    expected = contributions.mean(axis=0) / (contributions.std(axis=0, ddof=1) / np.sqrt(2))
    assert np.allclose(result["t"].squeeze(), expected)
    assert result["dof"] == 1

    # Spreading the first block's maps apart around the same mean must change nothing.
    dispersed = beta_maps.copy()
    dispersed[0] -= 10.0
    dispersed[1] += 10.0
    assert np.allclose(
        _permuted_ols(dispersed, exchangeability_blocks=groups)["t"],
        result["t"],
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


@pytest.mark.parametrize("use_weights", [False, True])
def test_permutation_batches_stay_within_their_memory_budget(use_weights, monkeypatch):
    """The batch size must reflect what a batch actually allocates.

    The CR2 sandwich holds several times more (batch x n_voxels) arrays than the unweighted
    path, so one count for both left it far over the budget it was handed. Measured rather
    than asserted from the source, because most of the allocation happens inside PyMARE.
    """
    import tracemalloc

    from nimare.meta import _permutation

    beta_maps = np.random.RandomState(0).normal(size=(6, 4000))
    groups = np.repeat(np.arange(3), 2)
    weights = np.array([1.0, 4.0, 9.0]) if use_weights else None

    budget = 1 << 20  # 1 MiB: small enough that several batches are needed
    monkeypatch.setattr(_permutation, "_MAX_BATCH_BYTES", budget)

    peaks = []
    real_maxima = _permutation._permutation_maxima

    def _measured(*args, **kwargs):
        tracemalloc.start()
        try:
            return real_maxima(*args, **kwargs)
        finally:
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

    monkeypatch.setattr(_permutation, "_permutation_maxima", _measured)

    _permutation._permuted_ols(
        beta_maps,
        exchangeability_blocks=groups,
        group_weights=weights,
        n_perm=40,
        random_state=0,
    )

    assert peaks
    # Slack for the null array and bookkeeping, but nowhere near the threefold overshoot an
    # undercounted budget produced.
    assert max(peaks) < 2 * budget


def test_null_correlation_matches_pymare_when_nothing_is_missing():
    """Complete data must keep giving exactly what PyMARE's own estimator gave."""
    rng = np.random.RandomState(0)
    maps = rng.normal(size=(4, 4000)) + 2.0 * rng.normal(size=4000)
    groups = np.array([0, 0, 1, 1])

    assert np.array_equal(
        _null_correlation(maps, groups),
        pymare.stats.estimate_null_correlation(maps, groups=groups),
    )


def test_null_correlation_ignores_the_zeros_that_stand_for_missing_coverage():
    """A shared coverage hole is a shared constant, which centering makes look correlated.

    Images 0 and 1 are independent but share a field of view and a strong shared signal.
    """
    rng = np.random.RandomState(0)
    maps = rng.normal(size=(4, 4000)) + 2.0 * rng.normal(size=4000)
    groups = np.array([0, 0, 1, 1])
    holed = maps.copy()
    holed[:2, :2000] = 0.0

    complete = _null_correlation(maps, groups)[0, 1]
    every_voxel = pymare.stats.estimate_null_correlation(holed, groups=groups)[0, 1]
    pairwise = _null_correlation(holed, groups)[0, 1]

    assert abs(complete) < 0.05
    assert every_voxel > 0.6
    assert abs(pairwise) < 0.25


def test_null_correlation_gives_up_when_no_two_images_overlap():
    """Two images that share no valid voxel leave nothing to estimate a correlation from."""
    maps = np.ones((2, 400))
    maps[0, 200:] = 0.0
    maps[1, :200] = 0.0

    assert _null_correlation(maps, np.array([0, 0])) is None
