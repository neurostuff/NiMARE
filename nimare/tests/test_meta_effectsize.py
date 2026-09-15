"""Tests for nimare.meta.cbma.effectsize (coordinate-based effect-size meta-analysis).

CBES reads magnitudes from effect-size images and reads coordinate tables only for where
studies were **silent**. The tests are organised around that split: what the image channel
does, what the silence channel does, what happens when one of them is missing, and what the
permutation null is and is not testing.
"""

import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.maskers import NiftiMasker

from nimare.correct import FDRCorrector, FWECorrector
from nimare.generate import create_effect_size_coordinate_studyset
from nimare.meta.cbma.effectsize import (
    _NULL_Z_STEP,
    CBES,
    DEFAULT_COVERAGE_RADIUS_MM,
    NULL_METHODS,
    _local_dersimonian_laird,
    _null_bin_edges,
    _stat_from_histogram,
    null_effect_variance,
    reported_minimum_z,
    reporting_cutoff_to_g,
)
from nimare.utils import mm2vox

TRUTH = (0, 0, 0)
TRUE_G = 0.5

# The field simulator's grid, chosen to coincide exactly with ``small_mask`` so the images it
# writes need no resampling and the truth is readable voxel for voxel.
SHAPE = (21, 21, 21)
ZOOMS = 4.0
EXTENT = 40.0
BLOB_FWHM = 10.0
AFFINE = np.array(
    [
        [ZOOMS, 0, 0, -EXTENT],
        [0, ZOOMS, 0, -EXTENT],
        [0, 0, ZOOMS, -EXTENT],
        [0, 0, 0, 1.0],
    ]
)


@pytest.fixture(scope="module")
def small_mask():
    """Build a small 4 mm box around the origin, so the fits in these tests stay quick."""
    return nib.Nifti1Image(np.ones(SHAPE, dtype=np.int32), AFFINE)


def make_studyset(image_dir, *, n_images=2, n_studies=20, effect=TRUE_G, seed=7, **kwargs):
    """Build a collection of field-simulated studies, the first ``n_images`` sharing maps.

    The field simulator rather than the point one, because the point simulator draws a value
    *at* the focus and never builds a map -- so it can produce neither the images this
    estimator requires nor the selection its silence channel corrects.
    """
    options = dict(
        sample_size=(20, 40),
        tau=0.1,
        simulate_field=True,
        noise_extent=EXTENT,
        field_zooms=ZOOMS,
        blob_fwhm=BLOB_FWHM,
    )
    options.update(kwargs)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    return create_effect_size_coordinate_studyset(
        [TRUTH],
        effect_sizes=effect,
        n_studies=n_studies,
        seed=seed,
        n_image_studies=n_images,
        image_dir=str(image_dir),
        **options,
    )


@pytest.fixture(scope="module")
def studyset(tmp_path_factory):
    """Build twenty studies with a true g of 0.5 at the origin; two share their maps."""
    return make_studyset(tmp_path_factory.mktemp("cbes_base"))


@pytest.fixture(scope="module")
def coordinates_only(tmp_path_factory):
    """Build the same collection with nobody sharing a map, which CBES has to refuse."""
    return make_studyset(tmp_path_factory.mktemp("cbes_nocoord"), n_images=0)


@pytest.fixture(scope="module")
def permutation_fit(studyset, small_mask):
    """One permutation fit at ``n_iters=20``, shared by the tests that all wanted the same one."""
    estimator = CBES(mask=small_mask, n_iters=20, threshold="reporting_threshold")
    return estimator, estimator.fit(studyset)


def truth_field(mask_img):
    """Return the effect the simulator actually built, in the masker's voxel order.

    Known exactly rather than estimated, which is the point of scoring against it: a reference
    built from the same maps that produced the peaks would condition the comparison on the very
    noise being measured.
    """
    grid = np.stack(np.indices(SHAPE), axis=-1) * ZOOMS - EXTENT
    sigma = BLOB_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    volume = TRUE_G * np.exp(-(grid**2).sum(axis=-1) / (2.0 * sigma**2))
    masker = NiftiMasker(mask_img).fit()
    return masker.transform(nib.Nifti1Image(volume.astype(np.float32), AFFINE)).ravel()


def value_at(result, name, xyz=TRUTH):
    """Read one map value at an xyz (mm) location."""
    img = result.get_map(name)
    ijk = mm2vox(np.array([xyz]), img.affine)[0]
    return float(img.get_fdata()[tuple(ijk)])


def arrays(result):
    """Return every map as a flat array, in the masker's voxel order."""
    return {name: result.get_map(name, return_type="array").ravel() for name in result.maps}


# --------------------------------------------------------------- numerical pieces


def test_null_effect_variance_shrinks_with_sample_size():
    """The variance a silence is judged against is the sampling variance at mu = 0."""
    variances = null_effect_variance(np.array([20.0, 80.0, 320.0]))
    assert np.all(np.diff(variances) < 0)
    # One-sample Hedges' variance at zero effect is about 1/N, and the bias correction pulls it
    # very slightly below.
    assert np.allclose(variances, 1.0 / np.array([20.0, 80.0, 320.0]), rtol=0.05)


def test_the_cutoff_conversion_distinguishes_the_two_designs():
    """A two-sample N is a total split into equal groups, so its cutoff is a different g.

    Worth a direct test rather than an inherited one: the last time these two designs shared a
    code path the sample size was reduced by mean instead of sum, which read ``[30, 30]`` as two
    groups of fifteen and inflated everything downstream by 39%.
    """
    one = reporting_cutoff_to_g([3.29], [40.0], design="one-sample")[0]
    two = reporting_cutoff_to_g([3.29], [40.0], design="two-sample")[0]
    # Same statistic, half the subjects per group, so the effect it implies is about twice as
    # large -- the ratio of the two designs' standard errors, sqrt(4/N) over sqrt(1/N).
    assert two / one == pytest.approx(2.0, rel=0.1)

    # A sign is irrelevant: the cutoff is a magnitude either way.
    assert reporting_cutoff_to_g([-3.29], [40.0])[0] == pytest.approx(one)

    # And a study too small for the conversion is refused rather than returned as a nan.
    with pytest.raises(ValueError, match="at least 4 subjects"):
        reporting_cutoff_to_g([3.29], [3.0], design="one-sample")
    with pytest.raises(ValueError, match="at least 5 subjects"):
        reporting_cutoff_to_g([3.29], [4.0], design="two-sample")
    with pytest.raises(ValueError, match="design must be"):
        reporting_cutoff_to_g([3.29], [40.0], design="paired")


def test_the_threshold_bound_reads_a_t_table_as_well_as_a_z_one():
    """A collection tabulating ``t`` must give the same bound as one tabulating the same tails.

    The bound lives on the z scale because that is what ``threshold`` is quoted on, so a ``t``
    has to be mapped to the z with the same tail probability rather than used as-is.
    """
    from nimare.transforms import t_to_z

    n = 30.0
    t_values = np.array([4.0, 5.5, 6.0])
    z_from_t = t_to_z(t_values, n - 1.0)

    as_z = pd.DataFrame({"id": ["a", "a", "a"], "z_stat": z_from_t})
    as_t = pd.DataFrame({"id": ["a", "a", "a"], "t_stat": t_values, "sample_size": [n] * 3})
    assert reported_minimum_z(as_z)["a"] == pytest.approx(reported_minimum_z(as_t)["a"])
    # The minimum, not the first or the largest.
    assert reported_minimum_z(as_t)["a"] == pytest.approx(float(z_from_t.min()))

    # No statistic column at all is an ordinary coordinate table, not an error.
    assert reported_minimum_z(pd.DataFrame({"id": ["a"]})).empty
    # Nor is a column that holds nothing usable.
    assert reported_minimum_z(pd.DataFrame({"id": ["a", "b"], "z_stat": [np.nan, 0.0]})).empty


def test_null_effect_variance_distinguishes_the_two_designs():
    """A two-sample N is a total split into groups, so its variance is about 4/N, not 1/N."""
    one = null_effect_variance(np.array([40.0]), design="one-sample")[0]
    two = null_effect_variance(np.array([40.0]), design="two-sample")[0]
    assert np.isclose(one, 1.0 / 40.0, rtol=0.05)
    assert np.isclose(two, 4.0 / 40.0, rtol=0.05)


def test_local_dl_reduces_to_dersimonian_laird():
    """With unit weights the local estimator must be the textbook DL estimator."""
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


def test_local_dl_is_zero_without_two_studies():
    """Heterogeneity is not estimable from a single study, so it is reported as zero."""
    zeros = np.zeros(1)
    tau2 = _local_dersimonian_laird(
        zeros, np.ones(1), np.ones(1), np.ones(1), np.ones(1), np.ones(1), np.ones(1)
    )
    assert tau2[0] == 0.0


def test_stat_from_histogram_finds_the_threshold_for_a_target_p():
    """The cluster-forming statistic is read off the null histogram, not assumed."""
    edges = _null_bin_edges()
    histogram = np.zeros(len(edges) - 1)
    # A flat null over |z| in [0, 1): a target p of 0.1 sits at the 90th percentile.
    histogram[: int(1.0 / _NULL_Z_STEP)] = 1.0
    assert np.isclose(_stat_from_histogram(0.1, histogram), 0.9, atol=2 * _NULL_Z_STEP)
    # An empty histogram cannot place a threshold, and says so rather than returning zero.
    assert not np.isfinite(_stat_from_histogram(0.1, np.zeros_like(histogram)))


def test_pooling_reduces_to_inverse_variance_weighting(studyset, small_mask):
    """With the silence switched off the pooled estimate is ordinary inverse-variance weighting."""
    from pymare.stats import weighted_least_squares

    estimator = CBES(mask=small_mask, null_method="none", selection_model="none")
    estimator.fit(studyset)
    fit = estimator._pool(estimator._focus_table_, estimator._image_studies_)

    voxel = int(np.argmax(fit["n_studies"]))
    g, v = [], []
    for _, cols, _, g_k, var_k in fit["contributions"]:
        hit = np.flatnonzero(cols == voxel)
        if hit.size:
            g.append(g_k[hit[0]])
            v.append(var_k[hit[0]])
    g, v = np.asarray(g), np.asarray(v)
    assert len(g) >= 2

    tau2 = float(fit["tau2"][voxel])
    expected, cov = weighted_least_squares(
        g[:, None], v[:, None], np.ones((len(g), 1)), tau2=tau2, return_cov=True
    )
    assert np.isclose(fit["g"][voxel], float(np.asarray(expected).ravel()[0]))
    assert np.isclose(fit["se"][voxel], float(np.sqrt(np.asarray(cov).ravel()[0])))


# ------------------------------------------------------------ options and refusals


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"design": "three-sample"}, "design must be"),
        ({"tau2_method": "reml"}, "tau2_method must be"),
        ({"selection_model": "tobit"}, "selection_model must be"),
        ({"null_method": "montecarlo"}, "null_method must be"),
        ({"se_method": "sandwich"}, "se_method must be"),
        ({"se_method": "hksj"}, "hksj"),
        ({"threshold": object()}, "threshold must be"),
    ],
)
def test_cbes_rejects_bad_parameters(kwargs, match):
    """Unusable options are refused at construction, not at fit time."""
    with pytest.raises(ValueError, match=match):
        CBES(**kwargs)


def test_the_removed_options_fail_loudly_rather_than_being_ignored():
    """A caller carrying an old configuration has to hear about it, not get a different fit.

    ``fwhm``, ``peak_bias``, ``peak_bias_scale``, ``stat_column``, ``kernel_min_weight`` and
    ``use_images`` all belonged to the design in which coordinate tables supplied magnitudes.
    Silently accepting them would mean accepting a request the estimator no longer honours.
    """
    for gone in (
        "fwhm",
        "peak_bias",
        "peak_bias_scale",
        "stat_column",
        "kernel_min_weight",
        "use_images",
    ):
        with pytest.raises(TypeError):
            CBES(**{gone: 1.0})


def test_the_only_null_is_the_image_permutation_one():
    """The coordinate magnitudes are gone, so there is nothing left to permute over them."""
    assert NULL_METHODS == ("permute-images", "none")
    for gone in ("permute-magnitudes", "approximate", "montecarlo"):
        with pytest.raises(ValueError, match="null_method must be"):
            CBES(null_method=gone)


def test_a_collection_with_no_shared_map_is_refused(coordinates_only, small_mask):
    """Coordinates carry no magnitude here, so a fit without an image is refused.

    Not returned as a map of zeros, and not quietly reduced to something else: the whole
    magnitude channel is the images, and a caller who supplied none asked for a fit that has no
    answer.
    """
    with pytest.raises(ValueError, match="at least one study supplying both a 'g' and a 'g_var'"):
        CBES(mask=small_mask, null_method="none").fit(coordinates_only)


def test_a_collection_with_no_coordinates_is_redirected_to_an_image_estimator(
    studyset, small_mask
):
    """With no coordinate table there is no silence to read, so CBES adds nothing over an IBMA."""
    stripped = copy.deepcopy(studyset.to_dict())
    for study in stripped["studies"]:
        for analysis in study["analyses"]:
            analysis["points"] = []
    from nimare.studyset import Studyset

    with pytest.raises(ValueError, match="images but no coordinates"):
        CBES(mask=small_mask, null_method="none").fit(
            Studyset(stripped, target=None, mask=small_mask)
        )


def test_a_collection_with_neither_coordinates_nor_images_still_raises(tmp_path, small_mask):
    """The missing-coordinates message must not shadow a collection that has nothing at all."""
    from nimare.studyset import Studyset

    empty = {
        "id": "empty",
        "name": "empty",
        "studies": [
            {
                "id": "s0",
                "name": "s0",
                "metadata": {"sample_sizes": [30]},
                "analyses": [
                    {"id": "s0-1", "name": "1", "metadata": {"sample_sizes": [30]}, "points": []}
                ],
            }
        ],
    }
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(empty))
    with pytest.raises(Exception):
        CBES(mask=small_mask, null_method="none").fit(
            Studyset(json.loads(path.read_text()), target=None, mask=small_mask)
        )


# --------------------------------------------------------------------- the maps


def test_cbes_produces_expected_maps(studyset, small_mask):
    """The documented maps are all present, and g lands near the truth at the focus."""
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    result = estimator.fit(studyset)

    expected = {
        "g",
        "se",
        "z",
        "p",
        "logp",
        "tau2",
        "n_studies",
        "n_eff",
        "dof",
        "prevalence",
        "g_marginal",
        "se_marginal",
    }
    assert expected <= set(result.maps)
    # Nothing from the design that pooled peak heights.
    assert not {"g_relative", "g_absolute"} & set(result.maps)

    assert value_at(result, "g") == pytest.approx(TRUE_G, abs=0.2)
    assert value_at(result, "se") > 0
    assert 0.0 < value_at(result, "prevalence") <= 1.0
    # p is 1 everywhere with no null built, rather than a normal-theory value.
    assert np.allclose(arrays(result)["p"], 1.0)


def test_g_is_on_the_scale_the_images_arrive_on(studyset, small_mask):
    """No unidentified constant survives: the magnitude comes from maps already in g units."""
    estimator = CBES(mask=small_mask, null_method="none", selection_model="none")
    result = estimator.fit(studyset)

    donors = estimator._image_studies_
    assert len(donors) == 2
    # With the silence off, the pooled value at a voxel must lie between the donors' own values.
    values = arrays(result)
    voxel = int(np.argmax(values["n_studies"]))
    donor_values = [g[voxel] for g, _, _ in donors.values()]
    assert min(donor_values) - 1e-9 <= values["g"][voxel] <= max(donor_values) + 1e-9


def test_dof_counts_the_censoring_roster_and_survives_a_single_image(tmp_path, small_mask):
    """``dof`` is the roster minus one, which is the only reference that survives one image.

    A Kish count over the pooling weights would be the image count, so ``dof`` would be zero
    with one donor and the recommended *t* interval would be ``nan`` everywhere.
    """
    one = make_studyset(tmp_path / "one", n_images=1, n_studies=12, seed=3)
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    result = estimator.fit(one)

    values = arrays(result)
    assert len(estimator._image_studies_) == 1
    assert np.isclose(values["n_eff"].max(), 1.0)
    covered = values["n_studies"] > 0
    assert covered.any()
    assert np.allclose(values["dof"][covered], len(estimator._sample_sizes_) - 1.0)
    assert values["dof"][covered].min() > 0


def test_fixed_effects_option_zeroes_tau2(studyset, small_mask):
    """``tau2_method="none"`` is a fixed-effects fit, so no heterogeneity is reported."""
    result = CBES(
        mask=small_mask, null_method="none", tau2_method="none", selection_model="none"
    ).fit(studyset)
    assert np.all(arrays(result)["tau2"] == 0.0)


def test_hartung_knapp_replaces_the_se_without_touching_the_estimate(studyset, small_mask):
    """HKSJ is a different standard error for the same weighted mean."""
    shared = dict(mask=small_mask, null_method="none", selection_model="none")
    model = CBES(**shared, se_method="model").fit(studyset)
    hksj = CBES(**shared, se_method="hksj").fit(studyset)

    a, b = arrays(model), arrays(hksj)
    assert np.allclose(a["g"], b["g"])
    covered = a["n_studies"] > 1
    assert covered.any()
    assert not np.allclose(a["se"][covered], b["se"][covered])


# ------------------------------------------------------------- the silence channel


def test_the_silence_channel_moves_the_estimate_toward_the_truth(tmp_path, small_mask):
    r"""Drop the peak heights, keep the silence: that is the whole design, so measure it.

    Scored against the field the simulator built, which is known exactly and is independent of
    the peaks the studies reported. ``g_marginal`` is the comparison to make: it is
    :math:`\\pi\\mu`, the estimand an inverse-variance mean of the images also reports, whereas
    ``g`` is :math:`\\mu`, the effect among the studies that have one, and is larger by
    construction.
    """
    truth = truth_field(small_mask)
    paired = []
    for seed in range(3):
        collection = make_studyset(tmp_path / f"silence{seed}", seed=seed)
        with_silence = arrays(
            CBES(mask=small_mask, null_method="none", threshold="reporting_threshold").fit(
                collection
            )
        )
        without = arrays(
            CBES(
                mask=small_mask,
                null_method="none",
                selection_model="none",
                threshold="reporting_threshold",
            ).fit(collection)
        )
        paired.append(
            (
                float(np.sqrt(np.mean((np.abs(with_silence["g_marginal"]) - truth) ** 2))),
                float(np.sqrt(np.mean((np.abs(without["g"]) - truth) ** 2))),
            )
        )

    silence_rmse, image_rmse = np.mean(paired, axis=0)
    assert silence_rmse < image_rmse
    # Every seed, not only the mean: three arms going the same way is the claim.
    assert all(a < b for a, b in paired)


def test_silence_is_only_read_where_no_focus_is_nearby(studyset, small_mask):
    """A study that reported near a voxel contributes neither a value nor a silence there.

    That is the design: a coordinate table says *where* a study reported, and its peak height
    is discarded, so a reporting coordinate study informs neither term.
    """
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    estimator.fit(studyset)

    table = estimator._focus_table_
    roster = list(estimator._sample_sizes_.index)
    active = np.arange(int(np.asarray(estimator.masker.mask_img.dataobj).astype(bool).sum()))
    col, pos, sign = estimator._indicator_entries(
        table, roster, active, active.size, image_ids=tuple(estimator._image_studies_)
    )
    indicator = np.zeros((len(roster), active.size))
    indicator[pos, col] = sign

    # An image reports everywhere, so it has no indicator to contribute at any voxel: its
    # magnitude enters through its value instead.
    for study_id in estimator._image_studies_:
        assert np.all(indicator[roster.index(study_id)] == 0.0)

    for study_id in set(table["id"]):
        row = indicator[roster.index(study_id)]
        # A study that reported names some voxels (-1) and is silent about others (+1)...
        assert (row == -1.0).any()
        assert (row == 1.0).any()
        # ...and says nothing at the voxels it reached but did not name.
        assert (row == 0.0).any()
        # It names exactly the voxels its own foci sit in.
        named = int((row == -1.0).sum())
        assert named <= int((table["id"] == study_id).sum())


def test_a_study_that_reported_nothing_at_all_is_silent_everywhere(studyset, small_mask):
    """A study with no focus never reaches ``inputs_``, but its silence is the strongest datum.

    ``_collect_inputs`` drops a study with no coordinates (neurostuff/NiMARE#294), so the
    roster has to come from the collection rather than from the coordinates table.
    """
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    estimator.fit(studyset)

    roster = set(estimator._sample_sizes_.index)
    reporting = set(estimator._focus_table_["id"].unique())
    assert roster >= reporting
    assert roster >= set(estimator._image_studies_)
    assert len(roster) == len(studyset.ids)


def test_silence_pulls_the_estimate_down_where_nobody_reported(tmp_path, small_mask):
    """Every study staying silent about a region is evidence the effect there is small."""
    collection = make_studyset(tmp_path / "quiet", n_images=1, n_studies=14, seed=11)
    shared = dict(mask=small_mask, null_method="none", threshold="reporting_threshold")
    with_silence = arrays(CBES(**shared).fit(collection))
    without = arrays(CBES(**shared, selection_model="none").fit(collection))

    # Judged on the marginal estimand, which is what the images-only fit also reports.
    quiet = with_silence["prevalence"] < np.percentile(with_silence["prevalence"], 25)
    assert quiet.any()
    assert np.median(np.abs(with_silence["g_marginal"][quiet])) < np.median(
        np.abs(without["g"][quiet])
    )


def test_the_threshold_decides_how_surprising_a_silence_is(tmp_path, small_mask):
    """A study that thresholded strictly tells us less by staying silent, so the fit moves."""
    collection = make_studyset(tmp_path / "thresh", n_studies=14, seed=5)
    shared = dict(mask=small_mask, null_method="none")
    lenient = arrays(CBES(**shared, threshold=2.5).fit(collection))
    strict = arrays(CBES(**shared, threshold=5.0).fit(collection))

    covered = lenient["n_studies"] > 0
    assert covered.any()
    # A stricter assumed cut makes silence weaker evidence, so more of the roster is read as
    # having an effect that simply failed to clear it.
    assert np.median(strict["prevalence"][covered]) > np.median(lenient["prevalence"][covered])


def test_threshold_can_name_a_metadata_field(studyset, small_mask):
    """Papers that state their threshold should not be forced through an assumed constant."""
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    estimator.fit(studyset)

    supplied = np.asarray(
        studyset.get_metadata(field="reporting_threshold", ids=list(studyset.ids)), dtype=float
    )
    assert np.allclose(np.sort(estimator._cutoffs_z_.values), np.sort(supplied))
    # And the likelihood gets them on the effect-size scale, not the z scale.
    expected = reporting_cutoff_to_g(estimator._cutoffs_z_.values, estimator._sample_sizes_.values)
    assert np.allclose(estimator._thresholds_.values, expected)


def test_threshold_metadata_field_must_exist(studyset, small_mask):
    """A misspelt field name is a silent no-op otherwise, and the fit would use the default."""
    with pytest.raises(ValueError, match="not a metadata field"):
        CBES(mask=small_mask, null_method="none", threshold="not_a_field").fit(studyset)


def test_the_threshold_is_never_inferred_from_the_reported_heights(studyset, small_mask):
    """The heights are not read at all, so there is nothing to infer a threshold from.

    The removed rules undid the order statistic on a study's smallest reported value, which
    cannot tell a voxelwise height cut from a cluster-forming one and overshot the second by
    about 1 z. Assuming a constant is both simpler and more accurate.
    """
    default = CBES(mask=small_mask, null_method="none")
    default.fit(studyset)
    from nimare.meta.cbma.effectsize import DEFAULT_REPORTING_THRESHOLD_Z

    assert np.allclose(default._cutoffs_z_.values, DEFAULT_REPORTING_THRESHOLD_Z)
    for gone in ("study-min", "pooled-min"):
        with pytest.raises(ValueError, match="not a metadata field"):
            CBES(mask=small_mask, null_method="none", threshold=gone).fit(studyset)


def scale_reported_statistics(studyset, mask, factor):
    """Return the collection with every reported statistic scaled by ``factor``.

    A pure scaling, not an affine map. Adding a constant is not monotone in ``|z|`` -- it pulls
    the negative peaks toward zero and so *lowers* the smallest reported magnitude, which is
    the one quantity the estimator still reads. Scaling keeps ``|z|`` ordered and moves the
    minimum in the direction asked for.
    """
    altered = copy.deepcopy(studyset.to_dict())
    for study in altered["studies"]:
        for analysis in study["analyses"]:
            for point in analysis.get("points", []):
                for value in point.get("values", []):
                    if value.get("kind") == "Z":
                        value["value"] = float(value["value"]) * factor
    from nimare.studyset import Studyset

    return Studyset(altered, target=None, mask=mask)


def test_the_reported_heights_never_reach_the_magnitude(studyset, small_mask):
    """Scale every reported statistic and the fit must not move: heights are not magnitudes.

    ``clamp_threshold=False`` so the one channel the statistics *do* still feed -- a bound on
    the reporting threshold -- is switched off, leaving nothing for them to touch.
    """
    shared = dict(
        mask=small_mask,
        null_method="none",
        threshold="reporting_threshold",
        clamp_threshold=False,
    )
    original = arrays(CBES(**shared).fit(studyset))
    for factor in (0.6, 3.0):
        altered = arrays(
            CBES(**shared).fit(scale_reported_statistics(studyset, small_mask, factor))
        )
        for name in ("g", "se", "prevalence", "g_marginal"):
            assert np.allclose(original[name], altered[name]), (factor, name)


def test_the_reported_heights_reach_the_threshold_and_only_the_threshold(studyset, small_mask):
    """The one thing the statistics are still read for is a *bound* on each study's cutoff.

    Anything a study reported cleared its cut, so its smallest reported value is an upper bound
    on that cut -- a hard inequality, not an inference, and it can only move a cutoff down.
    """
    # Scaled down so the smallest reported value falls below the cut the metadata states,
    # which is the only situation the bound can act in: a table that contradicts its own
    # stated threshold.
    contradictory = scale_reported_statistics(studyset, small_mask, 0.8)
    shared = dict(mask=small_mask, null_method="none", threshold="reporting_threshold")
    clamped = CBES(**shared)
    clamped.fit(contradictory)
    held = CBES(**shared, clamp_threshold=False)
    held.fit(contradictory)

    # The clamp never raises a cutoff, and lowers at least one on this collection.
    assert np.all(clamped._cutoffs_z_.values <= held._cutoffs_z_.values + 1e-9)
    assert np.any(clamped._cutoffs_z_.values < held._cutoffs_z_.values - 1e-9)

    # And the bound really is each study's own minimum, where that bites.
    minima = reported_minimum_z(clamped.inputs_["coordinates"]).reindex(clamped._cutoffs_z_.index)
    expected = np.minimum(held._cutoffs_z_.values, minima.fillna(np.inf).values)
    assert np.allclose(clamped._cutoffs_z_.values, expected)


def test_the_threshold_bound_is_a_no_op_on_tables_that_do_not_contradict_it(studyset, small_mask):
    """Raise every reported statistic and the bound becomes vacuous, changing nothing.

    This is the thin-table case -- a paper reporting only its strongest peaks -- where the
    smallest reported value sits far above the cut it actually applied. The clamp must do
    nothing there rather than guess.
    """
    thin = scale_reported_statistics(studyset, small_mask, 2.0)
    shared = dict(mask=small_mask, null_method="none", threshold="reporting_threshold")
    clamped = arrays(CBES(**shared).fit(thin))
    held = arrays(CBES(**shared, clamp_threshold=False).fit(thin))
    for name in ("g", "se", "prevalence"):
        assert np.array_equal(clamped[name], held[name])


def test_the_coverage_radius_is_assumed_rather_than_read(tmp_path, small_mask):
    """Assume the extent a focus stands in for, rather than reading it.

    Papers do not report cluster extent reliably, and assuming an extent is not the same thing
    as treating everything outside it as silence.
    """
    collection = make_studyset(tmp_path / "radius", n_studies=14, seed=13)
    shared = dict(mask=small_mask, null_method="none", threshold="reporting_threshold")
    tight = arrays(CBES(**shared, coverage_radius=8.0).fit(collection))
    wide = arrays(CBES(**shared, coverage_radius=28.0).fit(collection))

    covered = tight["n_studies"] > 0
    # A wider assumed extent means fewer studies count as silent, and prevalence rises with it
    # monotonically at every true value.
    assert np.median(wide["prevalence"][covered]) > np.median(tight["prevalence"][covered])
    # None is the documented default, not a fall-back to some other geometry.
    assert np.isclose(DEFAULT_COVERAGE_RADIUS_MM, 20.0)
    default = arrays(CBES(**shared, coverage_radius=None).fit(collection))
    fixed = arrays(CBES(**shared, coverage_radius=DEFAULT_COVERAGE_RADIUS_MM).fit(collection))
    assert np.allclose(default["g"], fixed["g"])


def test_coordinate_share_says_where_the_coordinate_channel_is_acting(studyset, small_mask):
    """Say whether the coordinate channel is doing anything at each voxel.

    The one diagnostic a reader cannot otherwise get. It is the fraction of the Fisher
    information about ``g`` contributed by the reporting indicators rather than the images'
    values, so 0 means the images carry the estimate alone and every coordinate caveat is moot
    there, and 1 means the indicators carry it.
    """
    result = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold").fit(
        studyset
    )
    values = arrays(result)
    share = values["coordinate_share"]

    assert np.all(share >= 0.0) and np.all(share <= 1.0)
    # It has to vary: a constant map would be telling the reader nothing.
    covered = values["n_studies"] > 0
    assert share[covered].std() > 0.01
    # Highest where the studies actually reported, which is the stratification the design rests
    # on -- the images carry the rest of the map.
    focus = int(np.argmax(np.abs(values["g"]) * covered))
    assert share[focus] > np.median(share[covered])


def test_coordinate_share_is_zero_when_the_coordinates_cannot_act(studyset, small_mask):
    """With the selection model off the tables are inert, so no share is reported at all."""
    result = CBES(mask=small_mask, null_method="none", selection_model="none").fit(studyset)
    assert "coordinate_share" not in set(result.maps)


def test_the_prevalence_is_not_fitted_where_no_indicator_identifies_it(
    tmp_path_factory, small_mask
):
    """With every study carrying an image the indicator is empty, so ``pi`` must stay at 1.

    Nothing but the reporting indicator separates "no effect in this study" from "a small
    effect plus noise", so with no indicator anywhere the prevalence is unidentified. Left free
    it does not merely wander -- a two-component mixture explains Gaussian noise as a mixture,
    and returned 0.577 against a true 1.0 here.
    """
    all_images = make_studyset(
        tmp_path_factory.mktemp("cbes_allimages"), n_images=20, n_studies=20
    )
    result = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold").fit(
        all_images
    )
    values = arrays(result)
    assert np.allclose(values["coordinate_share"], 0.0), "the indicator should be empty"
    assert np.allclose(values["prevalence"], 1.0)


def test_an_unidentified_prevalence_is_not_profiled_out_of_the_error(tmp_path_factory, small_mask):
    """Holding ``pi`` at 1 is only half of it; the Schur complement has to go too.

    The limit cannot be left to the algebra: with ``pi`` clamped just below 1 the cross block
    stays order one, so profiling would still subtract a term no parameter earned. The check is
    that the mixture fit then agrees with the same images fitted without a mixture at all,
    which is the model it has collapsed to.
    """
    all_images = make_studyset(
        tmp_path_factory.mktemp("cbes_allimages_se"), n_images=20, n_studies=20
    )
    shared = dict(mask=small_mask, null_method="none", threshold="reporting_threshold")
    mixture = arrays(CBES(**shared).fit(all_images))
    plain = arrays(CBES(selection_model="none", **shared).fit(all_images))
    finite = np.isfinite(mixture["se"]) & np.isfinite(plain["se"])
    assert finite.any()
    assert np.allclose(mixture["se"][finite], plain["se"][finite])
    assert np.allclose(mixture["g"], plain["g"])


def test_a_thin_indicator_still_fits_the_prevalence(studyset, small_mask):
    """The guard is for no evidence at all, not for little of it.

    The configuration the estimator is actually for -- a couple of images among many tables --
    must be unaffected, or the guard has quietly become a different estimator.
    """
    result = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold").fit(
        studyset
    )
    values = arrays(result)
    assert (values["coordinate_share"] > 0).any()
    assert (values["prevalence"] < 1.0).any(), "pi should still be free where silence speaks"


def test_the_marginal_map_is_the_conditional_one_times_the_prevalence(studyset, small_mask):
    """``g_marginal`` is the estimand an image-based meta-analysis reports; ``g`` is not."""
    result = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold").fit(
        studyset
    )
    values = arrays(result)
    assert np.allclose(values["g_marginal"], values["g"] * values["prevalence"])


def test_the_selection_model_can_be_switched_off_entirely(studyset, small_mask):
    """``selection_model="none"`` must make the coordinate tables inert, not merely quieter.

    It is the control arm the coordinates are credited against, so if a table still reached the
    estimate there the measured benefit would be against the wrong baseline.
    """
    stripped = copy.deepcopy(studyset.to_dict())
    for study in stripped["studies"]:
        for analysis in study["analyses"]:
            for point in analysis.get("points", []):
                point["coordinates"] = [0.0, 0.0, 0.0]
    from nimare.studyset import Studyset

    shared = dict(mask=small_mask, null_method="none", selection_model="none")
    original = arrays(CBES(**shared).fit(studyset))
    moved = arrays(CBES(**shared).fit(Studyset(stripped, target=None, mask=small_mask)))
    assert np.allclose(original["g"], moved["g"])
    assert "prevalence" not in original


def test_the_censored_likelihood_reads_silence_as_evidence_against_a_large_effect():
    """The model's defining behaviour, checked on the likelihood rather than on a fitted map.

    Adding silent studies must pull the optimum down: silence is improbable when the effect is
    large, so the more studies stay quiet the smaller the effect that best explains the data.

    The cutoff here is deliberately low relative to the reported effect, which puts the studies
    inside the *window of detectability* -- where the chance of reporting still responds to the
    magnitude. Far above that window silence is certain whatever the effect, so it informs the
    prevalence and not the magnitude; that is a property of the model rather than a defect, but
    it means a probe placed there cannot see this behaviour at all.
    """
    fitted, prevalences = [], []
    for n_silent in (0, 3, 9):
        n_studies = 3 + n_silent
        sample_sizes = np.full(n_studies, 30.0)
        null_var = null_effect_variance(sample_sizes, design="one-sample")[:, None]
        weights = np.zeros((n_studies, 1))
        g_obs = np.zeros((n_studies, 1))
        var_obs = np.ones((n_studies, 1))
        # +1 is a silence; 0 is "no indicator", which is what a value-bearing study carries.
        indicator = np.ones((n_studies, 1))
        for study, value in enumerate((0.8, 0.7, 0.9)):
            weights[study, 0] = 1.0
            g_obs[study, 0] = value
            var_obs[study, 0] = float(null_var[study, 0])
            indicator[study, 0] = 0.0

        estimator = CBES(null_method="none", max_iter=400)
        mu, pi, _, _, _ = estimator._fit_chunk(
            weights=weights,
            g_obs=g_obs,
            var_obs=var_obs,
            indicator=indicator,
            tau2=np.zeros(1),
            null_var=null_var,
            cutoffs=np.full((n_studies, 1), 0.55),
            start=np.array([0.8]),
        )
        fitted.append(float(mu[0]))
        prevalences.append(float(pi[0]))

    assert fitted[0] > fitted[1] > fitted[2], fitted
    # And the prevalence falls too: the mixture is free to explain silence either way.
    assert prevalences[0] > prevalences[-1], prevalences


def test_fit_chunk_ignores_studies_that_say_nothing_here():
    """A study with neither a value nor an indicator here must not move the fit at all.

    That is the third case the block carries, and it is the common one: an image study's own
    indicator, and any voxel an ``analysis_mask`` says a study never examined.
    """
    estimator = CBES(selection_model="zero-inflated", null_method="none")

    def run(n_mute):
        n_studies = 3 + n_mute
        weights = np.zeros((n_studies, 1))
        weights[:2, 0] = 1.0
        g_obs = np.zeros((n_studies, 1))
        g_obs[:2, 0] = [0.9, 1.1]
        var_obs = np.full((n_studies, 1), 1.0 / 30.0)
        indicator = np.zeros((n_studies, 1))
        indicator[2, 0] = 1.0  # one genuinely silent study
        mu, pi, _, _, _ = estimator._fit_chunk(
            weights=weights,
            g_obs=g_obs,
            var_obs=var_obs,
            indicator=indicator,
            tau2=np.zeros(1),
            null_var=np.full((n_studies, 1), 1.0 / 30.0),
            cutoffs=np.full((n_studies, 1), 1.2),
            start=np.array([1.0]),
        )
        return float(mu[0]), float(pi[0])

    assert run(0) == pytest.approx(run(9), abs=1e-9)


def test_a_report_and_a_silence_pull_the_magnitude_opposite_ways():
    """Both values of the reporting indicator are evidence, and they disagree.

    Dropping the report limb leaves the silences as the only evidence about the indicator, so
    the model reads the silence fraction against a denominator that excludes every study that
    reported -- and over-shrinks. Measured on a known truth, restoring it cut the rmse where
    the effect is largest from 0.091 to 0.074 and the bias from -0.060 to -0.042.
    """
    estimator = CBES(selection_model="zero-inflated", null_method="none", max_iter=400)

    def fit(sign):
        # One image supplying a value, and nine coordinate studies all carrying ``sign``.
        n_studies = 10
        weights = np.zeros((n_studies, 1))
        weights[0, 0] = 1.0
        g_obs = np.zeros((n_studies, 1))
        g_obs[0, 0] = 0.6
        var_obs = np.full((n_studies, 1), 1.0 / 30.0)
        indicator = np.zeros((n_studies, 1))
        indicator[1:, 0] = sign
        mu, _, _, _, _ = estimator._fit_chunk(
            weights=weights,
            g_obs=g_obs,
            var_obs=var_obs,
            indicator=indicator,
            tau2=np.zeros(1),
            null_var=np.full((n_studies, 1), 1.0 / 30.0),
            cutoffs=np.full((n_studies, 1), 0.6),
            start=np.array([0.6]),
        )
        return float(mu[0])

    alone, silent, reported = fit(0.0), fit(1.0), fit(-1.0)
    assert silent < alone < reported


# ----------------------------------------------------------------- analysis masks


@pytest.fixture(scope="module")
def roi_studyset(tmp_path_factory):
    """Build twelve studies; four examined only a slab, and are silent everywhere else.

    One study shares its map, which CBES requires. The slab sits at one end of the volume, away
    from the focus every whole-brain study reports at, so the partial studies' silence at the
    focus is uninformative and must not be read.
    """
    directory = tmp_path_factory.mktemp("cbes_roi")
    shape = (12, 12, 12)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -22.0
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), directory / "mask.nii.gz")

    slab = np.zeros(shape, np.int32)
    slab[:3] = 1
    nib.save(nib.Nifti1Image(slab, affine), directory / "slab.nii.gz")

    rng = np.random.default_rng(2)
    grid = np.indices(shape).astype(float)
    centre = (np.array(shape) - 1) / 2.0
    truth = 0.8 * np.exp(-sum((grid[i] - centre[i]) ** 2 for i in range(3)) / 8.0)
    observed = truth + rng.normal(0, 1 / np.sqrt(30), shape)
    nib.save(nib.Nifti1Image(observed.astype(np.float32), affine), directory / "donor_g.nii.gz")
    nib.save(
        nib.Nifti1Image(np.full(shape, 1.0 / 30.0, np.float32), affine),
        directory / "donor_var.nii.gz",
    )

    from nimare.studyset import Studyset

    studies = []
    for k in range(12):
        partial = k >= 8
        analysis = {
            "id": f"r{k}-1",
            "name": "1",
            "metadata": {"sample_sizes": [30]},
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
        if k == 0:
            analysis["images"] = [
                {
                    "url": str(directory / "donor_g.nii.gz"),
                    "filename": "donor_g.nii.gz",
                    "space": "MNI",
                    "value_type": "g",
                },
                {
                    "url": str(directory / "donor_var.nii.gz"),
                    "filename": "donor_var.nii.gz",
                    "space": "MNI",
                    "value_type": "g_var",
                },
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


def masked_index(studyset, xyz):
    """Return the position of an xyz (mm) location in the masker's voxel order."""
    masker = studyset.masker
    ijk = mm2vox(np.array([xyz]), masker.mask_img.affine)[0]
    mask = np.asarray(masker.mask_img.dataobj).astype(bool)
    lookup = np.full(mask.shape, -1, dtype=np.int64)
    lookup[mask] = np.arange(mask.sum())
    return int(lookup[tuple(ijk)])


def test_a_declared_analysis_mask_stops_silence_being_read_where_nobody_looked(roi_studyset):
    """An ROI study never looked outside its region, so its silence there is not evidence."""
    focus = masked_index(roi_studyset, (0.0, 0.0, 0.0))
    assert focus >= 0

    shared = dict(null_method="none")
    ignored = CBES(**shared).fit(roi_studyset)
    honoured = CBES(**shared, analysis_mask="analysis_mask").fit(roi_studyset)

    # Four studies stop arguing against the effect at the focus, so the prevalence must rise.
    assert (
        honoured.get_map("prevalence", return_type="array").ravel()[focus]
        > ignored.get_map("prevalence", return_type="array").ravel()[focus]
    )


def test_an_absent_analysis_mask_changes_nothing_but_says_so(roi_studyset, caplog):
    """Asking for a value type the collection does not carry is a silent no-op otherwise."""
    shared = dict(null_method="none")
    plain = arrays(CBES(**shared).fit(roi_studyset))
    with caplog.at_level("WARNING"):
        requested = arrays(CBES(**shared, analysis_mask="nope").fit(roi_studyset))
    assert np.allclose(plain["g"], requested["g"])
    assert "matches no image value type" in caplog.text


def test_the_analysis_mask_is_keyed_per_contrast_not_per_study(roi_studyset):
    """Coverage is decided per analysis, matching the unit the rest of the estimator uses."""
    estimator = CBES(null_method="none", analysis_mask="analysis_mask")
    estimator.fit(roi_studyset)
    keys = set(estimator._analysis_masks_)
    assert keys
    assert all(key in set(estimator._sample_sizes_.index) for key in keys)


# ---------------------------------------------------------------------- the null


def test_the_null_shuffles_image_values_within_a_study_and_never_between_them(
    studyset, small_mask
):
    """Exchangeability: values move among a study's own voxels, nothing moves between studies."""
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    estimator.fit(studyset)

    rng = np.random.default_rng(0)
    permuted = estimator._permute_image_values(rng)
    assert set(permuted) == set(estimator._image_studies_)
    for study_id, (g, var_g, usable) in estimator._image_studies_.items():
        new_g, new_var, new_usable = permuted[study_id]
        assert np.array_equal(usable, new_usable)
        # The same multiset of values, rearranged among this study's own usable voxels.
        assert np.allclose(np.sort(g[usable]), np.sort(new_g[new_usable]))
        assert np.allclose(np.sort(var_g[usable]), np.sort(new_var[new_usable]))


def test_the_null_leaves_the_coordinate_tables_exactly_alone(studyset, small_mask):
    """The silence pattern is identical in the observed fit and in every permutation.

    That is what makes the censoring term cancel between the two, and why a null over the
    coordinates is neither needed nor available.
    """
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    estimator.fit(studyset)
    before = estimator._focus_table_.copy()
    estimator._statistic(
        estimator._focus_table_,
        estimator._sample_sizes_,
        estimator._thresholds_,
        estimator._permute_image_values(np.random.default_rng(1)),
    )
    assert before.equals(estimator._focus_table_)


def test_a_collection_that_cannot_be_permuted_is_refused_rather_than_given_p_values(
    roi_studyset, caplog
):
    """A null that admits too few arrangements reads as a null result rather than as no test."""
    estimator = CBES(null_method="permute-images", n_iters=10)
    with caplog.at_level("WARNING"):
        estimator.fit(roi_studyset)
    # One donor over a 12^3 volume admits plenty of arrangements, so this one is testable --
    # the guard is exercised by stubbing the roster out.
    assert estimator._null_is_usable()

    estimator._image_studies_ = {}
    estimator._null_refusal_logged_ = False
    with caplog.at_level("WARNING"):
        assert not estimator._null_is_usable()
    assert "cannot be tested" in caplog.text


def test_the_null_counts_only_the_image_studies(studyset, small_mask):
    """Coordinates contribute no randomness, because the shuffle never touches them."""
    estimator = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold")
    estimator.fit(studyset)

    log10_states, contributing = estimator._null_has_states()
    assert contributing == len(estimator._image_studies_)
    assert log10_states > 4.0
    # Dropping every coordinate row must not change the count.
    estimator._focus_table_ = estimator._focus_table_.iloc[:0]
    assert estimator._null_has_states() == (log10_states, contributing)


def test_no_null_reports_no_p_values(studyset, small_mask):
    """``null_method="none"`` returns p = 1 everywhere, not a normal-theory p-value."""
    result = CBES(mask=small_mask, null_method="none").fit(studyset)
    assert np.allclose(arrays(result)["p"], 1.0)


def test_the_permutation_null_produces_usable_p_values(permutation_fit):
    """Check p is bounded by the permutation floor and refers each voxel to its own null."""
    estimator, result = permutation_fit
    p = arrays(result)["p"]
    assert np.all(p >= 1.0 / (1.0 + estimator.n_iters) - 1e-12)
    assert np.all(p <= 1.0)
    assert p.min() < 1.0  # something moved, so the null was actually built


def test_null_is_built_from_the_selected_statistic(studyset, small_mask):
    """The null refits the same statistic the observed map came from, selection model included."""
    estimator = CBES(
        mask=small_mask, null_method="permute-images", n_iters=5, selection_model="zero-inflated"
    )
    estimator.fit(studyset)
    fit, _ = estimator._statistic(
        estimator._focus_table_,
        estimator._sample_sizes_,
        estimator._thresholds_,
        estimator._permute_image_values(np.random.default_rng(0)),
    )
    # A permutation fit carries a prevalence, which only the selection model produces.
    assert "prevalence" in fit


def test_convergence_alone_does_not_make_a_voxel_significant(tmp_path, small_mask):
    """The null is about magnitude, not about foci piling up: this is not a convergence test."""
    # Every study reports at the same place with no effect anywhere, so the foci converge
    # perfectly and the images carry nothing.
    from nimare.studyset import Studyset

    directory = tmp_path / "flat"
    directory.mkdir()
    shape = (10, 10, 10)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -18.0
    nib.save(nib.Nifti1Image(np.ones(shape, np.int32), affine), directory / "mask.nii.gz")
    rng = np.random.default_rng(0)
    studies = []
    for k in range(10):
        analysis = {
            "id": f"f{k}-1",
            "name": "1",
            "metadata": {"sample_sizes": [30]},
            "points": [
                {
                    "space": "MNI",
                    "coordinates": [0.0, 0.0, 0.0],
                    "values": [{"kind": "Z", "value": 4.0}],
                }
            ],
            "images": [],
        }
        if k < 2:
            flat = rng.normal(0, 1 / np.sqrt(30), shape)
            nib.save(
                nib.Nifti1Image(flat.astype(np.float32), affine), directory / f"f{k}_g.nii.gz"
            )
            nib.save(
                nib.Nifti1Image(np.full(shape, 1 / 30.0, np.float32), affine),
                directory / f"f{k}_v.nii.gz",
            )
            analysis["images"] = [
                {
                    "url": str(directory / f"f{k}_g.nii.gz"),
                    "filename": "g",
                    "space": "MNI",
                    "value_type": "g",
                },
                {
                    "url": str(directory / f"f{k}_v.nii.gz"),
                    "filename": "v",
                    "space": "MNI",
                    "value_type": "g_var",
                },
            ]
        studies.append(
            {
                "id": f"f{k}",
                "name": f"f{k}",
                "metadata": {"sample_sizes": [30]},
                "analyses": [analysis],
            }
        )

    collection = Studyset(
        {"id": "flat", "name": "flat", "studies": studies},
        target=None,
        mask=str(directory / "mask.nii.gz"),
    )
    result = CBES(null_method="permute-images", n_iters=50, seed=0).fit(collection)
    p = arrays(result)["p"]
    focus = masked_index(collection, (0.0, 0.0, 0.0))
    # The foci converge exactly at the focus and the p there is unremarkable.
    assert p[focus] > 0.05


# ---------------------------------------------------------------- FWE correction


def test_correct_fwe_montecarlo(studyset, small_mask):
    """Voxel- and cluster-level corrected maps come out, with the names ALE uses."""
    estimator = CBES(mask=small_mask, null_method="none", n_iters=20, seed=0)
    result = estimator.fit(studyset)
    corrected = FWECorrector(method="montecarlo", n_iters=20).transform(result)

    for name in (
        "logp_level-voxel_corr-FWE_method-montecarlo",
        "logp_desc-size_level-cluster_corr-FWE_method-montecarlo",
        "logp_desc-mass_level-cluster_corr-FWE_method-montecarlo",
    ):
        assert name in corrected.maps
        values = corrected.get_map(name, return_type="array")
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0)


def test_correct_fwe_montecarlo_needs_a_fit(small_mask):
    """Correcting an unfitted estimator is a programming error, not an empty result."""
    with pytest.raises(ValueError, match="requires a fitted estimator"):
        CBES(mask=small_mask).correct_fwe_montecarlo(None)


def test_fwe_montecarlo_vfwe_only_returns_only_voxel_maps(permutation_fit):
    """``vfwe_only`` skips the cluster pass, so no cluster maps are emitted."""
    estimator, result = permutation_fit
    maps, _, _ = estimator.correct_fwe_montecarlo(result, n_iters=10, vfwe_only=True)
    assert "logp_level-voxel" in maps
    assert not any("cluster" in name for name in maps)


def test_cluster_null_is_built_during_fit(permutation_fit):
    """``fit`` builds the cluster nulls alongside the voxel one, so correction is nearly free."""
    estimator, _ = permutation_fit
    assert "values_desc-size_level-cluster_corr-fwe_method-montecarlo" in (
        estimator.null_distributions_
    )
    assert "values_desc-mass_level-cluster_corr-fwe_method-montecarlo" in (
        estimator.null_distributions_
    )
    assert np.isfinite(estimator.null_distributions_["cluster_forming_stat"])


def test_cluster_threshold_none_skips_the_cluster_null(studyset, small_mask):
    """Opting out of the cluster null must actually skip it, not build it quietly."""
    estimator = CBES(
        mask=small_mask, null_method="permute-images", n_iters=10, cluster_threshold=None
    )
    estimator.fit(studyset)
    assert "cluster_forming_stat" not in estimator.null_distributions_


def test_fwe_montecarlo_reuses_the_null_from_fit(studyset, small_mask):
    """The same iterations at the same threshold are reused rather than recomputed."""
    estimator = CBES(mask=small_mask, n_iters=20, seed=0)
    result = estimator.fit(studyset)
    before = estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
    estimator.correct_fwe_montecarlo(result, n_iters=estimator.n_iters, voxel_thresh=0.001)
    after = estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
    assert np.array_equal(before, after)


@pytest.mark.parametrize(
    "corrector,map_name",
    [
        (FDRCorrector(method="indep"), "logp_corr-FDR_method-indep"),
        (FWECorrector(method="bonferroni"), "logp_corr-FWE_method-bonferroni"),
    ],
)
def test_stock_correctors_work(permutation_fit, corrector, map_name):
    """The uncorrected p map is a real p map, so the stock correctors apply to it."""
    _, result = permutation_fit
    corrected = corrector.transform(result)
    assert map_name in corrected.maps
    assert np.all(np.isfinite(corrected.get_map(map_name, return_type="array")))


# ---------------------------------------------------------------- the description


def test_cbes_description_says_where_each_channel_comes_from(studyset, small_mask):
    """The description is what ends up in a methods section, so it has to name the design."""
    result = CBES(mask=small_mask, null_method="none", threshold="reporting_threshold").fit(
        studyset
    )
    description = result.description_

    assert "coordinate tables supplied only" in description
    assert "peak heights being discarded" in description
    assert "zero-inflated censored" in description
    assert "No effect-size images were imputed" in description
    assert "no p-values are reported" in description
    # Nothing left from the design that pooled peak heights.
    for gone in ("Gaussian kernel", "peak-height", "confidence interval of"):
        assert gone not in description


def test_the_description_reports_the_null_it_actually_ran(permutation_fit):
    """A reader has to be able to tell which hypothesis was tested."""
    _, result = permutation_fit
    description = result.description_
    assert "reassigned among its own voxels" in description
    assert "coordinate silence held fixed" in description


# ------------------------------------------------------------------ two-sample


@pytest.fixture(scope="module")
def two_group_studyset(tmp_path_factory):
    """Build a two-sample collection whose metadata gives per-group sizes."""
    collection = make_studyset(
        tmp_path_factory.mktemp("cbes_two"), n_studies=12, seed=21, design="two-sample"
    )
    raw = copy.deepcopy(collection.to_dict())
    for study in raw["studies"]:
        for analysis in study["analyses"]:
            total = int(np.sum(analysis["metadata"]["sample_sizes"]))
            halves = [total // 2, total - total // 2]
            analysis["metadata"]["sample_sizes"] = halves
        study["metadata"]["sample_sizes"] = halves
    from nimare.studyset import Studyset

    return Studyset(raw, target=None, mask=None)


def test_a_two_sample_design_gets_the_total_sample_size(two_group_studyset, small_mask):
    """Sum a two-sample design's per-group sizes rather than averaging them.

    ``[30, 30]`` means sixty subjects; reducing it by mean gave thirty, which was then read as
    two groups of fifteen and wrecked every variance downstream.
    """
    estimator = CBES(mask=small_mask, design="two-sample", null_method="none")
    estimator.fit(two_group_studyset)
    assert estimator._size_reduction() == "sum"

    supplied = two_group_studyset.get_metadata(
        field="sample_sizes", ids=list(two_group_studyset.ids)
    )
    totals = sorted(float(np.sum(value)) for value in supplied)
    assert sorted(estimator._sample_sizes_.values) == pytest.approx(totals)


# ----------------------------------------------------------------- the simulator


def test_simulator_respects_the_reporting_threshold(tmp_path):
    """A study reports a peak only where its statistic cleared the threshold it applied."""
    collection = make_studyset(tmp_path / "thresh_sim", n_studies=8, seed=1, threshold_z=4.5)
    values = []
    for analysis in collection.analyses:
        for point in analysis.points:
            for kind, value in point.values.items() if isinstance(point.values, dict) else ():
                if kind in ("z_stat", "value_z"):
                    values.append(abs(float(value)))
    reported = np.abs(collection.coordinates["z_stat"].dropna().values)
    assert reported.size
    assert reported.min() >= 4.5 - 1e-6


def test_simulator_writes_images_only_when_asked(tmp_path):
    """``n_image_studies`` is what makes a simulated collection a valid CBES input."""
    plain = make_studyset(tmp_path / "plain", n_images=0, n_studies=6, seed=2)
    assert not {"g", "g_var"} & set(plain.images.columns)

    donors = make_studyset(tmp_path / "donors", n_images=2, n_studies=6, seed=2)
    assert {"g", "g_var"} <= set(donors.images.columns)
    assert int(donors.images["g"].notna().sum()) == 2
    # The donors keep their coordinate tables: a paper that shares its maps still tabulates.
    assert len(donors.coordinates)


def test_simulator_refuses_images_without_a_field(tmp_path):
    """The point simulator never builds a map, so there is nothing to write."""
    with pytest.raises(ValueError, match="needs simulate_field=True"):
        create_effect_size_coordinate_studyset(
            [TRUTH], n_studies=4, simulate_field=False, n_image_studies=1, image_dir=str(tmp_path)
        )
    with pytest.raises(ValueError, match="needs image_dir"):
        create_effect_size_coordinate_studyset(
            [TRUTH], n_studies=4, simulate_field=True, n_image_studies=1
        )


def test_effect_size_images_survive_conversion_to_a_dataset(studyset, small_mask):
    """A Dataset built from the collection must still carry the g images CBES needs."""
    dataset = studyset.to_dataset()
    assert {"g", "g_var"} <= set(dataset.images.columns)
    assert int(dataset.images["g"].notna().sum()) == 2
    result = CBES(mask=small_mask, null_method="none").fit(dataset)
    assert np.isfinite(value_at(result, "g"))
