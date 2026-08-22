"""Test nimare.transforms."""

import logging
import re
import time

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from pymare.stats import log_chi2_sf
from scipy import stats

from nimare import transforms
from nimare.utils import DEFAULT_FLOAT_DTYPE


def test_ImageTransformer(testdata_ibma):
    """Smoke test on transforms.ImageTransformer."""
    dset = testdata_ibma
    z_files = dset.images["z"].tolist()
    z_transformer = transforms.ImageTransformer(target="z")
    new_dset = z_transformer.transform(dset)
    new_z_files = new_dset.images["z"].tolist()
    assert z_files[:-1] == new_z_files[:-1]
    # new z statistic map should have 3 dimensions
    assert len(nib.load(new_z_files[-1]).shape) == 3
    assert all([nzf is not None for nzf in new_z_files])

    varcope_files = dset.images["varcope"].tolist()
    varcope_p_transformer = transforms.ImageTransformer(target=["varcope", "p"])
    new_dset = varcope_p_transformer.transform(dset)
    new_varcope_files = new_dset.images["varcope"].tolist()
    assert not all([isinstance(vf, str) for vf in varcope_files])
    assert all([isinstance(vf, str) for vf in new_varcope_files])
    new_p_files = new_dset.images["p"].tolist()
    assert all([isinstance(pf, str) for pf in new_p_files])

    t_files = dset.images["t"].tolist()
    t_transformer = transforms.ImageTransformer(target="t")
    new_dset = t_transformer.transform(dset)
    new_t_files = new_dset.images["t"].tolist()
    assert t_files[:-1] == new_t_files[:-1]


@pytest.mark.parametrize("target", ["d", "g", "g_var"])
def test_resolve_transforms_standardized_effect_sizes(testdata_ibma, target):
    """Match the direct conversion from t and the sample size."""
    masker = testdata_ibma.masker
    row = testdata_ibma.images.iloc[0]
    sample_sizes = testdata_ibma.metadata.loc[
        testdata_ibma.metadata["id"] == row["id"], "sample_sizes"
    ].iloc[0]
    available = {"t": row["t"], "sample_sizes": sample_sizes}

    img = transforms.resolve_transforms(target, dict(available), masker)

    n = transforms.sample_sizes_to_sample_size(sample_sizes)
    d = transforms.t_to_d(masker.transform(available["t"]), n)
    g, g_var = transforms.d_to_g(d, n, return_variance=True)
    expected = {"d": d, "g": g, "g_var": g_var}[target]

    assert np.allclose(masker.transform(img), expected)


def test_resolve_transforms_reaches_g_from_z_alone(testdata_ibma):
    """A studyset with only z maps can still get to a standardized effect size, via t."""
    masker = testdata_ibma.masker
    row = testdata_ibma.images.iloc[0]
    sample_sizes = testdata_ibma.metadata.loc[
        testdata_ibma.metadata["id"] == row["id"], "sample_sizes"
    ].iloc[0]

    img = transforms.resolve_transforms("g", {"z": row["z"], "sample_sizes": sample_sizes}, masker)

    assert img is not None
    assert np.isfinite(masker.transform(img)).any()


@pytest.mark.parametrize("target", ["d", "g", "g_var"])
def test_resolve_transforms_needs_a_sample_size(testdata_ibma, target):
    """Without a sample size there is nothing to standardize by, so decline rather than guess."""
    row = testdata_ibma.images.iloc[0]

    assert transforms.resolve_transforms(target, {"t": row["t"]}, testdata_ibma.masker) is None


def test_resolve_transforms_warns_on_multiple_group_sample_sizes(testdata_ibma, caplog):
    """Warn when the one-sample conversion is applied to more than one group."""
    row = testdata_ibma.images.iloc[0]

    with caplog.at_level(logging.WARNING, logger="nimare.transforms"):
        transforms.resolve_transforms(
            "g", {"t": row["t"], "sample_sizes": [20, 20]}, testdata_ibma.masker
        )

    assert "one-sample" in caplog.text


def _p_map(masker, rng=None):
    """Write a p map over the masker's voxels, as an image object."""
    rng = np.random.RandomState(0) if rng is None else rng
    n_voxels = masker.transform(masker.mask_img).size
    return masker.inverse_transform(rng.uniform(1e-4, 1.0, size=n_voxels))


@pytest.mark.parametrize("target", ["z", "t", "g"])
def test_resolve_transforms_warns_when_z_comes_from_a_p_map(testdata_ibma, caplog, target):
    """A p map has no direction, so the z it yields is unsigned. Convert, but say so.

    The warning has to survive the recursion as well: 't', 'g' and the rest reach z first,
    and inherit its missing sign.
    """
    masker = testdata_ibma.masker
    available = {"p": _p_map(masker), "sample_sizes": [20]}

    with caplog.at_level(logging.WARNING, logger="nimare.transforms"):
        img = transforms.resolve_transforms(target, available, masker)

    assert img is not None
    assert "unsigned" in caplog.text
    assert "p map" in caplog.text
    if target == "z":
        values = masker.transform(img)
        assert (values[np.isfinite(values)] >= 0).all()


def test_resolve_transforms_takes_the_sign_from_t_without_a_sample_size(testdata_ibma, caplog):
    """A t map with no sample size cannot set the magnitude, but it can still set the sign."""
    masker = testdata_ibma.masker
    rng = np.random.RandomState(1)
    n_voxels = masker.transform(masker.mask_img).size
    t_values = rng.normal(size=n_voxels)
    t_values[:3] = [0.0, 5.0, -5.0]
    p_values = rng.uniform(1e-4, 0.9, size=n_voxels)
    available = {
        "t": masker.inverse_transform(t_values),
        "p": masker.inverse_transform(p_values),
    }

    with caplog.at_level(logging.WARNING, logger="nimare.transforms"):
        img = transforms.resolve_transforms("z", available, masker)

    z = masker.transform(img).squeeze()
    assert np.allclose(z, np.sign(t_values) * transforms.p_to_z(p_values), atol=1e-4)
    assert (z < 0).any()
    # A t of exactly 0 has no direction to give, and is a p of 1, whose z is 0 regardless.
    assert z[0] == 0
    # The map is signed, so the warning reserved for unsigned ones must stay quiet.
    assert "unsigned" not in caplog.text


def test_resolve_transforms_does_not_warn_when_z_comes_from_a_t_map(testdata_ibma, caplog):
    """The t path keeps the sign, so it must stay quiet. This is the asymmetry being fixed."""
    row = testdata_ibma.images.iloc[0]

    with caplog.at_level(logging.WARNING, logger="nimare.transforms"):
        img = transforms.resolve_transforms(
            "z", {"t": row["t"], "sample_sizes": [20]}, testdata_ibma.masker
        )

    assert img is not None
    assert "unsigned" not in caplog.text


def test_transform_images_names_the_analysis_in_the_unsigned_warning(
    testdata_ibma, caplog, tmp_path
):
    """Name the analysis, so a studyset of many maps says which one lost its sign."""
    masker = testdata_ibma.masker
    p_file = str(tmp_path / "p.nii.gz")
    _p_map(masker).to_filename(p_file)
    images_df = pd.DataFrame({"id": ["study1-1"], "p": [p_file]})

    with caplog.at_level(logging.WARNING, logger="nimare.transforms"):
        new_images_df = transforms.transform_images(
            images_df, target="z", masker=masker, out_dir=str(tmp_path)
        )

    assert new_images_df["z"].notnull().all()
    assert "study1-1: " in caplog.text
    assert "unsigned" in caplog.text


def test_transform_images(testdata_ibma):
    """Smoke test on transforms.transform_images."""
    dset = testdata_ibma
    z_files = dset.images["z"].tolist()
    new_images = transforms.transform_images(
        dset.images, target="z", masker=dset.masker, metadata_df=dset.metadata
    )
    new_z_files = new_images["z"].tolist()
    assert z_files[:-1] == new_z_files[:-1]
    # new z statistic map should have 3 dimensions
    assert len(nib.load(new_z_files[-1]).shape) == 3
    assert all([nzf is not None for nzf in new_z_files])

    varcope_files = dset.images["varcope"].tolist()
    new_images = transforms.transform_images(
        dset.images, target="varcope", masker=dset.masker, metadata_df=dset.metadata
    )
    new_varcope_files = new_images["varcope"].tolist()
    assert not all([isinstance(vf, str) for vf in varcope_files])
    assert all([isinstance(vf, str) for vf in new_varcope_files])


def test_sample_sizes_to_dof():
    """Unit tests for transforms.sample_sizes_to_dof."""
    sample_sizes = [20, 20, 20]
    dof = 57
    assert transforms.sample_sizes_to_dof(sample_sizes) == dof
    sample_sizes = [20]
    dof = 19
    assert transforms.sample_sizes_to_dof(sample_sizes) == dof


def test_sample_sizes_to_sample_size():
    """Unit tests for transforms.sample_sizes_to_sample_size."""
    sample_sizes = [20, 20, 20]
    sample_size = 60
    assert transforms.sample_sizes_to_sample_size(sample_sizes) == sample_size
    sample_sizes = [20]
    sample_size = 20
    assert transforms.sample_sizes_to_sample_size(sample_sizes) == sample_size


def test_t_to_z():
    """Smoke test for transforms.t_to_z."""
    t_arr = np.random.random(100)
    z_arr = transforms.t_to_z(t_arr, dof=20)
    assert z_arr.shape == t_arr.shape
    t_arr2 = transforms.z_to_t(z_arr, dof=20)
    assert np.allclose(t_arr, t_arr2)


def test_threshold_image_array_uses_thresholding_values():
    """threshold_image should retain image values only where thresholding values pass."""
    image = np.array([3.0, -2.0, 1.0, 4.0], dtype=np.float32)
    p_values = np.array([0.01, 0.2, 0.05, 0.049], dtype=np.float32)

    thresholded = transforms.threshold_image(
        image,
        threshold=0.05,
        thresholding_values=p_values,
        tail="lower",
    )

    np.testing.assert_array_equal(
        thresholded,
        np.array([3.0, 0.0, 1.0, 4.0], dtype=np.float32),
    )


def test_threshold_image_nifti_preserves_affine():
    """threshold_image should operate on Niimg-like inputs and preserve geometry."""
    image = nib.Nifti1Image(
        np.array([[[0.2, -0.8], [1.2, -2.4]]], dtype=np.float32),
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    thresholded = transforms.threshold_image(image, threshold=1.0, tail="two-sided")

    assert isinstance(thresholded, nib.Nifti1Image)
    np.testing.assert_array_equal(
        thresholded.get_fdata(dtype=np.float32),
        np.array([[[0.0, 0.0], [1.2, -2.4]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(thresholded.affine, image.affine)


NO_OUTPUT_PATTERN = re.compile(
    (
        r"^No clusters were found for ([\w\.0-9+-]+) at a threshold of [0-9]+\.[0-9]+$|"
        r"No Z or p map for ([\w-]+), skipping..."
    )
)


@pytest.mark.parametrize(
    "kwargs,drop_data,add_data",
    [
        ({"merge_strategy": "fill"}, "z", "p"),
        ({"merge_strategy": "replace"}, None, None),
        ({"merge_strategy": "demolish", "remove_subpeaks": True}, None, None),
        ({"merge_strategy": "fill", "two_sided": True}, "z", "p"),
        (
            {
                "merge_strategy": "demolish",
                "two_sided": True,
                "z_threshold": 1.9,
            },
            None,
            None,
        ),
        ({"merge_strategy": "demolish", "z_threshold": 10.0}, None, None),
    ],
)
def test_images_to_coordinates(tmp_path, caplog, testdata_ibma, kwargs, drop_data, add_data):
    """Test conversion of statistical images to coordinates."""
    # only catch warnings from the transforms logger
    caplog.set_level("WARNING", logger=transforms.LGR.name)

    img2coord = transforms.ImagesToCoordinates(**kwargs)
    tst_dset = testdata_ibma.copy()

    if add_data:
        tst_dset.images = transforms.transform_images(
            tst_dset.images,
            add_data,
            tst_dset.masker,
            tst_dset.metadata,
            tmp_path,
        )

    if drop_data:
        tst_dset.images = tst_dset.images.drop(columns=drop_data)

    new_dset = img2coord.transform(tst_dset)

    # metadata column "coordinate_source" should exist
    assert "coordinate_source" in new_dset.metadata.columns

    # get the studies that did not generate coordinates
    # either because the threshold was too high or
    # because there were no images to generate coordinates
    studies_without_coordinates = []
    for msg in caplog.messages:
        match = NO_OUTPUT_PATTERN.match(msg)
        if match:
            studies_without_coordinates.append(
                match.group(1) if match.group(1) else match.group(2)
            )

    # if there is not a z map for a study contrast, raise a warning
    # unless the strategy is fill since all studies already have coordinates
    if drop_data == "z" and add_data == "p" and img2coord.merge_strategy != "fill":
        assert "No Z map for" in caplog.messages[0]

        # if someone is trying to use two-sided on a study contrast with a p map, raise a warning
        if img2coord.two_sided:
            assert "Cannot use two_sided threshold using a p map for" in caplog.messages[0]

    # if two_sided was specified and z maps were used, there
    # should be peaks with negative values.
    if img2coord.two_sided and not drop_data and not add_data:
        assert np.any(new_dset.coordinates["z_stat"] < 0.0)

    # since testdata_ibma already has coordinate data for every study
    # this transformation should retain the same number of unique ids
    # unless the merge_strategy was demolish
    if img2coord.merge_strategy == "demolish":
        expected_studies_with_coordinates = set(
            tst_dset.images.loc[~tst_dset.images["z"].isnull(), "id"]
        ) - set(studies_without_coordinates)
    else:
        expected_studies_with_coordinates = set(tst_dset.coordinates["id"]).union(
            ["pain_01.nidm-1"]
        )

    assert set(new_dset.coordinates["id"]) == expected_studies_with_coordinates, set(
        new_dset.coordinates["id"]
    )


def test_ddimages_to_coordinates_merge_strategy(testdata_ibma):
    """Test different merging strategies."""
    img2coord = transforms.ImagesToCoordinates(z_threshold=1.9)

    # keep pain_01.nidm-1, pain_02.nidm-1, pain_03.nidm-1, pain_04.nidm-1
    tst_dset = testdata_ibma.slice(
        ["pain_01.nidm-1", "pain_02.nidm-1", "pain_03.nidm-1", "pain_04.nidm-1"]
    )

    # remove image data for pain_01.nidm-1 and pain_03.nidm-1
    # coordinate data for pain_01.nidm-1 and pain_02.nidm-1 are already removed
    tst_dset.images = tst_dset.images.query("id != 'pain_01.nidm-1'")
    tst_dset.images = tst_dset.images.query("id != 'pain_03.nidm-1'")

    # | study        | image | coordinate |
    # |--------------|-------|------------|
    # | pain_01.nidm | no    | no         |
    # | pain_02.nidm | yes   | no         |
    # | pain_03.nidm | no    | yes        |
    # | pain_04.nidm | yes   | yes        |

    # test 'fill' strategy
    # only pain_02.nidm should have new data.
    # pain_01.nidm, pain_03.nidm, and pain_04.nidm should remain the same
    img2coord.merge_strategy = "fill"
    fill_dset = img2coord.transform(tst_dset)
    # pain_01.nidm and pain_03.nidm should be unchanged
    assert set(fill_dset.coordinates.query("id != 'pain_02.nidm-1'")["x"]) == set(
        tst_dset.coordinates["x"]
    )
    # pain_02.nidm should be in the coordinates now
    assert "pain_02.nidm-1" in fill_dset.coordinates["id"].unique()

    # test 'replace' strategy
    # pain_02.nidm and pain_04.nidm should have new data,
    # but pain_01.nidm and pain_03.nidm should remain the same
    img2coord.merge_strategy = "replace"
    replace_dset = img2coord.transform(tst_dset)

    # pain_01.nidm should remain the same
    assert set(replace_dset.coordinates.query("id == 'pain_01.nidm-1'")["x"]) == set(
        tst_dset.coordinates.query("id == 'pain_01.nidm-1'")["x"]
    )
    # pain_02.nidm should be new
    assert "pain_02.nidm-1" in replace_dset.coordinates["id"].unique()
    # pain_03.nidm should remain the same
    assert set(replace_dset.coordinates.query("id == 'pain_03.nidm-1'")["x"]) == set(
        tst_dset.coordinates.query("id == 'pain_03.nidm-1'")["x"]
    )
    # pain_04.nidm should be new (and have different coordinates from the old version)
    assert set(replace_dset.coordinates.query("id == 'pain_04.nidm-1'")["x"]) != set(
        tst_dset.coordinates.query("id == 'pain_04.nidm-1'")["x"]
    )

    # test 'demolish' strategy
    # pain_03.nidm will be removed, and pain_02.nidm and pain_04.nidm will be new
    img2coord.merge_strategy = "demolish"
    demolish_dset = img2coord.transform(tst_dset)

    # pain_01.nidm should not be in the dset
    assert "pain_01.nidm-1" not in demolish_dset.coordinates["id"].unique()
    # pain_02.nidm should be new
    assert "pain_02.nidm-1" in demolish_dset.coordinates["id"].unique()
    # pain_03.nidm should not be in the dset
    assert "pain_03.nidm-1" not in demolish_dset.coordinates["id"].unique()
    # pain_04.nidm should be new (and have different coordinates from the old version)
    assert set(demolish_dset.coordinates.query("id == 'pain_04.nidm-1'")["x"]) != set(
        tst_dset.coordinates.query("id == 'pain_04.nidm-1'")["x"]
    )


@pytest.mark.parametrize(
    "z,tail,expected_p",
    [
        (0.0, "two", 1.0),
        (0.0, "one", 0.5),
        (1.959963, "two", 0.05),
        (1.959963, "one", 0.025),
        (-1.959963, "one", 0.975),
        (-1.959963, "two", 0.05),
        ([0.0, 1.959963, -1.959963], "two", [1.0, 0.05, 0.05]),
    ],
)
def test_z_to_p(z, tail, expected_p):
    """Test z to p conversion."""
    p = transforms.z_to_p(z, tail)

    assert np.all(np.isclose(p, expected_p))


@pytest.mark.parametrize(
    "tail,z_values,expected_equal",
    [
        ("two", np.array([50.0, -50.0]), True),
        ("one", np.array([50.0, -50.0]), False),
    ],
)
def test_z_to_p_clips_extreme_values_to_positive_floor(tail, z_values, expected_equal):
    """Extreme z-values should stay finite and avoid zero underflow across tails."""
    p = transforms.z_to_p(z_values, tail=tail)

    assert np.all(np.isfinite(p))
    assert p[0] > 0
    if expected_equal:
        assert p[1] > 0
        assert p[0] == p[1]
    else:
        assert p[1] == 1.0


def test_z_to_t_clips_extreme_tail_probabilities():
    """Extreme z-values should not produce infinite t-values from inverse CDF calls."""
    t_values = transforms.z_to_t(np.array([-50.0, 50.0]), dof=10)

    assert np.all(np.isfinite(t_values))
    assert t_values[0] < 0
    assert t_values[1] > 0


def test_t_to_z_is_unchanged_below_the_old_ceiling():
    """The log-space tail must reproduce the previous values wherever they were not capped.

    The old implementation floored its internal p-value at the machine epsilon of its dtype,
    which truncated |z| at about 8.13.
    """
    rng = np.random.default_rng(0)
    for dof in (5, 20, 100):
        t = rng.normal(0, 2.5, size=50_000)
        z = transforms.t_to_z(t, dof)
        # Matched tail probabilities, computed independently of the implementation.
        expected = np.sign(t) * stats.norm.isf(stats.t.sf(np.abs(t), dof))
        below = np.abs(z) < 8.0
        assert below.mean() > 0.99
        assert np.allclose(z[below], expected[below], atol=1e-10)


def test_t_to_z_keeps_going_past_the_old_ceiling():
    """A large t must no longer saturate, and must keep its sign."""
    t = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])

    z = transforms.t_to_z(t, 20)

    assert np.all(np.abs(z[[0, 1, 3, 4]]) > 8.13), "these all used to be clipped to 8.13"
    assert np.allclose(z, -z[::-1]), "must stay antisymmetric in t"
    assert z[2] == 0.0
    assert np.allclose(np.abs(z[[1, 4]]), [9.755, 14.630], atol=1e-3)
    # And the result is representable in the dtype the maps are stored in.
    assert np.all(np.isfinite(np.asarray(z, dtype=DEFAULT_FLOAT_DTYPE)))


@pytest.mark.parametrize("tail", ["one", "two"])
def test_nlogp_to_z_matches_p_to_z_while_p_is_representable(tail):
    """Agree with the ordinary conversion wherever the p-value is still a number."""
    p = np.array([0.5, 0.05, 1e-8, 1e-20, 1e-300])

    assert np.allclose(
        transforms.nlogp_to_z(np.log(p), tail=tail),
        transforms.p_to_z(p, tail=tail),
        atol=1e-10,
    )


def test_nlogp_to_z_passes_where_p_to_z_runs_out():
    """Past the smallest representable p, only the log-space form still carries a value."""
    # p = 1e-5000: zero in any float, so p_to_z can only return its floor.
    nlogp = -5000 * np.log(10)

    z = transforms.nlogp_to_z(nlogp, tail="one")

    assert 151 < z < 152
    assert transforms.p_to_z(np.array([0.0]), tail="one")[0] < 39
    # Round-trip through float32 storage, which holds z and -log10(p) alike.
    stored = DEFAULT_FLOAT_DTYPE(z)
    assert np.isfinite(stored)
    assert np.isclose(float(stored), z, rtol=1e-6)


@pytest.mark.parametrize("tail", ["one", "two"])
def test_z_to_nlogp_inverts_nlogp_to_z(tail):
    """The two log-space conversions must be inverses well past where a p-value dies."""
    z = np.linspace(0.0, 100.0, 501)

    assert np.allclose(transforms.nlogp_to_z(transforms.z_to_nlogp(z, tail), tail), z, atol=1e-8)


@pytest.mark.parametrize("tail", ["one", "two"])
def test_z_to_nlogp_matches_the_log_of_z_to_p(tail):
    """Agree with the ordinary conversion wherever the p-value is still a number."""
    z = np.array([-3.0, 0.0, 1.959963, 5.0, 20.0])

    assert np.allclose(
        transforms.z_to_nlogp(z, tail), np.log(transforms.z_to_p(z, tail)), atol=1e-12
    )


def test_t_to_nlogp_matches_the_tail_of_t_to_z():
    """The t tail and the z NiMARE reports from it must describe the same probability."""
    t = np.array([0.5, 2.0, 10.0, 50.0, 1000.0])

    nlogp = transforms.t_to_nlogp(t, 20, tail="one")

    assert np.allclose(nlogp, transforms.z_to_nlogp(transforms.t_to_z(t, 20), tail="one"))
    # Two tails are one added log(2), and the deep tail stays finite.
    assert np.allclose(transforms.t_to_nlogp(t, 20, tail="two"), nlogp + np.log(2.0))
    assert np.all(np.isfinite(nlogp))


@pytest.mark.parametrize("dof", [1, 2, 5, 10, 20, 50, 100, 200, 500])
def test_z_to_t_inverts_t_to_z_at_every_degree_of_freedom(dof):
    """The two directions must agree well past where either SciPy routine works.

    ``scipy.stats.t.isf`` needs a representable p-value and degrades before it runs out of
    one; ``scipy.stats.t.logsf`` underflows to -inf at high ``dof``. Both directions fall
    back to the power law the t tail becomes, so the round trip has to hold anyway.
    """
    t = np.array([-1000.0, -100.0, -20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 100.0, 1000.0])

    z = transforms.t_to_z(t, dof)
    assert np.all(np.isfinite(z)), "t_to_z used to return inf once t.logsf underflowed"

    back = transforms.z_to_t(z, dof)
    # 1% covers the narrow band where the p-value has underflowed but the expansion is not
    # yet tight; everywhere else this is 1e-11.
    assert np.allclose(back, t, rtol=1e-2)


def test_t_to_z_passes_where_the_log_survival_function_underflows():
    """``t.logsf`` returns -inf from t = 96 at dof = 500, which would put inf in a z map."""
    t = np.array([96.0, 1000.0])

    z = transforms.t_to_z(t, 500)

    assert np.all(np.isinf(stats.t.logsf(t, 500))), "fixture must be past SciPy's underflow"
    assert np.all(np.isfinite(z))
    assert np.all(np.diff(z) > 0), "must stay monotone in t"


def test_z_to_t_reaches_a_t_no_p_value_could_invert():
    """|z| = 50 on 10 dof matches a t of 8e54, which no representable p-value reaches.

    SciPy's inverse t returns 2.5e31 at best here, and infinity on some platforms.
    """
    t_values = transforms.z_to_t(np.array([-50.0, 50.0]), dof=10)

    assert np.all(np.isfinite(t_values))
    assert np.isclose(t_values[1], 8.047e54, rtol=1e-3)
    assert t_values[0] == -t_values[1]


def test_z_to_t_round_trips_past_the_old_ceiling():
    """z_to_t floored its p at machine epsilon, so t saturated from |z| = 8.13 on."""
    t = np.array([-1000.0, -100.0, -20.0, 0.0, 20.0, 100.0, 1000.0])
    z = transforms.t_to_z(t, 20)
    assert np.max(np.abs(z)) > 8.13, "fixture must reach past the old ceiling"

    assert np.allclose(transforms.z_to_t(z, 20), t, rtol=1e-6)


@pytest.mark.parametrize("dof", [1, 2, 7])
def test_chi2_to_nlogp_matches_the_general_implementation(dof):
    """The one-dof shortcut must agree with the general chi-squared tail it bypasses."""
    chi2_values = np.array([0.0, 0.5, 3.0, 100.0, 1400.0, 5000.0])

    nlogp = transforms.chi2_to_nlogp(chi2_values, dof)

    assert np.allclose(nlogp, log_chi2_sf(chi2_values, dof), atol=1e-10)
    assert np.all(nlogp <= 0)


def test_chi2_to_nlogp_passes_where_the_survival_function_underflows():
    """``chi2.logsf`` is -inf from a chi-squared of about 1416 on; this must not be."""
    chi2_values = np.array([1500.0, 1e4, 1e5])

    nlogp = transforms.chi2_to_nlogp(chi2_values, 1)

    assert np.all(np.isinf(stats.chi2.logsf(chi2_values, 1))), "fixture must be past the cliff"
    assert np.all(np.isfinite(nlogp))
    # The exact identity for one degree of freedom, computed independently.
    expected = np.log(2.0) + stats.norm.logcdf(-np.sqrt(chi2_values))
    assert np.allclose(nlogp, expected, rtol=1e-12)


def test_chi2_to_nlogp_is_cheap_for_one_degree_of_freedom():
    """The one-dof path is what makes voxelwise use affordable, so hold it to a budget.

    ``pymare.stats.log_chi2_sf`` reaches non-underflowing values through
    ``scipy.stats.chi2.logsf`` at about a microsecond each, which cost MKDAChi2 a 19x
    slowdown; going through ``erfc`` instead is about 25ns.
    """
    chi2_values = np.random.default_rng(0).chisquare(1, size=200_000)

    start = time.perf_counter()
    transforms.chi2_to_nlogp(chi2_values, 1)
    elapsed = time.perf_counter() - start

    # The general path takes ~0.4s for this many values; the budget is deliberately loose
    # enough to survive a loaded machine while still catching a regression to it.
    assert elapsed < 0.1, f"one-dof chi-squared tail took {elapsed:.3f}s for 200k values"


@pytest.mark.parametrize("tail", ["one", "two"])
def test_z_to_nlogp_agrees_with_the_log_ndtr_route(tail):
    """The erfc-based tail must match the log_ndtr one it replaced, across the join."""
    z = np.concatenate([np.linspace(-40.0, 40.0, 20001), np.array([50.0, 200.0])])

    nlogp = transforms.z_to_nlogp(z, tail)

    if tail == "two":
        expected = np.log(2.0) + stats.norm.logcdf(-np.abs(z))
    else:
        expected = stats.norm.logcdf(-z)
    assert np.all(np.isfinite(nlogp))
    assert np.allclose(nlogp, np.minimum(expected, 0.0), rtol=1e-12, atol=1e-12)
