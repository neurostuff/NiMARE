"""Test nimare.transforms."""
import copy
import re

import nibabel as nib
import numpy as np
import pytest

from nimare import transforms


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


NO_OUTPUT_PATTERN = re.compile(
    (
        r"^No clusters were found for ([\w-]+) at a threshold of [0-9]+\.[0-9]+$|"
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

    if add_data:
        tst_dset = copy.deepcopy(testdata_ibma)
        tst_dset.images = transforms.transform_images(
            tst_dset.images,
            add_data,
            tst_dset.masker,
            tst_dset.metadata,
            tmp_path,
        )
    else:
        tst_dset = testdata_ibma

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

        # if someone is trying to use two-sided on a study contrast with a p map
        # raise a warning
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
        expected_studies_with_coordinates = set(tst_dset.coordinates["id"]) - set(
            studies_without_coordinates
        )
    else:
        expected_studies_with_coordinates = set(tst_dset.coordinates["id"])

    assert set(new_dset.coordinates["id"]) == expected_studies_with_coordinates


def test_images_to_coordinates_merge_strategy(testdata_ibma):
    """Test different merging strategies."""
    img2coord = transforms.ImagesToCoordinates(z_threshold=1.9)

    # keep pain_01-1, pain_02-1, and pain_03-1
    tst_dset = testdata_ibma.slice(["pain_01-1", "pain_02-1", "pain_03-1"])
    # remove coordinate data for pain_02-1
    tst_dset.coordinates = tst_dset.coordinates.query("id != 'pain_02-1'")
    # remove image data for pain_01-1
    tst_dset.images = tst_dset.images.query("id != 'pain_01-1'")

    # |  study  | image | coordinate |
    # |---------|-------|------------|
    # | pain_01 | no    | yes        |
    # | pain_02 | yes   | no         |
    # | pain_03 | yes   | yes        |

    # test 'fill' strategy
    # only pain_02 should have new data, pain_01 and pain_03 should remain the same
    img2coord.merge_strategy = "fill"
    fill_dset = img2coord.transform(tst_dset)
    # pain_01 and pain_03 should be unchaged
    assert set(fill_dset.coordinates.query("id != 'pain_02-1'")["x"]) == set(
        tst_dset.coordinates["x"]
    )
    # pain_02 should be in the coordinates now
    assert "pain_02-1" in fill_dset.coordinates["id"].unique()

    # test 'replace' strategy
    # pain_02 and pain_03 should have new data, but pain_01 should remain the same
    img2coord.merge_strategy = "replace"
    replace_dset = img2coord.transform(tst_dset)

    # pain_01 should remain the same
    assert set(replace_dset.coordinates.query("id == 'pain_01-1'")["x"]) == set(
        tst_dset.coordinates.query("id == 'pain_01-1'")["x"]
    )
    # pain_02 should be new
    assert "pain_02-1" in replace_dset.coordinates["id"].unique()
    # pain_03 should be new (and have different coordinates from the old version)
    assert set(replace_dset.coordinates.query("id == 'pain_03-1'")["x"]) != set(
        tst_dset.coordinates.query("id == 'pain_03-1'")["x"]
    )

    # test 'demolish' strategy
    # pain_01 will be removed, and pain_02, and pain_03 will be new
    img2coord.merge_strategy = "demolish"
    demolish_dset = img2coord.transform(tst_dset)

    # pain_01 should not be in the dset
    assert "pain_01-1" not in demolish_dset.coordinates["id"].unique()
    # pain_02 should be new
    assert "pain_02-1" in demolish_dset.coordinates["id"].unique()
    # pain_03 should be new (and have different coordinates from the old version)
    assert set(demolish_dset.coordinates.query("id == 'pain_03-1'")["x"]) != set(
        tst_dset.coordinates.query("id == 'pain_03-1'")["x"]
    )


@pytest.mark.parametrize(
    "z,tail,expected_p",
    [
        (0.0, "two", 1.0),
        (0.0, "one", 0.5),
        (1.959963, "two", 0.05),
        (1.959963, "one", 0.025),
        (-1.959963, "two", 0.05),
        ([0.0, 1.959963, -1.959963], "two", [1.0, 0.05, 0.05]),
    ],
)
def test_z_to_p(z, tail, expected_p):
    """Test z to p conversion."""
    p = transforms.z_to_p(z, tail)

    assert np.all(np.isclose(p, expected_p))
