"""Test nimare.utils."""

import logging
import os
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nimare import utils
from nimare.meta.utils import _apply_liberal_mask


def test_clip_p_values_copy_parameter_controls_mutation():
    """P-value clipping should preserve inputs by default and mutate only when requested."""
    p_values = np.array([np.nan, 0.0, 1e-50, 0.5, 2.0], dtype=np.float32)
    original = p_values.copy()

    clipped = utils._clip_p_values(p_values)

    assert not np.shares_memory(clipped, p_values)
    np.testing.assert_array_equal(p_values, original)
    assert clipped[1] == utils._minimum_positive_float()
    assert clipped[2] == utils._minimum_positive_float()
    assert clipped[4] == np.float32(1.0)

    clipped_in_place = utils._clip_p_values(p_values, copy=False)

    assert np.shares_memory(clipped_in_place, p_values)
    assert p_values[1] == utils._minimum_positive_float()
    assert p_values[2] == utils._minimum_positive_float()
    assert p_values[4] == np.float32(1.0)


def test_p_to_logp_values_can_reuse_owned_array():
    """p-to-logp conversion should support in-place mutation for owned arrays."""
    p_values = np.array([1.0, 0.01, 0.0, np.nan], dtype=np.float32)

    logp_values = utils._p_to_logp_values(p_values, copy=False)

    assert np.shares_memory(logp_values, p_values)
    np.testing.assert_allclose(logp_values[:2], [0.0, 2.0], atol=1e-6)
    assert np.isfinite(logp_values[2])
    assert np.isnan(logp_values[3])


def test_description_references_support_underscore_keys():
    """Reference extraction should support citation keys with underscores."""
    bibtex = utils.get_description_references(
        "A JALE-derived workflow was used \\citep{Frahm_Monimu_Hoffstaedter}."
    )

    assert "@misc{Frahm_Monimu_Hoffstaedter" in bibtex
    assert "Juaml/Jale" in bibtex


def test_find_stem():
    """Test nimare.utils._find_stem."""
    test_array = [
        "/home/data/dataset/file1.nii.gz",
        "/home/data/dataset/file2.nii.gz",
        "/home/data/dataset/file3.nii.gz",
        "/home/data/dataset/file4.nii.gz",
        "/home/data/dataset/file5.nii.gz",
    ]
    stem = utils._find_stem(test_array)
    assert stem == "/home/data/dataset/file"

    test_array = [
        "/home/data/dataset/subfolder1/file1.nii.gz",
        "/home/data/dataset/subfolder1/file2.nii.gz",
        "/home/data/dataset/subfolder2/file3.nii.gz",
        "/home/data/dataset/subfolder2/file4.nii.gz",
        "/home/data/dataset/subfolder3/file5.nii.gz",
    ]
    stem = utils._find_stem(test_array)
    assert stem == "/home/data/dataset/subfolder"

    test_array = [
        "/home/data/file1_test-filename_test.nii.gz",
        "/home/data/file2_test-filename_test.nii.gz",
        "/home/data/file3_test-filename_test.nii.gz",
        "/home/data/file4_test-filename_test.nii.gz",
        "/home/data/file5_test-filename_test.nii.gz",
    ]
    stem = utils._find_stem(test_array)
    assert stem == "/home/data/file"

    test_array = [
        "souse",
        "youse",
        "house",
        "mouse",
        "louse",
    ]
    stem = utils._find_stem(test_array)
    assert stem == ""


def test_get_template():
    """Test nimare.utils.get_template."""
    # 1mm template
    img = utils.get_template(space="mni152_1mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    assert not nib.is_proxy(img.dataobj)
    img = utils.get_template(space="mni152_1mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)

    # 2mm template (default)
    img = utils.get_template(space="mni152_2mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="mni152_2mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)
    assert not nib.is_proxy(img.dataobj)

    # ALE template
    img = utils.get_template(space="ale_2mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="ale_2mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)
    assert not nib.is_proxy(img.dataobj)

    # Expect exceptions when incompatible spaces or masks are requested.
    with pytest.raises(ValueError):
        utils.get_template(space="something", mask=None)

    with pytest.raises(ValueError):
        utils.get_template(space="mni152_1mm", mask="gm")

    with pytest.raises(ValueError):
        utils.get_template(space="mni152_2mm", mask="gm")

    with pytest.raises(ValueError):
        utils.get_template(space="ale_2mm", mask="gm")


def test_mask_coverage_gm_uses_probability_template_with_binary_masker():
    """GM null space should not rely on binary analysis-mask intensities."""
    masker = utils.get_masker(utils.get_template(space="mni152_2mm", mask="brain"))

    brain_ijk = utils._mask_coverage_to_null_ijk(masker, mask_coverage="brain")
    gm_ijk = utils._mask_coverage_to_null_ijk(masker, mask_coverage="gm")

    assert 0 < gm_ijk.shape[0] < brain_ijk.shape[0]

    brain_mask = utils._mask_coverage_to_mask(masker, mask_coverage="brain")
    gm_mask = utils._mask_coverage_to_mask(masker, mask_coverage="gm")
    assert np.all(brain_mask[gm_mask])


def test_get_resource_path():
    """Test nimare.utils.get_resource_path."""
    print(utils.get_resource_path())
    assert op.isdir(utils.get_resource_path())


@pytest.mark.parametrize(
    "has_low_memory,memory_limit",
    [
        (True, "1gb"),
        (False, None),
    ],
)
def test_use_memmap(caplog, has_low_memory, memory_limit):
    """Test the memmapping decorator."""
    LGR = logging.getLogger(__name__)

    class DummyClass:
        def __init__(self, has_low_memory, memory_limit):
            self.has_low_memory = has_low_memory
            self.memory_limit = memory_limit

        @utils.use_memmap(LGR)
        def test_decorator(self):
            assert hasattr(self, "memmap_filenames")
            if self.has_low_memory:
                assert hasattr(self, "memory_limit")
                if self.memory_limit:
                    assert os.path.isfile(self.memmap_filenames[0])
                else:
                    assert self.memmap_filenames[0] is None
            return self.memmap_filenames

        @utils.use_memmap(LGR)
        def bad_justin_timberlake(self):
            raise ValueError("It's gonna be may!")

    my_class = DummyClass(has_low_memory, memory_limit)

    # make sure memmap file has been deleted
    my_class.test_decorator()
    first_memmap_filename = my_class.memmap_filenames[0]

    # run bad function
    with pytest.raises(ValueError):
        my_class.bad_justin_timberlake()
    assert "failed, removing" in caplog.text

    if hasattr(my_class, "memory_limit") and my_class.memory_limit:
        assert not os.path.isfile(first_memmap_filename)
        assert not os.path.isfile(my_class.memmap_filenames[0])
        # test when a function is called a new memmap file is created
        assert first_memmap_filename != my_class.memmap_filenames[0]


def test_validate_images_df_preserves_existing_relative_columns():
    """Absolute image columns should not duplicate an existing relative column."""
    image_df = pd.DataFrame(
        {
            "id": ["study-1"],
            "study_id": ["study"],
            "contrast_id": ["1"],
            "beta": ["C:/Users/runneradmin/nimare/tests/data/orig/study_beta.nii.gz"],
            "beta__relative": ["orig/study_beta.nii.gz"],
        }
    )

    validated = utils._validate_images_df(image_df)

    assert validated.columns.tolist().count("beta__relative") == 1
    assert validated.loc[0, "beta"] == image_df.loc[0, "beta"]
    assert validated.loc[0, "beta__relative"] == "orig/study_beta.nii.gz"


def test_tal2mni():
    """TODO: Get converted coords from official site."""
    test = np.array([[-44, 31, 27], [20, -32, 14], [28, -76, 28]])
    true = np.array(
        [
            [-45.83997568, 35.97904559, 23.55194326],
            [22.69248975, -31.34145016, 13.91284087],
            [31.53113226, -76.61685748, 33.22105166],
        ]
    )
    assert np.allclose(utils.tal2mni(test), true)


def test_mni2tal():
    """TODO: Get converted coords from official site."""
    test = np.array([[-44, 31, 27], [20, -32, 14], [28, -76, 28]])
    true = np.array(
        [[-42.3176, 26.0594, 29.7364], [17.4781, -32.6076, 14.0009], [24.7353, -75.0184, 23.3283]]
    )
    assert np.allclose(utils.mni2tal(test), true)


def test_vox2mm():
    """Test vox2mm."""
    test = np.array([[20, 20, 20], [0, 0, 0]])
    true = np.array([[-50.0, -86.0, -32.0], [-90.0, -126.0, -72.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(utils.vox2mm(test, aff), true)


def test_mm2vox():
    """Test mm2vox."""
    test = np.array([[20, 20, 20], [0, 0, 0]])
    true = np.array([[55.0, 73.0, 46.0], [45.0, 63.0, 36.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(utils.mm2vox(test, aff), true)


def test_apply_liberal_mask():
    """Test _apply_liberal_mask."""
    data = np.array([[1, 2, np.nan, np.nan], [4, np.nan, 6, 5], [0, 8, 9, 3]])
    true_data = [np.array([[1], [4]]), np.array([[2], [8]]), np.array([[6, 5], [9, 3]])]

    pred_data, _, _ = _apply_liberal_mask(data)

    assert len(pred_data) == len(true_data)

    for pred_val, true_val in zip(pred_data, true_data):
        assert np.array_equal(pred_val, true_val)
