"""Test nimare.utils."""
import logging
import os
import os.path as op

import nibabel as nib
import numpy as np
import pytest

from nimare import utils


def test_find_stem():
    """Test nimare.utils.find_stem."""
    test_array = [
        "/home/data/dataset/file1.nii.gz",
        "/home/data/dataset/file2.nii.gz",
        "/home/data/dataset/file3.nii.gz",
        "/home/data/dataset/file4.nii.gz",
        "/home/data/dataset/file5.nii.gz",
    ]
    stem = utils.find_stem(test_array)
    assert stem == "/home/data/dataset/file"

    test_array = [
        "/home/data/dataset/subfolder1/file1.nii.gz",
        "/home/data/dataset/subfolder1/file2.nii.gz",
        "/home/data/dataset/subfolder2/file3.nii.gz",
        "/home/data/dataset/subfolder2/file4.nii.gz",
        "/home/data/dataset/subfolder3/file5.nii.gz",
    ]
    stem = utils.find_stem(test_array)
    assert stem == "/home/data/dataset/subfolder"

    test_array = [
        "/home/data/file1_test-filename_test.nii.gz",
        "/home/data/file2_test-filename_test.nii.gz",
        "/home/data/file3_test-filename_test.nii.gz",
        "/home/data/file4_test-filename_test.nii.gz",
        "/home/data/file5_test-filename_test.nii.gz",
    ]
    stem = utils.find_stem(test_array)
    assert stem == "/home/data/file"

    test_array = [
        "souse",
        "youse",
        "house",
        "mouse",
        "louse",
    ]
    stem = utils.find_stem(test_array)
    assert stem == ""


def test_get_template():
    """Test nimare.utils.get_template."""
    img = utils.get_template(space="mni152_1mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="mni152_1mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="mni152_1mm", mask="gm")
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="mni152_2mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="mni152_2mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="mni152_2mm", mask="gm")
    assert isinstance(img, nib.Nifti1Image)


def test_get_resource_path():
    """Test nimare.utils.get_resource_path."""
    print(utils.get_resource_path())
    assert op.isdir(utils.get_resource_path())


@pytest.mark.parametrize(
    "has_low_memory,low_memory",
    [
        (True, True),
        (False, False),
    ],
)
def test_use_memmap(caplog, has_low_memory, low_memory):
    """Test the memmapping decorator."""
    LGR = logging.getLogger(__name__)

    class DummyClass:
        def __init__(self, has_low_memory, low_memory):
            self.has_low_memory = has_low_memory
            if has_low_memory:
                self.low_memory = low_memory

        @utils.use_memmap(LGR)
        def test_decorator(self):
            assert hasattr(self, "memmap_filenames")
            if self.has_low_memory:
                assert hasattr(self, "low_memory")
                if self.low_memory:
                    assert os.path.isfile(self.memmap_filenames[0])
                else:
                    assert self.memmap_filenames[0] is None
            return self.memmap_filenames

        @utils.use_memmap(LGR)
        def bad_justin_timberlake(self):
            raise ValueError("It's gonna be may!")

    my_class = DummyClass(has_low_memory, low_memory)

    # make sure memmap file has been deleted
    my_class.test_decorator()
    first_memmap_filename = my_class.memmap_filenames[0]

    # run bad function
    with pytest.raises(ValueError):
        my_class.bad_justin_timberlake()
    assert "failed, removing" in caplog.text

    if hasattr(my_class, "low_memory") and my_class.low_memory:
        assert not os.path.isfile(first_memmap_filename)
        assert not os.path.isfile(my_class.memmap_filenames[0])
        # test when a function is called a new memmap file is created
        assert first_memmap_filename != my_class.memmap_filenames[0]


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
    true = np.array([[50.0, -86.0, -32.0], [90.0, -126.0, -72.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(utils.vox2mm(test, aff), true)


def test_mm2vox():
    """Test mm2vox."""
    test = np.array([[20, 20, 20], [0, 0, 0]])
    true = np.array([[35.0, 73.0, 46.0], [45.0, 63.0, 36.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(utils.mm2vox(test, aff), true)
