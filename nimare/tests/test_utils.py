"""Test nimare.utils."""
import logging
import os
import os.path as op

import nibabel as nib
import pytest
from nilearn import input_data

from nimare import utils


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
            assert hasattr(self, "memmap_filename")
            if self.has_low_memory:
                assert hasattr(self, "low_memory")
                if self.low_memory:
                    assert os.path.isfile(self.memmap_filename)
                else:
                    assert self.memmap_filename is None
            return self.memmap_filename

        @utils.use_memmap(LGR)
        def bad_justin_timberlake(self):
            raise ValueError("It's gonna be may!")

    my_class = DummyClass(has_low_memory, low_memory)

    # make sure memmap file has been deleted
    my_class.test_decorator()
    first_memmap_filename = my_class.memmap_filename

    # run bad function
    with pytest.raises(ValueError):
        my_class.bad_justin_timberlake()
    assert "failed, removing" in caplog.text

    if hasattr(my_class, "low_memory") and my_class.low_memory:
        assert not os.path.isfile(first_memmap_filename)
        assert not os.path.isfile(my_class.memmap_filename)
        # test when a function is called a new memmap file is created
        assert first_memmap_filename != my_class.memmap_filename


def test_get_masker():
    """Smoke test for get_masker."""
    img = utils.get_template(space="mni152_2mm", mask="brain")
    masker = utils.get_masker(img)
    assert isinstance(masker, input_data.NiftiMasker)
    masker_ = input_data.NiftiMasker(img)
    masker = utils.get_masker(masker_)
    assert isinstance(masker, input_data.NiftiMasker)
    masker_ = input_data.NiftiLabelsMasker(img)
    with pytest.raises(ValueError):
        masker = utils.get_masker(masker_)
