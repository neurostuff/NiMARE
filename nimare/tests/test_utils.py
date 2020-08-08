"""
Test nimare.utils
"""
import os.path as op

import nibabel as nib

from nimare import utils


def test_get_template():
    """
    Test nimare.utils.get_template.
    """
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
    """
    Test nimare.utils.get_resource_path
    """
    print(utils.get_resource_path())
    assert op.isdir(utils.get_resource_path())
