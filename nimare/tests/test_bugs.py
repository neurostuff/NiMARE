"""Tests for pending bugs."""
import os

import pytest

import nimare
from nimare.utils import get_resource_path


def test_Dataset_gzipped_mask_xfail(testdata_cbma, tmp_path_factory):
    """Test that a gzipped mask file will prevent copying/saving."""
    maskfile = os.path.join(get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz")
    assert os.path.isfile(maskfile)
    tmpdir = tmp_path_factory.mktemp("test_ALE_analytic_null_unit")
    out_file = os.path.join(tmpdir, "file.pkl.gz")

    # The test dataset can be copied
    tmp_dset = testdata_cbma.copy()
    assert isinstance(tmp_dset, nimare.dataset.Dataset)
    tmp_dset.save(out_file)
    assert os.path.isfile(out_file)

    # The dataset, with a gzipped masker, cannot
    tmp_dset.masker = maskfile
    tmp_dset2 = tmp_dset.copy()

    tmp_dset.save(out_file)
