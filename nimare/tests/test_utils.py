"""
Test nimare.utils (Utility functions).
"""
import os.path as op
import math

import numpy as np
import nibabel as nib

from nimare import stats, utils


def test_null_to_p():
    """
    Test nimare.utils.stats.null_to_p.
    """
    data = np.arange(1, 101)
    assert math.isclose(stats.null_to_p(5, data, 'lower'), 0.05)
    assert math.isclose(stats.null_to_p(5, data, 'upper'), 0.95)
    assert math.isclose(stats.null_to_p(5, data, 'two'), 0.1)
    assert math.isclose(stats.null_to_p(95, data, 'lower'), 0.95)
    assert math.isclose(stats.null_to_p(95, data, 'upper'), 0.05)
    assert math.isclose(stats.null_to_p(95, data, 'two'), 0.1)


def test_get_template():
    """
    Test nimare.utils.utils.get_template.
    """
    img = utils.get_template(space='mni152_1mm', mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space='mni152_1mm', mask='brain')
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space='mni152_1mm', mask='gm')
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space='mni152_2mm', mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space='mni152_2mm', mask='brain')
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space='mni152_2mm', mask='gm')
    assert isinstance(img, nib.Nifti1Image)


def test_tal2mni():
    """
    TODO: Get converted coords from official site.
    """
    test = np.array([[-44, 31, 27],
                     [20, -32, 14],
                     [28, -76, 28]])
    true = np.array([[-45.83997568, 35.97904559, 23.55194326],
                     [22.69248975, -31.34145016, 13.91284087],
                     [31.53113226, -76.61685748, 33.22105166]])
    assert np.allclose(utils.tal2mni(test), true)


def test_mni2tal():
    """
    TODO: Get converted coords from official site.
    """
    test = np.array([[-44, 31, 27],
                     [20, -32, 14],
                     [28, -76, 28]])
    true = np.array([[-42.3176, 26.0594, 29.7364],
                     [17.4781, -32.6076, 14.0009],
                     [24.7353, -75.0184, 23.3283]])
    assert np.allclose(utils.mni2tal(test), true)


def test_vox2mm():
    """
    Test vox2mm
    """
    test = np.array([[20, 20, 20],
                     [0, 0, 0]])
    true = np.array([[50., -86., -32.],
                     [90., -126., -72.]])
    img = utils.get_template(space='mni152_2mm', mask=None)
    aff = img.affine
    assert np.array_equal(utils.vox2mm(test, aff), true)


def test_mm2vox():
    """
    Test mm2vox
    """
    test = np.array([[20, 20, 20],
                     [0, 0, 0]])
    true = np.array([[35., 73., 46.],
                     [45., 63., 36.]])
    img = utils.get_template(space='mni152_2mm', mask=None)
    aff = img.affine
    assert np.array_equal(utils.mm2vox(test, aff), true)


def test_get_resource_path():
    """
    Test nimare.utils.get_resource_path
    """
    print(utils.get_resource_path())
    assert op.isdir(utils.get_resource_path())
