"""
Test nimare.meta.ibma (image-based meta-analytic estimators).
"""
import os.path as op

from nilearn.input_data import NiftiLabelsMasker

import nimare
from nimare.meta import ibma
from nimare.correct import FDRCorrector
from ..utils import get_resource_path


def test_fishers(testdata):
    """
    Smoke test for Fisher's.
    """
    meta = ibma.Fishers()
    res = meta.fit(testdata['dset_z'])
    corr = FDRCorrector(method='indep', alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_z_perm(testdata):
    """
    Smoke test for z permutation.
    """
    meta = ibma.Stouffers(inference='rfx', null='empirical', n_iters=10)
    meta.fit(testdata['dset_z'])
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_stouffers_ffx(testdata):
    """
    Smoke test for Stouffer's FFX.
    """
    meta = ibma.Stouffers(inference='ffx', null='theoretical', n_iters=None)
    meta.fit(testdata['dset_z'])
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_stouffers_rfx(testdata):
    """
    Smoke test for Stouffer's RFX.
    """
    meta = ibma.Stouffers(inference='rfx', null='theoretical', n_iters=None)
    meta.fit(testdata['dset_z'])
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_weighted_stouffers(testdata):
    """
    Smoke test for Weighted Stouffer's.
    """
    meta = ibma.WeightedStouffers()
    meta.fit(testdata['dset_z'])
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_con_perm(testdata):
    """
    Smoke test for contrast permutation.
    """
    meta = ibma.RFX_GLM(null='empirical', n_iters=10)
    meta.fit(testdata['dset_betase'])
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_rfx_glm(testdata):
    """
    Smoke test for RFX GLM.
    """
    meta = ibma.RFX_GLM(null='theoretical', n_iters=None)
    meta.fit(testdata['dset_betase'])
    assert isinstance(meta.results, nimare.base.MetaResult)


def test_ibma_with_custom_masker(testdata):
    """ Ensure voxel-to-ROI reduction works. """
    atlas = op.join(get_resource_path(), 'atlases',
                    'HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')
    masker = NiftiLabelsMasker(atlas)
    meta = ibma.Fishers(mask=masker)
    meta.fit(testdata['dset_z'])
    assert isinstance(meta.results, nimare.base.MetaResult)
    assert meta.results.maps['z'].shape == (48, )
