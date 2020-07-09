"""
Test nimare.meta.cbma.mkda (KDA-based meta-analytic algorithms).
"""
import pytest

import nimare
from nimare.meta.cbma import mkda, kernel
from nimare.correct import FWECorrector, FDRCorrector


def test_mkda_density_kernel_instance_with_kwargs(testdata):
    """
    Smoke test for MKDADensity with a kernel transformer object, with kernel
    arguments provided, which should result in a warning, but the original
    object's parameters should remain untouched.
    """
    kern = kernel.MKDAKernel(r=2)
    meta = mkda.MKDADensity(kern, kernel__r=6)

    assert meta.kernel_transformer.get_params().get('r') == 2


def test_mkda_density_kernel_class(testdata):
    """
    Smoke test for MKDADensity with a kernel transformer class.
    """
    meta = mkda.MKDADensity(kernel.MKDAKernel, kernel__r=5)
    res = meta.fit(testdata['dset'])
    assert isinstance(res, nimare.base.MetaResult)


def test_mkda_density_kernel_instance(testdata):
    """
    Smoke test for MKDADensity with a kernel transformer object.
    """
    kern = kernel.MKDAKernel(r=5)
    meta = mkda.MKDADensity(kern)
    res = meta.fit(testdata['dset'])
    assert isinstance(res, nimare.base.MetaResult)


def test_mkda_density(testdata):
    """
    Smoke test for MKDADensity
    """
    meta = mkda.MKDADensity()
    res = meta.fit(testdata['dset'])
    corr = FWECorrector(method='montecarlo', voxel_thresh=0.001,
                        n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_mkda_chi2_fdr(testdata):
    """
    Smoke test for MKDAChi2
    """
    meta = mkda.MKDAChi2()
    res = meta.fit(testdata['dset'], testdata['dset'])
    corr = FDRCorrector(method='bh', alpha=0.001)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_mkda_chi2_fwe(testdata):
    """
    Smoke test for MKDAChi2
    """
    meta = mkda.MKDAChi2()
    res = meta.fit(testdata['dset'], testdata['dset'])
    corr = FWECorrector(method='montecarlo', n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)


def test_kda_density(testdata):
    """
    Smoke test for KDA
    """
    meta = mkda.KDA()
    res = meta.fit(testdata['dset'])
    corr = FWECorrector(method='montecarlo', n_iters=5, n_cores=1)
    cres = corr.transform(res)
    assert isinstance(res, nimare.base.MetaResult)
    assert isinstance(cres, nimare.base.MetaResult)
