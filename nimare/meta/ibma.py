"""
Image-based meta-analysis estimators
"""
from __future__ import division

import logging
from os import mkdir
import os.path as op
from shutil import rmtree

import numpy as np
import nibabel as nib
from scipy import stats
from nipype.interfaces import fsl
from nilearn.masking import unmask, apply_mask
import pymare

from ..base import MetaEstimator
from ..transforms import p_to_z

LGR = logging.getLogger(__name__)


class Fishers(MetaEstimator):
    """
    An image-based meta-analytic test using t- or z-statistic images.
    Requires z-statistic images, but will be extended to work with t-statistic
    images as well.

    Parameters
    ----------
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    References
    ----------
    * Fisher, R. A. (1934). Statistical methods for research workers.
      Statistical methods for research workers., (5th Ed).
      https://www.cabdirect.org/cabdirect/abstract/19351601205

    Notes
    -----
    Sum of -log P-values (from T/Zs converted to Ps)
    """
    _required_inputs = {
        'z_maps': ('image', 'z')
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        est = pymare.estimators.Fishers(input='z')
        est.fit(y=self.inputs_['z_maps'])
        est_summary = est.summary()
        results = {
            'z': est_summary.z,
            'p': est_summary.p,
        }
        return results


class Stouffers(MetaEstimator):
    """
    A t-test on z-statistic images. Requires z-statistic images.

    Parameters
    ----------
    inference : {'ffx', 'rfx'}, optional
        Whether to use fixed-effects inference (default) or random-effects
        inference.
    null : {'theoretical', 'empirical'}, optional
        Whether to use a theoretical null T distribution or an empirically-
        derived null distribution determined via sign flipping. Empirical null
        is only possible if ``inference = 'rfx'``.
    n_iters : :obj:`int` or :obj:`None`, optional
        The number of iterations to run in estimating the null distribution.
        Only used if ``inference = 'rfx'`` and ``null = 'empirical'``.
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    References
    ----------
    * Stouffer, S. A., Suchman, E. A., DeVinney, L. C., Star, S. A., &
      Williams Jr, R. M. (1949). The American Soldier: Adjustment during
      army life. Studies in social psychology in World War II, vol. 1.
      https://psycnet.apa.org/record/1950-00790-000
    """
    _required_inputs = {
        'z_maps': ('image', 'z')
    }

    def __init__(self, inference='ffx', null='theoretical', n_iters=None,
                 two_sided=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference = inference
        self.null = null
        self.n_iters = n_iters
        self.two_sided = two_sided

    def _fit(self, dataset):
        est = pymare.estimators.Stouffers(input='z')
        est.fit(y=self.inputs_['z_maps'])
        est_summary = est.summary()
        results = {
            'z': est_summary.z,
            'p': est_summary.p,
        }
        return results


class WeightedStouffers(MetaEstimator):
    """
    An image-based meta-analytic test using z-statistic images and
    sample sizes. Zs from bigger studies get bigger weights.

    Parameters
    ----------
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    References
    ----------
    * Zaykin, D. V. (2011). Optimally weighted Z‐test is a powerful method for
      combining probabilities in meta‐analysis. Journal of evolutionary
      biology, 24(8), 1836-1841.
      https://doi.org/10.1111/j.1420-9101.2011.02297.x
    """
    _required_inputs = {
        'z_maps': ('image', 'z'),
        'sample_sizes': ('metadata', 'sample_sizes')
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        sample_sizes = np.array([np.mean(n) for n in self.inputs_['sample_sizes']])
        weights = np.sqrt(sample_sizes)
        weight_maps = np.ones(self.inputs_['z_maps'].shape) * weights

        est = pymare.estimators.Stouffers(input='z')
        est.fit(y=self.inputs_['z_maps'], v=weight_maps)
        est_summary = est.summary()
        results = {
            'z': est_summary.z,
            'p': est_summary.p,
        }
        return results


class SampleSizeBased(MetaEstimator):
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
        'sample_sizes': ('metadata', 'sample_sizes')
    }

    def __init__(self, method='ml', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def _fit(self, dataset):
        sample_sizes = np.array([np.mean(n) for n in self.inputs_['sample_sizes']])
        n_maps = np.fill(self.inputs_['beta_maps'].shape) * sample_sizes

        est = pymare.estimators.SampleSizeBasedLikelihoodEstimator(method=self.method)
        est.fit(y=self.inputs_['z_maps'], n=n_maps)
        est_summary = est.summary()
        results = {
            'z': est_summary.z,
            'p': est_summary.p,
        }
        return results


class WeightedLeastSquares(MetaEstimator):
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
        'varcope_maps': ('image', 'varcope'),
    }

    def __init__(self, tau2=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau2 = tau2

    def _fit(self, dataset):
        pymare_dset = pymare.Dataset(y=self.inputs_['beta_maps'],
                                     v=self.inputs_['varcope_maps'])
        est = pymare.estimators.WeightedLeastSquares(tau2=self.tau2)
        est.fit(pymare_dset)
        est_summary = est.summary()
        results = {
            'z': est_summary.z,
            'p': est_summary.p,
        }
        return results


class Something(MetaEstimator):
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
        'varcope_maps': ('image', 'varcope'),
    }

    def __init__(self, estimator='DerSimonianLaird', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = estimator

    def _fit(self, dataset):
        if self.estimator = 'DerSimonianLaird':
            est = pymare.estimators.DerSimonianLaird()
        elif self.estimator = 'Hedges':
            est = pymare.estimators.Hedges()
        else:
            est = pymare.estimators.VarianceBasedLikelihoodEstimator()

        est.fit(y=self.inputs_['beta_maps'], v=self.inputs_['varcope_maps'])
        est_summary = est.summary()
        results = {
            'z': est_summary.z,
            'p': est_summary.p,
        }
        return results


class RandomEffectsGLM(MetaEstimator):
    """
    A t-test on contrast images. Requires contrast images.

    Parameters
    ----------
    null : {'theoretical', 'empirical'}, optional
        Whether to use a theoretical null T distribution or an empirically-
        derived null distribution determined via sign flipping.
        Default is 'theoretical'.
    n_iters : :obj:`int` or :obj:`None`, optional
        The number of iterations to run in estimating the null distribution.
        Only used if ``null = 'empirical'``.
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.
    """
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
    }

    def __init__(self, null='theoretical', n_iters=None, two_sided=True, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.null = null
        self.n_iters = n_iters
        self.two_sided = two_sided
        self.results = None

    def _fit(self, dataset):
        return rfx_glm(self.inputs_['beta_maps'], null=self.null,
                       n_iters=self.n_iters, two_sided=self.two_sided)


def rfx_glm(beta_maps, null='theoretical', n_iters=None, two_sided=True):
    """
    Run a random-effects (RFX) GLM on contrast maps.

    Parameters
    ----------
    beta_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast maps in the same space, after masking.
    null : {'theoretical', 'empirical'}, optional
        Whether to use a theoretical null T distribution or an empirically-
        derived null distribution determined via sign flipping.
        Default is 'theoretical'.
    n_iters : :obj:`int` or :obj:`None`, optional
        The number of iterations to run in estimating the null distribution.
        Only used if ``null = 'empirical'``.
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    Returns
    -------
    result : :obj:`dict`
        Dictionary object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    # Normalize contrast maps to have unit variance
    beta_maps = beta_maps / np.std(beta_maps, axis=1)[:, None]
    t_map, p_map = stats.ttest_1samp(beta_maps, popmean=0, axis=0)
    t_map[np.isnan(t_map)] = 0
    p_map[np.isnan(p_map)] = 1

    if not two_sided:
        # MATLAB one-tailed method
        p_map = stats.t.cdf(-t_map, df=beta_maps.shape[0] - 1)

    if null == 'empirical':
        k = beta_maps.shape[0]
        p_map = np.ones(t_map.shape)
        iter_t_maps = np.zeros((n_iters, t_map.shape[0]))

        data_signs = np.sign(beta_maps[beta_maps != 0])
        data_signs[data_signs < 0] = 0
        posprop = np.mean(data_signs)
        for i in range(n_iters):
            iter_beta_maps = np.copy(beta_maps)
            signs = np.random.choice(a=2, size=k, p=[1 - posprop, posprop])
            signs[signs == 0] = -1
            iter_beta_maps *= signs[:, None]
            iter_t_maps[i, :], _ = stats.ttest_1samp(iter_beta_maps, popmean=0,
                                                     axis=0)
        iter_t_maps[np.isnan(iter_t_maps)] = 0

        for voxel in range(iter_t_maps.shape[1]):
            p_map[voxel] = null_to_p(t_map[voxel], iter_t_maps[:, voxel])

        # Crop p-values of 0 or 1 to nearest values that won't evaluate to
        # 0 or 1. Prevents inf z-values.
        p_map[p_map < 1e-16] = 1e-16
        p_map[p_map > (1. - 1e-16)] = 1. - 1e-16
    elif null != 'theoretical':
        raise ValueError('Input null must be "theoretical" or "empirical".')

    # Convert p to z, preserving signs
    sign = np.sign(t_map)
    sign[sign == 0] = 1
    z_map = p_to_z(p_map, tail='two') * sign
    log_p_map = -np.log10(p_map)
    images = {'t': t_map,
              'z': z_map,
              'p': p_map,
              'logp': log_p_map}
    return images
