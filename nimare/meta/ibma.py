"""
Image-based meta-analysis estimators
"""
from __future__ import division

import logging

import numpy as np
from scipy import stats
import pymare

from ..base import MetaEstimator
from ..transforms import p_to_z
from ..stats import null_to_p

LGR = logging.getLogger(__name__)


class Fishers(MetaEstimator):
    """
    An image-based meta-analytic test using t- or z-statistic images.
    Requires z-statistic images, but will be extended to work with t-statistic
    images as well.

    Notes
    -----
    Requires ``z`` images.

    Warning
    -------
    This method does not currently calculate p-values correctly. Do not use.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Fisher, R. A. (1934). Statistical methods for research workers.
      Statistical methods for research workers., (5th Ed).
      https://www.cabdirect.org/cabdirect/abstract/19351601205

    See also
    --------
    :class:`pymare.estimators.Fishers`:
        The PyMARE estimator called by this class.
    """
    _required_inputs = {
        'z_maps': ('image', 'z')
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        pymare_dset = pymare.Dataset(y=self.inputs_['z_maps'])
        est = pymare.estimators.Fishers(input='z')
        est.fit(pymare_dset)
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
    use_sample_size : :obj:`bool`, optional
        Whether to use sample sizes for weights (i.e., "weighted Stouffer's")
        or not. Default is False.

    Notes
    -----
    Requires ``z`` images and optionally the sample size metadata field.

    Warning
    -------
    This method does not currently calculate p-values correctly. Do not use.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Stouffer, S. A., Suchman, E. A., DeVinney, L. C., Star, S. A., &
      Williams Jr, R. M. (1949). The American Soldier: Adjustment during
      army life. Studies in social psychology in World War II, vol. 1.
      https://psycnet.apa.org/record/1950-00790-000
    * Zaykin, D. V. (2011). Optimally weighted Z‐test is a powerful method for
      combining probabilities in meta‐analysis. Journal of evolutionary
      biology, 24(8), 1836-1841.
      https://doi.org/10.1111/j.1420-9101.2011.02297.x

    See also
    --------
    :class:`pymare.estimators.Stouffers`:
        The PyMARE estimator called by this class.
    """
    _required_inputs = {
        'z_maps': ('image', 'z'),
    }

    def __init__(self, use_sample_size=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sample_size = use_sample_size
        if self.use_sample_size:
            self._required_inputs['sample_sizes'] = ('metadata', 'sample_sizes')

    def _fit(self, dataset):
        if self.use_sample_size:
            sample_sizes = np.array([np.mean(n) for n in self.inputs_['sample_sizes']])
            weights = np.sqrt(sample_sizes)
            weight_maps = np.tile(weights, (self.inputs_['z_maps'].shape[1], 1)).T
            pymare_dset = pymare.Dataset(y=self.inputs_['z_maps'], v=weight_maps)
        else:
            pymare_dset = pymare.Dataset(y=self.inputs_['z_maps'])

        est = pymare.estimators.Stouffers(input='z')
        est.fit(pymare_dset)
        est_summary = est.summary()
        results = {
            'z': est_summary.z,
            'p': est_summary.p,
        }
        return results


class WeightedLeastSquares(MetaEstimator):
    """
    Weighted least-squares meta-regression.

    Provides the weighted least-squares estimate of the fixed effects given
    known/assumed between-study variance tau^2.
    When tau^2 = 0 (default), the model is the standard inverse-weighted
    fixed-effects meta-regression.

    Parameters
    ----------
    tau2 : :obj:`float` or 1D :class:`numpy.ndarray`, optional
        Assumed/known value of tau^2. Must be >= 0. Default is 0.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warning
    -------
    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Brockwell, S. E., & Gordon, I. R. (2001). A comparison of statistical
      methods for meta-analysis. Statistics in Medicine, 20(6), 825–840.
      https://doi.org/10.1002/sim.650

    See also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """
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
            'tau2': est_summary.tau2,
            'z': est_summary.get_fe_stats()['z'],
            'p': est_summary.get_fe_stats()['p'],
            'est': est_summary.get_fe_stats()['est'],
        }
        return results


class DerSimonianLaird(MetaEstimator):
    """
    DerSimonian-Laird meta-regression estimator.

    Estimates the between-subject variance tau^2 using the DerSimonian-Laird
    (1986) method-of-moments approach.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warning
    -------
    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
      Controlled clinical trials, 7(3), 177-188.
    * Kosmidis, I., Guolo, A., & Varin, C. (2017). Improving the accuracy of
      likelihood-based inference in meta-analysis and meta-regression.
      Biometrika, 104(2), 489–496. https://doi.org/10.1093/biomet/asx001

    See also
    --------
    :class:`pymare.estimators.DerSimonianLaird`:
        The PyMARE estimator called by this class.
    """
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
        'varcope_maps': ('image', 'varcope'),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        est = pymare.estimators.DerSimonianLaird()
        pymare_dset = pymare.Dataset(y=self.inputs_['beta_maps'],
                                     v=self.inputs_['varcope_maps'])
        est.fit(pymare_dset)
        est_summary = est.summary()
        results = {
            'tau2': est_summary.tau2,
            'z': est_summary.get_fe_stats()['z'],
            'p': est_summary.get_fe_stats()['p'],
            'est': est_summary.get_fe_stats()['est'],
        }
        return results


class Hedges(MetaEstimator):
    """
    Hedges meta-regression estimator.

    Estimates the between-subject variance tau^2 using the Hedges & Olkin (1985)
    approach.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    Warning
    -------
    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * Hedges LV, Olkin I. 1985. Statistical Methods for Meta‐Analysis.

    See also
    --------
    :class:`pymare.estimators.Hedges`:
        The PyMARE estimator called by this class.
    """
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
        'varcope_maps': ('image', 'varcope'),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, dataset):
        est = pymare.estimators.Hedges()
        pymare_dset = pymare.Dataset(y=self.inputs_['beta_maps'],
                                     v=self.inputs_['varcope_maps'])
        est.fit(pymare_dset)
        est_summary = est.summary()
        results = {
            'tau2': est_summary.tau2,
            'z': est_summary.get_fe_stats()['z'],
            'p': est_summary.get_fe_stats()['p'],
            'est': est_summary.get_fe_stats()['est'],
        }
        return results


class SampleSizeBasedLikelihood(MetaEstimator):
    """
    Likelihood-based estimator for estimates with known sample sizes but
    unknown sampling variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Parameters
    ----------
    method : {'ml', 'reml'}, optional
        The estimation method to use.
        Either 'ml' (for maximum-likelihood) or 'reml'
        (restricted maximum-likelihood). Default is 'ml'.

    Notes
    -----
    Requires ``beta`` images and sample size from metadata.

    Homogeneity of sigma^2 across studies is assumed.
    The ML and REML solutions are obtained via SciPy’s scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warning
    -------
    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    See also
    --------
    :class:`pymare.estimators.SampleSizeBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
        'sample_sizes': ('metadata', 'sample_sizes')
    }

    def __init__(self, method='ml', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def _fit(self, dataset):
        sample_sizes = np.array([np.mean(n) for n in self.inputs_['sample_sizes']])
        n_maps = np.tile(sample_sizes, (self.inputs_['beta_maps'].shape[1], 1)).T
        pymare_dset = pymare.Dataset(y=self.inputs_['beta_maps'], n=n_maps)
        est = pymare.estimators.SampleSizeBasedLikelihoodEstimator(method=self.method)
        est.fit(pymare_dset)
        est_summary = est.summary()
        results = {
            'tau2': est_summary.tau2,
            'z': est_summary.get_fe_stats()['z'],
            'p': est_summary.get_fe_stats()['p'],
            'est': est_summary.get_fe_stats()['est'],
        }
        return results


class VarianceBasedLikelihood(MetaEstimator):
    """
    A likelihood-based meta-analysis method for estimates with known variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    coefficients using the specified likelihood-based estimator (ML or REML).

    Parameters
    ----------
    method : {'ml', 'reml'}, optional
        The estimation method to use.
        Either 'ml' (for maximum-likelihood) or 'reml'
        (restricted maximum-likelihood). Default is 'ml'.

    Notes
    -----
    Requires ``beta`` and ``varcope`` images.

    The ML and REML solutions are obtained via SciPy’s scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warning
    -------
    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.

    References
    ----------
    * DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
      Controlled clinical trials, 7(3), 177-188.
    * Kosmidis, I., Guolo, A., & Varin, C. (2017). Improving the accuracy of
      likelihood-based inference in meta-analysis and meta-regression.
      Biometrika, 104(2), 489–496. https://doi.org/10.1093/biomet/asx001

    See also
    --------
    :class:`pymare.estimators.VarianceBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """
    _required_inputs = {
        'beta_maps': ('image', 'beta'),
        'varcope_maps': ('image', 'varcope'),
    }

    def __init__(self, method='ml', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def _fit(self, dataset):
        est = pymare.estimators.VarianceBasedLikelihoodEstimator(method=self.method)

        pymare_dset = pymare.Dataset(y=self.inputs_['beta_maps'],
                                     v=self.inputs_['varcope_maps'])
        est.fit(pymare_dset)
        est_summary = est.summary()
        results = {
            'tau2': est_summary.tau2,
            'z': est_summary.get_fe_stats()['z'],
            'p': est_summary.get_fe_stats()['p'],
            'est': est_summary.get_fe_stats()['est'],
        }
        return results


class TTest(MetaEstimator):
    """
    A t-test on contrast images, with optional empirical null distribution.
    Requires contrast images.

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

    Notes
    -----
    Requires ``beta`` images.

    Warning
    -------
    All image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis.
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
        return t_test(self.inputs_['beta_maps'], null=self.null,
                      n_iters=self.n_iters, two_sided=self.two_sided)


def t_test(beta_maps, null='theoretical', n_iters=None, two_sided=True):
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
