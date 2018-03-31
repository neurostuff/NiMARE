"""
Image-based meta-analysis estimators
"""
from __future__ import division

import numpy as np
from scipy import stats

from .base import IBMAEstimator
from ..base import MetaResult
from ...due import due, Doi, BibTeX


@due.dcite(BibTeX("""
           @article{stouffer1949american,
             title={The American soldier: Adjustment during army life.(Studies
                    in social psychology in World War II), Vol. 1},
             author={Stouffer, Samuel A and Suchman, Edward A and DeVinney,
                     Leland C and Star, Shirley A and Williams Jr, Robin M},
             year={1949},
             publisher={Princeton Univ. Press}
             }
           """),
           description='Stouffers citation.')
def stouffers(z_maps, mask, inference='ffx', null='theoretical', n_iters=None):
    """
    Run a Stouffer's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
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

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    if inference == 'rfx':
        if null == 'theoretical':
            t_map, p_map = stats.ttest_1samp(z_maps, popmean=0, axis=0)
        elif null == 'empirical':
            k = z_maps.shape[0]
            t_map, _ = stats.ttest_1samp(z_maps, popmean=0, axis=0)
            p_map = np.ones(t_map.shape)
            iter_t_maps = np.zeros((n_iters, t_map.shape[0]))

            data_signs = np.sign(z_maps[z_maps != 0])
            data_signs[data_signs < 0] = 0
            posprop = np.mean(data_signs)
            for i in range(n_iters):
                iter_z_maps = np.copy(z_maps)
                signs = np.random.choice(a=2, size=k, p=[1-posprop, posprop])
                signs[signs == 0] = -1
                iter_z_maps *= signs[:, None]
                iter_t_maps[i, :], _ = stats.ttest_1samp(iter_z_maps, popmean=0, axis=0)

            for voxel in range(iter_t_maps.shape[0]):
                p_value = (50 - np.abs(stats.percentileofscore(iter_t_maps[:, voxel], t_map[voxel]) - 50.)) * 2. / 100.
                p_map[voxel] = p_value
        else:
            raise ValueError('Input null must be "theoretical" or "empirical".')

        log_p_map = -np.log10(p_map)
        result = MetaResult(mask=mask, t=t_map, p=p_map, log_p=log_p_map)
    elif inference == 'ffx':
        if null == 'theoretical':
            k = z_maps.shape[0]
            z_map = np.sum(z_maps, axis=0) / np.sqrt(k)
            p_map = stats.norm.cdf(z_map, loc=0, scale=1)
            log_p_map = -np.log10(p_map)
            result = MetaResult(mask=mask, z=z_map, p=p_map, log_p=log_p_map)
        else:
            raise ValueError('Only theoretical null distribution may be used '
                             'for FFX Stouffers.')
    else:
        raise ValueError('Input inference must be "rfx" or "ffx".')
    return result


class Stouffers(IBMAEstimator):
    """
    A t-test on z-statistic images.

    Parameters
    ----------
    dataset : :obj:`nimare.dataset.Dataset`
        Dataset to analyze.
    inference : {'rfx', 'ffx'}
        Whether to run a random- or fixed-effects model.
    null : {'theoretical', 'empirical'}
        Whether to compare test statistics to theoretical or empirical null
        distribution. Empirical null distribution is only possible when
        inference is set to 'rfx'.

    Requirements:
        - z
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.inference = None
        self.null = None
        self.n_iters = None

    def fit(self, inference='ffx', null='theoretical', n_iters=None):
        self.inference = inference
        self.null = null
        self.n_iters = n_iters
        z_maps = self.dataset.get(self.ids, 'z')
        result = stouffers(z_maps, self.mask, inference=inference, null=null,
                           n_iters=n_iters)
        return result


@due.dcite(BibTeX("""
           @article{fisher1932statistical,
              title={Statistical methods for research workers, Edinburgh: Oliver and Boyd, 1925},
              author={Fisher, RA},
              journal={Google Scholar},
              year={1932}
              }
           """),
           description='Fishers citation.')
def fishers(z_maps, mask):
    """
    Run a Fisher's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    k = z_maps.shape[0]
    ffx_stat_map = -2 * np.sum(np.log10(stats.norm.cdf(-z_maps, loc=0, scale=1)),
                               axis=0)
    p_map = stats.chi2.sf(ffx_stat_map, 2*k)
    log_p_map = -np.log10(p_map)
    result = MetaResult(mask=mask, ffx_stat=ffx_stat_map, p=p_map, log_p=log_p_map)
    return result


class Fishers(IBMAEstimator):
    """
    An image-based meta-analytic test using t- or z-statistic images.
    Sum of -log P-values (from T/Zs converted to Ps)

    Requirements:
        - t OR z
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids

    def fit(self):
        z_maps = self.dataset.get(self.ids, 'z')
        result = fishers(z_maps, self.mask)
        return result


@due.dcite(BibTeX("""
           @article{zaykin2011optimally,
             title={Optimally weighted Z-test is a powerful method for combining
                    probabilities in meta-analysis},
             author={Zaykin, Dmitri V},
             journal={Journal of evolutionary biology},
             volume={24},
             number={8},
             pages={1836--1841},
             year={2011},
             publisher={Wiley Online Library}
             }
           """),
           description='Weighted Stouffers citation.')
def weighted_stouffers(z_maps, sample_sizes, mask):
    """
    Run a Stouffer's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    sample_sizes : (n_contrasts,) :obj:`numpy.ndarray`
        A 1D array of sample sizes associated with contrasts in ``z_maps``.
        Must be in same order as rows in ``z_maps``.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    assert z_maps.shape[0] == sample_sizes.shape[0]
    weighted_z_maps = z_maps * np.sqrt(sample_sizes)[:, None]
    ffx_stat_map = np.sum(weighted_z_maps, axis=0) / np.sqrt(np.sum(sample_sizes))
    p_map = stats.norm.cdf(-ffx_stat_map, loc=0, scale=1)
    log_p_map = -np.log10(p_map)
    result = MetaResult(mask=mask, ffx_stat=ffx_stat_map, p=p_map, log_p=log_p_map)
    return result


class WeightedStouffers(IBMAEstimator):
    """
    An image-based meta-analytic test using z-statistic images and sample sizes.
    Zs from bigger studies get bigger weight

    Requirements:
        - z
        - n
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids

    def fit(self):
        z_maps = self.dataset.get(self.ids, 'z')
        sample_sizes = self.dataset.get(self.ids, 'n')
        result = weighted_stouffers(z_maps, sample_sizes, self.mask)
        return result


def rfx_glm(con_maps, mask, null='theoretical', n_iters=None):
    """
    Run a random-effects (RFX) GLM on contrast maps.

    Parameters
    ----------
    con_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast maps in the same space, after masking.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
    null : {'theoretical', 'empirical'}, optional
        Whether to use a theoretical null T distribution or an empirically-
        derived null distribution determined via sign flipping.
    n_iters : :obj:`int` or :obj:`None`, optional
        The number of iterations to run in estimating the null distribution.
        Only used if ``null = 'empirical'``.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    # Normalize contrast maps to have unit variance
    con_maps = con_maps / np.std(con_maps, axis=1)[:, None]
    if null == 'theoretical':
        t_map, p_map = stats.ttest_1samp(con_maps, popmean=0, axis=0)
    elif null == 'empirical':
        k = con_maps.shape[0]
        t_map, _ = stats.ttest_1samp(con_maps, popmean=0, axis=0)
        p_map = np.ones(t_map.shape)
        iter_t_maps = np.zeros((n_iters, t_map.shape[0]))

        data_signs = np.sign(con_maps[con_maps != 0])
        data_signs[data_signs < 0] = 0
        posprop = np.mean(data_signs)
        for i in range(n_iters):
            iter_con_maps = np.copy(con_maps)
            signs = np.random.choice(a=2, size=k, p=[1-posprop, posprop])
            signs[signs == 0] = -1
            iter_con_maps *= signs[:, None]
            iter_t_maps[i, :], _ = stats.ttest_1samp(iter_con_maps, popmean=0, axis=0)

        for voxel in range(iter_t_maps.shape[0]):
            p_value = (50 - np.abs(stats.percentileofscore(iter_t_maps[:, voxel], t_map[voxel]) - 50.)) * 2. / 100.
            p_map[voxel] = p_value
    else:
        raise ValueError('Input null must be "theoretical" or "empirical".')
    log_p_map = -np.log10(p_map)
    result = MetaResult(mask=mask, t=t_map, p=p_map, log_p=log_p_map)
    return result


class RFX_GLM(IBMAEstimator):
    """
    A t-test on contrast images.

    Requirements:
        - con
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.null = None
        self.n_iters = None

    def fit(self, null='theoretical', n_iters=None):
        self.null = null
        self.n_iters = n_iters
        con_maps = self.dataset.get(self.ids, 'con')
        result = rfx_glm(con_maps, self.mask, null=self.null, n_iters=self.n_iters)
        return result


def ffx_glm(con_maps, var_maps, sample_sizes, mask, equal_var=True):
    """
    Run a Stouffer's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    con_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast maps in the same space, after masking.
    var_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast standard error maps in the same space, after
        masking. Must match shape and order of ``con_maps``.
    sample_sizes : (n_contrasts,) :obj:`numpy.ndarray`
        A 1D array of sample sizes associated with contrasts in ``con_maps``
        and ``var_maps``. Must be in same order as rows in ``con_maps`` and
        ``var_maps``.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
    equal_var : :obj:`bool`, optional
        Whether equal variance is assumed across contrasts. Default is True.
        False is not yet implemented.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    assert con_maps.shape[1] == var_maps.shape[1]
    assert con_maps.shape[0] == var_maps.shape[0] == sample_sizes.shape[0]
    if equal_var:
        weighted_con_maps = con_maps * sample_sizes[:, None]
        sum_weighted_con_map = np.sum(weighted_con_maps, axis=0)
        adj_con_map = (1. / np.sqrt(np.sum(sample_sizes))) * sum_weighted_con_map
        weighted_ss_maps = var_maps * (sample_sizes[:, None] - 1)
        sum_weighted_ss_map = np.sum(weighted_ss_maps, axis=0)
        est_ss_map = (1. / np.sqrt(np.sum(sample_sizes - 1))) * sum_weighted_ss_map
        ffx_stat_map = adj_con_map / np.sqrt(est_ss_map)
        dof = np.sum(sample_sizes - 1)
    else:
        raise Exception('Unequal variances not available yet.')

    p_map = stats.t.cdf(-ffx_stat_map, df=dof, loc=0, scale=1)
    log_p_map = -np.log10(p_map)
    result = MetaResult(mask=mask, ffx_stat=ffx_stat_map, p=p_map, log_p=log_p_map)
    return result


class FFX_GLM(IBMAEstimator):
    """
    An image-based meta-analytic test using contrast and standard error images.
    Don't estimate variance, just take from first level

    Requirements:
        - con
        - se
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.sample_sizes = None
        self.equal_var = None

    def fit(self, sample_sizes=None, equal_var=True):
        """
        Perform meta-analysis given parameters.
        """
        self.sample_sizes = sample_sizes
        self.equal_var = equal_var
        con_maps = self.dataset.get(self.ids, 'con')
        var_maps = self.dataset.get(self.ids, 'con_se')
        k = con_maps.shape[0]
        if self.sample_sizes is not None:
            sample_sizes = np.repeat(self.sample_sizes, k)
        else:
            sample_sizes = self.dataset.get(self.ids, 'n')
        result = ffx_glm(con_maps, var_maps, sample_sizes, self.mask, equal_var=self.equal_var)
        return result


class MFX_GLM(IBMAEstimator):
    """
    The gold standard image-based meta-analytic test. Uses contrast and standard
    error images.

    Requirements:
        - con
        - se
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids

    def fit(self):
        con_maps = self.dataset.get(self.ids, 'con')
        var_maps = self.dataset.get(self.ids, 'con_se')
