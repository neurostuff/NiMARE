"""
Image-based meta-analysis estimators
"""
from __future__ import division

import numpy as np
from scipy import stats

from .base import IBMAEstimator
from ..base import MetaResult


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
    def __init__(self, dataset, thresh=0.05, inference='ffx', null='theoretical', n_iters=None):
        self.sample = None
        self.mask = None
        self.dataset = dataset
        self.thresh = thresh
        self.inference = inference
        self.null = null
        self.n_iters = n_iters

    def fit(self, sample):
        if self.inference == 'rfx':
            if self.null == 'theoretical':
                z_maps = sample.get('z')
                t_map, p_map = stats.ttest_1samp(z_maps, popmean=0, axis=0)
                log_p_map = -np.log10(p_map)
                result = MetaResult(t=t_map, p=p_map, log_p=log_p_map, mask=self.mask)
            elif self.null == 'empirical':
                n_iters = self.n_iters
                z_maps = sample.get('z')
                k = z_maps.shape[0]
                t_map, _ = stats.ttest_1samp(z_maps, popmean=0, axis=0)
                p_map = np.ones(t_map.shape)
                iter_t_maps = np.zeros((n_iters, t_map.shape[0]))
                for i in n_iters:
                    iter_z_maps = np.copy(z_maps)
                    signs = np.random.choice(a=2, size=k, p=[0.5, 0.5])
                    signs[signs == 0] = -1
                    iter_z_maps *= signs[:, None]
                    iter_t_maps[i, :], _ = stats.ttest_1samp(iter_z_maps, popmean=0, axis=0)

                for voxel in range(iter_t_maps.shape[0]):
                    voxel_perc = stats.percentileofscore(iter_t_maps[:, voxel], t_map[voxel]) / 100.
                    p_map[voxel] = 1. - (np.abs(voxel_perc - 0.5) * 2)
                log_p_map = -np.log10(p_map)
                result = MetaResult(t=t_map, p=p_map, log_p=log_p_map, mask=self.mask)
            else:
                raise ValueError('Input null must be "theoretical" or "empirical".')
        elif self.inference == 'ffx':
            if self.null == 'theoretical':
                z_maps = self.sample.get('z')
                k = z_maps.shape[0]
                z_map = np.sum(z_maps, axis=0) / np.sqrt(k)
                p_map = stats.norm.cdf(z_map, loc=0, scale=1)
                log_p_map = -np.log10(p_map)
                result = MetaResult(z=z_map, p=p_map, log_p=log_p_map, mask=self.mask)
            else:
                raise ValueError('Only theoretical null distribution may be used '
                                 'for FFX Stouffers.')
        else:
            raise ValueError('Input inference must be "rfx" or "ffx".')
        return result


class Fishers(IBMAEstimator):
    """
    An image-based meta-analytic test using t- or z-statistic images.
    Sum of -log P-values (from T/Zs converted to Ps)

    Requirements:
        - t OR z
    """
    def __init__(self, dataset, thresh=0.05):
        self.sample = None
        self.mask = None
        self.dataset = dataset
        self.thresh = thresh

    def fit(self, sample):
        z_maps = sample.get('z')
        k = z_maps.shape[0]
        ffx_stat_map = -2 * np.sum(np.log10(stats.norm.cdf(-z_maps, loc=0,
                                                           scale=1)), axis=0)
        p_map = stats.chi2.cdf(ffx_stat_map, 2*k)
        log_p_map = -np.log10(p_map)
        result = MetaResult(ffx_stat=ffx_stat_map, p=p_map, log_p=log_p_map, mask=self.mask)
        return result


class WeightedStouffers(IBMAEstimator):
    """
    An image-based meta-analytic test using z-statistic images and sample sizes.
    Zs from bigger studies get bigger weight

    Requirements:
        - z
        - n
    """
    def __init__(self, dataset, thresh=0.05):
        self.sample = None
        self.mask = None
        self.dataset = dataset
        self.thresh = thresh

    def fit(self, sample):
        z_maps = sample.get('z')
        sample_sizes = sample.get('n')
        weighted_z_maps = z_maps * np.sqrt(sample_sizes)[:, None]
        ffx_stat_map = np.sum(weighted_z_maps, axis=0) / np.sqrt(np.sum(sample_sizes))
        p_map = stats.norm.cdf(-ffx_stat_map, loc=0, scale=1)
        log_p_map = -np.log10(p_map)
        result = MetaResult(ffx_stat=ffx_stat_map, p=p_map, log_p=log_p_map, mask=self.mask)
        return result


class RFX_GLM(IBMAEstimator):
    """
    A t-test on contrast images.

    Requirements:
        - con
    """
    def __init__(self, dataset, thresh=0.05, null='theoretical', n_iters=None):
        self.sample = None
        self.mask = None
        self.dataset = dataset
        self.thresh = thresh
        self.null = null
        self.n_iters = n_iters

    def fit(self, sample):
        con_maps = sample.get('con')
        if self.null == 'theoretical':
            t_map, p_map = stats.ttest_1samp(con_maps, popmean=0, axis=0)
            log_p_map = -np.log10(p_map)
            result = MetaResult(t=t_map, p=p_map, log_p=log_p_map, mask=self.mask)
        elif self.null == 'empirical':
            n_iters = self.n_iters
            z_maps = sample.get('z')
            k = z_maps.shape[0]
            t_map, _ = stats.ttest_1samp(con_maps, popmean=0, axis=0)
            p_map = np.ones(t_map.shape)
            iter_t_maps = np.zeros((n_iters, t_map.shape[0]))
            for i in n_iters:
                iter_con_maps = np.copy(con_maps)
                signs = np.random.choice(a=2, size=k, p=[0.5, 0.5])
                signs[signs == 0] = -1
                iter_con_maps *= signs[:, None]
                iter_t_maps[i, :], _ = stats.ttest_1samp(iter_con_maps, popmean=0, axis=0)

            for voxel in range(iter_t_maps.shape[0]):
                voxel_perc = stats.percentileofscore(iter_t_maps[:, voxel], t_map[voxel]) / 100.
                p_map[voxel] = 1. - (np.abs(voxel_perc - 0.5) * 2)
            log_p_map = -np.log10(p_map)
            result = MetaResult(t=t_map, p=p_map, log_p=log_p_map, mask=self.mask)
        else:
            raise ValueError('Input null must be "theoretical" or "empirical".')
        return result


class FFX_GLM(IBMAEstimator):
    """
    An image-based meta-analytic test using contrast and standard error images.
    Don't estimate variance, just take from first level

    Requirements:
        - con
        - se
    """
    def __init__(self, dataset, n_subjects=None, thresh=0.05, equal_var=True):
        self.dataset = dataset
        self.n_subjects = n_subjects
        self.thresh = thresh
        self.equal_var = equal_var

    def fit(self, sample):
        con_maps = sample.get('con')
        var_maps = sample.get('con_se')
        k = con_maps.shape[0]
        if self.n_subjects is not None:
            n_subjects = np.repeat(self.n_subjects, k)
        else:
            n_subjects = sample.get('n')

        if self.equal_var:
            weighted_con_maps = con_maps * n_subjects[:, None]
            sum_weighted_con_map = np.sum(weighted_con_maps, axis=0)
            weighted_ss_maps = var_maps * (n_subjects[:, None] - 1)
            est_ss_map = (1. / (n_subjects - 1)) * np.sum(weighted_ss_maps, axis=0)
            ffx_stat_map = (1. / np.sqrt(np.sum(n_subjects))) * sum_weighted_con_map / np.sqrt(est_ss_map)
            dof = np.sum(n_subjects - 1)
        else:
            raise Exception('Unequal variances not available yet.')

        p_map = stats.t.cdf(-ffx_stat_map, df=dof, loc=0, scale=1)
        log_p_map = -np.log10(p_map)
        result = MetaResult(ffx_stat=ffx_stat_map, p=p_map, log_p=log_p_map, mask=self.mask)
        return result


class MFX_GLM(IBMAEstimator):
    """
    The gold standard image-based meta-analytic test. Uses contrast and standard
    error images.

    Requirements:
        - con
        - se
    """
    def __init__(self, dataset, n_iters=10000, voxel_thresh=0.001, clust_thresh=0.05,
                 corr='FWE', verbose=True, n_cores=4):
        self.dataset = dataset
        self.n_iters = n_iters
        self.voxel_thresh = voxel_thresh
        self.clust_thresh = clust_thresh
        self.corr = corr
        self.verbose = verbose
        self.n_cores = n_cores

    def fit(self, sample):
        con_images = self.dataset.images
        se_images = self.dataset.images
        con_data = con_images  # xyz + contrast 4D array
        se_data = se_images  # xyz + contrast 4D array
