"""
Image-based meta-analysis estimators
"""
from __future__ import division

import numpy as np
from sklearn.preprocessing import normalize
from scipy import stats

from .base import IBMAEstimator
from ..base import MetaResult


class Stouffers(IBMAEstimator):
    """
    An image-based meta-analytic test using z-statistic images.
    Average Z, rescaled to N(0,1)
    """
    def __init__(self, n_iters=10000, voxel_thresh=0.001, clust_thresh=0.05,
                 corr='FWE', verbose=True, n_cores=4):
        self.sample = None
        self.mask = None
        self.n_iters = n_iters
        self.voxel_thresh = voxel_thresh
        self.clust_thresh = clust_thresh
        self.corr = corr
        self.verbose = verbose
        self.n_cores = n_cores

    def fit(self, sample):
        """
        look at xarray, or perhaps a dictionary-like object with keys for
        different input maps, but also some way of logging level of specificity
        (e.g., exps vs studies), study IDs, and mask.

        """
        self.sample = sample
        self.mask = sample.mask

        # Rescale z maps to mean of zero, variance of one
        z_maps = self.sample.get('z')
        z_maps = normalize(z_maps, axis=1)
        k = z_maps.shape[0]

        # Combine
        z_map = np.sqrt(k) * np.mean(z_maps, axis=0)
        p_map = stats.norm.sf(abs(z_map))*2

        result = MetaResult(z=z_map, p=p_map, mask=self.mask)
        return result


class StouffersRFX(IBMAEstimator):
    """
    An image-based meta-analytic test using z-statistic images.
    Submit Zs to one-sample t-test
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
        # Rescale z maps to mean of zero, variance of one
        z_maps = sample.get('z')
        t_map, p_map = stats.ttest_1samp(z_maps, popmean=0, axis=0)
        return t_map, p_map


class MFX_GLM(IBMAEstimator):
    """
    The gold standard image-based meta-analytic test. Uses contrast and standard
    error images.
    """
    def __init__(self, n_iters=10000, voxel_thresh=0.001, clust_thresh=0.05,
                 corr='FWE', verbose=True, n_cores=4):
        self.dataset = None
        self.n_iters = n_iters
        self.voxel_thresh = voxel_thresh
        self.clust_thresh = clust_thresh
        self.corr = corr
        self.verbose = verbose
        self.n_cores = n_cores

    def fit(self, sample, dataset):
        pass


class RFX_GLM(IBMAEstimator):
    """
    An image-based meta-analytic test using contrast images.
    Analyze per-study contrasts as "dat"
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
        pass


class FFX_GLM(IBMAEstimator):
    """
    An image-based meta-analytic test using contrast and standard error images.
    Don't estimate variance, just take from first level
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
        studies_matching_criteria = self.dataset.get(images='con AND se')


class Fishers(IBMAEstimator):
    """
    An image-based meta-analytic test using t- or z-statistic images.
    Sum of -log P-values (from T/Zs converted to Ps)
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
        pass


class WeightedStouffers(IBMAEstimator):
    """
    An image-based meta-analytic test using z-statistic images and sample sizes.
    Zs from bigger studies get bigger weight
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
        pass
