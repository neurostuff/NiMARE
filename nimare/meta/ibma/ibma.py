# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image-based meta-analysis estimators
"""
from ..base import MetaEstimator


class MFX_GLM(MetaEstimator):
    """
    The gold standard image-based meta-analytic test. Uses contrast and standard
    error images.
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


class RFX_GLM(MetaEstimator):
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


class FFX_GLM(MetaEstimator):
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


class Fishers(MetaEstimator):
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


class Stouffers(MetaEstimator):
    """
    An image-based meta-analytic test using z-statistic images.
    Average Z, rescaled to N(0,1)
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


class StouffersRFX(MetaEstimator):
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
        pass


class WeightedStouffers(MetaEstimator):
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
