"""
Coordinate-based meta-analysis estimators
"""
import warnings

import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask, unmask

from .base import CBMAEstimator
from .kernel import MKDAKernel, KDAKernel
from .utils import p_to_z
from ...stats import two_way, one_way, fdr
from ...due import due, Doi


@due.dcite(Doi('10.1093/scan/nsm015'), description='Introduces the MKDA algorithm.')
class MKDA(CBMAEstimator):
    """
    Multilevel kernel density analysis
    """
    def __init__(self, dataset, ids, ids2=None, kernel_estimator=MKDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()\
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.dataset = dataset

        k_est = kernel_estimator(self.dataset.coordinates, self.dataset.mask)
        ma_maps1 = k_est.transform(ids, **kernel_args)
        if ids2 is None:
            ids2 = list(set(self.dataset.ids) - set(ids))
        ma_maps2 = k_est.transform(ids2, **kernel_args)
        self.ma_maps = [ma_maps1, ma_maps2]
        self.ids = [ids, ids2]
        self.voxel_thresh = None
        self.corr = None
        self.n_iters = None
        self.images = {}

    def fit(self, voxel_thresh=0.01, q=0.05, corr='FDR', n_iters=10000, prior=0.5):
        self.voxel_thresh = voxel_thresh
        self.corr = corr
        self.n_iters = n_iters

        # Calculate different count variables
        n_selected = len(self.ids[0])
        n_unselected = len(self.ids[1])
        n_mappables = n_selected + n_unselected

        # Transform MA maps to 1d arrays
        ma_maps1 = np.vstack([apply_mask(m, self.dataset.mask) for m in self.ma_maps[0]])
        ma_maps2 = np.vstack([apply_mask(m, self.dataset.mask) for m in self.ma_maps[1]])
        ma_maps_all = np.vstack((ma_maps1, ma_maps2))

        n_selected_active_voxels = np.sum(ma_maps1, axis=0)
        n_unselected_active_voxels = np.sum(ma_maps2, axis=0)

        # Nomenclature for variables below: p = probability, F = feature present, g = given,
        # U = unselected, A = activation. So, e.g., pAgF = p(A|F) = probability of activation
        # in a voxel if we know that the feature is present in a study.
        pF = (n_selected * 1.0) / n_mappables
        pA = np.array(np.sum(ma_maps_all, axis=0) / n_mappables).squeeze()

        # Conditional probabilities
        pAgF = n_selected_active_voxels * 1.0 / n_selected
        pAgU = n_unselected_active_voxels * 1.0 / n_unselected
        pFgA = pAgF * pF / pA

        # Recompute conditionals with uniform prior
        pAgF_prior = prior * pAgF + (1 - prior) * pAgU
        pFgA_prior = pAgF * prior / pAgF_prior

        # One-way chi-square test for consistency of activation
        p_vals = one_way(np.squeeze(n_selected_active_voxels), n_selected)
        p_vals[p_vals < 1e-240] = 1e-240
        z_sign = np.sign(n_selected_active_voxels - np.mean(n_selected_active_voxels)).ravel()
        pAgF_z = p_to_z(p_vals, z_sign)
        fdr_thresh = fdr(p_vals, q)
        pAgF_z_FDR = pAgF_z.copy()
        pAgF_z_FDR[p_vals > fdr_thresh] = 0

        # Two-way chi-square for specificity of activation
        cells = np.squeeze(
            np.array([[n_selected_active_voxels, n_unselected_active_voxels],
                      [n_selected - n_selected_active_voxels,
                       n_unselected - n_unselected_active_voxels]]).T)
        p_vals = two_way(cells)
        p_vals[p_vals < 1e-240] = 1e-240
        z_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(p_vals, z_sign)
        fdr_thresh = fdr(p_vals, q)
        pFgA_z_FDR = pAgF_z.copy()
        pFgA_z_FDR[p_vals > fdr_thresh] = 0

        # Retain any images we may want to save or access later
        pA = unmask(pA, self.dataset.mask)
        pAgF = unmask(pAgF, self.dataset.mask)
        pFgA = unmask(pFgA, self.dataset.mask)
        pAgF_prior = unmask(pAgF_prior, self.dataset.mask)
        pFgA_prior = unmask(pFgA_prior, self.dataset.mask)
        pAgF_z = unmask(pAgF_z, self.dataset.mask)
        pFgA_z = unmask(pFgA_z, self.dataset.mask)
        pAgF_z_FDR = unmask(pAgF_z_FDR, self.dataset.mask)
        pFgA_z_FDR = unmask(pFgA_z_FDR, self.dataset.mask)
        self.images = {
            'pA': pA,
            'pAgF': pAgF,
            'pFgA': pFgA,
            ('pAgF_given_pF=%0.2f' % prior): pAgF_prior,
            ('pFgA_given_pF=%0.2f' % prior): pFgA_prior,
            'consistency_z': pAgF_z,
            'specificity_z': pFgA_z,
            ('pAgF_z_FDR_%s' % q): pAgF_z_FDR,
            ('pFgA_z_FDR_%s' % q): pFgA_z_FDR
        }


@due.dcite(Doi('10.1016/S1053-8119(03)00078-8'),
           description='Introduces the KDA algorithm.')
class KDA(CBMAEstimator):
    """
    Kernel density analysis
    """
    def __init__(self, dataset, ids, ids2=None, kernel_estimator=KDAKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()\
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.dataset = dataset

        k_est = kernel_estimator(self.dataset.coordinates, self.dataset.mask)
        ma_maps1 = k_est.transform(ids, **kernel_args)
        if ids2 is None:
            ids2 = list(set(self.dataset.ids) - set(ids))
        ma_maps2 = k_est.transform(ids2, **kernel_args)
        self.ma_maps = [ma_maps1, ma_maps2]
        self.ids = [ids, ids2]
        self.voxel_thresh = None
        self.corr = None
        self.n_iters = None
        self.images = {}

    def fit(self, sample, voxel_thresh=0.01, corr='FDR', n_iters=10000):
        pass
