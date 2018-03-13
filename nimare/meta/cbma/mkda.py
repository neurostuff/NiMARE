"""
Coordinate-based meta-analysis estimators
"""
import warnings

import numpy as np

from .base import CBMAEstimator
from .kernel import MKDAKernel, KDAKernel
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

        k_est = kernel_estimator(**kernel_args)
        red_df1 = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]
        red_df2 = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids2)]
        ma_maps1 = k_est.transform(red_df1)
        ma_maps2 = k_est.transform(red_df2)
        self.ma_maps = [ma_maps1, ma_maps2]
        self.dataset = dataset
        self.voxel_thresh = None
        self.corr = None
        self.n_iters = None
        self.images = {}

    def fit(self, voxel_thresh=0.01, corr='FDR', n_iters=10000):
        self.voxel_thresh = voxel_thresh
        self.corr = corr
        self.n_iters = n_iters

        # Calculate different count variables
        n_selected = len(self.selected_ids)
        n_unselected = np.sum(unselected_id_indices)
        n_mappables = n_selected + n_unselected

        n_selected_active_voxels = mt.data.dot(self.selected_id_indices)
        n_unselected_active_voxels = mt.data.dot(unselected_id_indices)

        # Nomenclature for variables below: p = probability, F = feature present, g = given,
        # U = unselected, A = activation. So, e.g., pAgF = p(A|F) = probability of activation
        # in a voxel if we know that the feature is present in a study.
        pF = (n_selected * 1.0) / n_mappables
        pA = np.array((mt.data.sum(axis=1) * 1.0) / n_mappables).squeeze()

        # Conditional probabilities
        pAgF = n_selected_active_voxels * 1.0 / n_selected
        pAgU = n_unselected_active_voxels * 1.0 / n_unselected
        pFgA = pAgF * pF / pA

        # Recompute conditionals with uniform prior
        pAgF_prior = prior * pAgF + (1 - prior) * pAgU
        pFgA_prior = pAgF * prior / pAgF_prior

        # One-way chi-square test for consistency of activation
        p_vals = stats.one_way(
            np.squeeze(n_selected_active_voxels), n_selected)
        p_vals[p_vals < 1e-240] = 1e-240
        z_sign = np.sign(
            n_selected_active_voxels - np.mean(
                n_selected_active_voxels)).ravel()
        pAgF_z = p_to_z(p_vals, z_sign)
        fdr_thresh = stats.fdr(p_vals, q)
        pAgF_z_FDR = imageutils.threshold_img(
            pAgF_z, fdr_thresh, p_vals, mask_out='above')

        # Two-way chi-square for specificity of activation
        cells = np.squeeze(
            np.array([[n_selected_active_voxels, n_unselected_active_voxels],
                      [n_selected - n_selected_active_voxels, n_unselected -
                       n_unselected_active_voxels]]).T)
        p_vals = stats.two_way(cells)
        p_vals[p_vals < 1e-240] = 1e-240
        z_sign = np.sign(pAgF - pAgU).ravel()
        pFgA_z = p_to_z(p_vals, z_sign)
        fdr_thresh = stats.fdr(p_vals, q)
        pFgA_z_FDR = imageutils.threshold_img(
            pFgA_z, fdr_thresh, p_vals, mask_out='above')

        # Retain any images we may want to save or access later
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
    def __init__(self, dataset, voxel_thresh=0.01, corr='FDR', n_iters=10000):
        self.dataset = dataset
        self.voxel_thresh = voxel_thresh
        self.corr = corr
        self.n_iters = n_iters

    def fit(self, sample):
        pass
