"""
Methods for estimating thresholded cluster maps from neuroimaging experiments
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).

NOTE: Currently imagining output from "dataset.get_coordinates" as a DataFrame
of peak coords and sample sizes/statistics (a la Neurosynth).
"""
from __future__ import division
import numpy as np
import pandas as pd
import nibabel as nib

from .base import KernelEstimator
from .utils import compute_ma, mem_smooth_64bit, get_kernel
from .transformations import xyz2ijk

__all__ = ['ALEKernel', 'MKDAKernel', 'KDAKernel']


class ALEKernel(KernelEstimator):
    """
    Generate ALE modeled activation images from coordinates and sample size.
    """
    def __init__(self, dataset):
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates
        self.fwhm = None
        self.n = None

    def transform(self, ids, fwhm=None, n=None):
        """
        Generate ALE modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        fwhm : :obj:`float`, optional
            Full-width half-max for Gaussian kernel, if you want to have a
            constant kernel across Contrasts. Mutually exclusive with ``n``.
        n : :obj:`int`, optional
            Sample size, used to derive FWHM for Gaussian kernel based on
            formulae from Eickhoff et al. (2012). This sample size overwrites
            the Contrast-specific sample sizes in the dataset, in order to hold
            kernel constant across Contrasts. Mutually exclusive with ``fwhm``.

        Returns
        -------
        imgs : :obj:`list` of `nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        self.fwhm = fwhm
        self.n = n
        if fwhm is not None and n is not None:
            raise ValueError('Only one of fwhm and n may be provided.')

        exp_dims = np.array(self.mask.shape) + np.array([30, 30, 30])
        sample_df = self.coordinates.loc[self.coordinates['id'].isin(ids)]
        imgs = []
        for i, (_, data) in enumerate(sample_df.groupby('id')):
            ijk = data[['i', 'j', 'k']].values.astype(int)
            if n is not None:
                n_subjects = n
            else:
                n_subjects = data['n'].values[0]

            assert np.isfinite(n_subjects), 'Sample size must be finite number'

            if fwhm is not None:
                temp_arr = np.zeros((31, 31, 31))
                temp_arr[15, 15, 15] = 1
                kern = mem_smooth_64bit(temp_arr, fwhm, self.mask)
            else:
                _, kern = get_kernel(n_subjects, self.mask)
            kernel_data = compute_ma(exp_dims, ijk, kern)
            img = nib.Nifti1Image(kernel_data, self.mask.affine)
            imgs.append(img)
        return imgs


class MKDAKernel(KernelEstimator):
    """
    Generate MKDA modeled activation images from coordinates.
    """
    def __init__(self, dataset):
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates
        self.r = None
        self.value = None

    def transform(self, ids, r=6, value=1):
        """
        Generate MKDA modeled activation images for each Contrast in dataset.
        For each Contrast, a binary sphere of radius ``r`` is placed around
        each coordinate. Voxels within overlapping regions between proximal
        coordinates are set to 1, rather than the sum.

        Parameters
        ----------
        ids : :obj:`list`
            A list of Contrast IDs for which to generate modeled activation
            images.
        r : :obj:`int`, optional
            Sphere radius, in mm.
        value : :obj:`int`, optional
            Value for sphere.

        Returns
        -------
        imgs : :obj:`list` of :obj:`nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        self.r = r
        self.value = value
        r = float(r)
        dims = self.mask.shape
        vox_dims = self.mask.header.get_zooms()

        sample_df = self.coordinates.loc[self.coordinates['id'].isin(ids)]
        imgs = []
        for i, (_, data) in enumerate(sample_df.groupby('id')):
            kernel_data = np.zeros(dims)
            for ijk in data[['i', 'j', 'k']].values:
                xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[i] + 0.01, 1) for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] = value
            img = nib.Nifti1Image(kernel_data, self.mask.affine)
            imgs.append(img)
        return imgs


class KDAKernel(KernelEstimator):
    """
    Generate KDA modeled activation images from coordinates.
    """
    def __init__(self, dataset):
        self.mask = dataset.mask
        self.coordinates = dataset.coordinates
        self.r = None
        self.value = None

    def transform(self, ids, r=6, value=1):
        """
        Generate KDA modeled activation images for each Contrast in dataset.
        Differs from MKDA images in that binary spheres are summed together in
        map (i.e., resulting image is not binary if coordinates are close to one
        another).

        Parameters
        ----------
        ids : :obj:`list`
            A list of Contrast IDs for which to generate modeled activation
            images.
        r : :obj:`int`, optional
            Sphere radius, in mm.
        value : :obj:`int`, optional
            Value for sphere.

        Returns
        -------
        imgs : :obj:`list` of :obj:`nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        self.r = r
        self.value = value
        r = float(r)
        dims = self.mask.shape
        vox_dims = self.mask.header.get_zooms()

        sample_df = self.coordinates.loc[self.coordinates['id'].isin(ids)]
        imgs = []
        for i, (_, data) in enumerate(sample_df.groupby('id')):
            kernel_data = np.zeros(dims)
            for ijk in data[['i', 'j', 'k']].values:
                xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[i] + 0.01, 1) for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] += value
            img = nib.Nifti1Image(kernel_data, self.mask.affine)
            imgs.append(img)
        return imgs
