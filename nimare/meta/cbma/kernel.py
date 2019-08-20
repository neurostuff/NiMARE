"""
Methods for estimating thresholded cluster maps from neuroimaging contrasts
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).

NOTE: Currently imagining output from "dataset.get_coordinates" as a DataFrame
of peak coords and sample sizes/statistics (a la Neurosynth).
"""
from __future__ import division
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import resample_to_img, math_img
from .utils import compute_ma, get_ale_kernel, peaks2maps
from ...utils import vox2mm, get_masker

from ...base import Transformer


__all__ = ['ALEKernel', 'MKDAKernel', 'KDAKernel', 'Peaks2MapsKernel']


class KernelTransformer(Transformer):
    """Base class for modeled activation-generating methods.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.
    """
    pass


class ALEKernel(KernelTransformer):
    """
    Generate ALE modeled activation images from coordinates and sample size.

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
    """
    def __init__(self, fwhm=None, n=None):
        if fwhm is not None and n is not None:
            raise ValueError('Only one of fwhm and n may be provided.')
        self.fwhm = fwhm
        self.n = n

    def transform(self, dataset, mask=None, masked=False):
        """
        Generate ALE modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        mask : img_like, optional
            Only used if dataset is a DataFrame.
        masked: :obj:`bool`, optional
            Return an array instead of a niimg.

        Returns
        -------
        imgs : :obj:`list` of `nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """

        if isinstance(dataset, pd.DataFrame):
            assert mask is not None, 'Argument "mask" must be provided if dataset is a DataFrame'
            mask = get_masker(mask).mask_img
            coordinates = dataset.copy()
        else:
            mask = dataset.masker.mask_img
            coordinates = dataset.coordinates

        if not masked:
            mask_data = mask.get_data().astype(float)
        else:
            mask_data = mask.get_data().astype(np.bool)

        imgs = []
        kernels = {}  # retain kernels in dictionary to speed things up
        for id_, data in coordinates.groupby('id'):
            ijk = np.vstack((data.i.values, data.j.values, data.k.values)).T.astype(int)
            if self.n is not None:
                n_subjects = self.n
            elif self.fwhm is None:
                n_subjects = data.n.astype(float).values[0]

            if self.fwhm is not None:
                assert np.isfinite(self.fwhm), 'FWHM must be finite number'
                if self.fwhm not in kernels.keys():
                    _, kern = get_ale_kernel(mask, fwhm=self.fwhm)
                    kernels[self.fwhm] = kern
                else:
                    kern = kernels[self.fwhm]
            else:
                assert np.isfinite(n_subjects), 'Sample size must be finite number'
                if n_subjects not in kernels.keys():
                    _, kern = get_ale_kernel(mask, n=n_subjects)
                    kernels[n_subjects] = kern
                else:
                    kern = kernels[n_subjects]
            kernel_data = compute_ma(mask.shape, ijk, kern)
            if not masked:
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)

        if masked:
            imgs = np.vstack(imgs)

        return imgs


class MKDAKernel(KernelTransformer):
    """
    Generate MKDA modeled activation images from coordinates.

    Parameters
    ----------
    r : :obj:`int`, optional
        Sphere radius, in mm.
    value : :obj:`int`, optional
        Value for sphere.
    """
    def __init__(self, r=10, value=1):
        self.r = float(r)
        self.value = value

    def transform(self, dataset, mask=None, masked=False):
        """
        Generate MKDA modeled activation images for each Contrast in dataset.
        For each Contrast, a binary sphere of radius ``r`` is placed around
        each coordinate. Voxels within overlapping regions between proximal
        coordinates are set to 1, rather than the sum.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        mask : img_like, optional
            Only used if dataset is a DataFrame.
        masked: :obj:`bool`, optional
            Return an array instead of a niimg.

        Returns
        -------
        imgs : :obj:`list` of `nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        if isinstance(dataset, pd.DataFrame):
            assert mask is not None, 'Argument "mask" must be provided if dataset is a DataFrame'
            mask = get_masker(mask).mask_img
            coordinates = dataset.copy()
        else:
            mask = dataset.masker.mask_img
            coordinates = dataset.coordinates

        if not masked:
            mask_data = mask.get_data().astype(float)
        else:
            mask_data = mask.get_data().astype(np.bool)

        dims = mask.shape
        vox_dims = mask.header.get_zooms()

        imgs = []
        for id_, data in coordinates.groupby('id'):
            kernel_data = np.zeros(dims)
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [slice(-self.r // vox_dims[i], self.r // vox_dims[i] + 0.01, 1) for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= self.r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] = self.value

            if not masked:
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)

        if masked:
            imgs = np.vstack(imgs)
        return imgs


class KDAKernel(KernelTransformer):
    """
    Generate KDA modeled activation images from coordinates.

    Parameters
    ----------
    r : :obj:`int`, optional
        Sphere radius, in mm.
    value : :obj:`int`, optional
        Value for sphere.
    """
    def __init__(self, r=6, value=1):
        self.r = float(r)
        self.value = value

    def transform(self, dataset, mask=None, masked=False):
        """
        Generate KDA modeled activation images for each Contrast in dataset.
        Differs from MKDA images in that binary spheres are summed together in
        map (i.e., resulting image is not binary if coordinates are close to one
        another).

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        mask : img_like, optional
            Only used if dataset is a DataFrame.
        masked: :obj:`bool`, optional
            Return an array instead of a niimg.

        Returns
        -------
        imgs : :obj:`list` of `nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        if isinstance(dataset, pd.DataFrame):
            assert mask is not None, 'Argument "mask" must be provided if dataset is a DataFrame'
            mask = get_masker(mask).mask_img
            coordinates = dataset.copy()
        else:
            mask = dataset.masker.mask_img
            coordinates = dataset.coordinates

        if not masked:
            mask_data = mask.get_data().astype(float)
        else:
            mask_data = mask.get_data().astype(np.bool)

        dims = mask.shape
        vox_dims = mask.header.get_zooms()

        imgs = []
        for id_, data in coordinates.groupby('id'):
            kernel_data = np.zeros(dims)
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [slice(-self.r // vox_dims[i], self.r // vox_dims[i] + 0.01, 1) for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= self.r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] += self.value

            if not masked:
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)
        if masked:
            imgs = np.vstack(imgs)
        return imgs


class Peaks2MapsKernel(KernelTransformer):
    """
    Generate peaks2maps modeled activation images from coordinates.
    """
    def __init__(self, resample_to_mask=True):
        self.resample_to_mask = resample_to_mask

    def transform(self, dataset, mask=None, masked=False):
        """
        Generate peaks2maps modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        ids : :obj:`list`
            A list of Contrast IDs for which to generate modeled activation
            images.
        masked : :obj:`boolean`
            Whether to mask the maps generated by peaks2maps

        Returns
        -------
        imgs : :obj:`list` of :obj:`nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        if isinstance(dataset, pd.DataFrame):
            assert mask is not None, 'Argument "mask" must be provided if dataset is a DataFrame'
            mask = get_masker(mask).mask_img
            coordinates = dataset.copy()
        else:
            mask = dataset.masker.mask_img
            coordinates = dataset.coordinates

        coordinates_list = []
        for id_, data in coordinates.groupby('id'):
            mm_coords = []
            for coord in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                mm_coords.append(vox2mm(coord, dataset.masker.mask_img.affine))
            coordinates_list.append(mm_coords)

        imgs = peaks2maps(coordinates_list, skip_out_of_bounds=True)

        if self.resample_to_mask:
            resampled_imgs = []
            for img in imgs:
                resampled_imgs.append(resample_to_img(img, dataset.masker.mask_img))
            imgs = resampled_imgs

        if masked:
            masked_images = []
            for img in imgs:
                if not self.resample_to_mask:
                    mask = resample_to_img(dataset.masker.mask_img,
                                           imgs[0], interpolation='nearest')
                else:
                    mask = dataset.masker.mask_img
                masked_images.append(math_img('map*mask', map=img, mask=mask))
            imgs = masked_images

        return imgs
