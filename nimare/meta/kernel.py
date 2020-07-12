"""
Methods for estimating thresholded cluster maps from neuroimaging contrasts
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).
"""
from __future__ import division
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import resample_to_img, math_img
from .utils import compute_ma, get_ale_kernel, peaks2maps
from ..transforms import vox2mm

from ..base import KernelTransformer


class ALEKernel(KernelTransformer):
    """
    Generate ALE modeled activation images from coordinates and sample size.

    Parameters
    ----------
    fwhm : :obj:`float`, optional
        Full-width half-max for Gaussian kernel, if you want to have a
        constant kernel across Contrasts. Mutually exclusive with
        ``sample_size``.
    sample_size : :obj:`int`, optional
        Sample size, used to derive FWHM for Gaussian kernel based on
        formulae from Eickhoff et al. (2012). This sample size overwrites
        the Contrast-specific sample sizes in the dataset, in order to hold
        kernel constant across Contrasts. Mutually exclusive with ``fwhm``.
    """
    def __init__(self, fwhm=None, sample_size=None):
        if fwhm is not None and sample_size is not None:
            raise ValueError('Only one of "fwhm" and "sample_size" may be provided.')
        self.fwhm = fwhm
        self.sample_size = sample_size

    def transform(self, dataset, masker=None, return_type='image'):
        """
        Generate ALE modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        masker : img_like, optional
            Only used if dataset is a DataFrame.
        return_type : {'image', 'array'}, optional
            Whether to return a niimg ('image') or a numpy array.
            Default is 'image'.

        Returns
        -------
        imgs : :obj:`list` of :class:`nibabel.nifti1.Nifti1Image` or :class:`numpy.ndarray`
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
        """
        if isinstance(dataset, pd.DataFrame):
            assert masker is not None, 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
            coordinates = dataset.copy()
        else:
            mask = dataset.masker.mask_img
            coordinates = dataset.coordinates

        if return_type == 'image':
            mask_data = mask.get_fdata().astype(float)
        elif return_type == 'array':
            mask_data = mask.get_fdata().astype(np.bool)
        else:
            raise ValueError('Argument "return_type" must be "image" or "array".')

        imgs = []
        kernels = {}  # retain kernels in dictionary to speed things up
        for id_, data in coordinates.groupby('id'):
            ijk = np.vstack((data.i.values, data.j.values, data.k.values)).T.astype(int)
            if self.sample_size is not None:
                sample_size = self.sample_size
            elif self.fwhm is None:
                sample_size = data.sample_size.astype(float).values[0]

            if self.fwhm is not None:
                assert np.isfinite(self.fwhm), 'FWHM must be finite number'
                if self.fwhm not in kernels.keys():
                    _, kern = get_ale_kernel(mask, fwhm=self.fwhm)
                    kernels[self.fwhm] = kern
                else:
                    kern = kernels[self.fwhm]
            else:
                assert np.isfinite(sample_size), 'Sample size must be finite number'
                if sample_size not in kernels.keys():
                    _, kern = get_ale_kernel(mask, sample_size=sample_size)
                    kernels[sample_size] = kern
                else:
                    kern = kernels[sample_size]
            kernel_data = compute_ma(mask.shape, ijk, kern)
            if return_type == 'image':
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)

        if return_type == 'array':
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

    def transform(self, dataset, masker=None, return_type='image'):
        """
        Generate MKDA modeled activation images for each Contrast in dataset.
        For each Contrast, a binary sphere of radius ``r`` is placed around
        each coordinate. Voxels within overlapping regions between proximal
        coordinates are set to 1, rather than the sum.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        masker : img_like, optional
            Only used if dataset is a DataFrame.
        return_type : {'image', 'array'}, optional
            Whether to return a niimg ('image') or a numpy array.
            Default is 'image'.

        Returns
        -------
        imgs : :obj:`list` of :class:`nibabel.Nifti1Image` or :class:`numpy.ndarray`
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
        """
        if isinstance(dataset, pd.DataFrame):
            assert masker is not None, 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
            coordinates = dataset.copy()
        else:
            mask = dataset.masker.mask_img
            coordinates = dataset.coordinates

        if return_type == 'image':
            mask_data = mask.get_fdata().astype(float)
        elif return_type == 'array':
            mask_data = mask.get_fdata().astype(np.bool)
        else:
            raise ValueError('Argument "return_type" must be "image" or "array".')

        dims = mask.shape
        vox_dims = mask.header.get_zooms()

        imgs = []
        for id_, data in coordinates.groupby('id'):
            kernel_data = np.zeros(dims)
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [slice(-self.r // vox_dims[i], self.r // vox_dims[i] + 0.01, 1)
                              for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= self.r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] = self.value

            if return_type == 'image':
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)

        if return_type == 'array':
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

    def transform(self, dataset, masker=None, return_type='image'):
        """
        Generate KDA modeled activation images for each Contrast in dataset.
        Differs from MKDA images in that binary spheres are summed together in
        map (i.e., resulting image is not binary if coordinates are close to one
        another).

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        masker : img_like, optional
            Only used if dataset is a DataFrame.
        return_type : {'image', 'array'}, optional
            Whether to return a niimg ('image') or a numpy array.
            Default is 'image'.

        Returns
        -------
        imgs : :obj:`list` of :class:`nibabel.Nifti1Image` or :class:`numpy.ndarray`
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
        """
        if isinstance(dataset, pd.DataFrame):
            assert masker is not None, 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
            coordinates = dataset.copy()
        else:
            mask = dataset.masker.mask_img
            coordinates = dataset.coordinates

        if return_type == 'image':
            mask_data = mask.get_fdata().astype(float)
        elif return_type == 'array':
            mask_data = mask.get_fdata().astype(np.bool)
        else:
            raise ValueError('Argument "return_type" must be "image" or "array".')

        dims = mask.shape
        vox_dims = mask.header.get_zooms()

        imgs = []
        for id_, data in coordinates.groupby('id'):
            kernel_data = np.zeros(dims)
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [slice(-self.r // vox_dims[i], self.r // vox_dims[i] + 0.01, 1)
                              for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= self.r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] += self.value

            if return_type == 'image':
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)
        if return_type == 'array':
            imgs = np.vstack(imgs)
        return imgs


class Peaks2MapsKernel(KernelTransformer):
    """
    Generate peaks2maps modeled activation images from coordinates.

    Parameters
    ----------
    resample_to_mask : :obj:`bool`, optional
        If True, will resample the MA maps to the mask's header.
        Default is True.
    """
    def __init__(self, resample_to_mask=True):
        self.resample_to_mask = resample_to_mask

    def transform(self, dataset, masker=None, return_type='image'):
        """
        Generate peaks2maps modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset for which to make images.
        masker : img_like, optional
            Only used if dataset is a DataFrame.
        return_type : {'image', 'array'}, optional
            Whether to return a niimg ('image') or a numpy array.
            Default is 'image'.

        Returns
        -------
        imgs : :obj:`list` of :class:`nibabel.Nifti1Image` or :class:`numpy.ndarray`
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
        """
        if isinstance(dataset, pd.DataFrame):
            assert masker is not None, 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
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
            mask = dataset.masker.mask_img
            resampled_imgs = []
            for img in imgs:
                resampled_imgs.append(resample_to_img(img, mask))
            imgs = resampled_imgs
        else:
            # Resample mask to data instead of data to mask
            mask = resample_to_img(dataset.masker.mask_img,
                                   imgs[0], interpolation='nearest')

        if return_type == 'array':
            imgs = masker.transform(imgs)
        else:
            masked_images = []
            for img in imgs:
                masked_images.append(math_img('map*mask', map=img, mask=mask))
            imgs = masked_images
        return imgs
