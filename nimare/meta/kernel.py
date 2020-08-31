"""
Methods for estimating thresholded cluster maps from neuroimaging contrasts
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).
"""
from __future__ import division

import logging
import os
from hashlib import md5

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image

from ..base import KernelTransformer
from ..transforms import vox2mm
from ..utils import get_masker
from .utils import compute_ma, get_ale_kernel, peaks2maps

LGR = logging.getLogger(__name__)


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

    def transform(self, dataset, masker=None, return_type="image"):
        """
        Generate ALE modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        masker : img_like, optional
            Only used if dataset is a DataFrame.
        return_type : {'array', 'image', 'dataset'}, optional
            Whether to return a numpy array ('array'), a list of niimgs ('image'), or
            a Dataset with MA images saved as files ('dataset').
            Default is 'dataset'.

        Returns
        -------
        imgs : (C x V) :class:`numpy.ndarray` or :obj:`list` of :class:`nibabel.Nifti1Image` or\
               :class:`nimare.dataset.Dataset`
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'dataset', a new Dataset object with modeled activation
            images saved to files and referenced in the Dataset.images attribute.

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for MA maps that will be saved by the transformer.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
        """
        if return_type not in ("array", "image", "dataset"):
            raise ValueError('Argument "return_type" must be "image", "array", or "dataset".')

        if isinstance(dataset, pd.DataFrame):
            assert (
                masker is not None
            ), 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
            coordinates = dataset.copy()
            assert (
                return_type != "dataset"
            ), "Input dataset must be a Dataset if return_type='dataset'."
        else:
            masker = dataset.masker if not masker else masker
            mask = masker.mask_img
            coordinates = dataset.coordinates

            # Determine MA map filenames. Must happen after parameters are set.
            self._infer_names(affine=md5(mask.affine).hexdigest())

            # Check for existing MA maps
            # Use coordinates to get IDs instead of Dataset.ids bc of possible mismatch
            # between full Dataset and contrasts with coordinates.
            if self.image_type in dataset.images.columns:
                files = dataset.get_images(ids=coordinates["id"].unique(), imtype=self.image_type)
                if all(f is not None for f in files):
                    LGR.debug("Files already exist. Using them.")
                    if return_type == "array":
                        return masker.transform(files)
                    elif return_type == "image":
                        return [nib.load(f) for f in files]
                    elif return_type == "dataset":
                        return dataset.copy()

        # Otherwise, generate the MA maps
        if return_type == "array":
            mask_data = mask.get_fdata().astype(np.bool)
        elif return_type == "image":
            mask_data = mask.get_fdata().astype(float)
        elif return_type == "dataset":
            dataset = dataset.copy()
            if dataset.basepath is None:
                raise ValueError(
                    "Dataset output path is not set. Set the path with Dataset.update_path()."
                )
            elif not os.path.isdir(dataset.basepath):
                raise ValueError(
                    "Output directory does not exist. "
                    "Set the path to an existing folder with Dataset.update_path()."
                )

        # Core code
        imgs = []
        kernels = {}  # retain kernels in dictionary to speed things up
        for id_, data in coordinates.groupby("id"):
            ijk = np.vstack((data.i.values, data.j.values, data.k.values)).T.astype(int)
            if self.sample_size is not None:
                sample_size = self.sample_size
            elif self.fwhm is None:
                sample_size = data.sample_size.astype(float).values[0]

            if self.fwhm is not None:
                assert np.isfinite(self.fwhm), "FWHM must be finite number"
                if self.fwhm not in kernels.keys():
                    _, kern = get_ale_kernel(mask, fwhm=self.fwhm)
                    kernels[self.fwhm] = kern
                else:
                    kern = kernels[self.fwhm]
            else:
                assert np.isfinite(sample_size), "Sample size must be finite number"
                if sample_size not in kernels.keys():
                    _, kern = get_ale_kernel(mask, sample_size=sample_size)
                    kernels[sample_size] = kern
                else:
                    kern = kernels[sample_size]
            kernel_data = compute_ma(mask.shape, ijk, kern)

            # Generic KernelTransformer code
            if return_type == "array":
                img = kernel_data[mask_data]
                imgs.append(img)
            elif return_type == "image":
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
                imgs.append(img)
            elif return_type == "dataset":
                img = nib.Nifti1Image(kernel_data, mask.affine)
                out_file = os.path.join(dataset.basepath, self.filename_pattern.format(id=id_))
                img.to_filename(out_file)
                dataset.images.loc[dataset.images["id"] == id_, self.image_type] = out_file

        if return_type == "array":
            return np.vstack(imgs)
        elif return_type == "image":
            return imgs
        elif return_type == "dataset":
            # Infer relative path
            dataset.images = dataset.images
            return dataset


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

    def transform(self, dataset, masker=None, return_type="image"):
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
        return_type : {'array', 'image', 'dataset'}, optional
            Whether to return a numpy array ('array'), a list of niimgs ('image'), or
            a Dataset with MA images saved as files ('dataset').
            Default is 'dataset'.

        Returns
        -------
        imgs : (C x V) :class:`numpy.ndarray` or :obj:`list` of :class:`nibabel.Nifti1Image` or\
               :class:`nimare.dataset.Dataset`
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'dataset', a new Dataset object with modeled activation
            images saved to files and referenced in the Dataset.images attribute.

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for MA maps that will be saved by the transformer.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
        """
        if return_type not in ("array", "image", "dataset"):
            raise ValueError('Argument "return_type" must be "image", "array", or "dataset".')

        if isinstance(dataset, pd.DataFrame):
            assert (
                masker is not None
            ), 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
            coordinates = dataset.copy()
            assert (
                return_type != "dataset"
            ), "Input dataset must be a Dataset if return_type='dataset'."
        else:
            masker = dataset.masker if not masker else masker
            mask = masker.mask_img
            coordinates = dataset.coordinates

            # Determine MA map filenames. Must happen after parameters are set.
            self._infer_names(affine=md5(mask.affine).hexdigest())

            # Check for existing MA maps
            # Use coordinates to get IDs instead of Dataset.ids bc of possible mismatch
            # between full Dataset and contrasts with coordinates.
            if self.image_type in dataset.images.columns:
                files = dataset.get_images(ids=coordinates["id"].unique(), imtype=self.image_type)
                if all(f is not None for f in files):
                    LGR.debug("Files already exist. Using them.")
                    if return_type == "array":
                        return masker.transform(files)
                    elif return_type == "image":
                        return [nib.load(f) for f in files]
                    elif return_type == "dataset":
                        return dataset.copy()

        # Otherwise, generate the MA maps
        if return_type == "array":
            mask_data = mask.get_fdata().astype(np.bool)
        elif return_type == "image":
            mask_data = mask.get_fdata().astype(type(self.value))
        elif return_type == "dataset":
            dataset = dataset.copy()
            if dataset.basepath is None:
                raise ValueError(
                    "Dataset output path is not set. Set the path with Dataset.update_path()."
                )
            elif not os.path.isdir(dataset.basepath):
                raise ValueError(
                    "Output directory does not exist. "
                    "Set the path to an existing folder with Dataset.update_path()."
                )

        # Core code
        dims = mask.shape
        vox_dims = mask.header.get_zooms()

        imgs = []
        for id_, data in coordinates.groupby("id"):
            kernel_data = np.zeros(dims, dtype=type(self.value))
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [
                    slice(-self.r // vox_dims[i], self.r // vox_dims[i] + 0.01, 1)
                    for i in range(len(ijk))
                ]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** 0.5 <= self.r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] = self.value

            # Generic KernelTransformer code
            if return_type == "array":
                img = kernel_data[mask_data]
                imgs.append(img)
            elif return_type == "image":
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
                imgs.append(img)
            elif return_type == "dataset":
                img = nib.Nifti1Image(kernel_data, mask.affine)
                out_file = os.path.join(dataset.basepath, self.filename_pattern.format(id=id_))
                img.to_filename(out_file)
                dataset.images.loc[dataset.images["id"] == id_, self.image_type] = out_file

        if return_type == "array":
            return np.vstack(imgs)
        elif return_type == "image":
            return imgs
        elif return_type == "dataset":
            # Infer relative path
            dataset.images = dataset.images
            return dataset


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
        # Set parameters
        self.r = float(r)
        self.value = value

    def transform(self, dataset, masker=None, return_type="dataset"):
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
        return_type : {'array', 'image', 'dataset'}, optional
            Whether to return a numpy array ('array'), a list of niimgs ('image'), or
            a Dataset with MA images saved as files ('dataset').
            Default is 'dataset'.

        Returns
        -------
        imgs : (C x V) :class:`numpy.ndarray` or :obj:`list` of :class:`nibabel.Nifti1Image` or\
               :class:`nimare.dataset.Dataset`
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'dataset', a new Dataset object with modeled activation
            images saved to files and referenced in the Dataset.images attribute.

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for MA maps that will be saved by the transformer.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
        """
        if return_type not in ("array", "image", "dataset"):
            raise ValueError('Argument "return_type" must be "image", "array", or "dataset".')

        if isinstance(dataset, pd.DataFrame):
            assert (
                masker is not None
            ), 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
            coordinates = dataset.copy()
            assert (
                return_type != "dataset"
            ), "Input dataset must be a Dataset if return_type='dataset'."
        else:
            masker = dataset.masker if not masker else masker
            mask = masker.mask_img
            coordinates = dataset.coordinates

            # Determine MA map filenames. Must happen after parameters are set.
            self._infer_names(affine=md5(mask.affine).hexdigest())

            # Check for existing MA maps
            # Use coordinates to get IDs instead of Dataset.ids bc of possible mismatch
            # between full Dataset and contrasts with coordinates.
            if self.image_type in dataset.images.columns:
                files = dataset.get_images(ids=coordinates["id"].unique(), imtype=self.image_type)
                if all(f is not None for f in files):
                    LGR.debug("Files already exist. Using them.")
                    if return_type == "array":
                        return masker.transform(files)
                    elif return_type == "image":
                        return [nib.load(f) for f in files]
                    elif return_type == "dataset":
                        return dataset.copy()

        # Otherwise, generate the MA maps
        if return_type == "array":
            mask_data = mask.get_fdata().astype(np.bool)
        elif return_type == "image":
            mask_data = mask.get_fdata().astype(type(self.value))
        elif return_type == "dataset":
            dataset = dataset.copy()
            if dataset.basepath is None:
                raise ValueError(
                    "Dataset output path is not set. Set the path with Dataset.update_path()."
                )
            elif not os.path.isdir(dataset.basepath):
                raise ValueError(
                    "Output directory does not exist. "
                    "Set the path to an existing folder with Dataset.update_path()."
                )

        # Core code
        dims = mask.shape
        vox_dims = mask.header.get_zooms()

        # Create MA maps
        imgs = []
        for id_, data in coordinates.groupby("id"):
            kernel_data = np.zeros(dims, dtype=type(self.value))
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [
                    slice(-self.r // vox_dims[i], self.r // vox_dims[i] + 0.01, 1)
                    for i in range(len(ijk))
                ]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** 0.5 <= self.r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] += self.value

            # Generic KernelTransformer code
            if return_type == "array":
                img = kernel_data[mask_data]
                imgs.append(img)
            elif return_type == "image":
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
                imgs.append(img)
            elif return_type == "dataset":
                img = nib.Nifti1Image(kernel_data, mask.affine)
                out_file = os.path.join(dataset.basepath, self.filename_pattern.format(id=id_))
                img.to_filename(out_file)
                dataset.images.loc[dataset.images["id"] == id_, self.image_type] = out_file

        if return_type == "array":
            return np.vstack(imgs)
        elif return_type == "image":
            return imgs
        elif return_type == "dataset":
            # Infer relative path
            dataset.images = dataset.images
            return dataset


class Peaks2MapsKernel(KernelTransformer):
    """
    Generate peaks2maps modeled activation images from coordinates.

    Parameters
    ----------
    resample_to_mask : :obj:`bool`, optional
        If True, will resample the MA maps to the mask's header.
        Default is True.
    """

    def __init__(self, resample_to_mask=True, model_dir="auto"):
        self.resample_to_mask = resample_to_mask
        # Use private attribute to hide value from get_params.
        # get_params will find model_dir=None, which is *very important* when a path is provided.
        self._model_dir = model_dir

    def transform(self, dataset, masker=None, return_type="image"):
        """
        Generate peaks2maps modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset for which to make images.
        masker : img_like, optional
            Only used if dataset is a DataFrame.
        return_type : {'array', 'image', 'dataset'}, optional
            Whether to return a numpy array ('array'), a list of niimgs ('image'), or
            a Dataset with MA images saved as files ('dataset').
            Default is 'dataset'.

        Returns
        -------
        imgs : (C x V) :class:`numpy.ndarray` or :obj:`list` of :class:`nibabel.Nifti1Image` or\
               :class:`nimare.dataset.Dataset`
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'dataset', a new Dataset object with modeled activation
            images saved to files and referenced in the Dataset.images attribute.

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for MA maps that will be saved by the transformer.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
        """
        if return_type not in ("array", "image", "dataset"):
            raise ValueError('Argument "return_type" must be "image", "array", or "dataset".')

        # Inferred filenames are invalid if mask is resampled to peaks2maps image space
        assert self.resample_to_mask or (
            return_type != "dataset"
        ), "Option resample_to_mask is required if return_type is 'dataset'."

        if isinstance(dataset, pd.DataFrame):
            assert (
                masker is not None
            ), 'Argument "masker" must be provided if dataset is a DataFrame'
            mask = masker.mask_img
            coordinates = dataset.copy()
            assert (
                return_type != "dataset"
            ), "Input dataset must be a Dataset if return_type='dataset'."
        else:
            masker = dataset.masker if not masker else masker
            mask = masker.mask_img
            coordinates = dataset.coordinates

            # Determine MA map filenames. Must happen after parameters are set.
            self._infer_names(affine=md5(mask.affine).hexdigest())

            # Check for existing MA maps
            # Use coordinates to get IDs instead of Dataset.ids bc of possible mismatch
            # between full Dataset and contrasts with coordinates.
            if self.image_type in dataset.images.columns:
                files = dataset.get_images(ids=coordinates["id"].unique(), imtype=self.image_type)
                if all(f is not None for f in files):
                    LGR.debug("Files already exist. Using them.")
                    if return_type == "array":
                        return masker.transform(files)
                    elif return_type == "image":
                        return [nib.load(f) for f in files]
                    elif return_type == "dataset":
                        return dataset.copy()

        # Otherwise, generate the MA maps
        if return_type == "dataset":
            dataset = dataset.copy()
            if dataset.basepath is None:
                raise ValueError(
                    "Dataset output path is not set. Set the path with Dataset.update_path()."
                )
            elif not os.path.isdir(dataset.basepath):
                raise ValueError(
                    "Output directory does not exist. "
                    "Set the path to an existing folder with Dataset.update_path()."
                )

        # Core code
        coordinates_list = []
        for id_, data in coordinates.groupby("id"):
            mm_coords = []
            for coord in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                mm_coords.append(vox2mm(coord, dataset.masker.mask_img.affine))
            coordinates_list.append(mm_coords)

        imgs = peaks2maps(coordinates_list, skip_out_of_bounds=True, model_dir=self._model_dir)

        if self.resample_to_mask:
            resampled_imgs = []
            for img in imgs:
                resampled_imgs.append(image.resample_to_img(img, mask))
            imgs = resampled_imgs
        else:
            # Resample mask to data instead of data to mask
            mask = image.resample_to_img(mask, imgs[0], interpolation="nearest")
            masker = get_masker(mask)

        # Generic KernelTransformer code
        if return_type == "array":
            return masker.transform(imgs)
        elif return_type == "image":
            masked_imgs = []
            for img in imgs:
                masked_imgs.append(image.math_img("map*mask", map=img, mask=mask))
            return masked_imgs
        elif return_type == "dataset":
            for i_id, (id_, _) in enumerate(coordinates.groupby("id")):
                img = imgs[i_id]
                out_file = os.path.join(dataset.basepath, self.filename_pattern.format(id=id_))
                img.to_filename(out_file)
                dataset.images.loc[dataset.images["id"] == id_, self.image_type] = out_file
            # Infer relative path
            dataset.images = dataset.images
            return dataset
