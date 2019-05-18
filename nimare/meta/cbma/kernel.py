"""
Methods for estimating thresholded cluster maps from neuroimaging contrasts
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).

NOTE: Currently imagining output from "dataset.get_coordinates" as a DataFrame
of peak coords and sample sizes/statistics (a la Neurosynth).
"""
from __future__ import division
import numpy as np
import nibabel as nib

from nilearn.image import resample_to_img, math_img
from .utils import compute_ma, get_ale_kernel, peaks2maps
from ...base.meta import KernelTransformer
from ...utils import vox2mm

__all__ = ['ALEKernel', 'MKDAKernel', 'KDAKernel', 'Peaks2MapsKernel']


class ALEKernel(KernelTransformer):
    """
    Generate ALE modeled activation images from coordinates and sample size.
    """
    def __init__(self, coordinates, mask):
        self.mask = mask
        self.coordinates = coordinates
        self.fwhm = None
        self.n = None

    def transform(self, ids, fwhm=None, n=None, masked=False):
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
        masked: :bool:, optional
            Return an array instead of a niimg.
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

        if not masked:
            mask_data = self.mask.get_data().astype(float)
        else:
            mask_data = self.mask.get_data().astype(np.bool)
        imgs = []
        kernels = {}
        for id_, data in self.coordinates.groupby('id'):
            # ijk = data[['i', 'j', 'k']].values.astype(int)
            ijk = np.vstack((data.i.values, data.j.values, data.k.values)).T.astype(int)
            if n is not None:
                n_subjects = n
            elif fwhm is None:
                n_subjects = data.n.astype(float).values[0]

            if fwhm is not None:
                assert np.isfinite(fwhm), 'FWHM must be finite number'
                if fwhm not in kernels.keys():
                    _, kern = get_ale_kernel(self.mask, fwhm=fwhm)
                    kernels[fwhm] = kern
                else:
                    kern = kernels[fwhm]
            else:
                assert np.isfinite(n_subjects), 'Sample size must be finite number'
                if n not in kernels.keys():
                    _, kern = get_ale_kernel(self.mask, n=n_subjects)
                    kernels[n] = kern
                else:
                    kern = kernels[n]
            kernel_data = compute_ma(self.mask.shape, ijk, kern)
            if not masked:
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, self.mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)

        if masked:
            imgs = np.vstack(imgs)

        return imgs


class MKDAKernel(KernelTransformer):
    """
    Generate MKDA modeled activation images from coordinates.
    """
    def __init__(self, coordinates, mask):
        self.mask = mask
        self.coordinates = coordinates
        self.r = None
        self.value = None

    def transform(self, ids, r=10, value=1, masked=False):
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
        if not masked:
            mask_data = self.mask.get_data()
        else:
            mask_data = self.mask.get_data().astype(np.bool)

        imgs = []
        for id_, data in self.coordinates.groupby('id'):
            kernel_data = np.zeros(dims)
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [slice(-r // vox_dims[i], r // vox_dims[i] + 0.01, 1) for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] = value

            if not masked:
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, self.mask.affine)
            else:
                img = kernel_data[mask_data]
            imgs.append(img)

        if masked:
            imgs = np.vstack(imgs)
        return imgs


class KDAKernel(KernelTransformer):
    """
    Generate KDA modeled activation images from coordinates.
    """
    def __init__(self, coordinates, mask):
        self.mask = mask
        self.coordinates = coordinates
        self.r = None
        self.value = None

    def transform(self, ids, r=6, value=1, masked=False):
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
        if not masked:
            mask_data = self.mask.get_data()
        else:
            mask_data = self.mask.get_data().astype(np.bool)

        imgs = []
        for id_, data in self.coordinates.groupby('id'):
            kernel_data = np.zeros(dims)
            for ijk in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                xx, yy, zz = [slice(-r // vox_dims[i], r // vox_dims[i] + 0.01, 1) for i in range(len(ijk))]
                cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
                sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
                sphere = np.round(sphere.T + ijk)
                idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1)
                sphere = sphere[idx, :].astype(int)
                kernel_data[tuple(sphere.T)] += value

            if not masked:
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, self.mask.affine)
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
    def __init__(self, coordinates, mask):
        self.mask = mask
        self.coordinates = coordinates

    def transform(self, ids, masked=False, resample_to_mask=True):
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
        coordinates_list = []
        for id_ in ids:
            data = self.coordinates.loc[self.coordinates['id'] == id_]
            mm_coords = []
            for coord in np.vstack((data.i.values, data.j.values, data.k.values)).T:
                mm_coords.append(vox2mm(coord, self.mask.affine))
            coordinates_list.append(mm_coords)

        imgs = peaks2maps(coordinates_list, skip_out_of_bounds=True)

        if resample_to_mask:
            resampled_imgs = []
            for img in imgs:
                resampled_imgs.append(resample_to_img(img, self.mask))
            imgs = resampled_imgs

        if masked:
            masked_images = []
            for img in imgs:
                if not resample_to_mask:
                    mask = resample_to_img(self.mask, imgs[0], interpolation='nearest')
                else:
                    mask = self.mask
                masked_images.append(math_img('map*mask', map=img, mask=mask))
            imgs = masked_images

        return imgs
