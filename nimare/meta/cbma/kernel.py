# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Methods for estimating thresholded cluster maps from neuroimaging experiments
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).

NOTE: Currently imagining output from "dataset.get_coordinates" as a DataFrame
of peak coords and sample sizes/statistics (a la Neurosynth).
"""
import pandas as pd

from .base import KernelEstimator
from .utils import compute_ma
from .transformations import xyz2ijk

__all__ = ['ALEKernel', 'MKDAKernel', 'KDAKernel']


class ALEKernel(KernelEstimator):
    """
    Generate ALE modeled activation images from coordinates and sample size.
    """
    def __init__(self, dataset):
        self.studies = dataset.get_coordinates()
        self.mask = dataset.mask_img

    def transform(self, fwhm=None, n=None):
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
        if fwhm is not None and n is not None:
            raise ValueError('Only one of fwhm and n may be provided.')

        temp_df = self.studies.copy()
        xyz = temp_df[['x', 'y', 'z']].values
        ijk = pd.DataFrame(xyz2ijk(xyz, self.mask.affine), columns=['i', 'j', 'k'])
        temp_df = pd.concat([temp_df, ijk], axis=1)
        for i, (name, data) in enumerate(temp_df.groupby('id')):
            ijk = data[['i', 'j', 'k']].values
            if n is not None:
                n_subjects = n
            else:
                n_subjects = data['n'].values[0]

            if fwhm is not None:
                kernel = smooth(data, fwhm, self.mask.affine)
            else:
                kernel = get_kernel(data, n_subjects, self.mask.affine)
            exp_dat = compute_ma(ijk, kernel)


class MKDAKernel(KernelEstimator):
    """
    Generate MKDA modeled activation images from coordinates.
    """
    def __init__(self, dataset):
        self.studies = dataset.get_coordinates()

    def transform(self, r=6):
        """
        Generate MKDA modeled activation images for each Contrast in dataset.
        Differs from KDA in that overlapping voxels are set to 1 (i.e.,
        resulting image remains binary).

        Parameters
        ----------
        r : :obj:`int`, optional
            Sphere radius, in mm.

        Returns
        -------
        imgs : :obj:`list` of `nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        pass


class KDAKernel(KernelEstimator):
    """
    Generate KDA modeled activation images from coordinates.
    """
    def __init__(self, dataset):
        self.studies = dataset.get_coordinates()

    def transform(self, r=6):
        """
        Generate KDA modeled activation images for each Contrast in dataset.
        Differs from MKDA images in that binary spheres are summed together in
        map (i.e., resulting image is not binary if coordinates are close to one
        another).

        Parameters
        ----------
        r : :obj:`int`, optional
            Sphere radius, in mm.

        Returns
        -------
        imgs : :obj:`list` of `nibabel.Nifti1Image`
            A list of modeled activation images (one for each of the Contrasts
            in the input dataset).
        """
        pass
