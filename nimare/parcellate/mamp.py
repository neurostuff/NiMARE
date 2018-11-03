"""
Meta-analytic activation modeling-based parcellation (MAMP).
"""
from ..base import Parcellator
from ..meta.cbma.kernel import ALEKernel
from ..due import due, Doi


@due.dcite(Doi('10.1016/j.neuroimage.2015.08.027'),
           description='Introduces the MAMP algorithm.')
class MAMP(Parcellator):
    """
    Meta-analytic activation modeling-based parcellation (MAMP).

    Parameters
    ----------
    text : :obj:`list` of :obj:`str`
        List of texts to use for parcellation.
    mask : :obj:`str` or :obj:`nibabel.Nifti1.Nifti1Image`
        Mask file or image.

    Notes
    -----
    MAMP works similarly to CBP, but skips the step of performing a MACM for
    each voxel. Here are the steps:
        1.  Create an MA map for each study in the dataset.
        2.  Concatenate MA maps across studies to create a 4D dataset.
        3.  Extract values across studies for voxels in mask, resulting in
            n_voxels X n_studies array.
        4.  Correlate "study series" between voxels to generate n_voxels X
            n_voxels correlation matrix.
        5.  Convert correlation coefficients to correlation distance (1 -r)
            values.
        6.  Perform clustering on correlation distance matrix.
    """
    def __init__(self, dataset, ids):
        pass

    def fit(self, target_mask, n_parcels=2, kernel_estimator=ALEKernel,
            **kwargs):
        """
        Run MAMP parcellation.

        Parameters
        ----------
        target_mask : img_like
            Image with binary mask for region of interest to be parcellated.
        n_parcels : :obj:`int` or array_like of :obj:`int`, optional
            Number of parcels to generate for ROI. If array_like, each parcel
            number will be evaluated and results for all will be returned.
            Default is 2.
        n_iters : :obj:`int`, optional
            Number of iterations to run for each parcel number.
            Default is 10000.
        n_cores : :obj:`int`, optional
            Number of cores to use for model fitting.

        Returns
        -------
        results
        """
        pass
