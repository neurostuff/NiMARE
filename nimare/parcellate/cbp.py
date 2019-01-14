"""
Coactivation-based parcellation
"""
from ..base import Parcellator
from ..meta.cbma.ale import SCALE
from ..due import due, Doi


@due.dcite(Doi('10.1002/hbm.22138'),
           description='Introduces CBP.')
class CoordCBP(Parcellator):
    """
    Coordinate-based coactivation-based parcellation

    Notes
    -----
    Here are the steps:
        1.  For each voxel in the mask, identify studies in dataset
            corresponding to that voxel. Selection criteria can be either
            based on a distance threshold (e.g., all studies with foci
            within 5mm of voxel) or based on a minimum number of studies
            (e.g., the 50 studies reporting foci closest to the voxel).
        2.  For each voxel, perform MACM (meta-analysis) using the
            identified studies.
        3.  Correlate statistical maps between voxel MACMs to generate
            n_voxels X n_voxels correlation matrix.
        4.  Convert correlation coefficients to correlation distance (1 -r)
            values.
        5.  Perform clustering on correlation distance matrix.
    """
    def __init__(self, dataset, ids):
        pass

    def fit(self, target_mask, method='min_distance', r=5, n_exps=50,
            n_parcels=2, meta_estimator=SCALE, **kwargs):
        """
        Run CBP parcellation.

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


class ImCBP(Parcellator):
    """
    Image-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids):
        pass

    def fit(self, target_mask, n_parcels=2):
        """
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
