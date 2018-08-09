"""
Coactivation-based parcellation
"""
from .base import Parcellator
from ..meta.cbma.kernel import ALEKernel
from ..due import due, Doi


@due.dcite(Doi('10.1002/hbm.22138'),
           description='Introduces CBP.')
class CoordCBP(Parcellator):
    """
    Coordinate-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids, kernel_estimator=ALEKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not
                  k.startswith('kernel__')}

        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]

        self.kernel_estimator = kernel_estimator
        self.kernel_arguments = kernel_args
        self.ids = ids

    def fit(self, target_mask, n_parcels=2, n_iters=10000, n_cores=4):
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


class ImCBP(Parcellator):
    """
    Image-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids):
        self.mask = dataset.mask
        self.ids = ids

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
