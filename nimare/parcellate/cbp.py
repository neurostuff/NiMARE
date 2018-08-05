"""
Coactivation-based parcellation
"""
from .base import Parcellator
from ..meta.cbma.kernel import ALEKernel
from ..due import due, Doi


@due.dcite(Doi('10.1002/hbm.22138'),
           description='Introduces CBP.')
class CBCBP(Parcellator):
    """
    Coordinate-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids, kernel_estimator=ALEKernel, **kwargs):
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

        self.mask = dataset.mask
        self.coordinates = dataset.coordinates.loc[dataset.coordinates['id'].isin(ids)]

        self.kernel_estimator = kernel_estimator
        self.kernel_arguments = kernel_args
        self.ids = ids

    def fit(self, target_mask, n_iters=10000, n_cores=4):
        """
        """
        pass


class IBCBP(Parcellator):
    """
    Image-based coactivation-based parcellation
    """
    def __init__(self, dataset, ids):
        self.mask = dataset.mask
        self.ids = ids

    def fit(self, target_mask):
        """
        """
        pass
