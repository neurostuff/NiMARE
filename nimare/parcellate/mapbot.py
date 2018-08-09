"""
Meta-analytic parcellation based on text (MAPBOT).
"""
from .base import Parcellator
from ..due import due, Doi


@due.dcite(Doi('10.1016/j.neuroimage.2017.06.032'),
           description='Introduces the MAPBOT algorithm.')
class MAPBOT(Parcellator):
    """
    Meta-analytic parcellation based on text (MAPBOT).

    Parameters
    ----------
    text : :obj:`list` of :obj:`str`
        List of texts to use for parcellation.
    mask : :obj:`str` or :obj:`nibabel.Nifti1.Nifti1Image`
        Mask file or image.
    """
    def __init__(self, text, mask):
        self.mask = mask
        self.text = text

    def fit(self, region_name, n_parcels=2):
        """
        Run MAPBOT parcellation.

        region_name : :obj:`str`
            Name of region for parcellation.
        n_parcels : :obj:`int`, optional
            Number of parcels to generate for ROI. If array_like, each parcel
            number will be evaluated and results for all will be returned.
            Default is 2.
        """
        pass
