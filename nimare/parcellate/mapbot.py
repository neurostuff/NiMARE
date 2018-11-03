"""
Meta-analytic parcellation based on text (MAPBOT).
"""
from ..base import Parcellator
from ..due import due, Doi


@due.dcite(Doi('10.1016/j.neuroimage.2017.06.032'),
           description='Introduces the MAPBOT algorithm.')
class MAPBOT(Parcellator):
    """
    Meta-analytic parcellation based on text (MAPBOT).

    Parameters
    ----------
    tfidf_df : :obj:`pandas.DataFrame`
        A DataFrame with feature counts for the model. The index is 'id',
        used for identifying studies. Other columns are features (e.g.,
        unigrams and bigrams from Neurosynth), where each value is the number
        of times the feature is found in a given article.
    coordinates_df : :obj:`pandas.DataFrame`
        A DataFrame with a list of foci in the dataset. The index is 'id',
        used for identifying studies. Additional columns include 'i', 'j' and
        'k' (the matrix indices of the foci in standard space).
    mask : :obj:`str` or :obj:`nibabel.Nifti1.Nifti1Image`
        Mask file or image.

    Notes
    -----
    MAPBOT uses both the reported foci for studies, as well as associated term
    weights.
    Here are the steps:
        1.  For each voxel in the mask, identify studies in dataset
            corresponding to that voxel. Selection criteria can be either
            based on a distance threshold (e.g., all studies with foci
            within 5mm of voxel) or based on a minimum number of studies
            (e.g., the 50 studies reporting foci closest to the voxel).
        2.  For each voxel, compute average frequency of each term across
            selected studies. This results in an n_voxels X n_terms frequency
            matrix F.
        3.  Compute n_voxels X n_voxels value matrix V:
            - D = (F.T * F) * ones(F)
            - V = F * D^-.5
        4.  Perform non-negative matrix factorization on value matrix.
    """
    def __init__(self, tfidf_df, coordinates_df, mask):
        pass

    def fit(self, target_mask, method='min_distance', r=5, n_exps=50,
            n_parcels=2):
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
