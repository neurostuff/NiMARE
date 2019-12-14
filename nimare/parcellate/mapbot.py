"""
Meta-analytic parcellation based on text (MAPBOT).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.spatial.distance import cdist
from nilearn.masking import apply_mask, unmask

from .base import Parcellator
from ..due import due
from .. import references


@due.dcite(references.MAPBOT, description='Introduces the MAPBOT algorithm.')
class MAPBOT(Parcellator):
    """
    Meta-analytic parcellation based on text (MAPBOT) [1]_.

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

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Yuan, Rui, et al. "MAPBOT: Meta-analytic parcellation based on text,
        and its application to the human thalamus." NeuroImage 157 (2017):
        716-732. https://doi.org/10.1016/j.neuroimage.2017.06.032
    """
    def __init__(self, tfidf_df, coordinates_df, mask):
        self.mask = mask
        self.tfidf_df = tfidf_df
        self.coordinates = coordinates_df

    def fit(self, target_mask, method='min_distance', r=5, n_exps=50,
            n_parcels=2):
        """
        Run MAPBOT parcellation.

        Parameters
        ----------
        region_name : :obj:`str`
            Name of region for parcellation.
        n_parcels : :obj:`int`, optional
            Number of parcels to generate for ROI. If array_like, each parcel
            number will be evaluated and results for all will be returned.
            Default is 2.
        """
        if not isinstance(n_parcels):
            n_parcels = [n_parcels]

        # Step 1: Build correlation matrix
        target_data = apply_mask(target_mask, self.mask)
        target_map = unmask(target_data, self.mask)
        target_data = target_map.get_data()
        mask_idx = np.vstack(np.where(target_data))
        n_voxels = mask_idx.shape[1]
        voxel_arr = np.zeros((n_voxels, np.sum(self.mask)))
        del voxel_arr  # currently unused

        ijk = self.coordinates[['i', 'j', 'k']].values
        temp_df = self.coordinates.copy()
        term_df = pd.DataFrame(columns=self.tfidf_df.columns,
                               index=range(n_voxels))
        for i_voxel in range(n_voxels):
            voxel = mask_idx[:, i_voxel]
            temp_df['distance'] = cdist(ijk, voxel)

            if method == 'min_studies':
                # number of studies
                temp_df2 = temp_df.groupby('id')[['distance']].min()
                temp_df2 = temp_df2.sort_values(by='distance')
                sel_ids = temp_df2.iloc[:n_exps].index.values
            elif method == 'min_distance':
                # minimum distance
                temp_df2 = temp_df.groupby('id')[['distance']].min()
                sel_ids = temp_df2.loc[temp_df2['distance'] < r].index.values

            # Build DT matrix
            voxel_df = self.tfidf_df.loc[self.tfidf_df.index.isin(sel_ids)]
            term_df.loc[i_voxel] = voxel_df.mean(axis=0)
        values = term_df.values
        d = np.dot(np.dot(values.T, values), np.ones((values.shape[0], 1)))
        values_prime = np.dot(values, d**-.5)
        for i_parc in n_parcels:
            model = NMF(n_components=i_parc, init='nndsvd', random_state=0)
            W = model.fit_transform(values_prime)
            H = model.components_
            del W, H  # not sure what's next
