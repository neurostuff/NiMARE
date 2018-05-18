"""
Methods for decoding unthresholded brain maps into text.
"""
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask

from .base import Decoder
from ..meta.cbma import MKDAChi2, MKDAKernel
from ..due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Describes decoding methods using GC-LDA.')
def gclda_decode_roi(model, roi, topic_priors=None, prior_weight=1):
    pass


@due.dcite(Doi('10.1038/nmeth.1635'),
           description='Introduces Neurosynth.')
class CorrelationDecoder(Decoder):
    """
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, img, features=None, frequency_threshold=0.001,
            meta_estimator=None, target_image='specificity_z'):
        """
        Parameters
        ----------
        img : :obj:`nibabel.Nifti1.Nifti1Image`
            Input image to decode. Must have same affine/dimensions as dataset
            mask.
        features : :obj:`list`, optional
            List of features in dataset annotations to use for decoding.
            Default is None, which uses all features available.
        frequency_threshold : :obj:`float`, optional
            Threshold to apply to dataset annotations. Values greater than or
            equal to the threshold as assigned as label+, while values below
            the threshold are considered label-. Default is 0.001.
        meta_estimator : initialized :obj:`nimare.meta.cbma.base.CBMAEstimator`, optional
            Defaults to MKDAChi2.
        target_image : :obj:`str`, optional
            Image from ``meta_estimator``'s results to use for decoding.
            Dependent on estimator.
        """
        # Check that input image is compatible with dataset
        assert np.array_equal(img.affine, self.dataset.mask.affine)

        # Load input data
        input_data = apply_mask(img, self.dataset.mask)

        if meta_estimator is None:
            meta_estimator = MKDAChi2(self.dataset)

        if features is None:
            features = self.dataset.annotations.columns.values

        out_df = pd.DataFrame(index=features, columns=['r'],
                              data=np.zeros(len(features)))
        out_df.index.name = 'feature'

        for feature in features:
            # TODO: Search for !feature to get ids2, if possible. Will compare
            # between label+ and label- without analyzing unlabeled studies.
            ids = self.dataset.get(features=[feature],
                                   frequency_threshold=frequency_threshold)
            meta_estimator.fit(ids, corr='FDR')
            feature_data = apply_mask(meta_estimator.results[target_image],
                                      self.dataset.mask)
            corr = np.corrcoef(feature_data, input_data)[0, 1]
            out_df.loc[feature, 'r'] = corr

        return out_df
