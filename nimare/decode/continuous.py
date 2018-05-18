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
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, img, features=None, frequency_threshold=0.001,
            meta_estimator=MKDAChi2, kernel_estimator=MKDAKernel,
            target_image='specificity_z', **kwargs):
        # Check that input image is compatible with dataset
        assert np.array_equal(img.affine, self.dataset.mask.affine)

        meta_args = {k.split('meta__')[-1]: v for k, v in kwargs.items() if
                     k.startswith('meta__')}
        kernel_args = {k: v for k, v in kwargs.items() if
                       k.startswith('kernel__')}
        kwargs = {k: v for k, v in kwargs.items() if not
                  (k.startswith('kernel__') or k.startswith('meta__'))}

        # Load input data
        input_data = apply_mask(img, self.dataset.mask)

        if features is None:
            features = self.dataset.annotations.columns.values

        out_df = pd.DataFrame(index=features, columns=['r'],
                              data=np.zeros(len(features)))
        out_df.index.name = 'feature'

        for feature in features:
            ids = self.dataset.get(features=[feature],
                                   frequency_threshold=frequency_threshold)
            meta = meta_estimator(self.dataset, ids, **meta_args,
                                  **kernel_args)
            meta.fit(corr='FDR')
            feature_data = apply_mask(meta.results[target_image],
                                      self.dataset.mask)
            corr = np.corrcoef(feature_data, input_data)[0, 1]
            out_df.loc[feature, 'r'] = corr

        return out_df
