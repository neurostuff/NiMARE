"""
Methods for decoding subsets of voxels (e.g., ROIs) or experiments (e.g., from
meta-analytic clustering on a database) into text.
"""
from .base import Decoder
from ...due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Citation for GCLDA decoding.')
class GCLDADiscreteDecoder(Decoder):
    def __init__(self, model, roi_img, topic_priors, prior_weight):
        pass


@due.dcite(Doi('10.1007/s00429-013-0698-0'),
           description='Citation for BrainMap-style decoding.')
class BrainMapDecoder(Decoder):
    """

    """
    def __init__(self, dataset, ids, frequency_threshold=0.001, u=0.05,
                 correction='fdr_bh'):
        pass


@due.dcite(Doi('10.1038/nmeth.1635'),
           description='Introduces Neurosynth.')
class NeurosynthDecoder(Decoder):
    """
    Performs discrete functional decoding according to Neurosynth's
    meta-analytic method. This does not employ correlations between
    unthresholded maps, which are the method of choice for decoding within
    Neurosynth and Neurovault.
    Metadata (i.e., feature labels) for studies within the selected sample
    (`ids`) are compared to the unselected studies remaining in the database
    (`dataset`).
    """
    def __init__(self, dataset, ids, frequency_threshold=0.001, prior=0.5,
                 u=0.05, correction='fdr_bh'):
        pass
