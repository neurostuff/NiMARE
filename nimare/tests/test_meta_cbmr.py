from nimare.meta.cbmr import CBMREstimator
from nimare.utils import standardize_field
import logging

def test_CBMREstimator(testdata_cbmr_full):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_full, metadata=["sample_sizes", 'avg_age'])
    cbmr = CBMREstimator(group_names='diagnosis', moderators=['standardized_sample_sizes', 'standardized_avg_age'], model='clustered_NB', penalty=False, lr=0.1, tol=1)
    # prep = cbmr._preprocess_input(dset)
    cbmr.fit(dataset=dset)
    