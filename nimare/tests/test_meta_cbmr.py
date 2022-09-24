from nimare.meta.cbmr import CBMREstimator
from nimare.utils import standardize_field
import logging

def test_CBMREstimator(testdata_cbmr_full):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_full, metadata=["sample_sizes", 'avg_age'])
    cbmr = CBMREstimator(group_names='diagnosis', moderators=['standardized_sample_sizes', 'standardized_avg_age'], spline_spacing=5, model='Poisson', penalty=False, lr=1e-2, tol=1e5, device='cuda')
    # prep = cbmr._preprocess_input(dset)
    cbmr.fit(dataset=dset)


    