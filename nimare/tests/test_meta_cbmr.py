from nimare.meta.cbmr import CBMREstimator
from nimare.utils import standardize_field
import logging

def test_CBMREstimator(testdata_cbmr_full):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_full, metadata=["sample_sizes", 'avg_age'])
    cbmr = CBMREstimator(group_names='diagnosis', moderators=['standardized_sample_sizes', 'standardized_avg_age'], spline_spacing=5, model='Poisson', penalty=False, lr=1e-2, tol=1e5, device='cuda')
    cbmr_res = cbmr.fit(dataset=dset)
    # p_map = cbmr_res.get_map('p')
    # p_vals = p_map.dataobj



    