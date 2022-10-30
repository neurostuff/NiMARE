from nimare.meta.cbmr import CBMREstimator, CBMRInference
from nimare.utils import standardize_field
import logging

# def test_CBMREstimator(testdata_cbmr_simulated):
#     logging.getLogger().setLevel(logging.DEBUG)
#     """Unit test for CBMR estimator."""
#     dset = standardize_field(dataset=testdata_cbmr_simulated, metadata=["sample_sizes", 'avg_age'])
#     cbmr = CBMREstimator(group_names='diagnosis', moderators=['standardized_sample_sizes', 'standardized_avg_age'], spline_spacing=5, model='Poisson', penalty=False, lr=1e-2, tol=1e4, device='cuda')
#     cbmr_res = cbmr.fit(dataset=dset)


def test_CBMRInference(testdata_cbmr_simulated):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_simulated, metadata=["sample_sizes", 'avg_age'])
    cbmr = CBMREstimator(group_names=['diagnosis', 'drug_status'], moderators=['standardized_sample_sizes', 'standardized_avg_age'], spline_spacing=10, model='NB', penalty=False, lr=1e-6, tol=1e6, device='cuda')
    cbmr_res = cbmr.fit(dataset=dset)
    inference = CBMRInference(CBMRResults=cbmr_res, t_con_group=[[[1,0,0,0],[0,0,1,0]], [1, 0, 0, 0]], t_con_moderator=[[[1,0],[0,1]], [1, -1]], device='cuda') # [[2, 0, 0, -2], [0, -2, 1, 1]]
    a = inference._contrast()
    
    # [[[1,0,0,0],[0,0,1,0]], [1, 0, 0, 0]]
    # [[[1,0],[0,1]], [1, -1]]





   