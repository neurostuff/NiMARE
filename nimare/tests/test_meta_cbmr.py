from nimare.meta.cbmr import CBMREstimator, CBMRInference
from nimare.utils import standardize_field
import logging

def test_CBMREstimator(testdata_cbmr_laird):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_laird, metadata=["publication_year", 'avg_age'])
    cbmr = CBMREstimator(group_names='diagnosis', moderators=['standardized_publication_year', 'standardized_avg_age'], spline_spacing=5, model='Poisson', penalty=False, lr=1e-2, tol=1e5, device='cuda')
    cbmr_res = cbmr.fit(dataset=dset)
    # p_map = cbmr_res.get_map('p')
    # p_vals = p_map.dataobj


def test_CBMRInference(testdata_cbmr_laird):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_laird, metadata=["publication_year", 'avg_age'])
    cbmr = CBMREstimator(group_names=['diagnosis', 'drug_status'], moderators=['standardized_publication_year', 'standardized_avg_age'], spline_spacing=5, model='Poisson', penalty=False, lr=1e-2, tol=1e5, device='cuda')
    cbmr_res = cbmr.fit(dataset=dset)
    inference = CBMRInference(CBMRResults=cbmr_res, spatial_homogeneity=True, t_con_group=[[1, 0, 0, 0], [0, 0, 0, 1]])
    a = inference._contrast()





    