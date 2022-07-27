from nimare.meta.cbmr import CBMREstimator


def test_CBMREstimator(testdata_cbmr):
    """Unit test for CBMR estimator."""
    cbmr = CBMREstimator(moderators=['sample_sizes', 'avg_age'], moderators_center=['sample_sizes', 'avg_age'], moderators_scale=['sample_sizes', 'avg_age'])
    prep = cbmr._preprocess_input(testdata_cbmr)
    fit = cbmr._fit(dataset=testdata_cbmr, model='Poisson', penalty=False, tol=1e8)
