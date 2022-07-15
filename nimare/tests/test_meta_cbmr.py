from nimare.meta.cbmr import CBMREstimator


def test_CBMREstimator(testdata_cbmr):

    cbmr = CBMREstimator(moderators=['sample_sizes', 'avg_age'])
    prep = cbmr._preprocess_input(testdata_cbmr)
    fit = cbmr._fit(dataset=testdata_cbmr, spline_spacing=5)


