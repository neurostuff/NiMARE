from nimare.meta.cbmr import CBMREstimator


def test_CBMREstimator(testdata_cbmr):

    cbmr = CBMREstimator()
    X = cbmr._preprocess_input(testdata_cbmr)
    cbmr.fit(testdata_cbmr)
