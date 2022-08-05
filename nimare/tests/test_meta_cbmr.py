from nimare.meta.cbmr import CBMREstimator
import logging

def test_CBMREstimator(testdata_cbmr_full, caplog):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    cbmr = CBMREstimator(group_names=['diagnosis'], moderators=['sample_sizes', 'avg_age'], model='Poisson', penalty=False, lr=0.1, tol=1e8)
    prep = cbmr._preprocess_input(testdata_cbmr_full)
    cbmr.fit(dataset=testdata_cbmr_full)
    print('1234')
    # with caplog.at_level(logging.DEBUG, logger="nimare.meta.cbma.base"):
    #     meta.fit(testdata_cbma)
    # assert "Loading pre-generated MA maps" not in caplog.text
