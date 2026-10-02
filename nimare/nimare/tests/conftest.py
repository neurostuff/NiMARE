import pathlib

def pytest_ignore_collect(collection_path, config):
    """Skip CBMR-related test modules that require statsmodels.

    The CBMR tests depend on the statsmodels package, which cannot load
    its compiled DLLs in the current corporate environment (Application
    Control policy blocks _cfa_simulation_smoother). To avoid collection
    errors we tell pytest to ignore any test file whose name contains the
    substring 	est_meta_cbmr.
    """
    try:
        filename = collection_path.name
    except Exception:
        filename = str(collection_path)
    return "test_meta_cbmr" in filename
