import pytest
from nimare.tests.conftest import testdata_cbma_full
from nimare.meta import cbma
from nimare import diagnostics

def test():
    # Since we can't easily get the pytest fixture, we'll import get_test_data_path
    import os
    from nimare.dataset import Dataset
    from nimare.tests.utils import get_test_data_path
    dset_file = os.path.join(get_test_data_path(), 'nidm_pain_dset.json')
    dset = Dataset(dset_file)
    meta = cbma.ALE()
    res = meta.fit(dset)
    jackknife = diagnostics.Jackknife(target_image='z', target_threshold=1.65, n_cores=1)
    results = jackknife.transform(res)
    print("Done")

if __name__ == '__main__':
    test()
