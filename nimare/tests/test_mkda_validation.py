"""Test validation in MKDAChi2.correct_fwe_montecarlo."""
import os
import pytest
import nimare
from nimare.meta.cbma.mkda import MKDAChi2
from nimare.correct import FWECorrector
from nimare.tests.utils import get_test_data_path

def test_correct_fwe_montecarlo_missing_inputs():
    """Test that correct_fwe_montecarlo raises error when inputs_ is missing."""
    # Load test dataset
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    ds = nimare.dataset.Dataset(dset_file)
    
    # Split into two datasets
    all_ids = ds.ids
    half_point = len(all_ids) // 2
    term_ids = all_ids[:half_point]
    notterm_ids = all_ids[half_point:]
    
    term_dset = ds.slice(term_ids)
    notterm_dset = ds.slice(notterm_ids)
    
    # Fit the estimator
    meta = MKDAChi2()
    results = meta.fit(term_dset, notterm_dset)
    
    # Corrupt the estimator by removing inputs_
    results.estimator.inputs_ = None
    
    # Try to run FWE correction - should raise ValueError
    corrector = FWECorrector(method='montecarlo', voxel_thresh=0.0005, n_iters=5, n_cores=1)
    
    with pytest.raises(ValueError, match="does not have valid inputs_"):
        corrector.transform(results)
        

def test_correct_fwe_montecarlo_missing_coordinates1():
    """Test that correct_fwe_montecarlo raises error when coordinates1 is missing."""
    # Load test dataset
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    ds = nimare.dataset.Dataset(dset_file)
    
    # Split into two datasets
    all_ids = ds.ids
    half_point = len(all_ids) // 2
    term_ids = all_ids[:half_point]
    notterm_ids = all_ids[half_point:]
    
    term_dset = ds.slice(term_ids)
    notterm_dset = ds.slice(notterm_ids)
    
    # Fit the estimator
    meta = MKDAChi2()
    results = meta.fit(term_dset, notterm_dset)
    
    # Corrupt the estimator by removing coordinates1
    del results.estimator.inputs_['coordinates1']
    
    # Try to run FWE correction - should raise ValueError
    corrector = FWECorrector(method='montecarlo', voxel_thresh=0.0005, n_iters=5, n_cores=1)
    
    with pytest.raises(ValueError, match="Missing 'coordinates1'"):
        corrector.transform(results)


def test_correct_fwe_montecarlo_none_coordinates1():
    """Test that correct_fwe_montecarlo raises error when coordinates1 is None."""
    # Load test dataset
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    ds = nimare.dataset.Dataset(dset_file)
    
    # Split into two datasets
    all_ids = ds.ids
    half_point = len(all_ids) // 2
    term_ids = all_ids[:half_point]
    notterm_ids = all_ids[half_point:]
    
    term_dset = ds.slice(term_ids)
    notterm_dset = ds.slice(notterm_ids)
    
    # Fit the estimator
    meta = MKDAChi2()
    results = meta.fit(term_dset, notterm_dset)
    
    # Corrupt the estimator by setting coordinates1 to None
    results.estimator.inputs_['coordinates1'] = None
    
    # Try to run FWE correction - should raise ValueError
    corrector = FWECorrector(method='montecarlo', voxel_thresh=0.0005, n_iters=5, n_cores=1)
    
    with pytest.raises(ValueError, match="Invalid 'coordinates1'"):
        corrector.transform(results)


def test_correct_fwe_montecarlo_empty_coordinates1():
    """Test that correct_fwe_montecarlo raises error when coordinates1 is empty."""
    import pandas as pd
    
    # Load test dataset
    dset_file = os.path.join(get_test_data_path(), "test_pain_dataset.json")
    ds = nimare.dataset.Dataset(dset_file)
    
    # Split into two datasets
    all_ids = ds.ids
    half_point = len(all_ids) // 2
    term_ids = all_ids[:half_point]
    notterm_ids = all_ids[half_point:]
    
    term_dset = ds.slice(term_ids)
    notterm_dset = ds.slice(notterm_ids)
    
    # Fit the estimator
    meta = MKDAChi2()
    results = meta.fit(term_dset, notterm_dset)
    
    # Corrupt the estimator by setting coordinates1 to an empty DataFrame
    empty_df = pd.DataFrame(columns=results.estimator.inputs_['coordinates1'].columns)
    results.estimator.inputs_['coordinates1'] = empty_df
    
    # Try to run FWE correction - should raise ValueError
    corrector = FWECorrector(method='montecarlo', voxel_thresh=0.0005, n_iters=5, n_cores=1)
    
    with pytest.raises(ValueError, match="DataFrame in estimator inputs_ is empty"):
        corrector.transform(results)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
