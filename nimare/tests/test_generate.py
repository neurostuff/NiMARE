import numpy as np
from ..generate import create_coordinate_dataset, create_source, create_foci

def test_create_foci():
    create_foci(2, 5, fwhm=10., rng=np.random.RandomState(seed=1939))

def test_create_source():
    pass

def test_create_coordinate_dataset():
    create_coordinate_dataset(
        foci=2, 
        fwhm=10.,
        sample_size_mean=30,
        sample_size_variance=10,
        studies=5,
        rng=np.random.RandomState(seed=1939)
    )