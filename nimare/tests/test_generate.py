import numpy as np
import pytest

from ..generate import create_coordinate_dataset, create_source, create_foci


@pytest.mark.parametrize(
    "foci,studies,fwhm",
    [
        (2, 5, 10.0),
    ],
)
def test_create_foci(foci, studies, fwhm):
    assert create_foci(foci, studies, fwhm, rng=np.random.RandomState(seed=1939))


def test_create_source():
    source_dict = create_source(foci=[{"x": [0], "y": [0], "z": [0]}], sample_sizes=[25])
    assert source_dict["study-0"]["contrasts"]["1"]["metadata"]["sample_sizes"] == [25]


def test_create_coordinate_dataset():
    assert create_coordinate_dataset(
        foci=2,
        fwhm=10.0,
        sample_size_mean=30,
        sample_size_variance=10,
        studies=5,
        rng=np.random.RandomState(seed=1939),
    )
