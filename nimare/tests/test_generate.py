import numpy as np
import pytest
from contextlib import ExitStack as does_not_raise

from ..generate import create_coordinate_dataset, create_source, create_foci
from ..dataset import Dataset


@pytest.mark.parametrize(
    "foci,studies,fwhm,rng,space,expectation",
    [
        (2, 5, 10.0, np.random.RandomState(seed=1939), "MNI", does_not_raise()),
        (np.array([[44, 61, 70]]), 5, 10.0, None, "MNI", does_not_raise()),
        (2, 5, [10.0, 6.0], None, "MNI", does_not_raise()),
        (2, 5, 10.0, None, "INVALID_SPACE", pytest.raises(NotImplementedError)),
        (2, 5, "INVALID_FWHM", None, "MNI", pytest.raises(ValueError)),
        (2, 5, [10.0], None, "MNI", pytest.raises(ValueError)),
    ],
)
def test_create_foci(foci, studies, fwhm, rng, space, expectation):
    with expectation:
        ground_truth_foci, foci_dict = create_foci(foci, studies, fwhm, rng=rng, space=space)
    if isinstance(expectation, does_not_raise):
        assert all(isinstance(key, int) for key in foci_dict)
        assert all(isinstance(coord, tuple) for coord in ground_truth_foci)


def test_create_source():
    source_dict = create_source(foci={0: [(0, 0, 0)]}, sample_sizes=[25])
    assert source_dict["study-0"]["contrasts"]["1"]["metadata"]["sample_sizes"] == [25]


def test_create_coordinate_dataset():
    n_foci = 2
    ground_truth_foci, dataset = create_coordinate_dataset(
        foci=n_foci,
        fwhm=10.0,
        sample_size_mean=30,
        sample_size_variance=10,
        studies=5,
        rng=np.random.RandomState(seed=1939),
    )
    assert len(ground_truth_foci) == n_foci
    assert isinstance(dataset, Dataset)
