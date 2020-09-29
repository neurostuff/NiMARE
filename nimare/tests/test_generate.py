import numpy as np
import pytest
from contextlib import ExitStack as does_not_raise

from ..generate import create_coordinate_dataset, create_source, create_foci
from ..dataset import Dataset


@pytest.mark.parametrize(
    "kwargs,expectation",
    [
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": 10.0,
                "studies": 5,
                "foci_coords": None,
                "foci_noise": 0,
                "foci_weights": [1, 0.1],
                "rng": np.random.RandomState(seed=1939),
                "space": "MNI",
            },
            does_not_raise(),
            id="test_foci_weights",
        ),
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": 10.0,
                "studies": 5,
                "foci_coords": [(0, 0, 0)],
                "foci_noise": 2,
                "foci_weights": None,
                "rng": None,
                "space": "MNI",
            },
            does_not_raise(),
            id="test_input_foci",
        ),
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": [10.0, 6.0],
                "studies": 5,
                "foci_coords": None,
                "foci_noise": 0,
                "foci_weights": None,
                "rng": None,
                "space": "MNI",
            },
            does_not_raise(),
            id="test_list_of_fwhm",
        ),
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": 10.0,
                "studies": 5,
                "foci_coords": None,
                "foci_noise": 0,
                "foci_weights": None,
                "rng": None,
                "space": "INVALID_SPACE",
            },
            pytest.raises(NotImplementedError),
            id="test_invalid_space",
        ),
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": "INVALID_FWHM",
                "studies": 5,
                "foci_coords": None,
                "foci_noise": 0,
                "foci_weights": None,
                "rng": None,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="test_invalid_fwhm",
        ),
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": [10.0],
                "studies": 5,
                "foci_coords": None,
                "foci_noise": 0,
                "foci_weights": None,
                "rng": None,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="test_invalid_list_of_fwhm",
        ),
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": 10.0,
                "studies": 5,
                "foci_coords": None,
                "foci_noise": 0,
                "foci_weights": "INVALID_FOCI_WEIGHTS",
                "rng": None,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="test_invalid_foci_weights",
        ),
        pytest.param(
            {
                "foci_num": 2,
                "fwhm": 10.0,
                "studies": 5,
                "foci_coords": None,
                "foci_noise": 0,
                "foci_weights": [0.1],
                "rng": None,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="test_wrong_length_foci_weights",
        ),
        pytest.param(
            {
                "foci_num": [10, 1, 5],
                "fwhm": [8.0],
                "studies": 3,
                "foci_coords": [(0, 0, 0)],
                "foci_noise": [10, 20, 30],
                "foci_weights": [0.1],
                "rng": None,
                "space": "MNI",
            },
            does_not_raise(),
            id="test_all_manual_specifications",
        ),
        pytest.param(
            {
                "foci_num": [1, 2, 2],
                "fwhm": [8.0, 10.0],
                "studies": 3,
                "foci_coords": None,
                "foci_noise": [10, 20, 30],
                "foci_weights": [0.1, 1],
                "rng": None,
                "space": "MNI",
            },
            does_not_raise(),
            id="test_foci_num_list_but_no_coords",
        ),
    ],
)
def test_create_foci(kwargs, expectation):
    with expectation:
        ground_truth_foci, foci_dict = create_foci(**kwargs)
    if isinstance(expectation, does_not_raise):
        assert all(isinstance(key, int) for key in foci_dict)
        assert all(isinstance(coord, tuple) for coord in ground_truth_foci)


def test_create_source():
    source_dict = create_source(foci={0: [(0, 0, 0)]}, sample_sizes=[25])
    assert source_dict["study-0"]["contrasts"]["1"]["metadata"]["sample_sizes"] == [25]


def test_create_coordinate_dataset():
    n_foci = 2
    ground_truth_foci, dataset = create_coordinate_dataset(
        foci_num=n_foci,
        fwhm=10.0,
        sample_size=30,
        sample_size_variance=10,
        studies=5,
        foci_coords=None,
        foci_noise=0,
        foci_weights=None,
        rng=np.random.RandomState(seed=1939),
    )
    assert len(ground_truth_foci) == n_foci
    assert isinstance(dataset, Dataset)
