import pytest
from contextlib import ExitStack as does_not_raise

from ..generate import create_coordinate_dataset, create_source, create_foci
from ..dataset import Dataset


@pytest.mark.parametrize(
    "kwargs,expectation",
    [
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": [1, 0.1],
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="list_foci_weights",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "n_studies": 5,
                "foci_coords": [(0, 0, 0)],
                "n_noise_foci": 2,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="specify_input_foci",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": [10.0, 6.0],
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="list_of_fwhm",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
                "space": "INVALID_SPACE",
            },
            pytest.raises(NotImplementedError),
            id="invalid_space",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": "INVALID_FWHM",
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="invalid_fwhm",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": [10.0],
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="invalid_list_of_fwhm",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": "INVALID_FOCI_WEIGHTS",
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="invalid_foci_weights",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": [0.1],
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="wrong_length_foci_weights",
        ),
        pytest.param(
            {
                "n_foci": [10, 1, 5],
                "fwhm": [8.0],
                "n_studies": 3,
                "foci_coords": [(0, 0, 0)],
                "n_noise_foci": [10, 20, 30],
                "foci_weights": [0.1],
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="all_manual_specifications",
        ),
        pytest.param(
            {
                "n_foci": [1, 2, 2],
                "fwhm": [8.0, 10.0],
                "n_studies": 3,
                "foci_coords": None,
                "n_noise_foci": [10, 20, 30],
                "foci_weights": [0.1, 1],
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="n_foci_list_but_no_coords",
        ),
        pytest.param(
            {
                "n_foci": [1, 0, 0],
                "fwhm": 10.0,
                "n_studies": 3,
                "foci_coords": None,
                "n_noise_foci": 5,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="zero_foci_for_some_studies",
        ),
        pytest.param(
            {
                "n_foci": 0,
                "fwhm": 10.0,
                "n_studies": 3,
                "foci_coords": [(0, 0, 0)],
                "n_noise_foci": None,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(TypeError),
            id="zero_foci_and_zero_noise_foci",
        ),
        pytest.param(
            {
                "n_foci": 0,
                "fwhm": 10.0,
                "n_studies": 3,
                "foci_coords": [(0, 0, 0)],
                "n_noise_foci": 10,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="only_noise_foci",
        ),
        pytest.param(
            {
                "n_foci": [1, 2],
                "fwhm": 10.0,
                "n_studies": 3,
                "foci_coords": [(0, 0, 0)],
                "n_noise_foci": 10,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="incorrect_n_foci_length",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "n_studies": 3,
                "foci_coords": "INVALID_FOCI_COORDS",
                "n_noise_foci": 10,
                "foci_weights": None,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="invalid_foci_coords",
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


@pytest.mark.parametrize(
    "kwargs,expectation",
    [
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "sample_size": 30,
                "sample_size_interval": 10,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
            },
            does_not_raise(),
            id="random_sample_size",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "sample_size": [30] * 5,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
            },
            does_not_raise(),
            id="specified_sample_size",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "sample_size": [30] * 4,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
            },
            pytest.raises(ValueError),
            id="incorrect_sample_size_list",
        ),
        pytest.param(
            {
                "n_foci": 2,
                "fwhm": 10.0,
                "sample_size": 30,
                "sample_size_interval": None,
                "n_studies": 5,
                "foci_coords": None,
                "n_noise_foci": 0,
                "foci_weights": None,
                "seed": 42,
            },
            pytest.raises(ValueError),
            id="not_specifiying_variance",
        ),
    ],
)
def test_create_coordinate_dataset(kwargs, expectation):
    with expectation:
        ground_truth_foci, dataset = create_coordinate_dataset(**kwargs)
    if isinstance(expectation, does_not_raise):
        assert isinstance(dataset, Dataset)
        assert len(dataset.ids) == kwargs["n_studies"]
