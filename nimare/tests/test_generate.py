"""Tests for the nimare.generate module."""
from contextlib import ExitStack as does_not_raise

import pytest
from numpy.random import RandomState

from ..dataset import Dataset
from ..generate import (
    _array_like,
    _create_foci,
    _create_source,
    create_coordinate_dataset,
    create_neurovault_dataset,
)


@pytest.mark.parametrize(
    "kwargs,expectation",
    [
        pytest.param(
            {
                "foci": [(0, 0, 0)],
                "foci_percentage": "60%",
                "fwhm": 10.0,
                "n_studies": 5,
                "n_noise_foci": 2,
                "rng": RandomState(seed=42),
                "space": "MNI",
            },
            does_not_raise(),
            id="specify_foci_coord",
        ),
        pytest.param(
            {
                "foci": 1,
                "foci_percentage": "60%",
                "fwhm": 10.0,
                "n_studies": 5,
                "n_noise_foci": 2,
                "rng": RandomState(seed=42),
                "space": "MNI",
            },
            does_not_raise(),
            id="integer_foci",
        ),
        pytest.param(
            {
                "foci": 0,
                "foci_percentage": "60%",
                "fwhm": 10.0,
                "n_studies": 5,
                "n_noise_foci": 0,
                "rng": RandomState(seed=42),
                "space": "MNI",
            },
            does_not_raise(),
            id="no_foci",
        ),
    ],
)
def test_create_foci(kwargs, expectation):
    """Smoke test for _create_foci."""
    with expectation:
        ground_truth_foci, foci_dict = _create_foci(**kwargs)
    if isinstance(expectation, does_not_raise):
        assert all(isinstance(key, int) for key in foci_dict)
        assert all(isinstance(coord, tuple) for coord in ground_truth_foci)


def test_create_source():
    """Smoke test for _create_source."""
    source_dict = _create_source(foci={0: [(0, 0, 0)]}, sample_sizes=[25])
    assert source_dict["study-0"]["contrasts"]["1"]["metadata"]["sample_sizes"] == [25]


@pytest.mark.parametrize(
    "kwargs,expectation",
    [
        pytest.param(
            {
                "foci": 2,
                "foci_percentage": 1.0,
                "fwhm": 10.0,
                "sample_size": (10, 20),
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="random_sample_size",
        ),
        pytest.param(
            {
                "foci": [(0, 0, 0), (0, 10, 10)],
                "foci_percentage": "100%",
                "fwhm": 10.0,
                "sample_size": [30] * 5,
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="specified_sample_size",
        ),
        pytest.param(
            {
                "foci": 2,
                "fwhm": 10.0,
                "sample_size": [30] * 4,
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="incorrect_sample_size_list",
        ),
        pytest.param(
            {
                "foci": 0,
                "foci_percentage": 1.0,
                "fwhm": 10.0,
                "sample_size": (10, 20),
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="no_foci",
        ),
        pytest.param(
            {
                "foci": 0,
                "foci_percentage": "50%",
                "fwhm": 10.0,
                "sample_size": (10, 20),
                "n_studies": 5,
                "n_noise_foci": 10,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="only_noise_foci",
        ),
        pytest.param(
            {
                "foci": 1,
                "foci_percentage": "50%",
                "fwhm": 10.0,
                "sample_size": (10, 20),
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            does_not_raise(),
            id="insufficient_foci",
        ),
        pytest.param(
            {
                "foci": "INVALID_FOCI",
                "foci_percentage": "50%",
                "fwhm": 10.0,
                "sample_size": (10, 20),
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="invalid_foci",
        ),
        pytest.param(
            {
                "foci": 1,
                "foci_percentage": "INVALID_PERCENT",
                "fwhm": 10.0,
                "sample_size": (10, 20),
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="invalid_percent",
        ),
        pytest.param(
            {
                "foci": 1,
                "foci_percentage": "60%",
                "fwhm": 10.0,
                "sample_size": "INVALID_SAMPLE_SIZE",
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "MNI",
            },
            pytest.raises(ValueError),
            id="invalid_sample_size",
        ),
        pytest.param(
            {
                "foci": 1,
                "foci_percentage": "60%",
                "fwhm": 10.0,
                "sample_size": 30,
                "n_studies": 5,
                "n_noise_foci": 0,
                "seed": 42,
                "space": "INVALID_SPACE",
            },
            pytest.raises(NotImplementedError),
            id="invalid_space",
        ),
    ],
)
def test_create_coordinate_dataset(kwargs, expectation):
    """Create a coordinate Dataset according to parameters."""
    with expectation:
        ground_truth_foci, dataset = create_coordinate_dataset(**kwargs)
    if isinstance(expectation, does_not_raise):
        assert isinstance(dataset, Dataset)
        assert len(dataset.ids) == kwargs["n_studies"]
        # test if the number of observed coordinates in the dataset is correct
        if _array_like(kwargs["foci"]):
            n_foci = len(kwargs["foci"])
        else:
            n_foci = kwargs["foci"]
        expected_coordinate_number = max(
            kwargs["n_studies"],
            (kwargs["n_studies"] * n_foci) + (kwargs["n_studies"] * kwargs["n_noise_foci"]),
        )
        assert len(dataset.coordinates) == expected_coordinate_number


def test_create_neurovault_dataset():
    """Test creating a neurovault dataset."""
    dset = create_neurovault_dataset(
        collection_ids=(8836,),
        contrasts={"animal": "as-Animal"},
    )
    expected_columns = {"beta", "t", "varcope", "z"}
    assert expected_columns.issubset(dset.images.columns)
