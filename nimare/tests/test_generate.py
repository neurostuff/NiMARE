"""Tests for the nimare.generate module."""

from contextlib import ExitStack as does_not_raise

import pytest
from numpy.random import RandomState

from nimare.dataset import Dataset
from nimare.generate import (
    _array_like,
    _create_foci,
    _create_source,
    create_coordinate_dataset,
    create_coordinate_studyset,
    create_neurovault_dataset,
    create_neurovault_studyset,
)
from nimare.nimads import Studyset


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


def test_create_coordinate_studyset():
    """Create a coordinate Studyset according to parameters."""
    ground_truth_foci, studyset = create_coordinate_studyset(
        foci=2,
        foci_percentage="60%",
        fwhm=10.0,
        sample_size=30,
        n_studies=5,
        n_noise_foci=1,
        seed=42,
        space="MNI",
    )

    assert isinstance(studyset, Studyset)
    assert len(studyset.ids) == 5
    assert len(ground_truth_foci) == 2
    assert not studyset.coordinates.empty


def test_create_neurovault_dataset():
    """Test creating a neurovault dataset."""
    dset = create_neurovault_dataset(
        collection_ids=(8836,),
        contrasts={"animal": "as-Animal"},
    )
    expected_columns = {"beta", "t", "varcope", "z"}
    assert expected_columns.issubset(dset.images.columns)


def test_create_neurovault_studyset():
    """Test creating a neurovault Studyset."""
    studyset = create_neurovault_studyset(
        collection_ids=(8836,),
        contrasts={"animal": "as-Animal"},
    )
    expected_columns = {"beta", "t", "varcope", "z"}
    assert isinstance(studyset, Studyset)
    assert expected_columns.issubset(studyset.images.columns)


def test_simulate_field_produces_real_peak_height_inflation():
    """The point simulator cannot validate a peak-height correction; the field one can.

    Drawing a value at the ground-truth location makes the reported statistic an unbiased
    estimate of the effect there, so the true inflation is exactly 1 and there is nothing for
    such a correction to recover. Selecting local maxima of a noisy field instead makes the
    reported height overstate the effect where it was found, which is the thing being
    corrected. Each peak carries the true effect at its own voxel, so it is measurable.
    """
    import numpy as np

    from nimare.generate import create_effect_size_coordinate_studyset
    from nimare.meta.cbma.effectsize import peak_stat_to_hedges_g

    sample_size = 25
    studyset = create_effect_size_coordinate_studyset(
        [(0, 0, 0)],
        effect_sizes=0.8,
        n_studies=25,
        sample_size=sample_size,
        seed=2,
        simulate_field=True,
        noise_extent=40.0,
        threshold_z=3.2905,
    )
    coordinates = studyset.coordinates
    assert len(coordinates) > 50
    assert "value_trueg" in coordinates.columns

    distance = np.sqrt((coordinates[["x", "y", "z"]].astype(float).to_numpy() ** 2).sum(axis=1))
    on_signal = distance <= 15.0
    assert on_signal.sum() >= 5

    z_values = np.abs(coordinates["z_stat"].astype(float).to_numpy())[on_signal]
    implied, _ = peak_stat_to_hedges_g(
        z_values, np.full(len(z_values), float(sample_size)), stat_type="z"
    )
    truth = np.abs(coordinates["value_trueg"].astype(float).to_numpy())[on_signal]

    # The reported statistic overstates the effect where its peak was found.
    inflation = truth.mean() / np.abs(implied).mean()
    assert 0.3 < inflation < 0.95

    # And most of what gets reported is noise, which is why peak magnitudes carry so little.
    assert on_signal.mean() < 0.3
