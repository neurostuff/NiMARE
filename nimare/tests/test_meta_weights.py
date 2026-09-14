"""Tests for nimare.meta.cbma.weights."""

import logging

import numpy as np
import pytest

from nimare.meta.cbma.weights import (
    StudyWeights,
    normalize_weights,
    resolve_weighting,
)
from nimare.studyset import Studyset


def _studyset(sample_sizes, extra_metadata=None):
    """Build a minimal Studyset with one analysis per study and given sample sizes."""
    studies = []
    for i_study, n in enumerate(sample_sizes):
        metadata = {}
        if n is not None:
            metadata["sample_sizes"] = [n]
        if extra_metadata is not None and extra_metadata[i_study] is not None:
            metadata.update(extra_metadata[i_study])
        studies.append(
            {
                "id": f"S{i_study}",
                "name": f"study {i_study}",
                "analyses": [
                    {
                        "id": f"A{i_study}",
                        "metadata": metadata,
                        "points": [{"space": "MNI", "coordinates": [i_study, 2, 3]}],
                    }
                ],
            }
        )
    return Studyset({"id": "weights-test", "studies": studies})


def test_sqrt_sample_size_weights():
    """The published weight is the square root of the contrast's sample size."""
    studyset = _studyset([16, 25, 100])
    weights = StudyWeights().raw_weights(studyset, studyset.ids)

    np.testing.assert_allclose(weights.to_numpy(), [4.0, 5.0, 10.0])
    assert list(weights.index) == list(studyset.ids)


@pytest.mark.parametrize(
    "transform,expected",
    [("sqrt", [4.0, 5.0]), ("linear", [16.0, 25.0]), ("none", [1.0, 1.0])],
)
def test_transform_options(transform, expected):
    """`transform` selects what is done to the sample size before weighting."""
    studyset = _studyset([16, 25])
    weights = StudyWeights(transform=transform).raw_weights(studyset, studyset.ids)
    np.testing.assert_allclose(weights.to_numpy(), expected)


def test_fixed_effects_discount_is_opt_in(caplog):
    """No inference_field means no contrast is discounted."""
    studyset = _studyset(
        [16, 25],
        extra_metadata=[{"inference": "fixed"}, {"inference": "random"}],
    )
    undiscounted = StudyWeights().raw_weights(studyset, studyset.ids)
    np.testing.assert_allclose(undiscounted.to_numpy(), [4.0, 5.0])

    discounted = StudyWeights(inference_field="inference").raw_weights(studyset, studyset.ids)
    np.testing.assert_allclose(discounted.to_numpy(), [4.0 * 0.75, 5.0])


@pytest.mark.parametrize("label", ["Fixed", "  FFX ", "fixed-effects", "FE"])
def test_fixed_effects_labels_match_case_insensitively(label):
    """Fixed-effects labels are matched after stripping case and whitespace."""
    studyset = _studyset([16, 25], extra_metadata=[{"inference": label}, {"inference": "rfx"}])
    weights = StudyWeights(inference_field="inference").raw_weights(studyset, studyset.ids)
    np.testing.assert_allclose(weights.to_numpy(), [3.0, 5.0])


def test_partial_inference_labels_warn(caplog):
    """Partial labelling promotes the unlabelled contrasts, so it has to be visible."""
    studyset = _studyset([16, 25], extra_metadata=[{"inference": "fixed"}, None])
    with caplog.at_level(logging.WARNING, logger="nimare.meta.cbma.weights"):
        StudyWeights(inference_field="inference").raw_weights(studyset, studyset.ids)
    assert "1 of 2 contrasts" in caplog.text


def test_unknown_inference_field_raises():
    """A misspelled metadata field must not silently mean 'no discount'."""
    studyset = _studyset([16, 25])
    with pytest.raises(ValueError, match="is not a metadata field"):
        StudyWeights(inference_field="nope").raw_weights(studyset, studyset.ids)


def test_missing_sample_size_imputes_mean_weight(caplog):
    """Missing weights take the mean of the valid ones, as the CANlab implementation does."""
    studyset = _studyset([16, 25, None])
    with caplog.at_level(logging.WARNING, logger="nimare.meta.cbma.weights"):
        weights = StudyWeights().raw_weights(studyset, studyset.ids)

    np.testing.assert_allclose(weights.to_numpy(), [4.0, 5.0, 4.5])
    assert "imputing the mean weight" in caplog.text


def test_zero_sample_size_is_treated_as_missing():
    """A zero sample size would zero out a contrast entirely; impute instead."""
    studyset = _studyset([16, 25, 0])
    weights = StudyWeights().raw_weights(studyset, studyset.ids)
    np.testing.assert_allclose(weights.to_numpy(), [4.0, 5.0, 4.5])


def test_on_missing_raise():
    """`on_missing='raise'` refuses rather than imputing."""
    studyset = _studyset([16, 25, None])
    with pytest.raises(ValueError, match="missing or non-positive weight"):
        StudyWeights(on_missing="raise").raw_weights(studyset, studyset.ids)


def test_all_weights_invalid_raises():
    """There is nothing to impute from when every contrast is unusable."""
    studyset = _studyset([0, 0])
    with pytest.raises(ValueError, match="No contrast has a usable weight"):
        StudyWeights().raw_weights(studyset, studyset.ids)


def test_no_sample_size_metadata_raises():
    """Asking for sample-size weighting without sample sizes is an error, not a warning."""
    studyset = _studyset([None, None])
    with pytest.raises(ValueError, match="has no 'sample_sizes' or 'sample_size' metadata"):
        StudyWeights().raw_weights(studyset, studyset.ids)


def test_explicit_weights_by_mapping():
    """A mapping of study ID to weight covers arbitrary analyst-supplied quality weights."""
    studyset = _studyset([16, 25])
    ids = list(studyset.ids)
    weights = StudyWeights(source={ids[0]: 3.0, ids[1]: 7.0}).raw_weights(studyset, ids)
    np.testing.assert_allclose(weights.to_numpy(), [3.0, 7.0])


def test_explicit_weights_by_array_checks_length():
    """A positional array must match the number of studies."""
    studyset = _studyset([16, 25])
    with pytest.raises(ValueError, match="3 weights for 2 studies"):
        StudyWeights(source=[1.0, 2.0, 3.0]).raw_weights(studyset, studyset.ids)


def test_uniform_source():
    """`source='uniform'` is the explicit way to ask for no weighting."""
    studyset = _studyset([16, 25])
    weights = StudyWeights(source="uniform").raw_weights(studyset, studyset.ids)
    np.testing.assert_allclose(weights.to_numpy(), [1.0, 1.0])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"source": "bogus"}, "Invalid source"),
        ({"transform": "log"}, "Invalid transform"),
        ({"reduce": "median"}, "Invalid reduce"),
        ({"on_missing": "drop"}, "Invalid on_missing"),
        ({"fixed_effects_discount": 0}, "positive, finite"),
        ({"fixed_effects_discount": np.nan}, "positive, finite"),
    ],
)
def test_invalid_arguments(kwargs, match):
    """Bad configuration fails at construction, not at fit time."""
    with pytest.raises(ValueError, match=match):
        StudyWeights(**kwargs)


def test_normalize_weights_sums_to_n():
    """Normalised weights sum to the number of contrasts."""
    weights = normalize_weights([4.0, 5.0, 10.0], 3)
    assert weights.sum() == pytest.approx(3.0)
    np.testing.assert_allclose(weights, 3 * np.array([4.0, 5.0, 10.0]) / 19.0)


def test_normalize_weights_is_exactly_one_when_uniform():
    """Equal inputs give exactly 1.0, so an equal-N analysis matches an unweighted one."""
    for value in (1.0, 7.0, 0.001):
        weights = normalize_weights([value] * 5, 5)
        assert np.all(weights == 1.0)


def test_normalize_weights_rejects_degenerate_totals():
    """A zero or non-finite total has no meaningful normalisation."""
    with pytest.raises(ValueError, match="positive, finite"):
        normalize_weights([0.0, 0.0], 2)


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("sample_size", StudyWeights), ("uniform", StudyWeights)],
)
def test_resolve_weighting(value, expected):
    """Strings are shorthand for a StudyWeights instance."""
    resolved = resolve_weighting(value)
    assert resolved is None if expected is None else isinstance(resolved, expected)


def test_resolve_weighting_passes_through_instances():
    """An explicit instance is used as given."""
    weights = StudyWeights(transform="linear")
    assert resolve_weighting(weights) is weights


def test_resolve_weighting_rejects_junk():
    """Anything else is a configuration error."""
    with pytest.raises(ValueError, match="Invalid weighting"):
        resolve_weighting("sqrt_n")
    with pytest.raises(TypeError, match="must be None"):
        resolve_weighting(3.0)


def test_counts_recorded_for_the_methods_description():
    """The description reports how many contrasts were discounted and imputed."""
    studyset = _studyset(
        [16, 25, None, 36],
        extra_metadata=[{"inference": "fixed"}, {"inference": "random"}, None, None],
    )
    weighting = StudyWeights(inference_field="inference")
    weighting.raw_weights(studyset, studyset.ids)

    assert weighting.n_fixed_effects_ == 1
    assert weighting.n_imputed_ == 1


def test_counts_reset_between_calls():
    """Counts describe the most recent call, not every call ever made."""
    weighting = StudyWeights(inference_field="inference")
    messy = _studyset([16, None], extra_metadata=[{"inference": "ffx"}, {"inference": "rfx"}])
    weighting.raw_weights(messy, messy.ids)
    assert (weighting.n_fixed_effects_, weighting.n_imputed_) == (1, 1)

    clean = _studyset([16, 25], extra_metadata=[{"inference": "rfx"}, {"inference": "rfx"}])
    weighting.raw_weights(clean, clean.ids)
    assert (weighting.n_fixed_effects_, weighting.n_imputed_) == (0, 0)
