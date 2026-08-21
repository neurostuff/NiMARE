"""End-to-end regression fixtures for CBMR fits.

These pin the whole path a user actually drives -- Studyset in, maps and tables out -- which
:mod:`nimare.tests.test_meta_cbmr_glm_equivalence` deliberately bypasses by constructing
inputs directly. Between them the two files answer different questions: the equivalence tests
prove the numbers are *correct*, against an independent GLM implementation, while these prove
they are *unchanged*, which is what catches an accidental behavior change during a refactor
that is meant to be a pure move.

.. warning::
    These fixtures pin an optimizer *trajectory*, not a converged fit. The Poisson likelihood
    has no interior maximum for coordinate-based foci -- because no experiment reports two
    foci in one voxel, the per-experiment-voxel counts are 0/1 and the coefficients run to the
    boundary rather than an optimum. The pinned settings below (notably ``tol=1e4``) stop the
    optimizer early and deterministically, which is reproducible but says nothing about the
    maximum likelihood estimate. Do not tighten ``tol`` here expecting better fixtures; it
    makes them worse. Correctness lives in the equivalence tests, which simulate counts with
    an interior maximum instead.

Regenerate after an intended behavior change::

    python -m nimare.tests.test_meta_cbmr_golden

and review the resulting diff rather than accepting it blindly.
"""

import os
import warnings

import numpy as np
import pytest

from nimare.tests.utils import get_test_data_path

try:
    import torch  # noqa: F401
except ImportError:
    warnings.warn("Torch not installed. CBMR golden fixture tests will be skipped.", stacklevel=2)
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta import CBMREstimator, models

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

FIXTURE_DIR = os.path.join(get_test_data_path(), "cbmr_golden")

GROUP_CATEGORIES = ["diagnosis", "drug_status"]
MODERATORS = ["standardized_sample_sizes", "standardized_avg_age"]

# Pinned so the fit is reproducible. These mirror what the rest of the CBMR suite uses; see the
# module warning for why the loose tolerance is deliberate rather than an oversight.
FIT_KWARGS = dict(
    group_categories=GROUP_CATEGORIES,
    spline_spacing=100,
    penalty=False,
    n_iter=200,
    lr=1,
    tol=1e4,
    device="cpu",
    random_state=100,
    generate_description=False,
)


def _configurations():
    """Return one entry per distinct fit path CBMR can take.

    Chosen to cover every branch rather than every combination: all three of ``_fit``,
    ``_fit_full`` and ``_fit_approximate``, each observation distribution, and the mixed
    global/voxelwise split.
    """
    return {
        "global-poisson": dict(
            moderator_effect="global", model=models.PoissonEstimator, moderators=MODERATORS
        ),
        "global-negbin": dict(
            moderator_effect="global",
            model=models.NegativeBinomialEstimator,
            moderators=MODERATORS,
        ),
        "global-clusterednegbin": dict(
            moderator_effect="global",
            model=models.ClusteredNegativeBinomialEstimator,
            moderators=MODERATORS,
        ),
        "voxelwise-full": dict(
            moderator_effect="voxelwise", backend="full", moderators=MODERATORS
        ),
        "voxelwise-approximate": dict(
            moderator_effect="voxelwise", backend="approximate", moderators=MODERATORS
        ),
        "mixed-full": dict(
            moderator_effect="mixed",
            backend="full",
            global_moderators=[MODERATORS[0]],
            voxelwise_moderators=[MODERATORS[1]],
        ),
    }


def _build_studyset():
    """Build the fixed synthetic Studyset every fixture is fitted to."""
    from nimare.generate import create_coordinate_studyset
    from nimare.transforms import StandardizeField

    _, studyset = create_coordinate_studyset(foci=10, sample_size=(20, 40), n_studies=40, seed=11)
    annotations = studyset.annotations_df.copy()
    n_rows = annotations.shape[0]
    pattern = [
        ("schizophrenia", "Yes"),
        ("schizophrenia", "No"),
        ("depression", "Yes"),
        ("depression", "No"),
    ]
    annotations[["diagnosis", "drug_status"]] = [pattern[i % 4] for i in range(n_rows)]
    annotations["sample_sizes"] = [studyset.metadata.sample_sizes[i][0] for i in range(n_rows)]
    annotations["avg_age"] = np.arange(n_rows, dtype=float)
    studyset.annotations_df = annotations
    return StandardizeField(fields=["sample_sizes", "avg_age"]).transform(studyset)


def _fit(name, studyset=None):
    """Fit one configuration and return its maps and tables."""
    studyset = studyset if studyset is not None else _build_studyset()
    estimator = CBMREstimator(**FIT_KWARGS, **_configurations()[name])
    result = estimator.fit(dataset=studyset)
    return result.maps, result.tables


def _flatten(maps, tables):
    """Flatten maps and tables into the arrays an ``.npz`` can hold."""
    payload = {}
    for map_name, values in maps.items():
        payload[f"map::{map_name}"] = np.asarray(values, dtype=np.float64)
    for table_name, frame in tables.items():
        payload[f"table::{table_name}::values"] = frame.to_numpy(dtype=np.float64)
        payload[f"table::{table_name}::columns"] = np.asarray(
            [str(c) for c in frame.columns], dtype=object
        )
        payload[f"table::{table_name}::index"] = np.asarray(
            [str(i) for i in frame.index], dtype=object
        )
    return payload


def _fixture_path(name):
    return os.path.join(FIXTURE_DIR, f"{name}.npz")


def regenerate():
    """Refit every configuration and overwrite the stored fixtures."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    studyset = _build_studyset()
    for name in _configurations():
        payload = _flatten(*_fit(name, studyset))
        np.savez_compressed(_fixture_path(name), **payload)
        print(f"wrote {_fixture_path(name)} ({len(payload)} arrays)")


@pytest.fixture(scope="module")
def golden_studyset():
    """Build the fixture Studyset once for the whole module."""
    return _build_studyset()


@pytest.mark.parametrize("name", sorted(_configurations()))
def test_cbmr_fit_matches_stored_fixture(name, golden_studyset):
    """Refitting a pinned configuration must reproduce its stored maps and tables."""
    path = _fixture_path(name)
    if not os.path.exists(path):
        pytest.skip(f"missing fixture {path}; run `python -m {__name__}` to create it")

    actual = _flatten(*_fit(name, golden_studyset))
    with np.load(path, allow_pickle=True) as stored:
        expected_keys = set(stored.files)
        assert set(actual) == expected_keys, (
            f"fixture key mismatch for {name}: "
            f"missing {sorted(expected_keys - set(actual))}, "
            f"unexpected {sorted(set(actual) - expected_keys)}"
        )
        for key in sorted(expected_keys):
            want, got = stored[key], actual[key]
            if want.dtype == object:
                np.testing.assert_array_equal(got, want, err_msg=f"{name}: {key}")
            else:
                np.testing.assert_allclose(
                    got, want, rtol=1e-6, atol=1e-9, err_msg=f"{name}: {key}"
                )


if __name__ == "__main__":
    regenerate()
