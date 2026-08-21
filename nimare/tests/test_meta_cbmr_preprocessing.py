"""Tests for CBMR's input preparation: masking, incidence filtering, and foci matrices.

These cover the part of CBMR that happens before any model is fitted -- turning reported
coordinates into an analysis mask, a spline basis and a foci matrix. It survived the move to a
formula-specified interface unchanged, and it is where a silent misalignment would do the most
damage, because an experiment's annotations have to line up with its row of the foci matrix.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr import CBMR

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")


def test_incidence_threshold_drops_low_incidence_voxels():
    """Voxels whose empirical focus rate is at or below the threshold should be dropped.

    Incidence filtering is what keeps the analysis mask to voxels the data can actually speak
    about. Its effect is not cosmetic: it decides the basis width and therefore the parameter
    count of every term.
    """
    estimator = CBMR("~ 1", incidence_threshold=0.25)
    estimator.inputs_ = {}
    mask_data = np.ones((3, 1, 1), dtype=bool)
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), np.eye(4))
    coordinates = pd.DataFrame(
        {"id": ["study-1", "study-2", "study-3"], "_cbmr_mask_index": [0, 0, 1]}
    )
    ids_by_group = {"Default": ["study-1", "study-2", "study-3", "study-4"]}

    thresholded = estimator._threshold_mask_by_incidence(
        mask_img, mask_data, coordinates, ids_by_group, n_mask_voxels=3
    )

    # Voxel 0 is hit by two of four experiments, voxel 1 by one, voxel 2 by none. Only the
    # first is strictly above 0.25.
    np.testing.assert_array_equal(np.asanyarray(thresholded.dataobj).ravel(), [1, 0, 0])
    np.testing.assert_allclose(estimator.inputs_["empirical_incidence_rate_roi"], [0.5, 0.25, 0.0])
    np.testing.assert_allclose(estimator.inputs_["empirical_incidence_rate"], [0.5])


def test_no_threshold_keeps_every_voxel():
    """``incidence_threshold=None`` should leave the mask alone."""
    estimator = CBMR("~ 1", incidence_threshold=None)
    estimator.inputs_ = {}
    mask_data = np.ones((3, 1, 1), dtype=bool)
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), np.eye(4))
    coordinates = pd.DataFrame({"id": ["study-1"], "_cbmr_mask_index": [0]})

    thresholded = estimator._threshold_mask_by_incidence(
        mask_img, mask_data, coordinates, {"Default": ["study-1"]}, n_mask_voxels=3
    )

    np.testing.assert_array_equal(np.asanyarray(thresholded.dataobj).ravel(), [1, 1, 1])


def test_invalid_incidence_threshold_is_rejected():
    """A threshold outside [0, 1) would drop everything or nothing; say so at construction."""
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="incidence_threshold"):
            CBMR("~ 1", incidence_threshold=bad)


def test_foci_matrices_count_foci_per_experiment_and_voxel():
    """Each cell must hold that experiment's focus count in that voxel, including repeats."""
    coordinates = pd.DataFrame(
        {
            "id": ["a", "a", "a", "b"],
            "_cbmr_mask_index": [0, 0, 2, 1],
        }
    )
    matrices = CBMR._build_group_foci_matrices(coordinates, {"g": ["a", "b"]}, 3)

    counts = matrices["g"]
    assert scipy.sparse.issparse(counts)
    np.testing.assert_array_equal(counts.toarray(), [[2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def test_experiments_without_in_mask_foci_keep_an_all_zero_row():
    """An experiment whose foci all fall outside the mask must stay in the matrix.

    Dropping it would silently misalign every later row against the annotations, so its
    moderator values would be attributed to a different study.
    """
    coordinates = pd.DataFrame({"id": ["a"], "_cbmr_mask_index": [1]})
    matrices = CBMR._build_group_foci_matrices(coordinates, {"g": ["a", "b", "c"]}, 2)

    counts = matrices["g"].toarray()
    assert counts.shape == (3, 2), "one row per experiment, foci or not"
    np.testing.assert_array_equal(counts[1], [0.0, 0.0])
    np.testing.assert_array_equal(counts[2], [0.0, 0.0])


def test_no_coordinates_still_yields_a_correctly_shaped_matrix():
    """An empty coordinate table is degenerate but should not change the matrix shape."""
    coordinates = pd.DataFrame({"id": [], "_cbmr_mask_index": []})
    matrices = CBMR._build_group_foci_matrices(coordinates, {"g": ["a", "b"]}, 4)

    assert matrices["g"].shape == (2, 4)
    assert matrices["g"].nnz == 0


@pytest.mark.parametrize("threshold", [None, 0.001])
def test_annotations_stay_aligned_with_the_foci_matrix(threshold):
    """The annotation order must match the foci rows, whatever filtering removed.

    The formula binds terms against the annotation table, so a mismatch here would silently
    pair one experiment's moderator values with another's foci -- which no downstream test could
    detect.
    """
    from nimare.generate import create_coordinate_studyset

    _, studyset = create_coordinate_studyset(foci=5, sample_size=(20, 40), n_studies=12, seed=3)
    annotations = studyset.annotations_df.copy()
    annotations["dx"] = ["a", "b"] * 6
    studyset.annotations_df = annotations

    # Push one experiment's foci far outside any brain mask.
    coordinates = studyset.coordinates
    target = coordinates["id"].iloc[0]
    coordinates.loc[coordinates["id"] == target, ["x", "y", "z"]] = 10_000

    estimator = CBMR(
        "~ s(dx)",
        spline_spacing=100,
        incidence_threshold=threshold,
        n_iter=5,
        generate_description=False,
    )
    estimator._collect_inputs(studyset, drop_invalid=True)
    estimator._preprocess_input(studyset)

    ids = estimator.inputs_["ids_by_group"]["Default"]
    foci = estimator.inputs_["foci_by_experiment"]["Default"]
    assert foci.shape[0] == len(ids)

    ordered = estimator._experiment_annotations(studyset)
    assert list(ordered["id"]) == list(ids)
    assert len(ordered) == foci.shape[0]


def test_coordinate_filtering_uses_mask_indices():
    """Filtering must record each focus's index into the masked voxel vector.

    That index is how a focus reaches the right column of the foci matrix; a raw voxel
    coordinate would point into the full volume instead.
    """
    estimator = CBMR("~ 1")
    mask_data = np.zeros((3, 1, 1), dtype=bool)
    mask_data[[0, 2], 0, 0] = True
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), np.eye(4))
    lookup, n_voxels = estimator._build_mask_lookup(mask_data)
    assert n_voxels == 2

    coordinates = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "space": ["MNI152NLin6Asym"] * 3,
        }
    )
    filtered = estimator._filter_coordinates_to_mask(coordinates, mask_img, mask_data, lookup)

    # The middle coordinate is outside the mask and drops out; the others map to columns 0 and 1.
    assert list(filtered["id"]) == ["a", "c"]
    np.testing.assert_array_equal(filtered["_cbmr_mask_index"].to_numpy(), [0, 1])
