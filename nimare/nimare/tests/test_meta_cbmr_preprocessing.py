"""Tests for CBMR's input preparation: masking, incidence filtering, and foci matrices.

Everything CBMR needs before a model is fitted comes off two things ``_collect_inputs`` produces
together: ``blocks_["coordinates"]``, a :class:`~nimare.studyset.blocks.CoordinateBlock`, and
``studyset_``, the selection it was resolved from. Because they come from the same narrowed view,
the block's rows *are* the studyset's analyses, in order -- which is what these tests mostly
exist to hold in place. Fetching coordinates and annotations separately and aligning them
afterwards is how one experiment's moderator values end up attributed to another's foci, and
nothing downstream of that could detect it.
"""

import nibabel as nib
import numpy as np
import pytest
import scipy.sparse

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr import CBMR
    from nimare.studyset.blocks import CoordinateBlock

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")


def _block(points_per_analysis, xyz, space=0, n_spaces=1):
    """Build a CoordinateBlock with the given foci grouped by analysis."""
    offsets = np.concatenate([[0], np.cumsum(points_per_analysis)]).astype(np.int64)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return CoordinateBlock(
        xyz=xyz,
        offsets=offsets,
        group_keys=np.asarray([f"analysis-{i}" for i in range(len(points_per_analysis))]),
        space=(
            np.full(len(xyz), space, dtype=np.int8)
            if n_spaces == 1
            else np.arange(len(xyz), dtype=np.int8) % n_spaces
        ),
        space_categories=[f"SPACE{i}" for i in range(max(n_spaces, 1))],
    )


def _unit_mask(shape):
    """Return an all-true mask image with an identity affine."""
    data = np.ones(shape, dtype=bool)
    return data, nib.Nifti1Image(data.astype(np.uint8), np.eye(4))


def test_focus_positions_come_from_the_block():
    """Rows and columns are read off the block, not recomputed from a frame.

    ``group_of_point`` gives each focus's analysis position and ``ijk`` its matrix index, the
    latter memoised on the block and documented to truncate exactly as ``mm2vox`` does. Both are
    what CBMR used to derive itself from a coordinate DataFrame.
    """
    estimator = CBMR("~ 1")
    mask_data, mask_img = _unit_mask((3, 1, 1))
    lookup, n_voxels = estimator._build_mask_lookup(mask_data)
    assert n_voxels == 3

    # Analysis 0 reports two foci (voxels 0 and 2), analysis 1 reports one (voxel 1).
    block = _block([2, 1], [[0, 0, 0], [2, 0, 0], [1, 0, 0]])
    rows, columns = estimator._focus_positions(block, mask_img, mask_data, lookup)

    np.testing.assert_array_equal(rows, [0, 0, 1])
    np.testing.assert_array_equal(columns, [0, 2, 1])


def test_foci_outside_the_mask_are_dropped():
    """Out-of-bounds and out-of-mask foci must not reach the matrix."""
    estimator = CBMR("~ 1")
    mask_data = np.zeros((3, 1, 1), dtype=bool)
    mask_data[[0, 2], 0, 0] = True
    mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), np.eye(4))
    lookup, n_voxels = estimator._build_mask_lookup(mask_data)
    assert n_voxels == 2

    # Voxel 1 is outside the mask; (99, 0, 0) is outside the volume entirely.
    block = _block([3], [[0, 0, 0], [1, 0, 0], [99, 0, 0]])
    rows, columns = estimator._focus_positions(block, mask_img, mask_data, lookup)

    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(columns, [0], err_msg="columns index the masked vector")


def test_foci_matrix_counts_repeats():
    """Two foci from one experiment in one voxel must give that cell a count of two."""
    counts = CBMR._foci_matrix(
        np.array([0, 0, 1]), np.array([0, 0, 2]), n_experiments=2, n_mask_voxels=3
    )
    assert scipy.sparse.issparse(counts)
    np.testing.assert_array_equal(counts.toarray(), [[2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_experiments_without_surviving_foci_keep_an_all_zero_row():
    """An experiment whose foci all fall outside the mask must stay in the matrix.

    Dropping it would shift every later row against the annotations, so its moderator values
    would be read off a different study.
    """
    counts = CBMR._foci_matrix(
        np.array([0]), np.array([1]), n_experiments=3, n_mask_voxels=2
    ).toarray()

    assert counts.shape == (3, 2), "one row per experiment, foci or not"
    np.testing.assert_array_equal(counts[1], [0.0, 0.0])
    np.testing.assert_array_equal(counts[2], [0.0, 0.0])


def test_no_surviving_foci_still_yields_a_correctly_shaped_matrix():
    """A degenerate selection should not change the matrix shape."""
    counts = CBMR._foci_matrix(
        np.array([], dtype=int), np.array([], dtype=int), n_experiments=2, n_mask_voxels=4
    )
    assert counts.shape == (2, 4)
    assert counts.nnz == 0


def test_incidence_threshold_drops_low_incidence_voxels():
    """Voxels whose empirical focus rate is at or below the threshold should be dropped.

    Incidence filtering keeps the analysis mask to voxels the data can speak about. Its effect is
    not cosmetic: it sets the basis width and therefore the parameter count of every term.
    """
    estimator = CBMR("~ 1", incidence_threshold=0.25)
    estimator.inputs_ = {}
    mask_data, mask_img = _unit_mask((3, 1, 1))

    # Voxel 0 is hit by two of four experiments, voxel 1 by one, voxel 2 by none.
    foci = CBMR._foci_matrix(
        np.array([0, 1, 2]), np.array([0, 0, 1]), n_experiments=4, n_mask_voxels=3
    )
    thresholded = estimator._threshold_mask_by_incidence(mask_img, mask_data, foci, 4)

    np.testing.assert_array_equal(np.asanyarray(thresholded.dataobj).ravel(), [1, 0, 0])
    np.testing.assert_allclose(estimator.inputs_["empirical_incidence_rate_roi"], [0.5, 0.25, 0.0])
    np.testing.assert_allclose(estimator.inputs_["empirical_incidence_rate"], [0.5])


def test_repeated_foci_count_once_toward_incidence():
    """Incidence is the fraction of experiments *reporting*, not the number of foci."""
    estimator = CBMR("~ 1", incidence_threshold=None)
    estimator.inputs_ = {}
    mask_data, mask_img = _unit_mask((2, 1, 1))

    # One experiment reporting three foci in voxel 0 is still one experiment.
    foci = CBMR._foci_matrix(
        np.array([0, 0, 0]), np.array([0, 0, 0]), n_experiments=2, n_mask_voxels=2
    )
    estimator._threshold_mask_by_incidence(mask_img, mask_data, foci, 2)

    np.testing.assert_allclose(estimator.inputs_["empirical_incidence_rate_roi"], [0.5, 0.0])


def test_no_threshold_keeps_every_voxel():
    """``incidence_threshold=None`` should leave the mask alone."""
    estimator = CBMR("~ 1", incidence_threshold=None)
    estimator.inputs_ = {}
    mask_data, mask_img = _unit_mask((3, 1, 1))
    foci = CBMR._foci_matrix(np.array([0]), np.array([0]), n_experiments=1, n_mask_voxels=3)

    thresholded = estimator._threshold_mask_by_incidence(mask_img, mask_data, foci, 1)

    np.testing.assert_array_equal(np.asanyarray(thresholded.dataobj).ravel(), [1, 1, 1])


def test_an_over_restrictive_threshold_is_reported():
    """Filtering everything away should say which knob to turn."""
    estimator = CBMR("~ 1", incidence_threshold=0.99)
    estimator.inputs_ = {}
    mask_data, mask_img = _unit_mask((2, 1, 1))
    foci = CBMR._foci_matrix(np.array([0]), np.array([0]), n_experiments=10, n_mask_voxels=2)

    with pytest.raises(ValueError, match="No voxels survived"):
        estimator._threshold_mask_by_incidence(mask_img, mask_data, foci, 10)


def test_invalid_incidence_threshold_is_rejected():
    """A threshold outside [0, 1) would drop everything or nothing; say so at construction."""
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="incidence_threshold"):
            CBMR("~ 1", incidence_threshold=bad)


def test_mixed_coordinate_spaces_are_refused():
    """Mixed spaces must be caught before the mask projection, not silently misprojected.

    The store harmonizes only when a target space is set, and neither the view context nor the
    Coordinates requirement sets one by default -- so unharmonized coordinates would reach the
    affine and land in the wrong voxels.
    """
    block = _block([2], [[0, 0, 0], [1, 0, 0]], n_spaces=2)
    with pytest.raises(ValueError, match="Mixed coordinate spaces"):
        CBMR._validate_block_space(block)


def test_a_single_space_is_accepted():
    """The ordinary case must pass."""
    CBMR._validate_block_space(_block([2], [[0, 0, 0], [1, 0, 0]]))


def test_missing_space_information_is_refused():
    """A block with no space at all cannot be projected either."""
    block = CoordinateBlock(
        xyz=np.zeros((0, 3)),
        offsets=np.array([0, 0], dtype=np.int64),
        group_keys=np.asarray(["analysis-0"]),
        space=np.array([], dtype=np.int8),
        space_categories=[],
    )
    with pytest.raises(ValueError, match="space information is missing"):
        CBMR._validate_block_space(block)


@pytest.mark.parametrize("threshold", [None, 0.001])
def test_annotations_are_already_aligned_with_the_foci_matrix(threshold):
    """The annotation rows must be the foci rows, without any reindexing.

    This is the guarantee the block interface buys. ``studyset_`` is the selection narrowed to
    analyses with usable coordinates, and the coordinate block was resolved from that same
    selection, so the two agree by construction rather than by a lookup that could be wrong.
    """
    from nimare.generate import create_coordinate_studyset

    _, studyset = create_coordinate_studyset(foci=5, sample_size=(20, 40), n_studies=12, seed=3)
    annotations = studyset.annotations_df.copy()
    annotations["dx"] = ["a", "b"] * 6
    studyset = studyset.with_annotations_df(annotations, name="groups", replace=True)

    # Move one experiment's foci far outside any brain mask. It has to retain *some* foci, or the
    # coordinates requirement drops it before CBMR sees it and this test asserts nothing.
    target = studyset.coordinates["id"].iloc[0]
    target_position = list(studyset.ids).index(target)
    studyset = (
        studyset.select_points((studyset.coordinates["id"] != target).to_numpy())
        .materialize_points()
        .with_points(
            [target_position],
            np.array([[10_000.0, 10_000.0, 10_000.0]]),
            space=studyset.space,
        )
    )

    estimator = CBMR(
        "~ s(dx)",
        spline_spacing=100,
        incidence_threshold=threshold,
        n_iter=5,
        generate_description=False,
    )
    estimator._collect_inputs(studyset, drop_invalid=True)
    estimator._preprocess_input(studyset)

    foci = estimator.inputs_["foci"]
    annotations = estimator._experiment_annotations()

    assert foci.shape[0] == len(estimator.studyset_.ids)
    assert list(annotations["id"]) == list(estimator.studyset_.ids)
    assert len(annotations) == foci.shape[0]
    # The relocated experiment is still a row, contributing nothing.
    assert foci[list(estimator.studyset_.ids).index(target)].nnz == 0
