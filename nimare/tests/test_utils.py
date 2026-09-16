"""Test nimare.utils."""

import logging
import os
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nimare import utils
from nimare.meta import utils as utils_meta
from nimare.meta.utils import _apply_liberal_mask


def test_clip_p_values_copy_parameter_controls_mutation():
    """P-value clipping should preserve inputs by default and mutate only when requested."""
    p_values = np.array([np.nan, 0.0, 1e-50, 0.5, 2.0], dtype=np.float32)
    original = p_values.copy()

    clipped = utils._clip_p_values(p_values)

    assert not np.shares_memory(clipped, p_values)
    np.testing.assert_array_equal(p_values, original)
    assert clipped[1] == utils._minimum_positive_float()
    assert clipped[2] == utils._minimum_positive_float()
    assert clipped[4] == np.float32(1.0)

    clipped_in_place = utils._clip_p_values(p_values, copy=False)

    assert np.shares_memory(clipped_in_place, p_values)
    assert p_values[1] == utils._minimum_positive_float()
    assert p_values[2] == utils._minimum_positive_float()
    assert p_values[4] == np.float32(1.0)


def test_p_to_logp_values_can_reuse_owned_array():
    """p-to-logp conversion should support in-place mutation for owned arrays."""
    p_values = np.array([1.0, 0.01, 0.0, np.nan], dtype=np.float32)

    logp_values = utils._p_to_logp_values(p_values, copy=False)

    assert np.shares_memory(logp_values, p_values)
    np.testing.assert_allclose(logp_values[:2], [0.0, 2.0], atol=1e-6)
    assert np.isfinite(logp_values[2])
    assert np.isnan(logp_values[3])


def test_description_references_support_underscore_keys():
    """Reference extraction should support citation keys with underscores."""
    bibtex = utils.get_description_references(
        "A JALE-derived workflow was used \\citep{Frahm_Monimu_Hoffstaedter}."
    )

    assert "@misc{Frahm_Monimu_Hoffstaedter" in bibtex
    assert "Juaml/Jale" in bibtex


def test_find_stem():
    """Test nimare.utils._find_stem."""
    test_array = [
        "/home/data/dataset/file1.nii.gz",
        "/home/data/dataset/file2.nii.gz",
        "/home/data/dataset/file3.nii.gz",
        "/home/data/dataset/file4.nii.gz",
        "/home/data/dataset/file5.nii.gz",
    ]
    stem = utils._find_stem(test_array)
    assert stem == "/home/data/dataset/file"

    test_array = [
        "/home/data/dataset/subfolder1/file1.nii.gz",
        "/home/data/dataset/subfolder1/file2.nii.gz",
        "/home/data/dataset/subfolder2/file3.nii.gz",
        "/home/data/dataset/subfolder2/file4.nii.gz",
        "/home/data/dataset/subfolder3/file5.nii.gz",
    ]
    stem = utils._find_stem(test_array)
    assert stem == "/home/data/dataset/subfolder"

    test_array = [
        "/home/data/file1_test-filename_test.nii.gz",
        "/home/data/file2_test-filename_test.nii.gz",
        "/home/data/file3_test-filename_test.nii.gz",
        "/home/data/file4_test-filename_test.nii.gz",
        "/home/data/file5_test-filename_test.nii.gz",
    ]
    stem = utils._find_stem(test_array)
    assert stem == "/home/data/file"

    test_array = [
        "souse",
        "youse",
        "house",
        "mouse",
        "louse",
    ]
    stem = utils._find_stem(test_array)
    assert stem == ""


def test_get_template():
    """Test nimare.utils.get_template."""
    # 1mm template
    img = utils.get_template(space="mni152_1mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    assert not nib.is_proxy(img.dataobj)
    img = utils.get_template(space="mni152_1mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)

    # 2mm template (default)
    img = utils.get_template(space="mni152_2mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="mni152_2mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)
    assert not nib.is_proxy(img.dataobj)

    # ALE template
    img = utils.get_template(space="ale_2mm", mask=None)
    assert isinstance(img, nib.Nifti1Image)
    img = utils.get_template(space="ale_2mm", mask="brain")
    assert isinstance(img, nib.Nifti1Image)
    assert not nib.is_proxy(img.dataobj)

    # Expect exceptions when incompatible spaces or masks are requested.
    with pytest.raises(ValueError):
        utils.get_template(space="something", mask=None)

    with pytest.raises(ValueError):
        utils.get_template(space="mni152_1mm", mask="gm")

    with pytest.raises(ValueError):
        utils.get_template(space="mni152_2mm", mask="gm")

    with pytest.raises(ValueError):
        utils.get_template(space="ale_2mm", mask="gm")


def test_mask_coverage_gm_uses_probability_template_with_binary_masker():
    """GM null space should not rely on binary analysis-mask intensities."""
    masker = utils.get_masker(utils.get_template(space="mni152_2mm", mask="brain"))

    brain_ijk = utils._mask_coverage_to_null_ijk(masker, mask_coverage="brain")
    gm_ijk = utils._mask_coverage_to_null_ijk(masker, mask_coverage="gm")

    assert 0 < gm_ijk.shape[0] < brain_ijk.shape[0]

    brain_mask = utils._mask_coverage_to_mask(masker, mask_coverage="brain")
    gm_mask = utils._mask_coverage_to_mask(masker, mask_coverage="gm")
    assert np.all(brain_mask[gm_mask])


def test_get_resource_path():
    """Test nimare.utils.get_resource_path."""
    print(utils.get_resource_path())
    assert op.isdir(utils.get_resource_path())


@pytest.mark.parametrize(
    "has_low_memory,memory_limit",
    [
        (True, "1gb"),
        (False, None),
    ],
)
def test_use_memmap(caplog, has_low_memory, memory_limit):
    """Test the memmapping decorator."""
    LGR = logging.getLogger(__name__)

    class DummyClass:
        def __init__(self, has_low_memory, memory_limit):
            self.has_low_memory = has_low_memory
            self.memory_limit = memory_limit

        @utils.use_memmap(LGR)
        def test_decorator(self):
            assert hasattr(self, "memmap_filenames")
            if self.has_low_memory:
                assert hasattr(self, "memory_limit")
                if self.memory_limit:
                    assert os.path.isfile(self.memmap_filenames[0])
                else:
                    assert self.memmap_filenames[0] is None
            return self.memmap_filenames

        @utils.use_memmap(LGR)
        def bad_justin_timberlake(self):
            raise ValueError("It's gonna be may!")

    my_class = DummyClass(has_low_memory, memory_limit)

    # make sure memmap file has been deleted
    my_class.test_decorator()
    first_memmap_filename = my_class.memmap_filenames[0]

    # run bad function
    with pytest.raises(ValueError):
        my_class.bad_justin_timberlake()
    assert "failed, removing" in caplog.text

    if hasattr(my_class, "memory_limit") and my_class.memory_limit:
        assert not os.path.isfile(first_memmap_filename)
        assert not os.path.isfile(my_class.memmap_filenames[0])
        # test when a function is called a new memmap file is created
        assert first_memmap_filename != my_class.memmap_filenames[0]


def test_validate_images_df_preserves_existing_relative_columns():
    """Absolute image columns should not duplicate an existing relative column."""
    image_df = pd.DataFrame(
        {
            "id": ["study-1"],
            "study_id": ["study"],
            "contrast_id": ["1"],
            "beta": ["C:/Users/runneradmin/nimare/tests/data/orig/study_beta.nii.gz"],
            "beta__relative": ["orig/study_beta.nii.gz"],
        }
    )

    validated = utils._validate_images_df(image_df)

    assert validated.columns.tolist().count("beta__relative") == 1
    assert validated.loc[0, "beta"] == image_df.loc[0, "beta"]
    assert validated.loc[0, "beta__relative"] == "orig/study_beta.nii.gz"


def test_tal2mni():
    """TODO: Get converted coords from official site."""
    test = np.array([[-44, 31, 27], [20, -32, 14], [28, -76, 28]])
    true = np.array(
        [
            [-45.83997568, 35.97904559, 23.55194326],
            [22.69248975, -31.34145016, 13.91284087],
            [31.53113226, -76.61685748, 33.22105166],
        ]
    )
    assert np.allclose(utils.tal2mni(test), true)


def test_mni2tal():
    """TODO: Get converted coords from official site."""
    test = np.array([[-44, 31, 27], [20, -32, 14], [28, -76, 28]])
    true = np.array(
        [[-42.3176, 26.0594, 29.7364], [17.4781, -32.6076, 14.0009], [24.7353, -75.0184, 23.3283]]
    )
    assert np.allclose(utils.mni2tal(test), true)


def test_vox2mm():
    """Test vox2mm."""
    test = np.array([[20, 20, 20], [0, 0, 0]])
    true = np.array([[-50.0, -86.0, -32.0], [-90.0, -126.0, -72.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(utils.vox2mm(test, aff), true)


def test_mm2vox():
    """Test mm2vox."""
    test = np.array([[20, 20, 20], [0, 0, 0]])
    true = np.array([[55.0, 73.0, 46.0], [45.0, 63.0, 36.0]])
    img = utils.get_template(space="mni152_2mm", mask=None)
    aff = img.affine
    assert np.array_equal(utils.mm2vox(test, aff), true)


def test_get_voxel_values_marks_out_of_bounds_indices():
    """Voxel lookup returns values and distinguishes clipped indices."""
    data = np.arange(27).reshape(3, 3, 3)
    ijk = np.array([[2, 0, 0], [3, 0, 0], [-1, 0, 0], [4294967298, 0, 0]])

    values, in_bounds = utils._get_voxel_values(data, ijk)

    assert values[0] == data[2, 0, 0]
    assert np.array_equal(in_bounds, [True, False, False, False])


def test_apply_liberal_mask():
    """Test _apply_liberal_mask."""
    data = np.array([[1, 2, np.nan, np.nan], [4, np.nan, 6, 5], [0, 8, 9, 3]])
    true_data = [np.array([[1], [4]]), np.array([[2], [8]]), np.array([[6, 5], [9, 3]])]

    pred_data, _, _ = _apply_liberal_mask(data)

    assert len(pred_data) == len(true_data)

    for pred_val, true_val in zip(pred_data, true_data):
        assert np.array_equal(pred_val, true_val)


def test_apply_liberal_mask_groups_voxels_that_are_not_adjacent():
    """Voxels sharing a coverage pattern belong in one bag, however they are ordered."""
    data = np.array(
        [
            [1.0, np.nan, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
    )

    values, voxel_masks, study_masks = _apply_liberal_mask(data)

    assert len(values) == 2
    # Bags come back in order of first appearance, so the {0, 2} bag leads.
    assert np.array_equal(voxel_masks[0], [0, 2])
    assert np.array_equal(study_masks[0], [0, 1, 2])
    assert np.array_equal(values[0], [[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]])

    assert np.array_equal(voxel_masks[1], [1])
    assert np.array_equal(study_masks[1], [1, 2])
    assert np.array_equal(values[1], [[4.0], [7.0]])


def test_apply_liberal_mask_treats_zeros_as_missing_and_drops_thin_bags():
    """Exact zeros mark a study as absent, and a voxel needs two studies to be fitted."""
    # Voxel 0 is covered by every study; voxel 1 only by the last, via an explicit zero.
    data = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 4.0]])

    values, voxel_masks, study_masks = _apply_liberal_mask(data)

    assert len(values) == 1
    assert np.array_equal(voxel_masks[0], [0])
    assert np.array_equal(study_masks[0], [0, 1, 2])


def test_apply_liberal_mask_partitions_every_covered_voxel():
    """Every voxel with at least two studies lands in exactly one bag."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(12, 400))
    data[rng.random(data.shape) < 0.3] = np.nan

    values, voxel_masks, study_masks = _apply_liberal_mask(data)

    covered = (~np.isnan(data)).sum(axis=0) >= 2
    assigned = np.concatenate(voxel_masks)
    assert np.array_equal(np.sort(assigned), np.flatnonzero(covered))

    for value, voxel_mask, study_mask in zip(values, voxel_masks, study_masks):
        # Every voxel in a bag really is covered by exactly the bag's studies.
        pattern = ~np.isnan(data[:, voxel_mask])
        assert np.array_equal(np.flatnonzero(pattern[:, 0]), study_mask)
        assert pattern[study_mask].all()
        assert np.array_equal(value, data[np.ix_(study_mask, voxel_mask)])


def test_liberal_mask_bags_and_values_compose_to_apply_liberal_mask():
    """The split entry points must cut the data exactly as the combined one does."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(8, 300))
    data[rng.random(data.shape) < 0.3] = np.nan
    mask = ~np.isnan(data) & (data != 0)

    values, voxel_masks, study_masks = _apply_liberal_mask(data)
    bags = utils_meta._liberal_mask_bags(mask)
    shared_values = utils_meta._liberal_mask_values(data, bags)

    assert len(bags) == len(values) > 1
    for value, voxel_mask, study_mask, (bag_voxels, bag_studies), shared in zip(
        values, voxel_masks, study_masks, bags, shared_values
    ):
        assert np.array_equal(voxel_mask, bag_voxels)
        assert np.array_equal(study_mask, bag_studies)
        assert np.array_equal(value, shared)


def test_reduce_idx_keeps_only_outermost_brace_pairs():
    """Braces nested inside an entry are discarded, whatever order they arrive in."""
    # In "{a{b}}{c}" the (2, 4) pair sits inside (0, 5); (6, 8) stands alone.
    braces = utils.find_braces("{a{b}}{c}")

    assert utils.reduce_idx(braces) == [(0, 5), (6, 8)]
    # The original implementation sorted its input, so shuffled input must not matter.
    assert utils.reduce_idx(braces[::-1]) == [(0, 5), (6, 8)]
    assert utils.reduce_idx([]) == []


def test_bibtex_reference_list_is_cached():
    """The packaged BibTeX file is parsed once, not on every ``fit``."""
    utils._bibtex_reference_list.cache_clear()
    first = utils._bibtex_reference_list()
    second = utils._bibtex_reference_list()

    assert first is second
    assert utils._bibtex_reference_list.cache_info().hits == 1
    assert all(entry.startswith("@") for entry in first)


def test_clip_logp_values_keeps_values_a_p_value_could_not_hold():
    """A -log10(p) must not be clipped to the range of the p-value it came from."""
    values = np.array([0.0, 44.85, 100.0, 1000.0, 5000.0])

    clipped = utils._clip_logp_values(values)

    assert np.allclose(clipped, values)
    assert clipped.dtype == np.dtype(utils.DEFAULT_FLOAT_DTYPE)


def test_clip_logp_values_still_bounds_the_non_finite_ends():
    """Only the ends a float cannot hold are clipped: negatives and infinity."""
    clipped = utils._clip_logp_values(np.array([-5.0, 0.0, 3.0, np.inf]))

    assert clipped[0] == 0.0, "p <= 1 cannot give a negative -log10(p)"
    assert clipped[2] == 3.0
    assert np.isfinite(clipped[3]) and clipped[3] > 1e30, "infinity becomes the dtype max"


def test_nlogp_to_logp_values_converts_nlogp_to_logp():
    """Natural log in, negated base-ten log out, with no p-value in between."""
    nlogp = np.array([0.0, -np.log(10.0), np.log(1e-45), -3000.0])

    logp = utils._nlogp_to_logp_values(nlogp)

    assert np.allclose(logp, [0.0, 1.0, 45.0, 3000.0 / np.log(10.0)], rtol=1e-6)
    assert logp.dtype == np.dtype(utils.DEFAULT_FLOAT_DTYPE)
    # The same tail through the p-value instead would have been clipped at 44.85.
    assert utils._p_to_logp_values(np.array([1e-300]))[0] < 45.0


def test_gpd_tail_p_engages_only_with_enough_exceedances():
    """The tail fit needs a tail, and says so by declining rather than fitting noise."""
    from nimare.meta.utils import _gpd_tail_p

    rng = np.random.default_rng(0)
    observed = np.array([12.0, 4.0])
    for n_iters in (20, 50):
        maxima = np.abs(rng.standard_normal(n_iters)) * 2.0 + 3.0
        assert _gpd_tail_p(observed, maxima) is None, n_iters

    maxima = np.abs(rng.standard_normal(500)) * 2.0 + 3.0
    fitted = _gpd_tail_p(observed, maxima)
    assert fitted is not None
    assert np.all((fitted >= 0) & (fitted <= 1))
    # Monotone: a larger statistic cannot be less significant.
    ordered = _gpd_tail_p(np.array([4.0, 8.0, 12.0]), maxima)
    assert ordered[0] > ordered[1] >= ordered[2]


def test_gpd_tail_p_keeps_the_empirical_tail_near_the_floor():
    """Below five times the empirical floor the fit is not trusted, so the empirical p is used.

    This test previously asserted that no corrected p could fall below five times the
    permutation floor, which encoded a bug rather than a policy: it conflated "do not trust the
    fitted tail here" with "never report a small p", and the empirical ``1 / (1 + n)`` is not
    manufactured significance -- it is exactly what the permutations support. What must never
    happen is a value *below* the empirical tail.
    """
    from nimare.meta.utils import _GPD_FLOOR_MULTIPLE, _gpd_tail_p

    rng = np.random.default_rng(1)
    n_iters = 500
    maxima = np.abs(rng.standard_normal(n_iters)) * 2.0 + 3.0
    floor = _GPD_FLOOR_MULTIPLE / (1.0 + n_iters)

    # An observation far past anything the permutations reached: the fit is not trusted this
    # far out, so the answer is the permutation tail itself.
    extreme = np.array([maxima.max() * 3.0])
    fitted = _gpd_tail_p(extreme, maxima)
    assert fitted is not None
    empirical = (1 + int((maxima >= extreme[0]).sum())) / (1.0 + n_iters)
    assert np.isclose(fitted[0], empirical)
    assert fitted[0] >= 1.0 / (1.0 + n_iters) - 1e-12
    assert fitted[0] < floor

    # A statistic below the fit's threshold is scored against the permutations themselves,
    # which cannot give less than one exceedance out of n + 1.
    modest = _gpd_tail_p(np.array([float(np.median(maxima))]), maxima)
    assert modest[0] > 0.25


def test_gpd_goodness_of_fit_returns_a_usable_p_value():
    """The fit is tested before it is trusted, so its test has to behave like a test."""
    from nimare.meta.utils import _gpd_goodness_of_fit

    rng = np.random.default_rng(2)
    from scipy import stats as sp_stats

    shape, scale = 0.1, 1.5
    genuine = sp_stats.genpareto.rvs(shape, loc=0.0, scale=scale, size=400, random_state=rng)
    p_good = _gpd_goodness_of_fit(genuine, shape, scale, n_boot=60, seed=0)
    assert 0.0 <= p_good <= 1.0
    # Data that is plainly not generalized Pareto should not pass as easily as data that is.
    wrong = np.abs(rng.standard_normal(400)) * 0.01 + 5.0
    p_bad = _gpd_goodness_of_fit(wrong, shape, scale, n_boot=60, seed=0)
    assert 0.0 <= p_bad <= 1.0
    assert p_bad < p_good


def test_gpd_tail_p_shortens_the_tail_and_gives_up_cleanly():
    """The retry path, and the surrender at the end of it."""
    from nimare.meta.utils import _gpd_tail_p

    # Degenerate tail: every extreme value identical, so the excesses are all zero and no
    # generalized Pareto can be fitted at any tail length.
    flat_tail = np.concatenate([np.linspace(0.0, 1.0, 400), np.full(200, 1.0)])
    assert _gpd_tail_p(np.array([5.0]), flat_tail) is None

    # Below the hard minimum it declines without trying at all.
    assert _gpd_tail_p(np.array([5.0]), np.linspace(0, 1, 99)) is None

    # A tail with a kink in it: fittable somewhere, but not at the length first attempted.
    rng = np.random.default_rng(3)
    body = np.abs(rng.standard_normal(500)) * 2.0
    kinked = np.sort(np.concatenate([body, body.max() + np.full(40, 4.0)]))
    result = _gpd_tail_p(np.array([kinked.max() * 1.2, float(np.median(kinked))]), kinked)
    # Either it found a shorter acceptable tail or it gave up; both are valid, and both must
    # come back as usable probabilities rather than as an exception.
    assert result is None or np.all((result >= 0) & (result <= 1))


def test_padded_flat_to_masked_agrees_with_the_unpadded_lookup():
    """Padding must not change which voxel an index names, only where it is safe to add."""
    rng = np.random.default_rng(0)
    mask = np.zeros((9, 11, 7), dtype=np.int16)
    mask[2:7, 3:9, 1:6] = 1
    mask_img = nib.Nifti1Image(mask, np.eye(4))
    offsets = utils_meta.sphere_kernel_offsets(2.0, (1.0, 1.0, 1.0))

    plain = utils_meta._get_mask_flat_to_masked(mask_img)
    padded, padded_shape, pad = utils_meta._padded_flat_to_masked(mask_img, offsets)

    assert np.all(padded_shape == np.array(mask.shape) + 2 * pad)
    assert padded.max() == plain.max()
    # Every in-image voxel resolves to the same masked index through either lookup.
    shape = np.array(mask.shape)
    ijk = np.stack([rng.integers(0, n, 200) for n in shape], axis=1)
    flat = ijk @ np.array([shape[1] * shape[2], shape[2], 1])
    padded_flat = (ijk + pad) @ np.array([padded_shape[1] * padded_shape[2], padded_shape[2], 1])
    assert np.array_equal(plain[flat], padded[padded_flat])


def test_padded_dilation_matches_a_bounds_checked_one_even_from_outside_the_image():
    """The point of the padding is that a focus near, or past, the edge needs no special case."""
    mask = np.zeros((9, 11, 7), dtype=np.int16)
    mask[2:7, 3:9, 1:6] = 1
    mask_img = nib.Nifti1Image(mask, np.eye(4))
    offsets = utils_meta.sphere_kernel_offsets(2.0, (1.0, 1.0, 1.0)).astype(np.int64)
    shape = np.array(mask.shape, dtype=np.int64)

    plain = utils_meta._get_mask_flat_to_masked(mask_img)
    padded, padded_shape, pad = utils_meta._padded_flat_to_masked(mask_img, offsets)
    padded_strides = np.array([padded_shape[1] * padded_shape[2], padded_shape[2], 1])
    flat_offsets = offsets @ padded_strides
    reach = np.abs(offsets).max(axis=0)

    # On the edge, one voxel outside, and far enough out to reach nothing.
    for focus in ([2, 3, 1], [0, 0, 0], [-1, 4, 3], [9, 4, 3], [-40, 4, 3]):
        focus = np.array(focus, dtype=np.int64)
        candidates = focus + offsets
        in_bounds = np.all((candidates >= 0) & (candidates < shape), axis=-1)
        flat = candidates @ np.array([shape[1] * shape[2], shape[2], 1])
        expected = plain[np.where(in_bounds, flat, 0)]
        expected = np.sort(expected[in_bounds & (expected >= 0)])

        if np.all((focus >= -reach) & (focus < shape + reach)):
            reached = padded[(focus + pad) @ padded_strides + flat_offsets]
            reached = np.sort(reached[reached >= 0])
        else:
            reached = np.array([], dtype=np.int32)  # dropped, and provably reaches nothing
        assert np.array_equal(expected, reached), focus


def test_gpd_tail_p_gives_up_when_the_fitter_raises_or_returns_nonsense(monkeypatch):
    """A fitter that fails, or succeeds with a degenerate answer, must not be trusted."""
    from nimare.meta.utils import _gpd_tail_p

    rng = np.random.default_rng(3)
    maxima = np.abs(rng.standard_normal(500)) * 2.0 + 3.0
    observed = np.array([maxima.max() * 2.0])
    assert _gpd_tail_p(observed, maxima) is not None  # fittable before it is sabotaged

    attempts = []

    def raising(*args, **kwargs):
        attempts.append("raise")
        raise RuntimeError("optimizer gave up")

    monkeypatch.setattr(utils_meta.stats.genpareto, "fit", raising)
    assert _gpd_tail_p(observed, maxima) is None
    # It shortened the tail and retried rather than surrendering on the first failure.
    assert len(attempts) > 1

    for bad in ((np.nan, 0.0, 1.0), (0.1, 0.0, np.inf), (0.1, 0.0, -1.0), (0.1, 0.0, 0.0)):
        monkeypatch.setattr(utils_meta.stats.genpareto, "fit", lambda *a, **k: bad)
        assert _gpd_tail_p(observed, maxima) is None, bad


def test_gpd_goodness_of_fit_discards_replicates_it_cannot_refit(monkeypatch):
    """A bootstrap replicate that will not refit is dropped, not counted as agreement."""
    from nimare.meta.utils import _gpd_goodness_of_fit

    rng = np.random.default_rng(4)
    excess = utils_meta.stats.genpareto.rvs(0.1, loc=0.0, scale=1.5, size=200, random_state=rng)
    n_boot = 25

    monkeypatch.setattr(
        utils_meta.stats.genpareto, "fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    assert _gpd_goodness_of_fit(excess, 0.1, 1.5, n_boot=n_boot) == 1 / (1 + n_boot)

    for bad in ((np.nan, 0.0, 1.0), (0.1, 0.0, -1.0), (0.1, 0.0, 0.0)):
        monkeypatch.setattr(utils_meta.stats.genpareto, "fit", lambda *a, **k: bad)
        assert _gpd_goodness_of_fit(excess, 0.1, 1.5, n_boot=n_boot) == 1 / (1 + n_boot), bad


def test_the_gpd_tail_falls_back_to_the_empirical_p_rather_than_clamping_up():
    """Below the range the fit is trusted in, the empirical p is the answer, not a floor.

    The code applies the fit only well above the permutation floor, having measured it
    anticonservative below that. It used to enforce that by raising the fitted value *up* to
    five times the floor, which for a statistic beyond every null maximum handed back 0.009980
    where the empirical p was 0.001996 -- five times too conservative, and worse than not
    fitting a tail at all.
    """
    from nimare.meta.utils import _GPD_FLOOR_MULTIPLE, _gpd_tail_p

    rng = np.random.default_rng(0)
    maxima = np.abs(rng.standard_normal(500)) * 2.0 + 3.0
    floor = _GPD_FLOOR_MULTIPLE / (1.0 + maxima.size)

    extreme = np.array([maxima.max() * 3.0])
    fitted = _gpd_tail_p(extreme, maxima)
    assert fitted is not None
    empirical = (1 + int((maxima >= extreme[0]).sum())) / (1 + maxima.size)
    assert np.isclose(fitted[0], empirical), (fitted[0], empirical)
    assert fitted[0] < floor  # i.e. it is no longer clamped up to the floor

    # Well above the floor the fit is still used, and still differs from the empirical tail.
    moderate = np.array([float(np.percentile(maxima, 90))])
    above = _gpd_tail_p(moderate, maxima)
    assert above is not None
    assert above[0] > 5 * floor
    assert not np.isclose(
        above[0], (1 + int((maxima >= moderate[0]).sum())) / (1 + maxima.size), rtol=1e-6
    )
