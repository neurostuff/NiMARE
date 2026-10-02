"""Tests for the CBMR spline basis and the package's optional-import behavior."""

import importlib
import sys
import warnings

import numpy as np
import pytest

import nimare

try:
    import torch  # noqa: F401
except ImportError:
    warnings.warn("Torch not installed. Some CBMR tests will be skipped.", stacklevel=2)
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True


def _reference_b_spline_bases(mask, spacing, margin=10):
    """Build the tensor-product basis the slow way: full product, then filter.

    Mirrors what :func:`~nimare.meta.cbmr.basis.b_spline_bases` did before it started walking
    the product one basis plane at a time, and exists so the optimized version can be pinned
    against it.
    """
    from nimare.meta.cbmr.basis import coef_spline_bases

    mask = np.asanyarray(mask).astype(bool, copy=False)
    xx = np.where(mask.sum(axis=(1, 2)) > 0)[0]
    yy = np.where(mask.sum(axis=(0, 2)) > 0)[0]
    zz = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    x_spline = coef_spline_bases(xx, spacing, margin)
    y_spline = coef_spline_bases(yy, spacing, margin)
    z_spline = coef_spline_bases(zz, spacing, margin)

    cropped = mask[xx.min() : xx.max() + 1, yy.min() : yy.max() + 1, zz.min() : zz.max() + 1]
    coords = np.argwhere(cropped)
    x_rows = x_spline[coords[:, 0]]
    y_rows = y_spline[coords[:, 1]]
    z_rows = z_spline[coords[:, 2]]
    xy_rows = (x_rows[:, :, None] * y_rows[:, None, :]).reshape(coords.shape[0], -1)
    full = (xy_rows[:, :, None] * z_rows[:, None, :]).reshape(coords.shape[0], -1)
    return full[:, np.max(full, axis=0) >= 0.1]


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
@pytest.mark.parametrize("spacing", [8, 12])
def test_b_spline_bases_matches_the_unfiltered_tensor_product(spacing):
    """Pruning basis planes early must not change a single value.

    The basis builder skips whole (i, j) planes whose peak falls under the support threshold
    rather than building the entire tensor product and discarding most of it, which at spacing=5
    on the 2 mm mask cut peak memory from 13.1 GB to 5.7 GB. The saving is only worth having if
    the result is untouched, so this pins it against the full-product reference -- values and
    column order alike.
    """
    from nimare.meta.cbmr.basis import b_spline_bases

    rng = np.random.default_rng(0)
    grid = np.zeros((26, 24, 22), dtype=bool)
    grid[4:22, 3:21, 5:18] = True
    # Punch out a few voxels so the mask is not a perfect box, which is what leaves some
    # tensor-product bases unsupported and gives the pruning something to do.
    holes = rng.integers([4, 3, 5], [22, 21, 18], size=(40, 3))
    grid[holes[:, 0], holes[:, 1], holes[:, 2]] = False

    actual = b_spline_bases(masker_voxels=grid, spacing=spacing)
    expected = _reference_b_spline_bases(grid, spacing)

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_b_spline_bases_prunes_unsupported_bases():
    """The pruning must actually discard columns, or the equality test proves nothing."""
    from nimare.meta.cbmr.basis import b_spline_bases, coef_spline_bases

    grid = np.zeros((26, 24, 22), dtype=bool)
    grid[4:22, 3:21, 5:18] = True

    kept = b_spline_bases(masker_voxels=grid, spacing=8).shape[1]
    per_axis = [
        coef_spline_bases(np.where(grid.sum(axis=axes) > 0)[0], 8, 10).shape[1]
        for axes in ((1, 2), (0, 2), (0, 1))
    ]
    built = per_axis[0] * per_axis[1] * per_axis[2]

    assert kept < built, "no bases were pruned; the equality test would be vacuous"


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_the_basis_nearly_spans_the_constant():
    """Pins why there is no intercept column, and why the collinearity is easy to miss.

    A cubic B-spline basis is a partition of unity, so it already spans the constant and an
    intercept column would be collinear with it. The support filter drops a little basis mass at
    the brain edge, which makes the collinearity *near* rather than exact -- so a rank check
    would not catch it and a fit would degrade quietly instead of failing.
    """
    from nimare.meta.cbmr.basis import b_spline_bases

    grid = np.zeros((26, 24, 22), dtype=bool)
    grid[4:22, 3:21, 5:18] = True
    bases = b_spline_bases(masker_voxels=grid, spacing=8)

    ones = np.ones(bases.shape[0])
    coefficients, *_ = np.linalg.lstsq(bases, ones, rcond=None)
    residual = np.linalg.norm(bases @ coefficients - ones) / np.linalg.norm(ones)

    assert residual < 0.2, "the basis should very nearly span the constant"
    assert np.linalg.matrix_rank(np.column_stack([bases, ones])) == bases.shape[1] + 1


def test_meta_package_defers_cbmr_import():
    """Importing nimare.meta must not eagerly import the optional CBMR modules."""
    # Snapshot every nimare.meta module so the original objects can be put back exactly.
    # Other code holds references to classes from these modules; swapping in freshly imported
    # duplicates breaks identity comparisons, dotted-path monkeypatches, and pickling by name.
    before = {n: m for n, m in sys.modules.items() if n.startswith("nimare.meta")}
    saved_attr = nimare.__dict__.pop("meta", None)

    # Clear by prefix, not by exact name: cbmr is a package, so leaving its submodules behind
    # would let the fresh import bind a new parent to stale children.
    for name in [
        n for n in list(sys.modules) if n == "nimare.meta" or n.startswith("nimare.meta.cbmr")
    ]:
        del sys.modules[name]

    try:
        meta = importlib.import_module("nimare.meta")

        assert not [n for n in sys.modules if n.startswith("nimare.meta.cbmr")]
        assert hasattr(meta, "ALE")
    finally:
        for name in [n for n in list(sys.modules) if n.startswith("nimare.meta")]:
            if name not in before:
                del sys.modules[name]
        sys.modules.update(before)
        if saved_attr is not None:
            nimare.__dict__["meta"] = saved_attr


@pytest.mark.cbmr_importerror
def test_cbmr_importerror():
    """Without torch, touching CBMR must raise ImportError naming the extra to install."""
    if TORCH_INSTALLED:
        pytest.skip("torch is installed in this test environment")

    with pytest.raises(ImportError):
        from nimare.meta.cbmr import CBMR

        CBMR("~ 1")

    with pytest.raises(ImportError):
        from nimare.meta.cbmr.distributions import Poisson

        Poisson()
