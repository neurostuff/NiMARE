"""Loop invariants for a CBMA Monte Carlo run, and the fused pass over them.

Lives beside :mod:`nimare.meta.cbma.null_utils` and
:mod:`nimare.meta.cbma.pairwise_utils` because it is CBMA-only, and is named for
what it holds rather than one underscore away from
:mod:`nimare.meta._permutation`, which is the unrelated IBMA sign-flip OLS.

A permutation changes only where the foci are. Everything else -- the group
boundaries, the kernel per group, the mask geometry -- is fixed for the run, so
it is compiled once into a plan and the inner loop becomes a single pass over
groups, foci and kernel offsets with no intermediate MA maps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from nimare.meta.utils import _get_mask_flat_to_masked, _kernel_to_sparse_support

__all__ = ["CoveragePlan", "PermutationPlan", "ale_plan_for", "kda_plan_for"]


# nogil matters here: ALE runs permutations on joblib's threading backend, and a
# compiled function that holds the GIL serialises them. The body touches only
# numeric arrays, so there is nothing to protect.
@njit(cache=True, nogil=True)
def _ale_stat_from_offsets(
    kernel_offsets,
    kernel_values,
    kernel_starts,
    group_kernel,
    ijk,
    offsets,
    shape,
    flat_to_masked,
    n_voxels,
):
    """One permutation's ALE statistic map, in a single pass.

    ``log_sums`` accumulates ``log1p(-value)`` per voxel across studies, which is
    the same reduction the sparse path performs; ``stamp`` marks which voxels the
    current study has already touched, so the within-study maximum is taken in
    place instead of by sorting.
    """
    log_sums = np.zeros(n_voxels, dtype=np.float64)
    scratch = np.zeros(n_voxels, dtype=np.float32)
    stamp = np.zeros(n_voxels, dtype=np.int64)
    touched = np.empty(n_voxels, dtype=np.int64)
    stride_y = shape[2]
    stride_x = shape[1] * shape[2]

    for g in range(offsets.shape[0] - 1):
        k = group_kernel[g]
        k_lo = kernel_starts[k]
        k_hi = kernel_starts[k + 1]
        n_touch = 0
        tag = g + 1
        for p in range(offsets[g], offsets[g + 1]):
            pi = ijk[p, 0]
            pj = ijk[p, 1]
            pk = ijk[p, 2]
            for m in range(k_lo, k_hi):
                i = pi + kernel_offsets[m, 0]
                j = pj + kernel_offsets[m, 1]
                kk = pk + kernel_offsets[m, 2]
                if i < 0 or j < 0 or kk < 0:
                    continue
                if i >= shape[0] or j >= shape[1] or kk >= shape[2]:
                    continue
                col = flat_to_masked[i * stride_x + j * stride_y + kk]
                if col < 0:
                    continue
                v = kernel_values[m]
                if stamp[col] != tag:
                    stamp[col] = tag
                    scratch[col] = v
                    touched[n_touch] = col
                    n_touch += 1
                elif v > scratch[col]:
                    scratch[col] = v
        for t in range(n_touch):
            col = touched[t]
            log_sums[col] += np.log1p(-np.float64(scratch[col]))

    return 1.0 - np.exp(log_sums)


@dataclass(frozen=True)
class PermutationPlan:
    """Loop invariants for a Monte Carlo run over one studyset selection."""

    offsets: np.ndarray
    group_kernel: np.ndarray
    kernel_offsets: np.ndarray
    kernel_values: np.ndarray
    kernel_starts: np.ndarray
    shape: np.ndarray
    flat_to_masked: np.ndarray
    n_voxels: int

    @property
    def n_foci(self):
        """Total number of foci the plan was compiled for."""
        return int(self.offsets[-1])

    @property
    def n_groups(self):
        """Number of studies the foci are grouped into."""
        return len(self.offsets) - 1

    def summary_stat(self, ijk, weights=None):
        """Compute the ALE statistic map for one permutation.

        ``weights`` is accepted so that both plan kinds share one signature; the
        ALE statistic is a union over studies and does not weight them.
        """
        return _ale_stat_from_offsets(
            self.kernel_offsets,
            self.kernel_values,
            self.kernel_starts,
            self.group_kernel,
            np.ascontiguousarray(ijk, dtype=np.int32),
            self.offsets,
            self.shape,
            self.flat_to_masked,
            self.n_voxels,
        )


def _plan_preamble(estimator, kernel_types, coordinates, block):
    """Return ``(kernel, mask, offsets)``, or ``None`` when no plan applies.

    Both builders need the same three things and decline for the same three
    reasons, so they ask once.
    """
    kernel = getattr(estimator, "kernel_transformer", None)
    if not isinstance(kernel, kernel_types):
        return None
    masker = getattr(estimator, "masker", None)
    if masker is None:
        return None
    offsets = _group_offsets(coordinates, block)
    if offsets is None:
        return None
    return kernel, masker.mask_img, offsets


def _mask_geometry(mask):
    """Return the flat-to-masked map, the shape, and the masked voxel count."""
    flat_to_masked = _get_mask_flat_to_masked(mask)
    return (
        np.ascontiguousarray(mask.shape, dtype=np.int32),
        np.ascontiguousarray(flat_to_masked, dtype=np.int32),
        int(flat_to_masked.max()) + 1 if flat_to_masked.size else 0,
    )


def ale_plan_for(estimator, coordinates, block=None):
    """Build a plan for an ALE-family estimator, or ``None`` if it does not apply.

    Returns ``None`` rather than raising when the estimator's kernel is not one
    this pass reproduces exactly, so the caller keeps its existing path.
    """
    from nimare.meta.kernel import ALEKernel
    from nimare.meta.utils import get_ale_kernel

    got = _plan_preamble(estimator, ALEKernel, coordinates, block)
    if got is None:
        return None
    kernel, mask, offsets = got
    starts = offsets[:-1]

    # One kernel per distinct width, and an index per group.
    if kernel.sample_size is not None:
        widths = np.full(len(starts), float(kernel.sample_size))
    elif kernel.fwhm is None:
        if "sample_size" not in coordinates.columns:
            return None
        widths = np.asarray(coordinates["sample_size"].values, dtype=float)[starts]
    else:
        widths = None

    supports = []
    if widths is None:
        _, dense = get_ale_kernel(mask, fwhm=kernel.fwhm)
        supports.append(_kernel_to_sparse_support(dense))
        group_kernel = np.zeros(len(starts), dtype=np.int64)
    else:
        if not np.isfinite(widths).all():
            return None
        rounded = np.round(widths).astype(np.int64)
        unique = np.unique(rounded)
        index_of = {int(value): i for i, value in enumerate(unique)}
        for value in unique:
            _, dense = get_ale_kernel(mask, sample_size=int(value))
            supports.append(_kernel_to_sparse_support(dense))
        group_kernel = np.asarray([index_of[int(v)] for v in rounded], dtype=np.int64)

    kernel_offsets = np.ascontiguousarray(np.concatenate([s[0] for s in supports]), dtype=np.int32)
    kernel_values = np.ascontiguousarray(
        np.concatenate([s[1] for s in supports]), dtype=np.float32
    )
    kernel_starts = np.concatenate(([0], np.cumsum([len(s[0]) for s in supports]))).astype(
        np.int64
    )

    flat_to_masked = _get_mask_flat_to_masked(mask)
    return PermutationPlan(
        offsets=offsets,
        group_kernel=group_kernel,
        kernel_offsets=kernel_offsets,
        kernel_values=kernel_values,
        kernel_starts=kernel_starts,
        shape=np.ascontiguousarray(mask.shape, dtype=np.int32),
        flat_to_masked=np.ascontiguousarray(flat_to_masked, dtype=np.int32),
        n_voxels=int(flat_to_masked.max()) + 1 if flat_to_masked.size else 0,
    )


@njit(cache=True, nogil=True)
def _weighted_coverage_from_offsets(
    kernel_offsets,
    ijk,
    offsets,
    weights,
    value,
    shape,
    flat_to_masked,
    n_voxels,
    sum_overlap,
):
    """One permutation's (M)KDA statistic map, in a single pass.

    Both statistics are a weighted sum across studies of the study's coverage:
    KDA weights every study equally, MKDA by its weight vector. With
    ``sum_overlap`` false a study contributes its value to a voxel once however
    many of its foci reach it, so ``stamp`` is all the bookkeeping needed. With
    it true -- KDA's semantics -- overlapping foci within a study each count, so
    there is nothing to track at all.
    """
    out = np.zeros(n_voxels, dtype=np.float64)
    stamp = np.zeros(n_voxels, dtype=np.int64)
    stride_y = shape[2]
    stride_x = shape[1] * shape[2]

    for g in range(offsets.shape[0] - 1):
        tag = g + 1
        contribution = weights[g] * value
        for p in range(offsets[g], offsets[g + 1]):
            pi = ijk[p, 0]
            pj = ijk[p, 1]
            pk = ijk[p, 2]
            for m in range(kernel_offsets.shape[0]):
                i = pi + kernel_offsets[m, 0]
                j = pj + kernel_offsets[m, 1]
                kk = pk + kernel_offsets[m, 2]
                if i < 0 or j < 0 or kk < 0:
                    continue
                if i >= shape[0] or j >= shape[1] or kk >= shape[2]:
                    continue
                col = flat_to_masked[i * stride_x + j * stride_y + kk]
                if col < 0:
                    continue
                if sum_overlap:
                    out[col] += contribution
                elif stamp[col] != tag:
                    stamp[col] = tag
                    out[col] += contribution
    return out


@dataclass(frozen=True)
class CoveragePlan:
    """Loop invariants for an (M)KDA permutation run."""

    offsets: np.ndarray
    kernel_offsets: np.ndarray
    value: float
    shape: np.ndarray
    flat_to_masked: np.ndarray
    n_voxels: int
    sum_overlap: bool = False

    @property
    def n_foci(self):
        """Total number of foci the plan was compiled for."""
        return int(self.offsets[-1])

    @property
    def n_groups(self):
        """Number of studies the foci are grouped into."""
        return len(self.offsets) - 1

    def summary_stat(self, ijk, weights=None):
        """Compute the statistic map for one permutation, given per-study weights.

        ``weights`` of ``None`` means weight every study equally, which is the
        KDA statistic; MKDA passes its weight vector.
        """
        if weights is None:
            weights = np.ones(self.n_groups, dtype=np.float64)
        return _weighted_coverage_from_offsets(
            self.kernel_offsets,
            np.ascontiguousarray(ijk, dtype=np.int32),
            self.offsets,
            np.ascontiguousarray(weights, dtype=np.float64),
            float(self.value),
            self.shape,
            self.flat_to_masked,
            self.n_voxels,
            bool(self.sum_overlap),
        )


def _group_offsets(coordinates, block=None):
    """CSR group boundaries for the foci, or ``None`` if they cannot be trusted.

    A :class:`~nimare.studyset.blocks.CoordinateBlock` already carries them --
    that is what the block is -- so when the estimator kept the one its inputs
    were built from, they are read rather than recovered. Falling back to the
    frame means proving contiguity from the analysis ids, which a caller that
    assembled its own frame may not have.
    """
    if block is not None and len(block) == len(coordinates):
        return np.asarray(block.offsets, dtype=np.int64)
    ids = np.asarray(coordinates["id"].values, dtype=str)
    if not len(ids):
        return None
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    if len(np.unique(ids)) != len(starts):
        return None
    return np.r_[starts, len(ids)].astype(np.int64)


def kda_plan_for(estimator, coordinates, block=None):
    """Build a plan for an (M)KDA-family estimator, or ``None`` if it does not apply."""
    from nimare.meta.kernel import KDAKernel, MKDAKernel
    from nimare.meta.utils import sphere_kernel_offsets

    got = _plan_preamble(estimator, (KDAKernel, MKDAKernel), coordinates, block)
    if got is None:
        return None
    kernel, mask, offsets = got
    # The same builder compute_kda_ma uses, so the fused pass and the observed
    # statistic cannot disagree about which voxels a focus reaches.
    kernel_offsets = sphere_kernel_offsets(kernel.r, mask.header.get_zooms())

    shape, flat_to_masked, n_voxels = _mask_geometry(mask)
    return CoveragePlan(
        offsets=offsets,
        kernel_offsets=kernel_offsets,
        value=float(kernel.value),
        shape=shape,
        flat_to_masked=flat_to_masked,
        n_voxels=n_voxels,
        sum_overlap=bool(getattr(kernel, "_sum_overlap", False)),
    )
