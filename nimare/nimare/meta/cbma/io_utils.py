"""Internal ALE IO and memory helpers."""

import gc
import os
import tempfile
import time

import numpy as np
import scipy.sparse as sp_sparse


def _estimate_csr_nbytes(ma_values):
    """Estimate the in-memory footprint of a CSR matrix."""
    ma_values = ma_values.tocsr(copy=False)
    return ma_values.data.nbytes + ma_values.indices.nbytes + ma_values.indptr.nbytes


def _get_available_memory_bytes():
    """Best-effort estimate of currently available system memory."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None

    if page_size <= 0 or available_pages <= 0:
        return None
    return int(page_size * available_pages)


def _determine_low_memory_chunk_bytes(available_bytes=None):
    """Choose a target per-chunk MA-map budget from available RAM."""
    if available_bytes is None:
        available_bytes = _get_available_memory_bytes()

    if available_bytes is None:
        return 256 * 1024**2

    return max(1 * 1024**2, min(512 * 1024**2, int(available_bytes * 0.1)))


def _copy_array_to_memmap(arr, filename):
    """Copy an array into a disk-backed memmap with the same dtype and shape."""
    arr = np.asarray(arr)
    mapped = np.memmap(filename, dtype=arr.dtype, mode="w+", shape=arr.shape)
    mapped[...] = arr
    return mapped


def _csr_to_memmap(ma_values, prefix):
    """Copy a CSR matrix into disk-backed arrays and return a CSR view plus temp files."""
    ma_values = ma_values.tocsr(copy=False)

    filenames = []
    for suffix in ("data", "indices", "indptr"):
        fd, filename = tempfile.mkstemp(prefix=f"{prefix}_{suffix}_", suffix=".mmap")
        os.close(fd)
        filenames.append(filename)

    data = _copy_array_to_memmap(ma_values.data, filenames[0])
    indices = _copy_array_to_memmap(ma_values.indices, filenames[1])
    indptr = _copy_array_to_memmap(ma_values.indptr, filenames[2])
    mapped = sp_sparse.csr_matrix(
        (data, indices, indptr),
        shape=ma_values.shape,
        copy=False,
    )
    return mapped, filenames


def _close_memmap_array(arr):
    """Close a numpy memmap backing file when present."""
    base = arr
    seen = set()
    while base is not None and id(base) not in seen:
        seen.add(id(base))
        mmap_obj = getattr(base, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
            break
        base = getattr(base, "base", None)


def _detach_csr_memmap_arrays(ma_values):
    """Detach CSR arrays from any memmap-backed storage before cleanup."""
    data = ma_values.data
    indices = ma_values.indices
    indptr = ma_values.indptr

    ma_values.data = np.empty(0, dtype=data.dtype)
    ma_values.indices = np.empty(0, dtype=indices.dtype)
    ma_values.indptr = np.zeros(ma_values.shape[0] + 1, dtype=indptr.dtype)

    for arr in (data, indices, indptr):
        _close_memmap_array(arr)


def _close_csr_memmaps(ma_values):
    """Close memmap-backed CSR arrays when present."""
    if hasattr(ma_values, "chunks"):
        for chunk in ma_values.chunks:
            _close_csr_memmaps(chunk)
        return

    if not sp_sparse.isspmatrix(ma_values):
        return

    _detach_csr_memmap_arrays(ma_values)


def _cleanup_temp_files(filenames):
    """Remove temporary files created for memmap-backed arrays."""
    for filename in filenames:
        if filename and os.path.isfile(filename):
            for i_try in range(5):
                try:
                    os.remove(filename)
                    break
                except PermissionError:
                    if i_try == 4:
                        raise
                    gc.collect()
                    time.sleep(0.05)


def _iter_study_id_chunks(coordinates, chunk_rows, start_idx=0):
    """Yield coordinate subsets spanning up to ``chunk_rows`` studies each."""
    study_ids = np.unique(coordinates["id"].values)
    for start in range(start_idx, study_ids.size, chunk_rows):
        chunk_ids = study_ids[start : start + chunk_rows]
        yield coordinates[coordinates["id"].isin(chunk_ids)]
