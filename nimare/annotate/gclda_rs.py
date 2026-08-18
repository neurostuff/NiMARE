"""Python loader for GCLDA models trained by the Rust ``gclda-train`` binary.

This module is the bridge between the Rust trainer (``rust/gclda``) and the
rest of NiMARE. It does not reimplement any GCLDA arithmetic -- it only:

- writes NiMARE's ``count_df``/``coordinates_df`` DataFrames into the TSV
  format ``gclda-train`` reads (:func:`export_gclda_tsvs`),
- invokes the ``gclda-train`` binary as a subprocess
  (:func:`train_gclda_rust`), and
- loads the resulting output directory back into a Python object
  (:func:`load_gclda_model`) that exposes exactly the attributes
  :mod:`nimare.decode` consumes, so ``gclda_decode_map``, ``gclda_decode_roi``,
  and ``gclda_encode`` work against a Rust-trained model unmodified.

The two large voxel-by-topic matrices (``p_topic_g_voxel_``,
``p_voxel_g_topic_``) can be tens to hundreds of megabytes at production
scale, so :func:`load_gclda_model` memory-maps every array it loads by
default (``mmap=True``) rather than reading it fully into memory.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.image import load_img

from nimare.utils import _mask_img_to_bool

# Mirrors gclda-train's own CLI defaults (rust/gclda/src/bin/gclda-train.rs),
# which in turn mirror GCLDAModel's constructor/fit defaults
# (nimare/annotate/gclda.py).
DEFAULT_TRAIN_PARAMS = {
    "n_topics": 100,
    "n_regions": 2,
    "symmetric": True,
    "alpha": 0.1,
    "beta": 0.01,
    "gamma": 0.01,
    "delta": 1.0,
    "dobs": 25.0,
    "roi_size": 50.0,
    "seed_init": 1,
    "peak_block_size": None,  # None -> gclda-train sizes the block from its byte budget
    "n_iters": 5000,
    "loglikely_freq": 10,
    "output_dtype": "f64",
    "threads": 0,
}

# Hyperparameters recorded in model.json that belong in GCLDAResult.params.
_PARAM_KEYS = (
    "n_topics",
    "n_regions",
    "symmetric",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "dobs",
    "roi_size",
    "seed_init",
    "n_iters",
    "loglikely_freq",
)

# Every file gclda-train's write_outputs (rust/gclda/src/output.rs) writes.
_REQUIRED_OUTPUT_FILES = (
    "p_topic_g_voxel.npy",
    "p_voxel_g_topic.npy",
    "p_topic_g_word.npy",
    "p_word_g_topic.npy",
    "n_word_tokens_word_by_topic.npy",
    "n_peak_tokens_doc_by_topic.npy",
    "regions_mu.npy",
    "regions_sigma.npy",
    "loglikelihood.tsv",
    "vocabulary.txt",
    "model.json",
)


class GCLDAResult:
    """A GCLDA model trained by the Rust trainer and loaded back into Python.

    Exposes exactly the attributes :mod:`nimare.decode` consumes
    (``mask``, ``p_topic_g_voxel_``, ``p_word_g_topic_``, ``vocabulary``,
    ``p_topic_g_word_``, ``p_voxel_g_topic_``), plus the remaining model
    state (counts, spatial parameters, log-likelihood history) needed to
    inspect or resume a fit. Instances are normally constructed via
    :func:`load_gclda_model` or :func:`train_gclda_rust`, not directly.

    Parameters
    ----------
    mask : :obj:`nibabel.nifti1.Nifti1Image`
        The brain mask the model was trained against.
    vocabulary : :obj:`list` of :obj:`str`
        Term strings, defining the ``W`` axis of the word-topic matrices.
    ids : :obj:`list` of :obj:`str`
        Document IDs, in the order used for the ``D`` axis of
        ``n_peak_tokens_doc_by_topic``.
    params : :obj:`dict`
        Training hyperparameters (``n_topics``, ``n_regions``, ``symmetric``,
        ``alpha``, ``beta``, ``gamma``, ``delta``, ``dobs``, ``roi_size``,
        ``seed_init``, ``n_iters``, ``loglikely_freq``).
    p_topic_g_voxel_ : (V, T) :obj:`numpy.ndarray`
        :math:`p(topic|voxel)`.
    p_voxel_g_topic_ : (V, T) :obj:`numpy.ndarray`
        :math:`p(voxel|topic)`.
    p_topic_g_word_ : (W, T) :obj:`numpy.ndarray`
        :math:`p(topic|word)`.
    p_word_g_topic_ : (W, T) :obj:`numpy.ndarray`
        :math:`p(word|topic)`.
    n_word_tokens_word_by_topic : (W, T) :obj:`numpy.ndarray`
        Word-token counts by word and topic.
    n_peak_tokens_doc_by_topic : (D, T) :obj:`numpy.ndarray`
        Peak-token counts by document and topic.
    regions_mu : (T, R, 3) :obj:`numpy.ndarray`
        Subregion means.
    regions_sigma : (T, R, 3, 3) :obj:`numpy.ndarray`
        Subregion covariances.
    loglikelihood : :obj:`dict`
        ``{"iter": [...], "x": [...], "w": [...], "total": [...]}``, matching
        ``GCLDAModel.loglikelihood``'s shape.
    """

    def __init__(
        self,
        mask,
        vocabulary,
        ids,
        params,
        p_topic_g_voxel_,
        p_voxel_g_topic_,
        p_topic_g_word_,
        p_word_g_topic_,
        n_word_tokens_word_by_topic,
        n_peak_tokens_doc_by_topic,
        regions_mu,
        regions_sigma,
        loglikelihood,
    ):
        self.mask = mask
        self.vocabulary = vocabulary
        self.ids = ids
        self.params = params
        self.p_topic_g_voxel_ = p_topic_g_voxel_
        self.p_voxel_g_topic_ = p_voxel_g_topic_
        self.p_topic_g_word_ = p_topic_g_word_
        self.p_word_g_topic_ = p_word_g_topic_
        self.n_word_tokens_word_by_topic = n_word_tokens_word_by_topic
        self.n_peak_tokens_doc_by_topic = n_peak_tokens_doc_by_topic
        self.regions_mu = regions_mu
        self.regions_sigma = regions_sigma
        self.loglikelihood = loglikelihood


def export_gclda_tsvs(count_df, coordinates_df, out_dir):
    """Write a word-count DataFrame and a coordinates DataFrame to TSVs.

    The output format matches what ``gclda-train``'s TSV reader
    (``rust/gclda/src/io/tsv.rs``) expects: the counts file's document ID is
    taken positionally as its first column, and the coordinates file's
    ``id``/``x``/``y``/``z`` columns are located by name.

    Parameters
    ----------
    count_df : :obj:`pandas.DataFrame`
        Word-count matrix, one row per document (indexed by document ID),
        one column per term.
    coordinates_df : :obj:`pandas.DataFrame`
        Peak coordinates, with an ``x``, ``y``, and ``z`` column, and either
        an ``id`` column or an index giving the associated document ID.
    out_dir : :obj:`str` or :obj:`os.PathLike`
        Directory to write ``counts.tsv`` and ``coordinates.tsv`` into
        (created if missing).

    Returns
    -------
    counts_path : :obj:`str`
        Path to the written counts TSV.
    coords_path : :obj:`str`
        Path to the written coordinates TSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count_df = count_df.copy()
    count_df.index = count_df.index.astype(str)
    counts_path = out_dir / "counts.tsv"
    count_df.to_csv(counts_path, sep="\t", index_label="id")

    coordinates_df = coordinates_df.copy()
    if "id" not in coordinates_df.columns:
        coordinates_df["id"] = coordinates_df.index
    coordinates_df["id"] = coordinates_df["id"].astype(str)
    coords_path = out_dir / "coordinates.tsv"
    coordinates_df[["id", "x", "y", "z"]].to_csv(coords_path, sep="\t", index=False)

    return str(counts_path), str(coords_path)


def _default_binary():
    """Locate the ``gclda-train`` binary next to this checkout, or on PATH."""
    import shutil

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "rust" / "gclda" / "target" / "release" / "gclda-train"
    if candidate.is_file():
        return str(candidate)

    found = shutil.which("gclda-train")
    if found:
        return found

    raise FileNotFoundError(
        "Could not locate the gclda-train binary. Build it with "
        "`cd rust/gclda && cargo build --release`, or pass `binary=` explicitly."
    )


def _resolve_mask_for_training(mask, out_dir):
    """Return a stable on-disk path for ``mask``, suitable for ``--mask``.

    If ``mask`` is already a path to an existing file, it is used as-is (and
    must remain readable for as long as the returned model directory is
    expected to be loadable without an explicit ``mask=`` override). If it is
    an in-memory image (or anything else :func:`nilearn.image.load_img`
    accepts), it is written to ``out_dir/mask.nii.gz`` so that the model
    directory is self-contained.
    """
    if isinstance(mask, (str, os.PathLike)) and os.path.isfile(mask):
        return str(mask)

    img = load_img(mask)
    mask_path = Path(out_dir) / "mask.nii.gz"
    img.to_filename(str(mask_path))
    return str(mask_path)


def train_gclda_rust(count_df, coordinates_df, mask, out_dir, binary=None, **params):
    """Train a GCLDA model with the Rust trainer and load the result back.

    Exports ``count_df``/``coordinates_df`` to TSVs, invokes ``gclda-train``
    as a subprocess, and loads the resulting output directory with
    :func:`load_gclda_model`.

    Parameters
    ----------
    count_df : :obj:`pandas.DataFrame`
        Word-count matrix, as accepted by :func:`export_gclda_tsvs`.
    coordinates_df : :obj:`pandas.DataFrame`
        Peak coordinates, as accepted by :func:`export_gclda_tsvs`.
    mask : :obj:`str`, :obj:`os.PathLike`, or image-like
        Brain mask. Anything :func:`nilearn.image.load_img` accepts.
    out_dir : :obj:`str` or :obj:`os.PathLike`
        Directory ``gclda-train`` writes its outputs into (created if
        missing). Input TSVs are staged in a separate temporary directory,
        not written here.
    binary : :obj:`str`, optional
        Path to the ``gclda-train`` binary. If ``None``, looked up next to
        this checkout (``rust/gclda/target/release/gclda-train``) or on
        ``PATH``.
    **params
        Training hyperparameters: ``n_topics``, ``n_regions``, ``symmetric``,
        ``alpha``, ``beta``, ``gamma``, ``delta``, ``dobs``, ``roi_size``,
        ``seed_init``, ``peak_block_size``, ``n_iters``, ``loglikely_freq``,
        ``output_dtype`` (``"f64"`` or ``"f32"``), ``threads``. Unspecified
        values fall back to :data:`DEFAULT_TRAIN_PARAMS`.

    Returns
    -------
    :class:`GCLDAResult`
    """
    binary = binary or _default_binary()

    unknown = set(params) - set(DEFAULT_TRAIN_PARAMS)
    if unknown:
        raise TypeError(f"Unknown GCLDA training parameter(s): {sorted(unknown)}")
    merged = dict(DEFAULT_TRAIN_PARAMS)
    merged.update(params)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = _resolve_mask_for_training(mask, out_dir)

    with tempfile.TemporaryDirectory(prefix="gclda_rs_inputs_") as stage_dir:
        counts_path, coords_path = export_gclda_tsvs(count_df, coordinates_df, stage_dir)

        cmd = [
            binary,
            "--counts",
            counts_path,
            "--coordinates",
            coords_path,
            "--mask",
            mask_path,
            "--out-dir",
            str(out_dir),
            "--n-topics",
            str(merged["n_topics"]),
            "--n-regions",
            str(merged["n_regions"]),
            "--symmetric",
            "true" if merged["symmetric"] else "false",
            "--alpha",
            str(merged["alpha"]),
            "--beta",
            str(merged["beta"]),
            "--gamma",
            str(merged["gamma"]),
            "--delta",
            str(merged["delta"]),
            "--dobs",
            str(merged["dobs"]),
            "--roi-size",
            str(merged["roi_size"]),
            "--seed-init",
            str(merged["seed_init"]),
            "--n-iters",
            str(merged["n_iters"]),
            "--loglikely-freq",
            str(merged["loglikely_freq"]),
            "--output-dtype",
            str(merged["output_dtype"]),
            "--threads",
            str(merged["threads"]),
        ]

        # Omitted when None so gclda-train derives the block size from its own
        # byte budget, which keeps the buffer roughly constant instead of
        # letting it grow linearly with n_topics.
        if merged["peak_block_size"] is not None:
            cmd += ["--peak-block-size", str(merged["peak_block_size"])]

        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"gclda-train failed (exit code {completed.returncode}).\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{completed.stderr}"
            )

    return load_gclda_model(out_dir, mask=mask_path, mmap=True)


def load_gclda_model(model_dir, mask=None, mmap=True):
    """Load a ``gclda-train`` output directory into a :class:`GCLDAResult`.

    Parameters
    ----------
    model_dir : :obj:`str` or :obj:`os.PathLike`
        Directory written by ``gclda-train`` (or :func:`train_gclda_rust`).
    mask : :obj:`str`, :obj:`os.PathLike`, image-like, or ``None``, optional
        The brain mask the model was trained against. If ``None`` (default),
        the path recorded in ``model.json``'s ``mask_path`` is used; this
        requires that path to still be readable.
    mmap : :obj:`bool`, optional
        If ``True`` (default), every ``.npy`` array is memory-mapped
        (``np.load(..., mmap_mode="r")``) rather than read fully into memory
        -- important for ``p_topic_g_voxel_``/``p_voxel_g_topic_``, which are
        ``V x T`` and can be tens to hundreds of megabytes at production
        scale. If ``False``, arrays are loaded fully resident.

    Returns
    -------
    :class:`GCLDAResult`
    """
    model_dir = Path(model_dir)

    missing = [name for name in _REQUIRED_OUTPUT_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{model_dir} is missing expected gclda-train output file(s): {missing}"
        )

    with open(model_dir / "model.json", encoding="utf-8") as fo:
        meta = json.load(fo)

    if mask is not None:
        mask_img = load_img(mask)
    else:
        mask_path = meta.get("mask_path")
        if not mask_path or not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"model.json's recorded mask_path ({mask_path!r}) does not exist. "
                "Pass `mask=` explicitly to load_gclda_model."
            )
        mask_img = load_img(mask_path)

    n_mask_voxels = int(_mask_img_to_bool(mask_img).sum())
    n_voxels_trained = int(meta["n_voxels"])
    if n_mask_voxels != n_voxels_trained:
        raise ValueError(
            f"Mask has {n_mask_voxels} nonzero voxels, but this model was trained "
            f"with {n_voxels_trained}. Pass the exact mask used for training."
        )

    mmap_mode = "r" if mmap else None

    def _load(name):
        return np.load(model_dir / name, mmap_mode=mmap_mode)

    p_topic_g_voxel_ = _load("p_topic_g_voxel.npy")
    p_voxel_g_topic_ = _load("p_voxel_g_topic.npy")
    p_topic_g_word_ = _load("p_topic_g_word.npy")
    p_word_g_topic_ = _load("p_word_g_topic.npy")
    n_word_tokens_word_by_topic = _load("n_word_tokens_word_by_topic.npy")
    n_peak_tokens_doc_by_topic = _load("n_peak_tokens_doc_by_topic.npy")
    regions_mu = _load("regions_mu.npy")
    regions_sigma = _load("regions_sigma.npy")

    # write_vocabulary (rust/gclda/src/output.rs) writes one term per line,
    # each terminated by "\n" -- including the last -- so a plain readlines
    # split would otherwise manufacture a phantom trailing empty-string term.
    with open(model_dir / "vocabulary.txt", encoding="utf-8") as fo:
        raw = fo.read()
    vocabulary = raw.split("\n")
    if vocabulary and vocabulary[-1] == "":
        vocabulary.pop()

    loglikelihood = {"iter": [], "x": [], "w": [], "total": []}
    loglik_path = model_dir / "loglikelihood.tsv"
    if loglik_path.stat().st_size > 0:
        loglik_df = pd.read_csv(loglik_path, sep="\t")
        for col in ("iter", "x", "w", "total"):
            if col in loglik_df.columns:
                loglikelihood[col] = loglik_df[col].tolist()

    params = {key: meta[key] for key in _PARAM_KEYS}

    return GCLDAResult(
        mask=mask_img,
        vocabulary=vocabulary,
        ids=list(meta["ids"]),
        params=params,
        p_topic_g_voxel_=p_topic_g_voxel_,
        p_voxel_g_topic_=p_voxel_g_topic_,
        p_topic_g_word_=p_topic_g_word_,
        p_word_g_topic_=p_word_g_topic_,
        n_word_tokens_word_by_topic=n_word_tokens_word_by_topic,
        n_peak_tokens_doc_by_topic=n_peak_tokens_doc_by_topic,
        regions_mu=regions_mu,
        regions_sigma=regions_sigma,
        loglikelihood=loglikelihood,
    )
