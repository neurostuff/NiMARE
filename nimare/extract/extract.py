"""Tools for downloading datasets."""

import contextlib
import hashlib
import itertools
import logging
import os
import os.path as op
import shutil
import tarfile
import time
import warnings
import zipfile
from glob import glob
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests

from nimare.dataset import Dataset
from nimare.extract.utils import (
    _download_zipped_file,
    _expand_df,
    _get_concept_reltype,
    _get_dataset_dir,
    _longify,
)
from nimare.utils import get_resource_path, load_json

LGR = logging.getLogger(__name__)

VALID_ENTITIES = {
    "coordinates.tsv.gz": ["data", "version"],
    "metadata.tsv.gz": ["data", "version"],
    "features.npz": ["data", "version", "vocab", "source", "type"],
    "vocabulary.txt": ["data", "version", "vocab"],
    "metadata.json": ["data", "version", "vocab"],
    "keys.tsv": ["data", "version", "vocab"],
}
VALID_FETCH_RETURN_TYPES = {"studyset", "dataset", "files"}

#: Root of the NeuroStore API.
NEUROSTORE_URL = "https://neurostore.org"
#: Index of the NeuroStore studyset releases, the maintained source of coordinate data.
NEUROSTORE_RELEASES_URL = f"{NEUROSTORE_URL}/api/neurostore-studyset-releases/"

NEUROSYNTH_DEPRECATION_MESSAGE = (
    "fetch_neurosynth downloads a frozen snapshot of the Neurosynth database "
    "(data extracted in 2018; files last repackaged in 2021) and is deprecated: it is "
    "kept only for reproducing published Neurosynth analyses and for the term "
    "annotations Neurosynth-based decoding needs. For up-to-date coordinate data, use "
    "nimare.extract.fetch_neurostore(), which downloads a NeuroStore studyset release "
    f"from {NEUROSTORE_RELEASES_URL}. Silence this warning with "
    "warnings.filterwarnings('ignore', message='fetch_neurosynth downloads a frozen')."
)


def _find_entities(filename, search_pairs, log=False):
    """Search file for any matching patterns of entities."""
    # Convert all string-based kwargs to lists
    search_pairs = {k: [v] if isinstance(v, str) else v for k, v in search_pairs.items()}
    search_pairs = [[f"{k}-{v_i}" for v_i in v] for k, v in search_pairs.items()]
    searches = list(itertools.product(*search_pairs))

    if log:
        LGR.info(f"Searching for any feature files matching the following criteria: {searches}")

    file_parts = filename.split("_")
    suffix = file_parts[-1]
    valid_entities_for_suffix = VALID_ENTITIES[suffix]
    for search in searches:
        temp_search = [term for term in search if term.split("-")[0] in valid_entities_for_suffix]
        if all(term in file_parts for term in temp_search):
            return True

    return False


# Entity keys that may appear in a database feature filename, e.g.
# "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz".
# Only these keys are recognized when parsing, so an unexpected filename segment
# cannot silently produce an incorrect entity mapping.
_SEARCHABLE_ENTITIES = ("version", "vocab", "source", "type")
_KNOWN_ENTITIES = ("data", *_SEARCHABLE_ENTITIES)


def _get_available_entities(database_file_manifest, data=None):
    """Collect the searchable entity values present in the database manifest.

    Only the known entity keys (``data`` plus the searchable ``version``/``vocab``/
    ``source``/``type``) are parsed from each feature filename; any other ``key-value``
    segment is ignored, so filename changes cannot silently mismap entities.

    Parameters
    ----------
    database_file_manifest : :obj:`list` of :obj:`dict`
        Parsed ``database_file_manifest.json``; each entry has a ``"features"``
        list of dictionaries with a ``"features"`` filename.
    data : :obj:`str` or None, optional
        If provided, restrict results to feature files for this data source
        (e.g. ``"neurosynth"``). Default is None (all sources).

    Returns
    -------
    :obj:`dict`
        Mapping of each searchable entity (``"version"``, ``"vocab"``,
        ``"source"``, ``"type"``) to a sorted list of its distinct values
        across the matching feature files. Entities with no values are omitted.
    """
    values = {key: set() for key in _SEARCHABLE_ENTITIES}
    for database in database_file_manifest:
        for feature_dict in database.get("features", []):
            entities = {}
            for part in feature_dict.get("features", "").split("_"):
                key, sep, value = part.partition("-")
                if sep and key in _KNOWN_ENTITIES:
                    entities[key] = value
            if data is not None and entities.get("data") != data:
                continue
            for key in _SEARCHABLE_ENTITIES:
                if key in entities:
                    values[key].add(entities[key])
    return {key: sorted(vals) for key, vals in values.items() if vals}


def _fetch_database(search_pairs, database_url, out_dir, overwrite=False):
    """Fetch generic database."""
    res_dir = get_resource_path()
    database_file_manifest = load_json(op.join(res_dir, "database_file_manifest.json"))

    out_dir = op.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    found_databases = []
    found_files = []
    log = True
    for database in database_file_manifest:
        coordinates_file = database["coordinates"]
        metadata_file = database["metadata"]
        if not _find_entities(coordinates_file, search_pairs, log=log):
            log = False
            continue

        log = False

        feature_dicts = database["features"]
        for feature_dict in feature_dicts:
            features_file = feature_dict["features"]
            # Other files associated with features have subset of entities,
            # so unnecessary to search them if we assume that the hard-coded manifest is valid.
            if not _find_entities(features_file, search_pairs):
                continue
            else:
                out_coordinates_file = op.join(out_dir, coordinates_file)
                out_metadata_file = op.join(out_dir, metadata_file)
                out_feature_dict = {k: op.join(out_dir, v) for k, v in feature_dict.items()}

                db_found = [
                    i_db
                    for i_db, db_dct in enumerate(found_databases)
                    if db_dct["coordinates"] == out_coordinates_file
                ]
                if len(db_found):
                    assert len(db_found) == 1

                    found_databases[db_found[0]]["features"].append(out_feature_dict)
                else:
                    found_databases.append(
                        {
                            "coordinates": out_coordinates_file,
                            "metadata": out_metadata_file,
                            "features": [out_feature_dict],
                        }
                    )
                found_files += [coordinates_file, metadata_file, *feature_dict.values()]

    if not found_databases:
        data_value = search_pairs.get("data")
        available = _get_available_entities(database_file_manifest, data=data_value)
        scope = f"data-{data_value}" if data_value else "the requested database"
        if available:
            options = "; ".join(f"{key}: {', '.join(vals)}" for key, vals in available.items())
            detail = f" Available options for {scope} are: {options}."
        else:
            detail = (
                f" No matching entries were found for {scope}; "
                "see the database file manifest for valid patterns."
            )
        raise ValueError(f"No files matched the query {search_pairs}.{detail}")

    found_files = sorted(list(set(found_files)))
    for found_file in found_files:
        print(f"Downloading {found_file}", flush=True)

        url = op.join(database_url, found_file + "?raw=true")
        out_file = op.join(out_dir, found_file)

        if op.isfile(out_file) and not overwrite:
            print("File exists and overwrite is False. Skipping.")
            continue

        with open(out_file, "wb") as fo:
            u = urlopen(url)

            block_size = 8192
            while True:
                buffer = u.read(block_size)
                if not buffer:
                    break
                fo.write(buffer)

    return found_databases


def _materialize_found_databases(found_databases, return_type, target):
    """Convert downloaded database manifests into NiMARE objects when requested."""
    from nimare.io import convert_neurosynth_to_dataset

    if return_type == "files":
        return found_databases

    if return_type not in VALID_FETCH_RETURN_TYPES:
        raise ValueError(
            f"Invalid return_type '{return_type}'. "
            f"Expected one of: {', '.join(sorted(VALID_FETCH_RETURN_TYPES))}."
        )

    from nimare.dataset import _quiet_dataset_deprecation

    materialized = []
    for database in found_databases:
        # A studyset request only passes through Dataset; don't warn about the detour.
        silence = (
            contextlib.nullcontext() if return_type == "dataset" else _quiet_dataset_deprecation()
        )
        with silence:
            dataset = convert_neurosynth_to_dataset(
                coordinates_file=database["coordinates"],
                metadata_file=database["metadata"],
                annotations_files=database["features"],
                target=target,
            )
        if return_type == "dataset":
            materialized.append(dataset)
        else:
            from nimare.nimads import Studyset

            materialized.append(Studyset.from_dataset(dataset))

    return materialized


def fetch_neurostore_releases(url=None, timeout=30):
    """List the NeuroStore studyset releases available for download.

    .. versionadded:: 0.22.0

    Parameters
    ----------
    url : :obj:`str` or None, optional
        Release index to query. Default is :data:`NEUROSTORE_RELEASES_URL`.
    timeout : :obj:`int` or :obj:`float`, optional
        Timeout, in seconds, for the request. Default is 30.

    Returns
    -------
    releases : :obj:`list` of :obj:`dict`
        One entry per release, as returned by the NeuroStore API. Each entry has at
        least ``"version"`` (e.g. ``"2026-09"`` or ``"nightly"``), ``"release_type"``,
        ``"built_at"``, ``"study_count"``, and ``"download_path"``.

    See Also
    --------
    fetch_neurostore : Download one of these releases as a Studyset.
    """
    url = url or NEUROSTORE_RELEASES_URL

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    releases = payload.get("results", []) if isinstance(payload, dict) else payload
    if not releases:
        raise ValueError(f"No NeuroStore studyset releases were found at {url}.")

    return list(releases)


def _release_sort_key(release):
    """Order releases by build time, falling back to the version string."""
    return (str(release.get("built_at") or ""), str(release.get("version") or ""))


def _select_neurostore_release(releases, version):
    """Pick the requested release from the release index.

    ``"latest"`` selects the most recently built dated release, ignoring the rolling
    ``"nightly"`` build; ``"nightly"`` selects the nightly build; anything else is
    matched against the releases' ``"version"`` fields.
    """
    if version == "latest":
        dated = [rel for rel in releases if rel.get("release_type") != "nightly"]
        candidates = dated or list(releases)
        return max(candidates, key=_release_sort_key)

    matches = [rel for rel in releases if str(rel.get("version")) == str(version)]
    if not matches:
        available = ", ".join(sorted(str(rel.get("version")) for rel in releases))
        raise ValueError(
            f"No NeuroStore studyset release named '{version}'. "
            f"Available versions are: {available} (or 'latest')."
        )

    return max(matches, key=_release_sort_key)


def _download_neurostore_archive(release, out_file, url, timeout=30, verify_checksum=True):
    """Stream a release archive to disk, checking its checksum when one is published."""
    download_path = release.get("download_url") or release.get("download_path")
    if not download_path:
        raise ValueError(
            f"NeuroStore release '{release.get('version')}' does not advertise a download path."
        )

    download_url = urljoin(url, download_path)
    LGR.info(f"Downloading NeuroStore studyset release from {download_url}")

    digest = hashlib.sha256()
    with requests.get(download_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(out_file, "wb") as fo:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:  # filter out keep-alive chunks
                    continue
                digest.update(chunk)
                fo.write(chunk)

    expected_checksum = release.get("archive_checksum")
    if verify_checksum and expected_checksum and digest.hexdigest() != expected_checksum:
        os.remove(out_file)
        raise ValueError(
            f"Checksum mismatch for NeuroStore release '{release.get('version')}': "
            f"expected {expected_checksum}, got {digest.hexdigest()}."
        )

    return out_file


def _extract_neurostore_archive(archive_file, out_dir):
    """Extract a release archive and return the directory holding ``studyset.json``."""
    if op.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    with tarfile.open(archive_file, "r:gz") as tf:
        # ``data`` refuses absolute paths and paths escaping the destination.
        if hasattr(tarfile, "data_filter"):
            tf.extractall(out_dir, filter="data")
        else:  # pragma: no cover - Python builds without PEP 706 filters
            tf.extractall(out_dir)

    return _find_parquet_studyset_dir(out_dir)


def _find_parquet_studyset_dir(root):
    """Return the directory containing ``studyset.json``, which archives nest one level."""
    if op.isfile(op.join(root, "studyset.json")):
        return root

    candidates = [
        op.join(root, entry)
        for entry in sorted(os.listdir(root))
        if op.isfile(op.join(root, entry, "studyset.json"))
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not find a parquet studyset (a directory with studyset.json) in {root}."
        )

    return candidates[0]


def fetch_neurostore(
    version="latest",
    data_dir=None,
    overwrite=False,
    return_type="studyset",
    target="mni152_2mm",
    url=None,
    timeout=30,
    verify_checksum=True,
):
    """Download a NeuroStore studyset release.

    .. versionadded:: 0.22.0

    NeuroStore releases are the maintained source of coordinate data for NiMARE, and
    should be preferred over the frozen Neurosynth and NeuroQuery snapshots downloaded
    by :func:`~nimare.extract.fetch_neurosynth` and
    :func:`~nimare.extract.fetch_neuroquery`. Releases are published as archives of
    parquet tables at https://neurostore.org/api/neurostore-studyset-releases/ and load
    directly into the columnar tables a :class:`~nimare.nimads.Studyset` stores.

    Parameters
    ----------
    version : :obj:`str`, optional
        Release to download. ``"latest"`` (the default) takes the most recent dated
        release, ``"nightly"`` takes the rolling nightly build, and any other value is
        matched against the published version strings (e.g. ``"2026-09"``).
        Call :func:`~nimare.extract.fetch_neurostore_releases` to see what is available.
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default, files are downloaded in the
        home directory. A subfolder, named ``neurostore``, will be created in
        ``data_dir``, with one subfolder per downloaded release.
    overwrite : :obj:`bool`, optional
        Whether to re-download and re-extract a release that is already on disk.
        Default is False.
    return_type : {"studyset", "files"}, optional
        Type of object to return. The default, ``"studyset"``, returns a
        :class:`~nimare.nimads.Studyset`; ``"files"`` returns the path to the extracted
        release directory without loading it.
    target : :obj:`str`, optional
        Target template space for the returned Studyset. Default is ``"mni152_2mm"``.
        Ignored when ``return_type="files"``.
    url : :obj:`str` or None, optional
        Release index to query. Default is :data:`NEUROSTORE_RELEASES_URL`.
    timeout : :obj:`int` or :obj:`float`, optional
        Timeout, in seconds, for each request. Default is 30.
    verify_checksum : :obj:`bool`, optional
        Whether to check the downloaded archive against the checksum published with the
        release. Default is True.

    Returns
    -------
    :class:`~nimare.nimads.Studyset` or :obj:`str`
        The downloaded studyset, or the path to the extracted release directory when
        ``return_type="files"``.

    Notes
    -----
    Reading the parquet tables requires ``pyarrow``, which NiMARE declares as the
    ``parquet`` extra (``pip install nimare[parquet]``). ``return_type="files"`` works
    without it.

    Examples
    --------
    Download the most recent release and run a meta-analysis on part of it::

        studyset = fetch_neurostore()
        print(studyset)

    See Also
    --------
    fetch_neurostore_releases : List the releases available for download.
    nimare.io.fetch_neurostore_studyset : Download a single studyset from NeuroStore.
    """
    valid_return_types = {"studyset", "files"}
    if return_type not in valid_return_types:
        raise ValueError(
            f"Invalid return_type '{return_type}'. "
            f"Expected one of: {', '.join(sorted(valid_return_types))}."
        )

    url = url or NEUROSTORE_RELEASES_URL
    releases = fetch_neurostore_releases(url=url, timeout=timeout)
    release = _select_neurostore_release(releases, version)
    release_version = str(release.get("version"))

    data_dir = _get_dataset_dir("neurostore", data_dir=data_dir)
    release_dir = op.join(data_dir, release_version)

    if op.isdir(release_dir) and not overwrite:
        studyset_dir = _find_parquet_studyset_dir(release_dir)
        LGR.info(f"Using cached NeuroStore release '{release_version}' in {studyset_dir}.")
    else:
        # basename: the name is taken from the API response, and only ever names a
        # file inside the download directory.
        archive_name = op.basename(
            release.get("archive_name") or f"neurostore-studyset-{release_version}.tar.gz"
        )
        archive_file = op.join(data_dir, archive_name)
        try:
            _download_neurostore_archive(
                release,
                archive_file,
                url=url,
                timeout=timeout,
                verify_checksum=verify_checksum,
            )
            studyset_dir = _extract_neurostore_archive(archive_file, release_dir)
        finally:
            # The extracted tables are what callers use; the archive is just transport.
            if op.isfile(archive_file):
                os.remove(archive_file)

    if return_type == "files":
        return studyset_dir

    from nimare.nimads import Studyset

    return Studyset(studyset_dir, target=target)


def fetch_neurosynth(
    data_dir=None,
    version="7",
    overwrite=False,
    return_type="studyset",
    target="mni152_2mm",
    **kwargs,
):
    """Download the frozen Neurosynth database files.

    .. deprecated:: 0.22.0

        Neurosynth's data files are a frozen snapshot: the data were extracted in
        2018, and the files were last repackaged in 2021. This function is kept for
        reproducing published Neurosynth analyses (and for the term annotations that
        Neurosynth-based decoding needs), and will be removed in NiMARE 1.0.0. For
        up-to-date coordinate data, use :func:`~nimare.extract.fetch_neurostore`.

    .. versionchanged:: 0.0.10

        * Use new format for Neurosynth and NeuroQuery files.
        * Change "path" parameter to "data_dir".

    .. versionadded:: 0.0.4

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default, files are downloaded in home directory.
        A subfolder, named ``neurosynth``, will be created in ``data_dir``, which is where the
        files will be located.
    version : str or list, optional
        The version to fetch. The default is "7" (Neurosynth's latest version).
    overwrite : bool, optional
        Whether to overwrite existing files or not. Default is False.
    return_type : {"studyset", "dataset", "files"}, optional
        Type of object to return after downloading. The default is ``"studyset"``.
        Use ``"dataset"`` for the legacy Dataset return type or ``"files"`` to return the
        downloaded file manifest without conversion.
    target : {'mni152_2mm', 'ale_2mm'}, optional
        Target template space used when constructing Dataset or Studyset outputs.
        Ignored when ``return_type="files"``.
    kwargs : dict, optional
        Keyword arguments to select relevant feature files.
        Valid kwargs include: source, vocab, type.
        Each kwarg may be a string or a list of strings.
        For most Neurosynth term-based workflows, including the decoding examples in NiMARE,
        use ``source="abstract"`` and ``vocab="terms"``.
        If no kwargs are provided, all feature files for the specified database version will be
        downloaded, including multiple annotation sets.

    Returns
    -------
    outputs : :obj:`list`
        List of downloaded databases, returned as Studysets, Datasets, or file-manifest
        dictionaries depending on ``return_type``.

    Notes
    -----
    This function was adapted from neurosynth.base.dataset.download().

    The ``source``, ``vocab``, and ``type`` keyword arguments are selectors for annotation files:

    - ``source`` identifies which text source the annotations came from.
      For Neurosynth, the available source is currently ``"abstract"``.
    - ``vocab`` identifies the annotation vocabulary.
      ``"terms"`` selects term-level tf-idf features, while ``"LDA50"``, ``"LDA100"``,
      ``"LDA200"``, and ``"LDA400"`` select topic-model features for versions 6 and 7.
    - ``type`` identifies the feature representation.
      ``"tfidf"`` is used for term annotations, while ``"weight"`` is used for LDA topics.

    Only combinations present in NiMARE's database manifest are valid.
    For Neurosynth, the supported combinations are:

    ======== ========= ========
    source   vocab     type
    ======== ========= ========
    abstract terms     tfidf
    abstract LDA50     weight
    abstract LDA100    weight
    abstract LDA200    weight
    abstract LDA400    weight
    ======== ========= ========

    Versions 3, 4, and 5 only provide ``abstract`` + ``terms`` + ``tfidf``.
    The LDA vocabularies are only available for versions 6 and 7.

    Examples
    --------
    Fetch the abstract-derived term annotations used by most Neurosynth decoding workflows::

        fetch_neurosynth(version="7", source="abstract", vocab="terms")

    .. warning::
        ``return_type="dataset"`` is deprecated and will be removed in NiMARE 1.0.0.
        Prefer the default ``return_type="studyset"``.

    .. warning::
        Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
        format. Old code using this function **will not work** with the new version.

    See Also
    --------
    fetch_neurostore : Download an up-to-date NeuroStore studyset release.
    """
    warnings.warn(NEUROSYNTH_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)

    URL = (
        "https://github.com/neurosynth/neurosynth-data/blob/"
        "209c33cd009d0b069398a802198b41b9c488b9b7/"
    )
    dataset_name = "neurosynth"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)

    kwargs["data"] = dataset_name
    kwargs["version"] = version

    found_databases = _fetch_database(kwargs, URL, data_dir, overwrite=overwrite)

    return _materialize_found_databases(found_databases, return_type=return_type, target=target)


def fetch_neuroquery(
    data_dir=None,
    version="1",
    overwrite=False,
    return_type="studyset",
    target="mni152_2mm",
    **kwargs,
):
    """Download the latest data files from NeuroQuery.

    .. versionadded:: 0.0.10

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default, files are downloaded in home directory.
    version : str or list, optional
        The version to fetch. The default is "7" (Neurosynth's latest version).
    url : None or str, optional
        Specific URL to download. If not None, overrides URL to current data.
        If you want to fetch Neurosynth's data from *before* the 2021 reorganization,
        you will need to use this argument.
    kwargs
        Keyword arguments to select relevant feature files.
        Valid kwargs include: source, vocab, type.
        Each kwarg may be a string or a list of strings.
        If no kwargs are provided, all feature files for the specified database version will be
        downloaded.
    return_type : {"studyset", "dataset", "files"}, optional
        Type of object to return after downloading. The default is ``"studyset"``.
        Use ``"dataset"`` for the legacy Dataset return type or ``"files"`` to return the
        downloaded file manifest without conversion.
    target : {'mni152_2mm', 'ale_2mm'}, optional
        Target template space used when constructing Dataset or Studyset outputs.
        Ignored when ``return_type="files"``.

    Returns
    -------
    outputs : :obj:`list`
        List of downloaded databases, returned as Studysets, Datasets, or file-manifest
        dictionaries depending on ``return_type``.

    Notes
    -----
    This function was adapted from neurosynth.base.dataset.download().

    .. warning::
        ``return_type="dataset"`` is deprecated and will be removed in NiMARE 1.0.0.
        Prefer the default ``return_type="studyset"``.

    .. note::
        Like Neurosynth's, NeuroQuery's data files are a frozen snapshot. For up-to-date
        coordinate data, use :func:`~nimare.extract.fetch_neurostore`.

    See Also
    --------
    fetch_neurostore : Download an up-to-date NeuroStore studyset release.
    """
    URL = (
        "https://github.com/neuroquery/neuroquery_data/blob/"
        "4580f86267fb7c14ac1f601e298cbed898d79f2d/data/"
    )
    dataset_name = "neuroquery"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)

    kwargs["data"] = dataset_name
    kwargs["version"] = version

    found_databases = _fetch_database(kwargs, URL, data_dir, overwrite=overwrite)

    return _materialize_found_databases(found_databases, return_type=return_type, target=target)


def download_nidm_pain(data_dir=None, overwrite=False):
    """Download NIDM Results for 21 pain studies from NeuroVault for tests.

    .. versionadded:: 0.0.2

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default, files are downloaded in home directory.
        A subfolder, named ``neuroquery``, will be created in ``data_dir``, which is where the
        files will be located.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.

    Returns
    -------
    data_dir : :obj:`str`
        Updated data directory pointing to dataset files.
    """
    url = "https://neurovault.org/collections/1425/download"

    dataset_name = "nidm_21pain"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    desc_file = op.join(data_dir, "description.txt")
    if op.isfile(desc_file) and overwrite is False:
        return data_dir

    # Download
    fname = op.join(data_dir, url.split("/")[-1])
    _download_zipped_file(url, filename=fname)

    # Unzip
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    collection_folders = [f for f in glob(op.join(data_dir, "*")) if ".nidm" not in f]
    collection_folders = [f for f in collection_folders if op.isdir(f)]
    if len(collection_folders) > 1:
        raise Exception(f"More than one folder found: {', '.join(collection_folders)}")
    else:
        folder = collection_folders[0]
    zip_files = glob(op.join(folder, "*.zip"))
    for zf in zip_files:
        fn = op.splitext(op.basename(zf))[0]
        with zipfile.ZipFile(zf, "r") as zip_ref:
            zip_ref.extractall(op.join(data_dir, fn))

    os.remove(fname)
    shutil.rmtree(folder)

    with open(desc_file, "w") as fo:
        fo.write("21 pain studies in NIDM-results packs.")
    return data_dir


def download_cognitive_atlas(data_dir=None, overwrite=False):
    """Download Cognitive Atlas ontology and extract IDs and relationships.

    .. versionadded:: 0.0.2

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default, files are downloaded in home directory.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.

    Returns
    -------
    out_dict : :obj:`dict`
        Dictionary with two keys: 'ids' and 'relationships'. Each points to a
        csv file. The 'ids' file contains CogAt identifiers, canonical names,
        and aliases, sorted by alias length (number of characters).
        The 'relationships' file contains associations between CogAt items,
        with three columns: input, output, and rel_type (relationship type).
    """
    from cognitiveatlas.api import get_concept, get_disorder, get_task

    dataset_name = "cognitive_atlas"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)

    ids_file = op.join(data_dir, "cogat_aliases.csv")
    rels_file = op.join(data_dir, "cogat_relationships.csv")
    if overwrite or not all([op.isfile(f) for f in [ids_file, rels_file]]):
        concepts = get_concept(silent=True).pandas
        tasks = get_task(silent=True).pandas
        disorders = get_disorder(silent=True).pandas

        # Identifiers and aliases
        long_concepts = _longify(concepts)
        long_tasks = _longify(tasks)

        # Disorders currently lack aliases
        disorders["name"] = disorders["name"].str.lower()
        disorders = disorders.assign(alias=disorders["name"])
        disorders = disorders[["id", "name", "alias"]]

        # Combine into aliases DataFrame
        aliases = pd.concat((long_concepts, long_tasks, disorders), axis=0)
        aliases = _expand_df(aliases)
        aliases = aliases.replace("", np.nan)
        aliases = aliases.dropna(axis=0)
        aliases = aliases.reset_index(drop=True)

        # Relationships
        relationship_list = []
        for i, id_ in enumerate(concepts["id"].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, "isSelf"]
            relationship_list.append(row)
            concept = get_concept(id=id_, silent=True).json
            for rel in concept["relationships"]:
                reltype = _get_concept_reltype(rel["relationship"], rel["direction"])
                if reltype is not None:
                    row = [id_, rel["id"], reltype]
                    relationship_list.append(row)

        for i, id_ in enumerate(tasks["id"].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, "isSelf"]
            relationship_list.append(row)
            task = get_task(id=id_, silent=True).json
            for rel in task["concepts"]:
                row = [id_, rel["concept_id"], "measures"]
                relationship_list.append(row)
                row = [rel["concept_id"], id_, "measuredBy"]
                relationship_list.append(row)

        for i, id_ in enumerate(disorders["id"].unique()):
            if i % 100 == 0:
                time.sleep(5)
            row = [id_, id_, "isSelf"]
            relationship_list.append(row)
            disorder = get_disorder(id=id_, silent=True).json
            for rel in disorder["disorders"]:
                if rel["relationship"] == "ISA":
                    rel_type = "isA"
                else:
                    rel_type = rel["relationship"]
                row = [id_, rel["id"], rel_type]
                relationship_list.append(row)

        relationships = pd.DataFrame(
            columns=["input", "output", "rel_type"], data=relationship_list
        )
        ctp_df = concepts[["id", "id_concept_class"]]
        ctp_df = ctp_df.assign(rel_type="inCategory")
        ctp_df.columns = ["input", "output", "rel_type"]
        ctp_df["output"].replace("", np.nan, inplace=True)
        ctp_df.dropna(axis=0, inplace=True)
        relationships = pd.concat((ctp_df, relationships))
        relationships = relationships.reset_index(drop=True)
        aliases.to_csv(ids_file, index=False)
        relationships.to_csv(rels_file, index=False)
    out_dict = {"ids": ids_file, "relationships": rels_file}

    return out_dict


def download_abstracts(dataset, email):
    """Download the abstracts for a list of PubMed IDs.

    Uses the BioPython package.

    .. versionadded:: 0.0.2

    Parameters
    ----------
    dataset : :obj:`~nimare.dataset.Dataset` or :obj:`~nimare.nimads.Studyset`
        A Dataset or Studyset where study IDs correspond to PMIDs.
    email : :obj:`str`
        Email address to use to call the PubMed API

    Returns
    -------
    dataset : :obj:`~nimare.dataset.Dataset` or :obj:`~nimare.nimads.Studyset`

    .. warning::
        Passing a :class:`~nimare.dataset.Dataset` is deprecated and will be removed in NiMARE
        1.0.0. Prefer passing a :class:`~nimare.nimads.Studyset`.

    This function assumes that the dataset uses identifiers in the format
    [PMID-EXPID]. Thus, the ``study_id`` column of the
    :py:attr:`~nimare.dataset.Dataset.texts` DataFrame should correspond to PMID.
    """
    try:
        from Bio import Entrez, Medline
    except ImportError:
        raise Exception("Module biopython is required for downloading abstracts from PubMed.")

    from nimare.nimads import Studyset

    Entrez.email = email

    if isinstance(dataset, Dataset):
        pmids = dataset.texts["study_id"].astype(str).tolist()
        pmids = sorted(list(set(pmids)))
    elif isinstance(dataset, Studyset):
        pmids = dataset.texts["study_id"].astype(str).tolist()
        pmids = sorted(list(set(pmids)))
    elif isinstance(dataset, list):
        pmids = [str(pmid) for pmid in dataset]
    else:
        raise Exception(f"Dataset type not recognized: {type(dataset)}")

    records = []
    # PubMed only allows you to search ~1000 at a time. I chose 900 to be safe.
    chunks = [pmids[x : x + 900] for x in range(0, len(pmids), 900)]
    for i, chunk in enumerate(chunks):
        LGR.info(f"Downloading chunk {i + 1} of {len(chunks)}")
        h = Entrez.efetch(db="pubmed", id=chunk, rettype="medline", retmode="text")
        records += list(Medline.parse(h))

    # Pull data for studies with abstracts
    data = [[study["PMID"], study["AB"]] for study in records if study.get("AB", None)]
    df = pd.DataFrame(columns=["study_id", "abstract"], data=data)
    if not isinstance(dataset, Dataset):
        if isinstance(dataset, Studyset):
            abstract_by_study = df.drop_duplicates("study_id").set_index("study_id")["abstract"]
            store = dataset.store
            rows, abstracts = [], []
            for row in range(store.n_analyses):
                study_id = str(store.study_key[store.study_idx[row]])
                abstract = abstract_by_study.get(study_id, None)
                if abstract is None:
                    continue
                rows.append(row)
                abstracts.append(abstract)
            if not rows:
                return dataset
            return dataset.with_texts(rows, "abstract", abstracts)
        return df

    dataset.texts = pd.merge(
        dataset.texts, df, left_on="study_id", right_on="study_id", how="left"
    )
    return dataset
