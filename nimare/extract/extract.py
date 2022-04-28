"""Tools for downloading datasets."""
import itertools
import json
import logging
import os
import os.path as op
import shutil
import tarfile
import time
import zipfile
from glob import glob
from io import BytesIO
from lzma import LZMAFile
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from nimare.dataset import Dataset
from nimare.extract.utils import (
    _download_zipped_file,
    _expand_df,
    _get_concept_reltype,
    _get_dataset_dir,
    _longify,
)
from nimare.utils import get_resource_path

LGR = logging.getLogger(__name__)

VALID_ENTITIES = {
    "coordinates.tsv.gz": ["data", "version"],
    "metadata.tsv.gz": ["data", "version"],
    "features.npz": ["data", "version", "vocab", "source", "type"],
    "vocabulary.txt": ["data", "version", "vocab"],
    "metadata.json": ["data", "version", "vocab"],
    "keys.tsv": ["data", "version", "vocab"],
}


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


def _fetch_database(search_pairs, database_url, out_dir, overwrite=False):
    """Fetch generic database."""
    res_dir = get_resource_path()
    with open(op.join(res_dir, "database_file_manifest.json"), "r") as fo:
        database_file_manifest = json.load(fo)

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


def fetch_neurosynth(data_dir=None, version="7", overwrite=False, **kwargs):
    """Download the latest data files from NeuroSynth.

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
    kwargs : dict, optional
        Keyword arguments to select relevant feature files.
        Valid kwargs include: source, vocab, type.
        Each kwarg may be a string or a list of strings.
        If no kwargs are provided, all feature files for the specified database version will be
        downloaded.

    Returns
    -------
    found_databases : :obj:`list` of :obj:`dict`
        List of dictionaries indicating datasets downloaded.
        Each list entry is a different database, containing a dictionary with three keys:
        "coordinates", "metadata", and "features". "coordinates" and "metadata" will be filenames.
        "features" will be a list of dictionaries, each containing "id", "vocab", and "features"
        keys with associated files.

    Notes
    -----
    This function was adapted from neurosynth.base.dataset.download().

    Warnings
    --------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    URL = (
        "https://github.com/neurosynth/neurosynth-data/blob/"
        "209c33cd009d0b069398a802198b41b9c488b9b7/"
    )
    dataset_name = "neurosynth"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)

    kwargs["data"] = dataset_name
    kwargs["version"] = version

    found_databases = _fetch_database(kwargs, URL, data_dir, overwrite=overwrite)

    return found_databases


def fetch_neuroquery(data_dir=None, version="1", overwrite=False, **kwargs):
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

    Returns
    -------
    found_databases : :obj:`list` of :obj:`dict`
        List of dictionaries indicating datasets downloaded.
        Each list entry is a different database, containing a dictionary with three keys:
        "coordinates", "metadata", and "features". "coordinates" and "metadata" will be filenames.
        "features" will be a list of dictionaries, each containing "id", "vocab", and "features"
        keys with associated files.

    Notes
    -----
    This function was adapted from neurosynth.base.dataset.download().
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

    return found_databases


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
    dataset : :obj:`~nimare.dataset.Dataset`
        A Dataset object where IDs are in the form PMID-EXPID
    email : :obj:`str`
        Email address to use to call the PubMed API

    Returns
    -------
    dataset : :obj:`~nimare.dataset.Dataset`

    Warnings
    --------
    This function assumes that the dataset uses identifiers in the format
    [PMID-EXPID]. Thus, the ``study_id`` column of the
    :py:attr:`~nimare.dataset.Dataset.texts` DataFrame should correspond to PMID.
    """
    try:
        from Bio import Entrez, Medline
    except ImportError:
        raise Exception("Module biopython is required for downloading abstracts from PubMed.")

    Entrez.email = email

    if isinstance(dataset, Dataset):
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
        return df

    dataset.texts = pd.merge(
        dataset.texts, df, left_on="study_id", right_on="study_id", how="left"
    )
    return dataset


def download_peaks2maps_model(data_dir=None, overwrite=False):
    """Download the trained Peaks2Maps model from OHBM 2018.

    .. deprecated:: 0.0.11
        `download_peaks2maps_model` will be removed in NiMARE 0.0.13.

    .. versionadded:: 0.0.2

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str` or None, optional
        Where to put the trained model.
        If None, then download to the automatic NiMARE data directory.
        Default is None.
    overwrite : bool, optional
        Whether to overwrite an existing model or not. Default is False.

    Returns
    -------
    data_dir : str
        Path to folder containing model.
    """
    url = "https://zenodo.org/record/1257721/files/ohbm2018_model.tar.xz?download=1"

    temp_dataset_name = "peaks2maps_model_ohbm2018__temp"
    data_dir = _get_dataset_dir("", data_dir=data_dir)
    temp_data_dir = _get_dataset_dir(temp_dataset_name, data_dir=data_dir)

    dataset_name = "peaks2maps_model_ohbm2018"
    if dataset_name not in data_dir:  # allow data_dir to include model folder
        data_dir = temp_data_dir.replace(temp_dataset_name, dataset_name)

    desc_file = op.join(data_dir, "description.txt")
    if op.isfile(desc_file) and overwrite is False:
        shutil.rmtree(temp_data_dir)
        return data_dir

    LGR.info("Downloading the model (this is a one-off operation)...")
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    f = BytesIO()

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024 * 1024
    wrote = 0
    for data in tqdm(
        r.iter_content(block_size),
        total=np.ceil(total_size // block_size),
        unit="MB",
        unit_scale=True,
    ):
        wrote = wrote + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception("Download interrupted")

    f.seek(0)
    LGR.info(f"Uncompressing the model to {temp_data_dir}...")
    tf_file = tarfile.TarFile(fileobj=LZMAFile(f), mode="r")
    tf_file.extractall(temp_data_dir)

    os.rename(op.join(temp_data_dir, "ohbm2018_model"), data_dir)
    shutil.rmtree(temp_data_dir)

    with open(desc_file, "w") as fo:
        fo.write("The trained Peaks2Maps model from OHBM 2018.")

    LGR.debug(f"Dataset moved to {data_dir}")

    return data_dir
