"""Tools for downloading datasets."""
import logging
import math
import os
import os.path as op
import shutil
import sys
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

from ..dataset import Dataset
from .utils import (
    _download_zipped_file,
    _expand_df,
    _get_concept_reltype,
    _get_dataset_dir,
    _longify,
)

LGR = logging.getLogger(__name__)


def fetch_neurosynth(path=".", url=None, unpack=False):
    """Download the latest data files from NeuroSynth.

    Parameters
    ----------
    path : str
        Location to save the retrieved data files. Defaults to current directory.
    url : None or str, optional
        Specific URL to download. If not None, overrides URL to current data.
    unpack : bool, optional
        If True, unzips the data file post-download. Defaults to False.

    Notes
    -----
    This function was originally neurosynth.base.dataset.download().
    """
    if url is None:
        url = (
            "https://github.com/neurosynth/neurosynth-data/blob/master/current_data.tar.gz?"
            "raw=true"
        )
    if os.path.exists(path) and os.path.isdir(path):
        basename = os.path.basename(url).split("?")[0]
        filename = os.path.join(path, basename)
    else:
        filename = path

    f = open(filename, "wb")

    u = urlopen(url)
    file_size = int(u.headers["Content-Length"][0])
    print("Downloading the latest Neurosynth files: {0} bytes: {1}".format(url, file_size))

    bytes_dl = 0
    block_size = 8192
    while True:
        buffer = u.read(block_size)
        if not buffer:
            break
        bytes_dl += len(buffer)
        f.write(buffer)
        p = float(bytes_dl) / file_size
        status = r"{0}  [{1:.2%}]".format(bytes_dl, p)
        status = status + chr(8) * (len(status) + 1)
        sys.stdout.write(status)

    f.close()

    if unpack:
        tarfile.open(filename, "r:gz").extractall(os.path.dirname(filename))


def download_nidm_pain(data_dir=None, overwrite=False, verbose=1):
    """Download NIDM Results for 21 pain studies from NeuroVault for tests.

    Parameters
    ----------
    data_dir : :obj:`str`, optional
        Location in which to place the studies. Default is None, which uses the
        package's default path for downloaded data.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.
    verbose : :obj:`int`, optional
        Default is 1.

    Returns
    -------
    data_dir : :obj:`str`
        Updated data directory pointing to dataset files.
    """
    url = "https://neurovault.org/collections/1425/download"

    dataset_name = "nidm_21pain"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
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
        raise Exception("More than one folder found: {0}".format(", ".join(collection_folders)))
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


def download_mallet(data_dir=None, overwrite=False, verbose=1):
    """Download the MALLET toolbox for LDA topic modeling.

    Parameters
    ----------
    data_dir : :obj:`str`, optional
        Location in which to place MALLET. Default is None, which uses the
        package's default path for downloaded data.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.
    verbose : :obj:`int`, optional
        Default is 1.

    Returns
    -------
    data_dir : :obj:`str`
        Updated data directory pointing to MALLET files.
    """
    url = "http://mallet.cs.umass.edu/dist/mallet-2.0.7.tar.gz"

    temp_dataset_name = "mallet__temp"
    temp_data_dir = _get_dataset_dir(temp_dataset_name, data_dir=data_dir, verbose=verbose)

    dataset_name = "mallet"
    data_dir = temp_data_dir.replace(temp_dataset_name, dataset_name)

    desc_file = op.join(data_dir, "description.txt")
    if op.isfile(desc_file) and overwrite is False:
        shutil.rmtree(temp_data_dir)
        return data_dir

    mallet_file = op.join(temp_data_dir, op.basename(url))
    _download_zipped_file(url, mallet_file)

    with tarfile.open(mallet_file) as tf:
        tf.extractall(path=temp_data_dir)

    os.rename(op.join(temp_data_dir, "mallet-2.0.7"), data_dir)

    os.remove(mallet_file)
    shutil.rmtree(temp_data_dir)

    with open(desc_file, "w") as fo:
        fo.write("The MALLET toolbox for latent Dirichlet allocation.")

    if verbose > 0:
        print("\nDataset moved to {}\n".format(data_dir))

    return data_dir


def download_cognitive_atlas(data_dir=None, overwrite=False, verbose=1):
    """Download Cognitive Atlas ontology and extract IDs and relationships.

    Parameters
    ----------
    data_dir : :obj:`str`, optional
        Location in which to place Cognitive Atlas files.
        Default is None, which uses the package's default path for downloaded
        data.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.
    verbose : :obj:`int`, optional
        Default is 1.

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
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

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
    """Download the abstracts for a list of PubMed IDs. Uses the BioPython package.

    Parameters
    ----------
    dataset : :obj:`nimare.dataset.Dataset`
        A Dataset object where IDs are in the form PMID-EXPID
    email : :obj:`str`
        Email address to use to call the PubMed API

    Returns
    -------
    dataset : :obj:`nimare.dataset.Dataset`

    Warning
    -------
    This function assumes that the dataset uses identifiers in the format
    [PMID-EXPID]. Thus, the ``study_id`` column of the
    :obj:`nimare.dataset.Dataset.texts` DataFrame should correspond to PMID.
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
        raise Exception("Dataset type not recognized: {0}".format(type(dataset)))

    records = []
    # PubMed only allows you to search ~1000 at a time. I chose 900 to be safe.
    chunks = [pmids[x : x + 900] for x in range(0, len(pmids), 900)]
    for i, chunk in enumerate(chunks):
        LGR.info("Downloading chunk {0} of {1}".format(i + 1, len(chunks)))
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


def download_peaks2maps_model(data_dir=None, overwrite=False, verbose=1):
    """Download the trained Peaks2Maps model from OHBM 2018.

    Parameters
    ----------
    data_dir : None or str, optional
        Where to put the trained model.
        If None, then download to the automatic NiMARE data directory.
        Default is None.
    overwrite : bool, optional
        Whether to overwrite an existing model or not. Default is False.
    verbose : int, optional
        Verbosity level. Default is 1.

    Returns
    -------
    data_dir : str
        Path to folder containing model.
    """
    url = "https://zenodo.org/record/1257721/files/ohbm2018_model.tar.xz?download=1"

    temp_dataset_name = "peaks2maps_model_ohbm2018__temp"
    data_dir = _get_dataset_dir("", data_dir=data_dir, verbose=verbose)
    temp_data_dir = _get_dataset_dir(temp_dataset_name, data_dir=data_dir, verbose=verbose)

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
        total=math.ceil(total_size // block_size),
        unit="MB",
        unit_scale=True,
    ):
        wrote = wrote + len(data)
        f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception("Download interrupted")

    f.seek(0)
    LGR.info("Uncompressing the model to {}...".format(temp_data_dir))
    tf_file = tarfile.TarFile(fileobj=LZMAFile(f), mode="r")
    tf_file.extractall(temp_data_dir)

    os.rename(op.join(temp_data_dir, "ohbm2018_model"), data_dir)
    shutil.rmtree(temp_data_dir)

    with open(desc_file, "w") as fo:
        fo.write("The trained Peaks2Maps model from OHBM 2018.")

    if verbose > 0:
        print("\nDataset moved to {}\n".format(data_dir))

    return data_dir
