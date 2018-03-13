"""
Classes for representing datasets of images and/or coordinates.
"""
from __future__ import print_function
import json
import time
import gzip
import pickle
from os import mkdir
from os.path import join, isdir, isfile
import pandas as pd
from pyneurovault import api
from ..utils import get_resource_path
from .extract import (NeuroVaultDataSource, NeurosynthDataSource,
                      BrainSpellDataSource)


def to_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def download_combined_database(out_dir, overwrite=False):
    """
    Download coordinates/annotations from brainspell and images/annotations
    from Neurovault.

    Currently, the largest barrier is the lack of links between experiments
    (tables) in brainspell/NeuroSynth and those in NeuroVault. The closest we
    have is overall papers, via DOIs.

    Additional problems:
    -   Does NeuroVault have standard error maps?
        -   If so, I doubt there's any way to associate a given SE map and beta
            map within a collection.
    -   How should space be handled?
        -   Should everything be resliced and transformed to the same space at
            this stage or later on?
        -   How can we link a target template (for images) to a target space
            (for coordinates)?
        -   Should we even allow custom targets? Maybe we just limit it to 2mm
            and 1mm MNI templates.

    Parameters
    ----------
    out_dir : :obj:`str`
        Folder in which to write out Dataset object and subfolders containing
        images.
    overwrite: :obj:`bool`, optional
        Whether to overwrite existing database, if one exists in `out_dir`.
        Defaults to False.
    """
    # Download collections metadata from Neurovault
    collections_file = join(out_dir, 'neurovault_collections.csv')
    if overwrite or not isfile(collections_file):
        colls_df = api.get_collections()
        colls_df.to_csv(collections_file, index=False, encoding='utf-8')
    else:
        colls_df = pd.read_csv(collections_file, encoding='utf-8')

    # Only include collections from published papers (or preprints)
    papers_file = join(out_dir, 'neurovault_papers.csv')
    if overwrite or not isfile(papers_file):
        paper_df = colls_df.dropna(subset=['DOI'])
        paper_df.to_csv(papers_file, index=False, encoding='utf-8')
    else:
        paper_df = pd.read_csv(papers_file, encoding='utf-8')

    # Get metadata for individual images from valid collections
    papers_metadata_file = join(out_dir, 'neurovault_papers_metadata.csv')
    if overwrite or not isfile(papers_metadata_file):
        valid_collections = sorted(paper_df['collection_id'].tolist())

        # Sleep between get_images calls to avoid spamming Neurovault
        image_dfs = []
        for chunk in to_chunks(valid_collections, 500):
            image_dfs.append(api.get_images(collection_pks=chunk))
            time.sleep(10)

        image_df = pd.concat(image_dfs)
        image_df.to_csv(papers_metadata_file, index=False, encoding='utf-8')
    else:
        image_df = pd.read_csv(papers_metadata_file, encoding='utf-8')

    # Reduce images database according to additional criteria
    # Only keep unthresholded, MNI, group level fMRI maps
    red_df = image_df.loc[image_df['modality'] == 'fMRI-BOLD']
    red_df = red_df.loc[red_df['image_type'] == 'statistic_map']
    red_df = red_df.loc[red_df['analysis_level'] == 'group']
    red_df = red_df.loc[red_df['is_thresholded'] is False]
    red_df = red_df.loc[red_df['not_mni'] is False]

    # Look for relevant metadata
    red_df = red_df.dropna(subset=['cognitive_paradigm_cogatlas'])

    ## MFX/FFX GLMs need contrast (beta) + standard error
    mffx_df = red_df.loc[red_df['map_type'] == 'univariate-beta map']

    ## RFX GLMs need contrast (beta)
    rfx_df = red_df.loc[red_df['map_type'] == 'univariate-beta map']

    ## Stouffer's, Stouffer's RFX, and Fisher's IBMAs can use Z maps.
    # T and F maps can be transformed into Z maps, but T maps need sample size.
    # Only keep test statistic maps
    acc_map_types = ['Z map', 'T map', 'F map']
    st_df = red_df.loc[red_df['map_type'].isin(acc_map_types)]
    keep_idx = st_df['map_type'].isin(['Z map', 'F map'])
    keep_idx2 = (st_df['map_type'] == 'T map') & ~pd.isnull(st_df['number_of_subjects'])
    keep_idx = keep_idx | keep_idx2
    st_df = st_df.loc[keep_idx]

    ## Weighted Stouffer's IBMAs need Z + sample size.
    st_df['id_str'] = st_df['image_id'].astype(str).str.zfill(6)

    if not isdir(out_dir):
        mkdir(out_dir)
        api.download_images(out_dir, red_df, target=None, resample=False)
    elif overwrite:
        # clear out out_dir
        raise Exception('Currently not prepared to overwrite database.')
        api.download_images(out_dir, red_df, target=None, resample=False)


class Database(object):
    """
    Storage container for a coordinate- and/or image-based meta-analytic
    dataset/database.

    Parameters
    ----------
    dataset_file : :obj:`str`
        Json file containing dictionary with database information.
    """
    def __init__(self, database_file):
        with open(database_file, 'r') as fo:
            self.data = json.load(fo)

    def get_dataset(self, search='', algorithm=None, target=None):
        """
        Retrieve files and/or metadata from the current Dataset.

        Should this work like a grabbit Layout's get method?

        Parameters
        ----------
        search : :obj:`str`
            Search term for selecting contrasts within database.
        target : :obj:`str`
            Target space for outputted images and coordinates.

        Returns
        -------
        dset : :obj:`nimare.dataset.Dataset`
            A Dataset object containing selection of database.

        """
        if algorithm:
            req_data = algorithm.req_data
            temp = [stud for stud in self.data if stud.has_data(req_data)]


class Dataset(object):
    """
    Storage container for a coordinate- and/or image-based meta-analytic
    dataset/database.

    Parameters
    ----------
    dataset_file : :obj:`str`
        Json file containing dictionary with database information.
    """
    def __init__(self, dataset_file):
        with open(dataset_file, 'r') as fo:
            self.data = json.load(fo)

    def has_data(self, dat_str):
        """
        Check if an experiment has necessary data (e.g., sample size or some
        image type).
        """
        dat_str = dat_str.split(' AND ')
        for ds in dat_str:
            try:
                self.data.get(ds, None)
            except:
                raise Exception('Nope')

    def get(self, search='', algorithm=None):
        """
        Retrieve files and/or metadata from the current Dataset.

        Should this work like a grabbit Layout's get method?

        Parameters
        ----------
        search : :obj:`str`
            Search term for selecting contrasts within database.
        target : :obj:`str`
            Target space for outputted images and coordinates.

        Returns
        -------
        dset : :obj:`nimare.dataset.Dataset`
            A Dataset object containing selection of dataset.
        """
        if algorithm:
            req_data = algorithm.req_data
            temp = [stud for stud in self.data if stud.has_data(req_data)]

    def get_studies(self):
        pass

    def get_metadata(self):
        pass

    def get_images(self):
        pass

    def get_coordinates(self):
        pass

    def save(self, filename):
        """
        Pickle the Dataset instance to the provided file.
        If the filename ends with 'z', gzip will be used to write out a
        compressed file. Otherwise, an uncompressed file will be created.
        """
        if filename.endswith('z'):
            with gzip.GzipFile(filename, 'wb') as file_object:
                pickle.dump(self, file_object)
        else:
            with open(filename, 'wb') as file_object:
                pickle.dump(self, file_object)

    @classmethod
    def load(cls, filename):
        """
        Load a pickled Dataset instance from file.
        If the filename ends with 'z', it will be assumed that the file is
        compressed, and gzip will be used to load it. Otherwise, it will
        be assumed that the file is not compressed.
        """
        if filename.endswith('z'):
            try:
                with gzip.GzipFile(filename, 'rb') as file_object:
                    dataset = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with gzip.GzipFile(filename, 'rb') as file_object:
                    dataset = pickle.load(file_object, encoding='latin')
        else:
            try:
                with open(filename, 'rb') as file_object:
                    dataset = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with open(filename, 'rb') as file_object:
                    dataset = pickle.load(file_object, encoding='latin')

        if not isinstance(dataset, Dataset):
            raise IOError('Pickled object must be `nimare.dataset.dataset.Dataset`, '
                          'not {0}'.format(type(dataset)))

        return dataset
