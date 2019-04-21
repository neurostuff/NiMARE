"""
Classes and functions for data retrieval.
"""
import time
from os import mkdir
import os.path as op

from abc import ABCMeta, abstractmethod
from six import with_metaclass
import pandas as pd
from pyneurovault import api


__all__ = ['NeuroVaultDataSource', 'NeurosynthDataSource',
           'BrainSpellDataSource']


class DataSource(with_metaclass(ABCMeta)):
    """
    Base class for DataSource hierarchy.

    Warnings
    --------
    This method is not yet implemented.
    """

    @abstractmethod
    def get_data(self, level='contrast', tags=None, dois=None, **kwargs):
        pass


class NeuroVaultDataSource(DataSource):
    """
    Interface with NeuroVault data.

    Warnings
    --------
    This method is not yet implemented.
    """

    def get_data(self, **kwargs):
        pass

    def _get_collections(self):
        pass

    def _get_images(self):
        pass


class NeurosynthDataSource(DataSource):
    """
    Interface with Neurosynth data.

    Warnings
    --------
    This method is not yet implemented.
    """
    pass

    def get_data(self, **kwargs):
        pass


class BrainSpellDataSource(DataSource):
    """
    Interface with BrainSpell data.

    Warnings
    --------
    This method is not yet implemented.
    """
    pass

    def get_data(self, **kwargs):
        pass


def _to_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
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

    Warnings
    --------
    This method is not yet implemented.
    """
    # Download collections metadata from Neurovault
    collections_file = op.join(out_dir, 'neurovault_collections.csv')
    if overwrite or not op.isfile(collections_file):
        colls_df = api.get_collections()
        colls_df.to_csv(collections_file, index=False, encoding='utf-8')
    else:
        colls_df = pd.read_csv(collections_file, encoding='utf-8')

    # Only include collections from published papers (or preprints)
    papers_file = op.join(out_dir, 'neurovault_papers.csv')
    if overwrite or not op.isfile(papers_file):
        paper_df = colls_df.dropna(subset=['DOI'])
        paper_df.to_csv(papers_file, index=False, encoding='utf-8')
    else:
        paper_df = pd.read_csv(papers_file, encoding='utf-8')

    # Get metadata for individual images from valid collections
    papers_metadata_file = op.join(out_dir, 'neurovault_papers_metadata.csv')
    if overwrite or not op.isfile(papers_metadata_file):
        valid_collections = sorted(paper_df['collection_id'].tolist())

        # Sleep between get_images calls to avoid spamming Neurovault
        image_dfs = []
        for chunk in _to_chunks(valid_collections, 500):
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

    # MFX/FFX GLMs need contrast (beta) + standard error
    mffx_df = red_df.loc[red_df['map_type'] == 'univariate-beta map']
    del mffx_df

    # RFX GLMs need contrast (beta)
    rfx_df = red_df.loc[red_df['map_type'] == 'univariate-beta map']
    del rfx_df

    # Stouffer's, Stouffer's RFX, and Fisher's IBMAs can use Z maps.
    # T and F maps can be transformed into Z maps, but T maps need sample size.
    # Only keep test statistic maps
    acc_map_types = ['Z map', 'T map', 'F map']
    st_df = red_df.loc[red_df['map_type'].isin(acc_map_types)]
    keep_idx = st_df['map_type'].isin(['Z map', 'F map'])
    keep_idx2 = (st_df['map_type'] == 'T map') & ~pd.isnull(st_df['number_of_subjects'])
    keep_idx = keep_idx | keep_idx2
    st_df = st_df.loc[keep_idx]

    # Weighted Stouffer's IBMAs need Z + sample size.
    st_df['id_str'] = st_df['image_id'].astype(str).str.zfill(6)

    if not op.isdir(out_dir):
        mkdir(out_dir)
        api.download_images(out_dir, red_df, target=None, resample=False)
    elif overwrite:
        # clear out out_dir
        raise Exception('Currently not prepared to overwrite database.')
        api.download_images(out_dir, red_df, target=None, resample=False)
