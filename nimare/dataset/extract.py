"""
Classes and functions for data retrieval.
"""
import re
import time
import json
from os import mkdir
import os.path as op

from abc import ABCMeta, abstractmethod
from six import with_metaclass
import pandas as pd
import numpy as np
from pyneurovault import api

from nimare.dataset import Database
from ..utils import get_resource_path, tal2mni


__all__ = ['NeuroVaultDataSource', 'NeurosynthDataSource', 'BrainSpellDataSource',
           'convert_sleuth', 'convert_sleuth_to_database', 'convert_sleuth_to_dict']


class DataSource(with_metaclass(ABCMeta)):
    ''' Base class for DataSource hierarchy. '''

    @abstractmethod
    def get_data(self, level='contrast', tags=None, dois=None, **kwargs):
        pass


class NeuroVaultDataSource(DataSource):
    ''' Interface with NeuroVault data. '''

    def get_data(self, **kwargs):
        pass

    def _get_collections(self):
        pass

    def _get_images(self):
        pass


class NeurosynthDataSource(DataSource):
    ''' Interface with Neurosynth data. '''
    pass

    def get_data(self, **kwargs):
        pass


class BrainSpellDataSource(DataSource):
    ''' Interface with BrainSpell data. '''
    pass

    def get_data(self, **kwargs):
        pass


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

    if not op.isdir(out_dir):
        mkdir(out_dir)
        api.download_images(out_dir, red_df, target=None, resample=False)
    elif overwrite:
        # clear out out_dir
        raise Exception('Currently not prepared to overwrite database.')
        api.download_images(out_dir, red_df, target=None, resample=False)


def convert_sleuth_to_dict(text_file):
    filename = op.basename(text_file)
    study_name, _ = op.splitext(filename)
    with open(text_file, 'r') as file_object:
        data = file_object.read()
    data = [line.rstrip() for line in re.split('\n\r|\r\n|\n|\r', data)]
    data = [line for line in data if line]
    # First line indicates stereotactic space. The rest are studies, ns, and coords.
    space = data[0].replace(' ', '').replace('//Reference=', '')
    if space not in ['MNI', 'TAL']:
        raise Exception('Space {0} unknown. Options supported: '
                        'MNI or TAL.'.format(space))
    # Split into experiments
    data = data[1:]
    metadata_idx = [i for i, line in enumerate(data) if line.startswith('//')]
    exp_idx = np.split(metadata_idx, np.where(np.diff(metadata_idx) != 1)[0] + 1)
    start_idx = [tup[0] for tup in exp_idx]
    end_idx = start_idx[1:] + [len(data) + 1]
    split_idx = zip(start_idx, end_idx)
    dict_ = {}
    for i_exp, exp_idx in enumerate(split_idx):
        exp_data = data[exp_idx[0]:exp_idx[1]]
        if exp_data:
            study_info = exp_data[0].replace('//', '').strip()
            study_name = study_info.split(':')[0]
            contrast_name = ':'.join(study_info.split(':')[1:]).strip()
            sample_size = int(exp_data[1].replace(' ', '').replace('//Subjects=', ''))
            xyz = exp_data[2:]  # Coords are everything after study info and sample size
            xyz = [row.split('\t') for row in xyz]
            correct_shape = np.all([len(coord) == 3 for coord in xyz])
            if not correct_shape:
                all_shapes = np.unique([len(coord) for coord in xyz]).astype(
                    str)  # pylint: disable=no-member
                raise Exception('Coordinates for study "{0}" are not all correct length. '
                                'Lengths detected: {1}.'.format(study_info,
                                                                ', '.join(all_shapes)))

            try:
                xyz = np.array(xyz, dtype=float)
            except:
                # Prettify xyz
                strs = [[str(e) for e in row] for row in xyz]
                lens = [max(map(len, col)) for col in zip(*strs)]
                fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
                table = '\n'.join([fmt.format(*row) for row in strs])
                raise Exception('Conversion to numpy array failed for study "{0}". '
                                'Coords:\n{1}'.format(study_info, table))

            x, y, z = list(xyz[:, 0]), list(xyz[:, 1]), list(xyz[:, 2])

            if study_name not in dict_.keys():
                dict_[study_name] = {'contrasts': {}}
            dict_[study_name]['contrasts'][contrast_name] = {
                'coords': {},
                'sample_sizes': [],
            }
            dict_[study_name]['contrasts'][contrast_name]['coords']['space'] = space
            dict_[study_name]['contrasts'][contrast_name]['coords']['x'] = x
            dict_[study_name]['contrasts'][contrast_name]['coords']['y'] = y
            dict_[study_name]['contrasts'][contrast_name]['coords']['z'] = z
            dict_[study_name]['contrasts'][contrast_name]['sample_sizes'] = [sample_size]
    return dict_


def convert_sleuth(text_file, out_file):
    """
    Convert Sleuth output text file into json.
    """
    dict_ = convert_sleuth_to_dict(text_file)

    with open(out_file, 'w') as fo:
        json.dump(dict_, fo, indent=4, sort_keys=True)


def convert_sleuth_to_database(text_file):
    dict_ = convert_sleuth_to_dict(text_file)
    return Database(dict_)
