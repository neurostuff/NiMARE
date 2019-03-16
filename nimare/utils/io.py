"""
Input/Output operations.
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

from ..dataset import Database
from .utils import get_resource_path, tal2mni


__all__ = ['convert_sleuth', 'convert_sleuth_to_database',
           'convert_sleuth_to_dict']


def convert_sleuth_to_dict(text_file):
    """
    Convert Sleuth text file to a dictionary.

    Parameters
    ----------
    text_file : :obj:`str`
        Path to Sleuth-format text file.

    Returns
    -------
    dict_ : :obj:`dict`
        NiMARE-organized dictionary containing experiment information from text
        file.
    """
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

    Parameters
    ----------
    text_file : :obj:`str`
        Path to Sleuth-format text file.
    out_file : :obj:`str`
        Path to output json file.
    """
    if isinstance(text_file, str):
        text_files = [text_file]
    elif isinstance(text_file, list):
        text_files = text_file
    else:
        raise ValueError('Unsupported type for parameter "text_file": '
                         '{0}'.format(type(text_file)))
    dict_ = {}
    for text_file in text_files:
        temp_dict = convert_sleuth_to_dict(text_file)
        dict_ = {**dict_, **temp_dict}

    with open(out_file, 'w') as fo:
        json.dump(dict_, fo, indent=4, sort_keys=True)


def convert_sleuth_to_database(text_file):
    """
    Convert Sleuth output text file into dictionary and create NiMARE Database
    with dictionary.

    Parameters
    ----------
    text_file : :obj:`str`
        Path to Sleuth-format text file.

    Returns
    -------
    :obj:`nimare.dataset.Database`
        Database object containing experiment information from text_file.
    """
    if isinstance(text_file, str):
        text_files = [text_file]
    elif isinstance(text_file, list):
        text_files = text_file
    else:
        raise ValueError('Unsupported type for parameter "text_file": '
                         '{0}'.format(type(text_file)))
    dict_ = {}
    for text_file in text_files:
        temp_dict = convert_sleuth_to_dict(text_file)
        dict_ = {**dict_, **temp_dict}
    return Database(dict_)
