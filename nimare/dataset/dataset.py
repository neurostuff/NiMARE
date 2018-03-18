"""
Classes for representing datasets of images and/or coordinates.
"""
from __future__ import print_function
import json
import gzip
import pickle

import numpy as np
import pandas as pd

from ..utils import tal2mni, mni2tal


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
    target : :obj:`str`
        Desired coordinate space for coordinates. Names follow NIDM convention.
    """
    def __init__(self, dataset_file, target='Icbm Mni152 Linear'):
        with open(dataset_file, 'r') as f_obj:
            self.data = json.load(f_obj)
        self.coordinates = None
        self.space = target
        self._load_coordinates()

    def _load_coordinates(self):
        """
        """
        # Required columns
        columns = ['id', 'study_id', 'contrast_id', 'x', 'y', 'z', 'n', 'space']
        core_columns = columns[:]  # Used in experiment for loop

        all_dfs = []
        for pid in self.data.keys():
            for expid in self.data[pid]['experiments'].keys():
                if 'coords' not in self.data[pid]['experiments'][expid].keys():
                    continue

                exp_columns = core_columns[:]
                exp = self.data[pid]['experiments'][expid]

                # Required info (ids, x, y, z, space)
                n_coords = len(exp['coords']['x'])
                rep_id = np.array([['{0}-{1}'.format(pid, expid), pid, expid]] * n_coords).T
                sample_size = exp.get('sample_sizes', np.nan)
                if not isinstance(sample_size, list):
                    sample_size = [sample_size]
                sample_size = np.array([n for n in sample_size if n != None])
                sample_size = np.mean(sample_size)
                sample_size = np.array([sample_size] * n_coords)
                space = exp['coords'].get('space')
                space = np.array([space] * n_coords)
                temp_data = np.vstack((rep_id,
                                       np.array(exp['coords']['x']),
                                       np.array(exp['coords']['y']),
                                       np.array(exp['coords']['z']),
                                       sample_size,
                                       space))

                # Optional information
                for k in list(set(exp['coords'].keys()) - set(columns)):
                    k_data = exp['coords'][k]
                    if not isinstance(k_data, list):
                        k_data = np.array([k_data] * n_coords)
                    exp_columns.append(k)

                    if k not in columns:
                        columns.append(k)
                    temp_data = np.vstack((temp_data, k_data))

                # Place data in list of dataframes to merge
                all_dfs.append(pd.DataFrame(temp_data.T, columns=exp_columns))
        df = pd.concat(all_dfs, axis=0, join='outer')
        df = df[columns]
        df = df.replace(to_replace='None', value=np.nan)
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)

        # Now to apply transformations!
        if 'mni' in self.space.lower():
            transform = {'TAL': tal2mni,
                         }
        elif 'tal' in self.space.lower():
            transform = {'MNI': mni2tal,
                         }

        for trans in transform.keys():
            alg = transform[trans]
            idx = df['space'] == trans
            df.loc[idx, ['x', 'y', 'z']] = alg(df.loc[idx, ['x', 'y', 'z']].values)
            df.loc[idx, 'space'] = self.space
        self.coordinates = df

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
