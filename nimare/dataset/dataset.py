"""
Classes for representing datasets of images and/or coordinates.
"""
from __future__ import print_function
import json
import gzip
import pickle
from os.path import join

import numpy as np
import pandas as pd
import nibabel as nib

from ..utils import tal2mni, mni2tal, mm2vox, get_template


class Database(object):
    """
    Storage container for a coordinate- and/or image-based meta-analytic
    dataset/database.

    Parameters
    ----------
    source : :obj:`str`
        JSON file containing dictionary with database information or the dict() object
    """
    def __init__(self, source):
        if isinstance(source, str):
            with open(source, 'r') as f_obj:
                self.data = json.load(f_obj)
        elif isinstance(source, dict):
            self.data = source
        else:
            raise Exception("`source` needs to be a file path or a dictionary")
        ids = []
        for pid in self.data.keys():
            for cid in self.data[pid]['contrasts'].keys():
                ids.append('{0}-{1}'.format(pid, cid))
        self.ids = ids

    def get_dataset(self, ids=None, search='', algorithm=None, target='mni152_2mm'):
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
        #if algorithm:
        #    req_data = algorithm.req_data
        #    temp = [stud for stud in self.data if stud.has_data(req_data)]
        return Dataset(self, ids=ids, target=target)


class Dataset(object):
    """
    Storage container for a coordinate- and/or image-based meta-analytic
    dataset/database.

    Parameters
    ----------
    database : :obj:`nimare.dataset.Database`
        Database object to be transformed into a dataset.
    ids : :obj:`list`
        List of contrast IDs to be taken from the database and kept in the dataset.
    target : :obj:`str`
        Desired coordinate space for coordinates. Names follow NIDM convention.
    """
    def __init__(self, database, ids=None, target='mni152_2mm',
                 mask_file=None):
        if mask_file is None:
            mask_img = get_template(target, mask='brain')
        else:
            mask_img = nib.load(mask_file)
        self.mask = mask_img

        if ids is None:
            self.data = database.data
            ids = database.ids
        else:
            data = {}
            for id_ in ids:
                pid, expid = id_.split('-')
                if pid not in data.keys():
                    data[pid] = database.data[pid]
                    data[pid]['contrasts'] = {}
                data[pid]['contrasts'][expid] = database.data[pid]['contrasts'][expid]
            self.data = data
        self.ids = ids
        self.coordinates = None
        self.space = target
        self._load_coordinates()

    def _load_coordinates(self):
        """
        """
        # Required columns
        columns = ['id', 'study_id', 'contrast_id', 'x', 'y', 'z', 'n', 'space']
        core_columns = columns[:]  # Used in contrast for loop

        all_dfs = []
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                if 'coords' not in self.data[pid]['contrasts'][expid].keys():
                    continue

                exp_columns = core_columns[:]
                exp = self.data[pid]['contrasts'][expid]

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
                con_df = pd.DataFrame(temp_data.T, columns=exp_columns)
                all_dfs.append(con_df)
        df = pd.concat(all_dfs, axis=0, join='outer')
        df = df[columns].reset_index()
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
        xyz = df[['x', 'y', 'z']].values
        ijk = pd.DataFrame(mm2vox(xyz, self.mask.affine), columns=['i', 'j', 'k'])
        df = pd.concat([df, ijk], axis=1)
        self.coordinates = df

    def has_data(self, dat_str):
        """
        Check if an contrast has necessary data (e.g., sample size or some
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

    def save(self, filename, compress=True):
        """
        Pickle the Dataset instance to the provided file.

        Parameters
        ----------
        filename : :obj:`str`
            File to which dataset will be saved.
        compress : :obj:`bool`, optional
            If True, the file will be compressed with gzip. Otherwise, the
            uncompressed version will be saved. Default = True.
        """
        if compress:
            with gzip.GzipFile(filename, 'wb') as file_object:
                pickle.dump(self, file_object)
        else:
            with open(filename, 'wb') as file_object:
                pickle.dump(self, file_object)

    @classmethod
    def load(cls, filename, compressed=True):
        """
        Load a pickled Dataset instance from file.

        Parameters
        ----------
        filename : :obj:`str`
            Name of file containing dataset.
        compressed : :obj:`bool`, optional
            If True, the file is assumed to be compressed and gzip will be used
            to load it. Otherwise, it will assume that the file is not
            compressed. Default = True.

        Returns
        -------
        dataset : :obj:`nimare.dataset.Dataset`
            Loaded dataset object.
        """
        if compressed:
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
