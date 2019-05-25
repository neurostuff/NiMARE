"""
Classes for representing datasets of images and/or coordinates.
"""
from __future__ import print_function
import json
import gzip
import copy
import pickle
import logging

import numpy as np
import pandas as pd
import nibabel as nib

from ..utils import tal2mni, mni2tal, mm2vox, get_template

LGR = logging.getLogger(__name__)


class Dataset(object):
    """
    Storage container for a coordinate- and/or image-based meta-analytic
    dataset/database.

    Parameters
    ----------
    source : :obj:`str`
        JSON file containing dictionary with database information or the dict()
        object
    target : :obj:`str`
        Desired coordinate space for coordinates. Names follow NIDM convention.
    """
    def __init__(self, source, target='mni152_2mm', mask_file=None):
        if isinstance(source, str):
            with open(source, 'r') as f_obj:
                self.data = json.load(f_obj)
        elif isinstance(source, dict):
            self.data = source
        else:
            raise Exception("`source` needs to be a file path or a dictionary")

        # Datasets are organized by study, then experiment
        # To generate unique IDs, we combine study ID with experiment ID
        raw_ids = []
        for pid in self.data.keys():
            for cid in self.data[pid]['contrasts'].keys():
                raw_ids.append('{0}-{1}'.format(pid, cid))
        self.ids = raw_ids

        if mask_file is None:
            mask_img = get_template(target, mask='brain')
        else:
            mask_img = nib.load(mask_file)
        self.mask = mask_img
        self.space = target
        self._load_coordinates()
        self._load_images()
        self._load_annotations()
        self._load_text()

    def slice(self, ids):
        """
        Return a reduced dataset with only requested IDs.
        """
        new_dset = copy.deepcopy(self)
        new_dset.ids = ids
        new_dset.coordinates = new_dset.coordinates.loc[new_dset.coordinates['id'].isin(ids)]
        new_dset.images = new_dset.images.loc[new_dset.images['id'].isin(ids)]
        new_dset.annotations = new_dset.annotations.loc[new_dset.annotations['id'].isin(ids)]
        new_dset.text = new_dset.text.loc[new_dset.text['id'].isin(ids)]
        temp_data = {}
        for id_ in ids:
            pid, expid = id_.split('-')
            if pid not in temp_data.keys():
                temp_data[pid] = self.data[pid].copy()  # make sure to copy
                temp_data[pid]['contrasts'] = {}
            temp_data[pid]['contrasts'][expid] = self.data[pid]['contrasts'][expid]
        new_dset.data = temp_data
        return new_dset

    def _load_annotations(self):
        """
        Load labels in Dataset into DataFrame.
        """
        # Required columns
        columns = ['id', 'study_id', 'contrast_id']

        # build list of ids
        all_ids = []
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)
                all_ids.append([id_, pid, expid])

        id_df = pd.DataFrame(columns=columns, data=all_ids)
        id_df = id_df.set_index('id', drop=False)

        label_dict = {}
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)

                if 'labels' not in self.data[pid]['contrasts'][expid].keys():
                    continue

                label_dict[id_] = exp['labels']

        label_df = pd.DataFrame.from_dict(label_dict, orient='index')
        df = pd.merge(id_df, label_df, left_index=True, right_index=True, how='outer')

        df = df.reset_index(drop=True)
        df = df.replace(to_replace='None', value=np.nan)
        self.annotations = df

    def _load_text(self):
        """
        Load texts in Dataset into a DataFrame.
        """
        # Required columns
        columns = ['id', 'study_id', 'contrast_id']

        # build list of ids
        all_ids = []
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)
                all_ids.append([id_, pid, expid])

        id_df = pd.DataFrame(columns=columns, data=all_ids)
        id_df = id_df.set_index('id', drop=False)

        exp_dict = {}
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)

                if 'texts' not in self.data[pid]['contrasts'][expid].keys():
                    continue

                exp_dict[id_] = exp['texts']

        text_df = pd.DataFrame.from_dict(exp_dict, orient='index')
        df = pd.merge(id_df, text_df, left_index=True, right_index=True, how='outer')

        df = df.reset_index(drop=True)
        df = df.replace(to_replace='None', value=np.nan)
        self.texts = df

    def _load_images(self):
        """
        Load images in Dataset into a DataFrame.
        """
        columns = ['id', 'study_id', 'contrast_id']

        # build list of ids
        all_ids = []
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)
                all_ids.append([id_, pid, expid])

        id_df = pd.DataFrame(columns=columns, data=all_ids)
        id_df = id_df.set_index('id', drop=False)

        image_dict = {}
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)

                if 'images' not in self.data[pid]['contrasts'][expid].keys():
                    continue

                image_dict[id_] = exp['images']

        image_df = pd.DataFrame.from_dict(image_dict, orient='index')
        df = pd.merge(id_df, image_df, left_index=True, right_index=True, how='outer')

        df = df.reset_index(drop=True)
        df = df.replace(to_replace='None', value=np.nan)
        self.images = df

    def _load_coordinates(self):
        """
        Load coordinates in Dataset into DataFrame.
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

                # collect sample size if available
                sample_size = exp.get('sample_sizes', np.nan)
                if not isinstance(sample_size, list):
                    sample_size = [sample_size]
                sample_size = np.array([n for n in sample_size if n])
                if len(sample_size):
                    sample_size = np.mean(sample_size)
                    sample_size = np.array([sample_size] * n_coords)
                else:
                    sample_size = np.array([np.nan] * n_coords)

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

        df = pd.concat(all_dfs, axis=0, join='outer', sort=False)
        df = df[columns].reset_index(drop=True)
        df = df.replace(to_replace='None', value=np.nan)
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)

        # Now to apply transformations!
        if 'mni' in self.space.lower() or 'ale' in self.space.lower():
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

    def get(self):
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

        Warnings
        --------
        This method is not yet implemented.
        """
        pass

    def get_labels(self, ids=None):
        """
        Extract list of labels for which studies in Dataset have annotations.

        Parameters
        ----------
        ids : list, optional
            A list of IDs in the Dataset for which to find labels. Default is
            None, in which case all labels are returned.

        Returns
        -------
        labels : list
            List of labels for which there are annotations in the Dataset.
        """
        id_cols = ['id', 'study_id', 'contrast_id']
        labels = [c for c in self.annotations.columns if c not in id_cols]
        if ids is not None:
            temp_annotations = self.annotations.loc[self.annotations['id'].isin(ids)]
            res = temp_annotations[labels].any(axis=0)
            labels = res.loc[res].index.tolist()

        return labels

    def get_studies_by_label(self, labels=None, label_threshold=0.5):
        """
        Extract list of studies with a given label.

        Parameters
        ----------
        labels : list, optional
            List of labels to use to search Dataset. If a contrast has all of
            the labels above the threshold, it will be returned.
            Default is None.
        label_threshold : float, optional
            Default is 0.5.

        Returns
        -------
        found_ids : list
            A list of IDs from the Dataset found by the search criteria.
        """
        if isinstance(labels, str):
            labels = [labels]
        elif labels is None:
            # For now, labels are all we can search by.
            return self.ids
        elif not isinstance(labels, list):
            raise ValueError('Argument "labels" cannot be {0}'.format(type(labels)))

        id_cols = ['id', 'study_id', 'contrast_id']
        found_labels = [l for l in labels if l in self.annotations.columns]
        temp_annotations = self.annotations[id_cols + found_labels]
        found_rows = (temp_annotations[found_labels] >= label_threshold).all(axis=1)
        if any(found_rows):
            found_ids = temp_annotations.loc[found_rows, 'id'].tolist()
        else:
            found_ids = []
        return found_ids

    def get_studies_by_mask(self, mask):
        """
        Extract list of studies with at least one coordinate in mask.

        Parameters
        ----------
        mask : img_like
            Mask across which to search for coordinates.

        Returns
        -------
        found_ids : list
            A list of IDs from the Dataset with at least one focus in the mask.
        """
        from scipy.spatial.distance import cdist
        if isinstance(mask, str):
            mask = nib.load(mask)

        if not np.array_equal(self.mask.affine, mask.affine):
            from nilearn.image import resample_to_img
            mask = resample_to_img(mask, self.mask)
        mask_ijk = np.vstack(np.where(mask.get_data())).T
        distances = cdist(mask_ijk, self.coordinates[['i', 'j', 'k']].values)
        distances = np.any(distances == 0, axis=0)
        found_ids = self.coordinates.loc[distances, 'id'].unique()
        return found_ids

    def get_studies_by_coordinate(self, xyz, r=20):
        """
        Extract list of studies with at least one focus within radius r of
        requested coordinates.

        Parameters
        ----------
        xyz : (X x 3) array_like
            List of coordinates against which to find studies.
        r : float, optional
            Radius (in mm) within which to find studies. Default is 20mm.

        Returns
        -------
        found_ids : list
            A list of IDs from the Dataset with at least one focus within
            radius r of requested coordinates.
        """
        from scipy.spatial.distance import cdist
        assert xyz.shape[1] == 3 and xyz.ndim == 2
        distances = cdist(xyz, self.coordinates[['x', 'y', 'z']].values)
        distances = np.any(distances <= r, axis=0)
        found_ids = self.coordinates.loc[distances, 'id'].unique()
        return found_ids

    def get_metadata(self):
        """
        Get metadata from Dataset.

        Warnings
        --------
        This method is not yet implemented.
        """
        pass

    def get_images(self, ids, imtype):
        """
        Get images of a certain type for a subset of studies in the dataset.

        Warnings
        --------
        This method is not yet implemented.
        """
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
