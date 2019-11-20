"""
Classes for representing datasets of images and/or coordinates.
"""
from __future__ import print_function
import json
import copy
import logging
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib

from .base import NiMAREBase
from .utils import (tal2mni, mni2tal, mm2vox, get_template, listify,
                    try_prepend, find_stem, get_masker)

LGR = logging.getLogger(__name__)


class Dataset(NiMAREBase):
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
    mask : `str`, `Nifti1Image`, or any nilearn `Masker`
        Mask(er) to use. If None, uses the target space image, with all
        non-zero voxels included in the mask.
    """
    _id_cols = ['id', 'study_id', 'contrast_id']

    def __init__(self, source, target='mni152_2mm', mask=None):
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

        # Set up Masker
        if mask is None:
            mask = get_template(target, mask='brain')
        self.masker = get_masker(mask)
        self.space = target
        self._load_coordinates()
        self._load_images()
        self._load_annotations()
        self._load_texts()
        self._load_metadata()

    def slice(self, ids):
        """
        Return a reduced dataset with only requested IDs.

        Parameters
        ----------
        ids : array_like
            List of study IDs to include in new dataset

        Returns
        -------
        new_dset : :obj:`nimare.dataset.Dataset`
            Redcued Dataset containing only requested studies.
        """
        new_dset = copy.deepcopy(self)
        new_dset.ids = ids
        new_dset.coordinates = new_dset.coordinates.loc[new_dset.coordinates['id'].isin(ids)]
        new_dset.images = new_dset.images.loc[new_dset.images['id'].isin(ids)]
        new_dset.annotations = new_dset.annotations.loc[new_dset.annotations['id'].isin(ids)]
        new_dset.texts = new_dset.texts.loc[new_dset.texts['id'].isin(ids)]
        temp_data = {}
        for id_ in ids:
            pid, expid = id_.split('-')
            if pid not in temp_data.keys():
                temp_data[pid] = self.data[pid].copy()  # make sure to copy
                temp_data[pid]['contrasts'] = {}
            temp_data[pid]['contrasts'][expid] = self.data[pid]['contrasts'][expid]
        new_dset.data = temp_data
        return new_dset

    def update_path(self, new_path):
        """
        Update paths to images. Prepends new path to the relative path for
        files in Dataset.images.

        Parameters
        ----------
        new_path : :obj:`str`
            Path to prepend to relative paths of files in Dataset.images.
        """
        relative_path_cols = [c for c in self.images if c.endswith('__relative')]
        for col in relative_path_cols:
            abs_col = col.replace('__relative', '')
            if abs_col in self.images.columns:
                LGR.info('Overwriting images column {}'.format(abs_col))
            self.images[abs_col] = self.images[col].apply(try_prepend, prefix=new_path)

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

        exp_dict = {}
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)

                if 'labels' not in self.data[pid]['contrasts'][expid].keys():
                    continue

                exp_dict[id_] = exp['labels']

        temp_df = pd.DataFrame.from_dict(exp_dict, orient='index')
        df = pd.merge(id_df, temp_df, left_index=True, right_index=True, how='outer')

        df = df.reset_index(drop=True)
        df = df.replace(to_replace='None', value=np.nan)
        self.annotations = df

    def _load_metadata(self):
        """
        Load metadata in Dataset into DataFrame.
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

                if 'metadata' not in self.data[pid]['contrasts'][expid].keys():
                    continue

                exp_dict[id_] = exp['metadata']

        temp_df = pd.DataFrame.from_dict(exp_dict, orient='index')
        df = pd.merge(id_df, temp_df, left_index=True, right_index=True, how='outer')

        df = df.reset_index(drop=True)
        df = df.replace(to_replace='None', value=np.nan)
        self.metadata = df

    def _load_texts(self):
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

        temp_df = pd.DataFrame.from_dict(exp_dict, orient='index')
        df = pd.merge(id_df, temp_df, left_index=True, right_index=True, how='outer')

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

        exp_dict = {}
        for pid in self.data.keys():
            for expid in self.data[pid]['contrasts'].keys():
                exp = self.data[pid]['contrasts'][expid]
                id_ = '{0}-{1}'.format(pid, expid)

                if 'images' not in self.data[pid]['contrasts'][expid].keys():
                    continue

                exp_dict[id_] = exp['images']

        temp_df = pd.DataFrame.from_dict(exp_dict, orient='index')
        valid_suffices = ['.brik', '.head', '.nii', '.img', '.hed']
        file_cols = []
        for col in temp_df.columns:
            vals = [v for v in temp_df[col].values if isinstance(v, str)]
            fc = any([any([vs in v for vs in valid_suffices]) for v in vals])
            if fc:
                file_cols.append(col)

        # Clean up temp_df
        # Find out which columns have full paths and which have relative paths
        abs_cols = []
        for col in file_cols:
            files = temp_df[col].tolist()
            abspaths = [f == op.abspath(f) for f in files if isinstance(f, str)]
            if all(abspaths):
                abs_cols.append(col)
            elif not any(abspaths):
                temp_df = temp_df.rename(columns={col: col + '__relative'})
            else:
                raise ValueError('Mix of absolute and relative paths detected '
                                 'for "{0}" images'.format(col))

        # Set relative paths from absolute ones
        if len(abs_cols):
            all_files = list(np.ravel(temp_df[abs_cols].values))
            all_files = [f for f in all_files if isinstance(f, str)]
            shared_path = find_stem(all_files)
            LGR.info('Shared path detected: "{0}"'.format(shared_path))
            for abs_col in abs_cols:
                temp_df[abs_col + '__relative'] = temp_df[abs_col].apply(
                    lambda x: x.split(shared_path)[1] if isinstance(x, str) else x)

        df = pd.merge(id_df, temp_df, left_index=True, right_index=True, how='outer')
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
                sample_size = exp['metadata'].get('sample_sizes', np.nan)
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
            transform = {'MNI': None,
                         'TAL': tal2mni,
                         'Talairach': tal2mni,
                         }
        elif 'tal' in self.space.lower():
            transform = {'MNI': mni2tal,
                         'TAL': None,
                         'Talairach': None,
                         }
        else:
            raise ValueError('Unrecognized space: {0}'.format(self.space))

        found_spaces = df['space'].unique()
        for found_space in found_spaces:
            if found_space not in transform.keys():
                LGR.warning('Not applying transforms to coordinates in '
                            'unrecognized space "{0}"'.format(found_space))
            alg = transform.get(found_space, None)
            idx = df['space'] == found_space
            if alg:
                df.loc[idx, ['x', 'y', 'z']] = alg(df.loc[idx, ['x', 'y', 'z']].values)
            df.loc[idx, 'space'] = self.space

        xyz = df[['x', 'y', 'z']].values
        ijk = pd.DataFrame(mm2vox(xyz, self.masker.mask_img.affine),
                           columns=['i', 'j', 'k'])
        df = pd.concat([df, ijk], axis=1)
        self.coordinates = df

    def get(self, dict_):
        """
        Retrieve files and/or metadata from the current Dataset.

        Parameters
        ----------
        dict_ : :obj:`dict`
            Dictionary specifying images or metadata to collect

        Returns
        -------
        results : :obj:`dict`
            A dictionary of lists of requested data.
        """
        results = {}
        results['id'] = self.ids
        keep_idx = np.arange(len(self.ids), dtype=int)
        for k in dict_:
            vals = dict_[k]
            if vals[0] == 'image':
                temp = self.get_images(imtype=vals[1])
            elif vals[0] == 'metadata':
                temp = self.get_metadata(field=vals[1])
            else:
                raise ValueError('Input "{}" not understood.'.format(vals[0]))
            results[k] = temp
            temp_keep_idx = np.where([t is not None for t in temp])[0]
            keep_idx = np.intersect1d(keep_idx, temp_keep_idx)

        # reduce
        if len(keep_idx) != len(self.ids):
            LGR.info('Retaining {0}/{1} studies'.format(len(keep_idx),
                                                        len(self.ids)))

        for k in results:
            results[k] = [results[k][i] for i in keep_idx]
        return results

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
        if not isinstance(ids, list) and ids is not None:
            ids = listify(ids)

        result = [c for c in self.annotations.columns if c not in self._id_cols]
        if ids is not None:
            temp_annotations = self.annotations.loc[self.annotations['id'].isin(ids)]
            res = temp_annotations[result].any(axis=0)
            result = res.loc[res].index.tolist()

        return result

    def get_texts(self, ids=None, text_type='abstract'):
        """
        Extract list of texts of a given type for selected IDs.

        Parameters
        ----------
        ids : list, optional
            A list of IDs in the Dataset for which to find texts. Default is
            None, in which case all texts of requested type are returned.
        text_type : str, optional
            Type of text to extract. Corresponds to column name in
            Dataset.texts DataFrame. Default is 'abstract'.

        Returns
        -------
        texts : list
            List of texts of requested type for selected IDs.
        """
        return_first = False
        if not isinstance(ids, list) and ids is not None:
            return_first = True
            ids = listify(ids)

        text_types = [c for c in self.texts.columns if c not in self._id_cols]
        if text_type not in text_types:
            raise ValueError('Text type "{0}" not found.\nAvailable types: '
                             '{1}'.format(text_type, ', '.join(text_types)))

        if ids is not None:
            result = self.texts[text_type].loc[self.texts['id'].isin(ids)]
        else:
            result = self.texts[text_type]

        if return_first:
            return result[0]
        else:
            return result

        return result

    def get_metadata(self, ids=None, field='sample_sizes'):
        """
        Get metadata from Dataset.

        Parameters
        ----------
        ids : list, optional
            A list of IDs in the Dataset for which to find texts. Default is
            None, in which case all texts of requested type are returned.
        field : str, optional
            Metadata field to extract. Corresponds to column name in
            Dataset.metadata DataFrame. Default is 'sample_sizes'.

        Returns
        -------
        metadata : list
            List of values of requested type for selected IDs.
        """
        return_first = False
        if not isinstance(ids, list) and ids is not None:
            return_first = True
            ids = listify(ids)

        md_fields = [c for c in self.metadata.columns if c not in self._id_cols]
        if field not in md_fields:
            raise ValueError('Metadata field "{0}" not found.\nAvailable fields: '
                             '{1}'.format(field, ', '.join(md_fields)))

        if ids is not None:
            result = self.metadata[field].loc[self.metadata['id'].isin(ids)].tolist()
        else:
            result = self.metadata[field].tolist()

        if return_first:
            return result[0]
        else:
            return result

    def get_images(self, ids=None, imtype='z'):
        """
        Get images of a certain type for a subset of studies in the dataset.

        Parameters
        ----------
        ids : list, optional
            A list of IDs in the Dataset for which to find texts. Default is
            None, in which case all texts of requested type are returned.
        imtype : str, optional
            Type of image to extract. Corresponds to column name in
            Dataset.images DataFrame. Default is 'z'.

        Returns
        -------
        images : list
            List of images of requested type for selected IDs.
        """
        return_first = False
        if not isinstance(ids, list) and ids is not None:
            return_first = True
            ids = listify(ids)

        imtypes = [c for c in self.images.columns if c not in self._id_cols]
        if imtype not in imtypes:
            raise ValueError('Image type "{0}" not found.\nAvailable types: '
                             '{1}'.format(imtype, ', '.join(imtypes)))

        if ids is not None:
            result = self.images[imtype].loc[self.images['id'].isin(ids)].tolist()
        else:
            result = self.images[imtype].tolist()

        if return_first:
            return result[0]
        else:
            return result

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

        found_labels = [l for l in labels if l in self.annotations.columns]
        temp_annotations = self.annotations[self._id_cols + found_labels]
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

        curr_mask = self.mask.mask_img
        if not np.array_equal(curr_mask.affine, mask.affine):
            from nilearn.image import resample_to_img
            mask = resample_to_img(mask, curr_mask)
        mask_ijk = np.vstack(np.where(mask.get_data())).T
        distances = cdist(mask_ijk, self.coordinates[['i', 'j', 'k']].values)
        distances = np.any(distances == 0, axis=0)
        found_ids = list(self.coordinates.loc[distances, 'id'].unique())
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
        found_ids = list(self.coordinates.loc[distances, 'id'].unique())
        return found_ids
