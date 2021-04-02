"""Classes for representing datasets of images and/or coordinates."""
import copy
import json
import logging
import os.path as op

import numpy as np
import pandas as pd
from nilearn._utils import load_niimg

from .base import NiMAREBase
from .utils import (
    dict_to_coordinates,
    dict_to_df,
    get_masker,
    get_template,
    listify,
    mm2vox,
    try_prepend,
    validate_df,
    validate_images_df,
)

LGR = logging.getLogger(__name__)


class Dataset(NiMAREBase):
    """Storage container for a coordinate- and/or image-based meta-analytic dataset/database.

    Parameters
    ----------
    source : :obj:`str` or :obj:`dict`
        JSON file containing dictionary with database information or the dict()
        object
    target : :obj:`str`, optional
        Desired coordinate space for coordinates. Names follow NIDM convention.
        Default is 'mni152_2mm' (MNI space with 2x2x2 voxels).
    mask : :obj:`str`, :class:`nibabel.nifti1.Nifti1Image`, \
    :class:`nilearn.input_data.NiftiMasker` or similar, or None, optional
        Mask(er) to use. If None, uses the target space image, with all
        non-zero voxels included in the mask.

    Attributes
    ----------
    ids : 1D :class:`numpy.ndarray`
        Identifiers
    masker : :class:`nilearn.input_data.NiftiMasker` or similar
        Masker object defining the space and location of the area of interest
        (e.g., 'brain').
    space : :obj:`str`
        Standard space. Same as ``target`` parameter.
    annotations : :class:`pandas.DataFrame`
        Labels describing studies
    coordinates : :class:`pandas.DataFrame`
        Peak coordinates from studies
    images : :class:`pandas.DataFrame`
        Images from studies
    metadata : :class:`pandas.DataFrame`
        Metadata describing studies
    texts : :class:`pandas.DataFrame`
        Texts associated with studies

    Notes
    -----
    Images loaded into a Dataset are assumed to be in the same space.
    If images have different resolutions or affines from the Dataset's masker,
    then they will be resampled automatically, at the point where they're used,
    by :obj:`Dataset.masker`.
    """

    _id_cols = ["id", "study_id", "contrast_id"]

    def __init__(self, source, target="mni152_2mm", mask=None):
        if isinstance(source, str):
            with open(source, "r") as f_obj:
                data = json.load(f_obj)
        elif isinstance(source, dict):
            data = source
        else:
            raise Exception("`source` needs to be a file path or a dictionary")

        # Datasets are organized by study, then experiment
        # To generate unique IDs, we combine study ID with experiment ID
        # build list of ids
        id_columns = ["id", "study_id", "contrast_id"]
        all_ids = []
        for pid in data.keys():
            for expid in data[pid]["contrasts"].keys():
                id_ = "{0}-{1}".format(pid, expid)
                all_ids.append([id_, pid, expid])
        id_df = pd.DataFrame(columns=id_columns, data=all_ids)
        id_df = id_df.set_index("id", drop=False)
        self._ids = id_df.index.values

        # Set up Masker
        if mask is None:
            mask = get_template(target, mask="brain")
        self.masker = mask
        self.space = target

        self.annotations = dict_to_df(id_df, data, key="labels")
        self.coordinates = dict_to_coordinates(data, masker=self.masker, space=self.space)
        self.images = dict_to_df(id_df, data, key="images")
        self.metadata = dict_to_df(id_df, data, key="metadata")
        self.texts = dict_to_df(id_df, data, key="text")
        self.basepath = None

    @property
    def ids(self):
        """numpy.ndarray: 1D array of identifiers in Dataset.

        The associated setter for this property is private, as ``Dataset.ids`` is immutable.
        """
        return self.__ids

    @ids.setter
    def _ids(self, ids):
        ids = np.sort(np.asarray(ids))
        assert isinstance(ids, np.ndarray) and ids.ndim == 1
        self.__ids = ids

    @property
    def masker(self):
        """:class:`nilearn.input_data.NiftiMasker` or similar: Masker object.

        Defines the space and location of the area of interest (e.g., 'brain').
        """
        return self.__masker

    @masker.setter
    def masker(self, mask):
        mask = get_masker(mask)
        if hasattr(self, "masker") and not np.array_equal(
            self.masker.mask_img.affine, mask.mask_img.affine
        ):
            LGR.info(
                "New masker does not match old masker. "
                "Space is assumed to be the same, but coordinates will "
                "be transformed to new matrix."
            )
            coords = self.coordinates
            coords[["i", "j", "k"]] = mm2vox(coords[["x", "y", "z"]], mask.mask_img.affine)
            self.coordinates = coords
        self.__masker = mask

    @property
    def annotations(self):
        """:class:`pandas.DataFrame`: Labels describing studies in the dataset.

        Each study/experiment has its own row.
        Columns correspond to individual labels (e.g., 'emotion'), and may
        be prefixed with a feature group including two underscores
        (e.g., 'Neurosynth_TFIDF__emotion').
        """
        return self.__annotations

    @annotations.setter
    def annotations(self, df):
        validate_df(df)
        self.__annotations = df.sort_values(by="id")

    @property
    def coordinates(self):
        """:class:`pandas.DataFrame`: Coordinates in the dataset.

        Each study has one row for each peak.
        Columns include ['x', 'y', 'z'] (peak locations in mm),
        ['i', 'j', 'k'] (peak locations in voxel index based on Dataset's space),
        and 'space' (Dataset's space).
        """
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, df):
        validate_df(df)
        self.__coordinates = df.sort_values(by="id")

    @property
    def images(self):
        """:class:`pandas.DataFrame`: Images in the dataset.

        Each image type has its own column (e.g., 'z') with absolute paths to
        files and each study has its own row.
        Additionally, relative paths to image files are stored in columns with
        the suffix '__relative' (e.g., 'z__relative').

        Warnings
        --------
        Images are assumed to be in the same space, although they may have
        different resolutions and affines. Images will be resampled as needed
        at the point where they are used, via :obj:`Dataset.masker`.
        """
        return self.__images

    @images.setter
    def images(self, df):
        validate_df(df)
        self.__images = validate_images_df(df).sort_values(by="id")

    @property
    def metadata(self):
        """:class:`pandas.DataFrame`: Metadata describing studies in the dataset.

        Each metadata field has its own column (e.g., 'sample_sizes') and each study
        has its own row.
        """
        return self.__metadata

    @metadata.setter
    def metadata(self, df):
        validate_df(df)
        self.__metadata = df.sort_values(by="id")

    @property
    def texts(self):
        """:class:`pandas.DataFrame`: Texts in the dataset.

        Each text type has its own column (e.g., 'abstract') and each study
        has its own row.
        """
        return self.__texts

    @texts.setter
    def texts(self, df):
        validate_df(df)
        self.__texts = df.sort_values(by="id")

    def slice(self, ids):
        """Create a new dataset with only requested IDs.

        Parameters
        ----------
        ids : array_like
            List of study IDs to include in new dataset

        Returns
        -------
        new_dset : :obj:`nimare.dataset.Dataset`
            Reduced Dataset containing only requested studies.
        """
        new_dset = copy.deepcopy(self)
        new_dset._ids = ids
        new_dset.annotations = new_dset.annotations.loc[new_dset.annotations["id"].isin(ids)]
        new_dset.coordinates = new_dset.coordinates.loc[new_dset.coordinates["id"].isin(ids)]
        new_dset.images = new_dset.images.loc[new_dset.images["id"].isin(ids)]
        new_dset.metadata = new_dset.metadata.loc[new_dset.metadata["id"].isin(ids)]
        new_dset.texts = new_dset.texts.loc[new_dset.texts["id"].isin(ids)]
        return new_dset

    def update_path(self, new_path):
        """Update paths to images.

        Prepends new path to the relative path for files in Dataset.images.

        Parameters
        ----------
        new_path : :obj:`str`
            Path to prepend to relative paths of files in Dataset.images.
        """
        self.basepath = op.abspath(new_path)
        df = self.images
        relative_path_cols = [c for c in df if c.endswith("__relative")]
        for col in relative_path_cols:
            abs_col = col.replace("__relative", "")
            if abs_col in df.columns:
                LGR.info("Overwriting images column {}".format(abs_col))
            df[abs_col] = df[col].apply(try_prepend, prefix=self.basepath)
        self.images = df

    def copy(self):
        """Create a copy of the Dataset."""
        return copy.deepcopy(self)

    def get(self, dict_):
        """Retrieve files and/or metadata from the current Dataset.

        Parameters
        ----------
        dict_ : :obj:`dict`
            Dictionary specifying images or metadata to collect.
            Keys should be variables to be used as keys for results dictionary.
            Values should be tuples with two values:
            type (e.g., 'image' or 'metadata') and specific field corresponding
            to column of type-specific DataFrame (e.g., 'z' or 'sample_sizes').

        Returns
        -------
        results : :obj:`dict`
            A dictionary of lists of requested data.

        Examples
        --------
        >>> dset.get({'z_maps': ('image', 'z'), 'sample_sizes': ('metadata', 'sample_sizes')})
        """
        results = {}
        results["id"] = self.ids
        keep_idx = np.arange(len(self.ids), dtype=int)
        for k in dict_:
            vals = dict_[k]
            if vals[0] == "image":
                temp = self.get_images(imtype=vals[1])
            elif vals[0] == "metadata":
                temp = self.get_metadata(field=vals[1])
            elif vals[0] == "coordinates":
                temp = [self.coordinates.loc[self.coordinates["id"] == id_] for id_ in self.ids]
            else:
                raise ValueError('Input "{}" not understood.'.format(vals[0]))
            results[k] = temp
            temp_keep_idx = np.where([t is not None for t in temp])[0]
            keep_idx = np.intersect1d(keep_idx, temp_keep_idx)

        # reduce
        if len(keep_idx) != len(self.ids):
            LGR.info("Retaining {0}/{1} studies".format(len(keep_idx), len(self.ids)))

        for k in results:
            results[k] = [results[k][i] for i in keep_idx]
            if dict_.get(k, [None])[0] == "coordinates":
                results[k] = pd.concat(results[k])
        return results

    def get_labels(self, ids=None):
        """Extract list of labels for which studies in Dataset have annotations.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find labels. Default is
            None, in which case all labels are returned.

        Returns
        -------
        labels : :obj:`list`
            List of labels for which there are annotations in the Dataset.
        """
        if not isinstance(ids, list) and ids is not None:
            ids = listify(ids)

        result = [c for c in self.annotations.columns if c not in self._id_cols]
        if ids is not None:
            temp_annotations = self.annotations.loc[self.annotations["id"].isin(ids)]
            res = temp_annotations[result].any(axis=0)
            result = res.loc[res].index.tolist()

        return result

    def get_texts(self, ids=None, text_type=None):
        """Extract list of texts of a given type for selected IDs.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find texts. Default is
            None, in which case all texts of requested type are returned.
        text_type : :obj:`str`, optional
            Type of text to extract. Corresponds to column name in
            Dataset.texts DataFrame. Default is None.

        Returns
        -------
        texts : :obj:`list`
            List of texts of requested type for selected IDs.
        """
        # Rename variables
        value = text_type
        df = self.texts

        return_first = False
        if isinstance(ids, str) and value is not None:
            return_first = True
        ids = listify(ids)

        available_types = [c for c in df.columns if c not in self._id_cols]
        if (value is not None) and (value not in available_types):
            raise ValueError(
                'Text type "{0}" not found.\n'
                "Available types: "
                "{1}".format(value, ", ".join(available_types))
            )

        if value is not None:
            if ids is not None:
                result = df[value].loc[df["id"].isin(ids)].tolist()
            else:
                result = df[value].tolist()
        else:
            if ids is not None:
                result = {v: df[v].loc[df["id"].isin(ids)].tolist() for v in available_types}
                result = {k: v for k, v in result.items() if any(v)}
            else:
                result = {v: df[v].tolist() for v in available_types}
            result = list(result.keys())

        if return_first:
            return result[0]
        else:
            return result

        return result

    def get_metadata(self, ids=None, field=None):
        """Get metadata from Dataset.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find texts. Default is
            None, in which case all texts of requested type are returned.
        field : :obj:`str`, optional
            Metadata field to extract. Corresponds to column name in
            Dataset.metadata DataFrame. Default is None.

        Returns
        -------
        metadata : :obj:`list`
            List of values of requested type for selected IDs.
        """
        # Rename variables
        value = field
        df = self.metadata

        return_first = False
        if isinstance(ids, str) and value is not None:
            return_first = True
        ids = listify(ids)

        available_types = [c for c in df.columns if c not in self._id_cols]
        if (value is not None) and (value not in available_types):
            raise ValueError(
                'Metadata field "{0}" not found.\n'
                "Available fields: "
                "{1}".format(field, ", ".join(available_types))
            )

        if value is not None:
            if ids is not None:
                result = df[value].loc[df["id"].isin(ids)].tolist()
            else:
                result = df[value].tolist()
        else:
            if ids is not None:
                result = {v: df[v].loc[df["id"].isin(ids)].tolist() for v in available_types}
                result = {k: v for k, v in result.items() if any(v)}
            else:
                result = {v: df[v].tolist() for v in available_types}
            result = list(result.keys())

        if return_first:
            return result[0]
        else:
            return result

    def get_images(self, ids=None, imtype=None):
        """Get images of a certain type for a subset of studies in the dataset.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find texts. Default is
            None, in which case all texts of requested type are returned.
        imtype : :obj:`str`, optional
            Type of image to extract. Corresponds to column name in
            Dataset.images DataFrame. Default is None.

        Returns
        -------
        images : :obj:`list`
            List of images of requested type for selected IDs.
        """
        # Rename variables
        value = imtype
        df = self.images

        return_first = False
        if isinstance(ids, str) and value is not None:
            return_first = True
        ids = listify(ids)

        metadata_fields = ["space"]
        available_types = [c for c in df.columns if c not in self._id_cols]
        available_types = [c for c in available_types if not c.endswith("__relative")]
        available_types = [c for c in available_types if c not in metadata_fields]
        if (value is not None) and (value not in available_types):
            raise ValueError(
                'Image type "{0}" not found.\n'
                "Available types: "
                "{1}".format(value, ", ".join(available_types))
            )

        if value is not None:
            if ids is not None:
                result = self.images[value].loc[self.images["id"].isin(ids)].tolist()
            else:
                result = self.images[value].tolist()
        else:
            if ids is not None:
                result = {v: df[v].loc[df["id"].isin(ids)].tolist() for v in available_types}
                result = {k: v for k, v in result.items() if any(v)}
            else:
                result = {v: df[v].tolist() for v in available_types}
            result = list(result.keys())

        if return_first:
            return result[0]
        else:
            return result

    def get_studies_by_label(self, labels=None, label_threshold=0.5):
        """Extract list of studies with a given label.

        Parameters
        ----------
        labels : :obj:`list`, optional
            List of labels to use to search Dataset. If a contrast has all of
            the labels above the threshold, it will be returned.
            Default is None.
        label_threshold : :obj:`float`, optional
            Default is 0.5.

        Returns
        -------
        found_ids : :obj:`list`
            A list of IDs from the Dataset found by the search criteria.
        """
        if isinstance(labels, str):
            labels = [labels]
        elif labels is None:
            # For now, labels are all we can search by.
            return self.ids
        elif not isinstance(labels, list):
            raise ValueError('Argument "labels" cannot be {0}'.format(type(labels)))

        found_labels = [label for label in labels if label in self.annotations.columns]
        temp_annotations = self.annotations[self._id_cols + found_labels]
        found_rows = (temp_annotations[found_labels] >= label_threshold).all(axis=1)
        if any(found_rows):
            found_ids = temp_annotations.loc[found_rows, "id"].tolist()
        else:
            found_ids = []
        return found_ids

    def get_studies_by_mask(self, mask):
        """Extract list of studies with at least one coordinate in mask.

        Parameters
        ----------
        mask : img_like
            Mask across which to search for coordinates.

        Returns
        -------
        found_ids : :obj:`list`
            A list of IDs from the Dataset with at least one focus in the mask.
        """
        from scipy.spatial.distance import cdist

        mask = load_niimg(mask)

        dset_mask = self.masker.mask_img
        if not np.array_equal(dset_mask.affine, mask.affine):
            from nilearn.image import resample_to_img

            mask = resample_to_img(mask, dset_mask, interpolation="nearest")
        mask_ijk = np.vstack(np.where(mask.get_fdata())).T
        distances = cdist(mask_ijk, self.coordinates[["i", "j", "k"]].values)
        distances = np.any(distances == 0, axis=0)
        found_ids = list(self.coordinates.loc[distances, "id"].unique())
        return found_ids

    def get_studies_by_coordinate(self, xyz, r=20):
        """Extract list of studies with at least one focus within radius of requested coordinates.

        Parameters
        ----------
        xyz : (X x 3) array_like
            List of coordinates against which to find studies.
        r : :obj:`float`, optional
            Radius (in mm) within which to find studies. Default is 20mm.

        Returns
        -------
        found_ids : :obj:`list`
            A list of IDs from the Dataset with at least one focus within
            radius r of requested coordinates.
        """
        from scipy.spatial.distance import cdist

        xyz = np.array(xyz)
        assert xyz.shape[1] == 3 and xyz.ndim == 2
        distances = cdist(xyz, self.coordinates[["x", "y", "z"]].values)
        distances = np.any(distances <= r, axis=0)
        found_ids = list(self.coordinates.loc[distances, "id"].unique())
        return found_ids
