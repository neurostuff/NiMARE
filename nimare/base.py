"""Base classes for NiMARE."""
import gzip
import inspect
import logging
import multiprocessing as mp
import pickle
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from hashlib import md5

import nibabel as nb
import numpy as np
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn.image import concat_imgs, resample_to_img

from .results import MetaResult
from .utils import get_masker, mm2vox

LGR = logging.getLogger(__name__)


class NiMAREBase(metaclass=ABCMeta):
    """Base class for NiMARE.

    This class contains a few features that are useful throughout the library:

    - Custom __repr__ method for printing the object.
    - A private _check_ncores method to check if the common n_cores argument is valid.
    - get_params from scikit-learn, with which parameters provided at __init__ can be viewed.
    - set_params from scikit-learn, with which parameters provided at __init__ can be overwritten.
      I'm not sure that this is actually used or useable in NiMARE.
    - save to save the object to a Pickle file.
    - load to load an instance of the object from a Pickle file.

    TODO: Actually write/refactor class methods. They mostly come directly from sklearn
    https://github.com/scikit-learn/scikit-learn/blob/
    2a1e9686eeb203f5fddf44fd06414db8ab6a554a/sklearn/base.py#L141
    """

    def __init__(self):
        pass

    def __repr__(self):
        """Show basic NiMARE class representation.

        Specifically, this shows the name of the class, along with any parameters
        that are **not** set to the default.
        """
        # Get default parameter values for the object
        signature = inspect.signature(self.__init__)
        defaults = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        # Eliminate any sub-parameters (e.g., parameters for a MetaEstimator's KernelTransformer),
        # as well as default values
        params = self.get_params()
        params = {k: v for k, v in params.items() if "__" not in k}
        params = {k: v for k, v in params.items() if defaults.get(k) != v}

        # Convert to strings
        param_strs = []
        for k, v in params.items():
            if isinstance(v, str):
                # Wrap string values in single quotes
                param_str = f"{k}='{v}'"
            else:
                # Keep everything else as-is based on its own repr
                param_str = f"{k}={v}"
            param_strs.append(param_str)

        rep = f"{self.__class__.__name__}({', '.join(param_strs)})"
        return rep

    def _check_ncores(self, n_cores):
        """Check number of cores used for method."""
        if n_cores <= 0:
            n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                f"Desired number of cores ({n_cores}) greater than number "
                f"available ({mp.cpu_count()}). Setting to {mp.cpu_count()}."
            )
            n_cores = mp.cpu_count()
        return n_cores

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : :obj:`bool`, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : :obj:`dict`
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def save(self, filename, compress=True):
        """Pickle the class instance to the provided file.

        Parameters
        ----------
        filename : :obj:`str`
            File to which object will be saved.
        compress : :obj:`bool`, optional
            If True, the file will be compressed with gzip. Otherwise, the
            uncompressed version will be saved. Default = True.
        """
        if compress:
            with gzip.GzipFile(filename, "wb") as file_object:
                pickle.dump(self, file_object)
        else:
            with open(filename, "wb") as file_object:
                pickle.dump(self, file_object)

    @classmethod
    def load(cls, filename, compressed=True):
        """Load a pickled class instance from file.

        Parameters
        ----------
        filename : :obj:`str`
            Name of file containing object.
        compressed : :obj:`bool`, optional
            If True, the file is assumed to be compressed and gzip will be used
            to load it. Otherwise, it will assume that the file is not
            compressed. Default = True.

        Returns
        -------
        obj : class object
            Loaded class object.
        """
        if compressed:
            try:
                with gzip.GzipFile(filename, "rb") as file_object:
                    obj = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with gzip.GzipFile(filename, "rb") as file_object:
                    obj = pickle.load(file_object, encoding="latin")
        else:
            try:
                with open(filename, "rb") as file_object:
                    obj = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with open(filename, "rb") as file_object:
                    obj = pickle.load(file_object, encoding="latin")

        if not isinstance(obj, cls):
            raise IOError(f"Pickled object must be {cls}, not {type(obj)}")

        return obj


class Estimator(NiMAREBase):
    """Estimators take in Datasets and return MetaResults.

    All Estimators must have a ``_fit`` method implemented, which applies algorithm-specific
    methods to a dataset and returns a dictionary of arrays to be converted into a MetaResult.
    Users will interact with the ``_fit`` method by calling the user-facing ``fit`` method.
    ``fit`` takes in a ``Dataset``, calls ``_validate_input``, then ``_preprocess_input``,
    then ``_fit``, and finally converts the dictionary returned by ``_fit`` into a ``MetaResult``.
    """

    # Inputs that must be available in input Dataset. Keys are names of
    # attributes to set; values are strings indicating location in Dataset.
    _required_inputs = {}

    def _validate_input(self, dataset, drop_invalid=True):
        """Search for, and validate, required inputs as necessary."""
        if not hasattr(dataset, "slice"):
            raise ValueError(
                f"Argument 'dataset' must be a valid Dataset object, not a {type(dataset)}."
            )

        if self._required_inputs:
            data = dataset.get(self._required_inputs, drop_invalid=drop_invalid)
            # Do not overwrite existing inputs_ attribute.
            # This is necessary for PairwiseCBMAEstimator, which validates two sets of coordinates
            # in the same object.
            # It makes the *strong* assumption that required inputs will not changes within an
            # Estimator across fit calls, so all fields of inputs_ will be overwritten instead of
            # retaining outdated fields from previous fit calls.
            if not hasattr(self, "inputs_"):
                self.inputs_ = {}

            for k, v in data.items():
                if v is None:
                    raise ValueError(
                        f"Estimator {self.__class__.__name__} requires input dataset to contain "
                        f"{k}, but no matching data were found."
                    )
                self.inputs_[k] = v

    def _preprocess_input(self, dataset):
        """Perform any additional preprocessing steps on data in self.inputs_."""
        pass

    def fit(self, dataset, drop_invalid=True):
        """Fit Estimator to Dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset object to analyze.
        drop_invalid : :obj:`bool`, optional
            Whether to automatically ignore any studies without the required data or not.
            Default is False.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Estimator fitting.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            Inputs used in _fit.

        Notes
        -----
        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Estimators' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.
        """
        self._validate_input(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        maps = self._fit(dataset)

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        self.results = MetaResult(self, masker, maps)
        return self.results

    @abstractmethod
    def _fit(self, dataset):
        """Apply estimation to dataset and output results.

        Must return a dictionary of results, where keys are names of images
        and values are ndarrays.
        """
        pass


class MetaEstimator(Estimator):
    """Base class for meta-analysis methods in :mod:`~nimare.meta`.

    .. versionchanged:: 0.0.8

        * [REF] Use saved MA maps, when available.

    .. versionadded:: 0.0.3

    """

    def __init__(self, *args, **kwargs):
        mask = kwargs.get("mask")
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.resample = kwargs.get("resample", False)
        self.memory_limit = kwargs.get("memory_limit", None)

        # defaults for resampling images (nilearn's defaults do not work well)
        self._resample_kwargs = {"clip": True, "interpolation": "linear"}
        self._resample_kwargs.update(
            {k.split("resample__")[1]: v for k, v in kwargs.items() if k.startswith("resample__")}
        )

    def _preprocess_input(self, dataset):
        """Preprocess inputs to the Estimator from the Dataset as needed."""
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nb.load(mask_img)

        # Ensure that protected values are not included among _required_inputs
        assert "aggressive_mask" not in self._required_inputs.keys(), "This is a protected name."

        # A dictionary to collect masked image data, to be further reduced by the aggressive mask.
        temp_image_inputs = {}

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "image":
                # If no resampling is requested, check if resampling is required
                if not self.resample:
                    check_imgs = {img: nb.load(img) for img in self.inputs_[name]}
                    _check_same_fov(**check_imgs, reference_masker=mask_img, raise_error=True)
                    imgs = list(check_imgs.values())
                else:
                    # resampling will only occur if shape/affines are different
                    # making this harmless if all img shapes/affines are the same as the reference
                    imgs = [
                        resample_to_img(nb.load(img), mask_img, **self._resample_kwargs)
                        for img in self.inputs_[name]
                    ]

                # input to NiFtiLabelsMasker must be 4d
                img4d = concat_imgs(imgs, ensure_ndim=4)

                # Mask required input images using either the dataset's mask or the estimator's.
                temp_arr = masker.transform(img4d)

                # An intermediate step to mask out bad voxels.
                # Can be dropped once PyMARE is able to handle masked arrays or missing data.
                nonzero_voxels_bool = np.all(temp_arr != 0, axis=0)
                nonnan_voxels_bool = np.all(~np.isnan(temp_arr), axis=0)
                good_voxels_bool = np.logical_and(nonzero_voxels_bool, nonnan_voxels_bool)

                data = masker.transform(img4d)

                temp_image_inputs[name] = data
                if "aggressive_mask" not in self.inputs_.keys():
                    self.inputs_["aggressive_mask"] = good_voxels_bool
                else:
                    # Remove any voxels that are bad in any image-based inputs
                    self.inputs_["aggressive_mask"] = np.logical_or(
                        self.inputs_["aggressive_mask"],
                        good_voxels_bool,
                    )

            elif type_ == "coordinates":
                # Try to load existing MA maps
                if hasattr(self, "kernel_transformer"):
                    self.kernel_transformer._infer_names(affine=md5(mask_img.affine).hexdigest())
                    if self.kernel_transformer.image_type in dataset.images.columns:
                        files = dataset.get_images(
                            ids=self.inputs_["id"],
                            imtype=self.kernel_transformer.image_type,
                        )
                        if all(f is not None for f in files):
                            self.inputs_["ma_maps"] = files

                # Calculate IJK matrix indices for target mask
                # Mask space is assumed to be the same as the Dataset's space
                # These indices are used directly by any KernelTransformer
                xyz = self.inputs_["coordinates"][["x", "y", "z"]].values
                ijk = mm2vox(xyz, mask_img.affine)
                self.inputs_["coordinates"][["i", "j", "k"]] = ijk

        # Further reduce image-based inputs to remove "bad" voxels
        # (voxels with zeros or NaNs in any studies)
        if "aggressive_mask" in self.inputs_.keys():
            n_bad_voxels = (
                self.inputs_["aggressive_mask"].size - self.inputs_["aggressive_mask"].sum()
            )
            if n_bad_voxels:
                LGR.warning(
                    f"Masking out {n_bad_voxels} additional voxels. "
                    "The updated masker is available in the Estimator.masker attribute."
                )
        for name, raw_masked_data in temp_image_inputs.items():
            self.inputs_[name] = raw_masked_data[:, self.inputs_["aggressive_mask"]]


class Transformer(NiMAREBase):
    """Transformers take in Datasets and return Datasets.

    Initialize with hyperparameters.
    """

    def __init__(self):
        pass

    @abstractmethod
    def transform(self, dataset):
        """Add stuff to transformer."""
        # Using attribute check instead of type check to allow fake Datasets for testing.
        if not hasattr(dataset, "slice"):
            raise ValueError(
                f"Argument 'dataset' must be a valid Dataset object, not a {type(dataset)}"
            )


class Decoder(NiMAREBase):
    """Base class for decoders in :mod:`~nimare.decode`.

    .. versionadded:: 0.0.3

    """

    __id_cols = ["id", "study_id", "contrast_id"]

    def _validate_input(self, dataset, drop_invalid=True):
        """Search for, and validate, required inputs as necessary."""
        if not hasattr(dataset, "slice"):
            raise ValueError(
                f"Argument 'dataset' must be a valid Dataset object, not a {type(dataset)}."
            )

        if self._required_inputs:
            data = dataset.get(self._required_inputs, drop_invalid=drop_invalid)
            # Do not overwrite existing inputs_ attribute.
            # This is necessary for PairwiseCBMAEstimator, which validates two sets of coordinates
            # in the same object.
            # It makes the *strong* assumption that required inputs will not changes within an
            # Estimator across fit calls, so all fields of inputs_ will be overwritten instead of
            # retaining outdated fields from previous fit calls.
            if not hasattr(self, "inputs_"):
                self.inputs_ = {}

            for k, v in data.items():
                if v is None:
                    raise ValueError(
                        f"Estimator {self.__class__.__name__} requires input dataset to contain "
                        f"{k}, but no matching data were found."
                    )
                self.inputs_[k] = v

    def _preprocess_input(self, dataset):
        """Select features for model based on requested features and feature_group.

        This also takes into account which features have at least one study in the
        Dataset with the feature.
        """
        # Reduce feature list as desired
        if self.feature_group is not None:
            if not self.feature_group.endswith("__"):
                self.feature_group += "__"
            feature_names = self.inputs_["annotations"].columns.values
            feature_names = [f for f in feature_names if f.startswith(self.feature_group)]
            if self.features is not None:
                features = [f.split("__")[-1] for f in feature_names if f in self.features]
            else:
                features = feature_names
        else:
            if self.features is None:
                features = self.inputs_["annotations"].columns.values
            else:
                features = self.features

        features = [f for f in features if f not in self.__id_cols]
        n_features_orig = len(features)

        # At least one study in the dataset much have each label
        counts = (self.inputs_["annotations"][features] > self.frequency_threshold).sum(0)
        features = counts[counts > 0].index.tolist()
        if not len(features):
            raise Exception("No features identified in Dataset!")
        elif len(features) < n_features_orig:
            LGR.info(f"Retaining {len(features)}/({n_features_orig} features.")

        self.features_ = features

    def fit(self, dataset, drop_invalid=True):
        """Fit Decoder to Dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset object to analyze.
        drop_invalid : :obj:`bool`, optional
            Whether to automatically ignore any studies without the required data or not.
            Default is True.


        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Decoder fitting.

        Notes
        -----
        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Decoders' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.

        Selection of features based on requested features and feature group is performed in
        `Decoder._preprocess_input`.
        """
        self._validate_input(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        self._fit(dataset)

    @abstractmethod
    def _fit(self, dataset):
        """Apply decoding to dataset and output results.

        Must return a DataFrame, with one row for each feature.
        """
        pass
