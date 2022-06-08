"""Base classes for NiMARE."""
import gzip
import inspect
import logging
import pickle
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from nimare.results import MetaResult

LGR = logging.getLogger(__name__)


class NiMAREBase(metaclass=ABCMeta):
    """Base class for NiMARE.

    This class contains a few features that are useful throughout the library:

    - Custom __repr__ method for printing the object.
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
    methods to a Dataset and returns a dictionary of arrays to be converted into a MetaResult.

    Users will interact with the ``_fit`` method by calling the user-facing ``fit`` method.
    ``fit`` takes in a ``Dataset``, calls ``_collect_inputs``, then ``_preprocess_input``,
    then ``_fit``, and finally converts the dictionary returned by ``_fit`` into a ``MetaResult``.
    """

    # Inputs that must be available in input Dataset. Keys are names of
    # attributes to set; values are strings indicating location in Dataset.
    _required_inputs = {}

    def _collect_inputs(self, dataset, drop_invalid=True):
        """Search for, and validate, required inputs as necessary.

        This method populates the ``inputs_`` attribute.

        .. versionchanged:: 0.0.12

            Renamed from ``_validate_input``.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
        drop_invalid : :obj:`bool`, optional
            Whether to automatically drop any studies in the Dataset without valid data or not.
            Default is True.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            A dictionary of required inputs for the Estimator, extracted from the Dataset.
            The actual inputs collected in this attribute are determined by the
            ``_required_inputs`` variable that should be specified in each child class.
        """
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

    @abstractmethod
    def _preprocess_input(self, dataset):
        """Perform any additional preprocessing steps on data in self.inputs_.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            The Dataset
        """
        pass

    @abstractmethod
    def _fit(self, dataset):
        """Apply estimation to dataset and output results.

        Must return a dictionary of results, where keys are names of images
        and values are ndarrays.
        """
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
        self._collect_inputs(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        maps = self._fit(dataset)

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        return MetaResult(self, masker, maps)
