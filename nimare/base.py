"""
Base classes for datasets.
"""
import gzip
import pickle
import inspect
import logging
import multiprocessing as mp
from collections import defaultdict
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from six import with_metaclass

from .results import MetaResult
from .utils import get_masker


LGR = logging.getLogger(__name__)


class NiMAREBase(with_metaclass(ABCMeta)):
    """
    Base class for NiMARE.
    """
    def __init__(self):
        """
        TODO: Actually write/refactor class methods. They mostly come directly from sklearn
        https://github.com/scikit-learn/scikit-learn/blob/
        2a1e9686eeb203f5fddf44fd06414db8ab6a554a/sklearn/base.py#L141
        """
        pass

    def _check_ncores(self, n_cores):
        """
        Check number of cores used for method.
        """
        if n_cores == -1:
            n_cores = mp.cpu_count()
        elif n_cores > mp.cpu_count():
            LGR.warning(
                'Desired number of cores ({0}) greater than number '
                'available ({1}). Setting to {1}.'.format(n_cores,
                                                          mp.cpu_count()))
            n_cores = mp.cpu_count()
        return n_cores

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
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
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
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
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def save(self, filename, compress=True):
        """
        Pickle the class instance to the provided file.

        Parameters
        ----------
        filename : :obj:`str`
            File to which object will be saved.
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
        Load a pickled class instance from file.

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
                with gzip.GzipFile(filename, 'rb') as file_object:
                    obj = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with gzip.GzipFile(filename, 'rb') as file_object:
                    obj = pickle.load(file_object, encoding='latin')
        else:
            try:
                with open(filename, 'rb') as file_object:
                    obj = pickle.load(file_object)
            except UnicodeDecodeError:
                # Need to try this for python3
                with open(filename, 'rb') as file_object:
                    obj = pickle.load(file_object, encoding='latin')

        if not isinstance(obj, cls):
            raise IOError('Pickled object must be {0}, '
                          'not {1}'.format(cls, type(obj)))

        return obj


class Estimator(NiMAREBase):
    """Estimators take in Datasets and return MetaResults
    """

    # Inputs that must be available in input Dataset. Keys are names of
    # attributes to set; values are strings indicating location in Dataset.
    _required_inputs = {}

    def _validate_input(self, dataset):
        """
        Search for, and validate, required inputs as necessary.
        """
        if not hasattr(dataset, 'slice'):
            raise ValueError('Argument "dataset" must be a valid Dataset '
                             'object, not a {0}'.format(type(dataset)))

        if self._required_inputs:
            data = dataset.get(self._required_inputs)
            self.inputs_ = {}
            for k, v in data.items():
                if v is None:
                    raise ValueError(
                        "Estimator {0} requires input dataset to contain {1}, but "
                        "none were found.".format(self.__class__.__name__, k))
                self.inputs_[k] = v

    def _preprocess_input(self, dataset):
        """
        Perform any additional preprocessing steps on data in self.inputs_
        """
        pass

    def fit(self, dataset):
        """
        Fit Estimator to Dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset object to analyze.

        Returns
        -------
        :obj:`nimare.results.MetaResult`
            Results of Estimator fitting.

        Notes
        -----
        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Estimators' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.
        """
        self._validate_input(dataset)
        self._preprocess_input(dataset)
        maps = self._fit(dataset)

        if hasattr(self, 'masker') and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        self.results = MetaResult(self, masker, maps)
        return self.results

    @abstractmethod
    def _fit(self, dataset):
        """
        Apply estimation to dataset and output results. Must return a
        dictionary of results, where keys are names of images and values are
        ndarrays.
        """
        pass


class MetaEstimator(Estimator):
    """Base class for meta-analysis methods in :mod:`nimare.meta`.
    """
    def __init__(self, *args, **kwargs):
        mask = kwargs.get('mask')
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

    def _preprocess_input(self, dataset):
        """Preprocess inputs to the Estimator from the Dataset as needed.
        """
        masker = self.masker or dataset.masker
        for name, (type_, _) in self._required_inputs.items():
            if type_ == 'image':
                # Mask required input images using either the dataset's mask or
                # the estimator's.
                self.inputs_[name] = masker.transform(self.inputs_[name])
            elif type_ == 'coordinates':
                self.inputs_[name] = dataset.coordinates.copy()


class CBMAEstimator(MetaEstimator):
    """Base class for coordinate-based meta-analysis methods.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    *args
        Optional arguments to the :obj:`nimare.base.MetaEstimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`nimare.base.MetaEstimator`
        __init__ (called automatically).
    """
    def __init__(self, kernel_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get kernel transformer
        kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                       if k.startswith('kernel__')}

        # Allow both instances and classes for the kernel transformer input.
        if not issubclass(kernel_transformer, KernelTransformer) and \
                not issubclass(type(kernel_transformer), KernelTransformer):
            raise ValueError('Argument "kernel_transformer" must be a kind of '
                             'KernelTransformer')
        elif not inspect.isclass(kernel_transformer) and kernel_args:
            LGR.warning('Argument "kernel_transformer" has already been '
                        'initialized, so kernel arguments will be ignored: '
                        '{}'.format(', '.join(kernel_args.keys())))
        elif inspect.isclass(kernel_transformer):
            kernel_transformer = kernel_transformer(**kernel_args)
        self.kernel_transformer = kernel_transformer

    def _preprocess_input(self, dataset):
        """Mask required input images using either the dataset's mask or the
        estimator's. Also, insert required metadata into coordinates DataFrame.
        """
        super()._preprocess_input(dataset)

        # All extra (non-ijk) parameters for a kernel should be overrideable as
        # parameters to __init__, so we can access them with get_params()
        kt_args = list(self.kernel_transformer.get_params().keys())

        # Integrate "sample_size" from metadata into DataFrame so that
        # kernel_transformer can access it.
        if 'sample_size' in kt_args:
            if 'sample_sizes' in dataset.get_metadata():
                # Extract sample sizes and make DataFrame
                sample_sizes = dataset.get_metadata(field='sample_sizes', ids=dataset.ids)
                # we need an extra layer of lists
                sample_sizes = [[ss] for ss in sample_sizes]
                sample_sizes = pd.DataFrame(index=dataset.ids, data=sample_sizes,
                                            columns=['sample_sizes'])
                sample_sizes['sample_size'] = sample_sizes['sample_sizes'].apply(np.mean)
                # Merge sample sizes df into coordinates df
                self.inputs_['coordinates'] = self.inputs_['coordinates'].merge(
                    right=sample_sizes, left_on='id', right_index=True,
                    sort=False, validate='many_to_one', suffixes=(False, False),
                    how='left')
            else:
                LGR.warning('Metadata field "sample_sizes" not found. '
                            'Set a constant sample size as a kernel transformer '
                            'argument, if possible.')


class Transformer(NiMAREBase):
    """Transformers take in Datasets and return Datasets

    Initialize with hyperparameters.
    """
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, dataset):
        """Add stuff to transformer.
        """
        if not hasattr(dataset, 'slice'):
            raise ValueError('Argument "dataset" must be a valid Dataset '
                             'object, not a {0}'.format(type(dataset)))


class KernelTransformer(Transformer):
    """Base class for modeled activation-generating methods in
    :mod:`nimare.meta.cbma.kernel`.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.

    Notes
    -----
    This base class exists solely to allow CBMA algorithms to check the class
    of their kernel_transformer parameters.

    All extra (non-ijk) parameters for a given kernel should be overrideable as
    parameters to __init__, so we can access them with get_params() and also
    apply them to datasets with missing data.
    """
    def __init__(self):
        pass


class Decoder(NiMAREBase):
    """Base class for decoders in :mod:`nimare.decode`.
    """
    __id_cols = ['id', 'study_id', 'contrast_id']

    def _preprocess_input(self, dataset):
        """
        Perform any additional preprocessing steps on data in self.inputs_
        """
        # Reduce feature list as desired
        if self.feature_group is not None:
            if not self.feature_group.endswith('__'):
                self.feature_group += '__'
            feature_names = dataset.annotations.columns.values
            feature_names = [f for f in feature_names if f.startswith(self.feature_group)]
            if self.features is not None:
                features = [f.split('__')[-1] for f in feature_names if f in self.features]
            else:
                features = feature_names
        else:
            if self.features is None:
                features = dataset.annotations.columns.values
            else:
                features = self.features
        features = [f for f in features if f not in self.__id_cols]
        # At least one study in the dataset much have each label
        counts = (dataset.annotations[features] > self.frequency_threshold).sum(0)
        features = counts[counts > 0].index.tolist()
        self.features_ = features

    def fit(self, dataset):
        """
        Fit Estimator to Dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset object to analyze.

        Returns
        -------
        :obj:`nimare.results.MetaResult`
            Results of Estimator fitting.

        Notes
        -----
        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Estimators' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.
        """
        self._preprocess_input(dataset)
        self._fit(dataset)

    @abstractmethod
    def _fit(self, dataset):
        """
        Apply estimation to dataset and output results. Must return a
        dictionary of results, where keys are names of images and values are
        ndarrays.
        """
        pass
