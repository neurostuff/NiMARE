"""
Base classes for datasets.
"""
import os
import gzip
import copy
import pickle
import logging
import multiprocessing as mp
from collections import defaultdict
from abc import ABCMeta

import inspect
import nilearn as nl
from six import with_metaclass

LGR = logging.getLogger(__name__)


class NiMAREBase(with_metaclass(ABCMeta)):
    def __init__(self):
        """
        TODO: Actually write/refactor class methods. They mostly come directly from sklearn
        https://github.com/scikit-learn/scikit-learn/blob/2a1e9686eeb203f5fddf44fd06414db8ab6a554a/sklearn/base.py#L141
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
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
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


class MetaResult(object):
    """
    Base class for meta-analytic results.
    """
    def __init__(self, estimator, mask, maps=None):
        self.estimator = estimator
        self.mask = mask
        self.maps = maps or {}

    def get_map(self, name, return_type='image'):
        m = self.maps.get(name)
        if m is None:
            raise ValueError("No map with name '{}' found.".format(name))
        return nl.masking.unmask(m, self.mask) if return_type == 'image' else m

    def save_maps(self, output_dir='.', prefix='', prefix_sep='_',
                  names=None):
        """
        Save results to files.

        Parameters
        ----------
        output_dir : :obj:`str`, optional
            Output directory in which to save results. If the directory doesn't
            exist, it will be created. Default is current directory.
        prefix : :obj:`str`, optional
            Prefix to prepent to output file names. Default is none.
        prefix_sep : :obj:`str`, optional
            Separator to add between prefix and default file names. Default is
            _.
        """
        if prefix == '':
            prefix_sep = ''

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        names = names or list(self.maps.keys())
        maps = {k: self.get_map(k) for k in names}

        for imgtype, img in maps.items():
            filename = prefix + prefix_sep + imgtype + '.nii.gz'
            outpath = os.path.join(output_dir, filename)
            img.to_filename(outpath)

    def copy(self):
        new = MetaResult(copy.copy(self.estimator),
                         copy.copy(self.mask),
                         copy.deepcopy(self.maps))
        return new
