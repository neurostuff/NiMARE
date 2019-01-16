"""Base classes for meta-analyses.
"""
from os import makedirs
from os.path import exists, join
from collections import defaultdict

from abc import ABCMeta, abstractmethod
from six import with_metaclass

from nilearn.masking import unmask


class MetaResult(object):
    """Base class for meta-analytic results.
    Will contain slots for different kinds of results maps (e.g., z-map, p-map)
    """
    def __init__(self, estimator, mask=None, **kwargs):
        self.estimator = estimator
        self.mask = mask
        self.images = {}
        for key, array in kwargs.items():
            self.images[key] = unmask(array, self.mask)

    def save_results(self, output_dir='.', prefix='', prefix_sep='_'):
        """Save results to files.
        """
        if prefix == '':
            prefix_sep = ''

        if not exists(output_dir):
            makedirs(output_dir)

        for imgtype, img in self.images.items():
            filename = prefix + prefix_sep + imgtype + '.nii.gz'
            outpath = join(output_dir, filename)
            img.to_filename(outpath)


class MetaEstimator(with_metaclass(ABCMeta)):
    """
    Base class for meta-analysis estimators.

    TODO: Actually write/refactor class methods. They mostly come directly from sklearn
    https://github.com/scikit-learn/scikit-learn/blob/afe540c7f2cbad259dd333e6744b088213180bee/sklearn/base.py#L176

    At minimum, we want:
      - get_params: return dict of model parameters and values
      - set_params: overwrite parameters in model from dict

    """
    def __init__(self):
        pass

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
        init_signature = signature(init)
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

    @abstractmethod
    def fit(self, sample):
        """Run meta-analysis on dataset.
        """
        pass


class CBMAEstimator(MetaEstimator):
    """Base class for coordinate-based meta-analysis methods.
    """
    pass


class KernelEstimator(with_metaclass(ABCMeta)):
    """Base class for modeled activation-generating methods.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.
    """
    @abstractmethod
    def transform(self):
        """
        Generate a modeled activation map for each of the contrasts in a
        dataset.
        """
        pass


class IBMAEstimator(MetaEstimator):
    """Base class for image-based meta-analysis methods.
    """
    pass
