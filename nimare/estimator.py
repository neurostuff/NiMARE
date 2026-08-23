"""Base class for estimators."""

from abc import abstractmethod

from joblib import Memory

from nimare.base import NiMAREBase
from nimare.results import MetaResult
from nimare.studyset import normalize_collection


class Estimator(NiMAREBase):
    """Estimators take in collections and return fitted result objects.

    All Estimators must have a ``_fit`` method implemented, which applies algorithm-specific
    methods to a collection and returns a dictionary of arrays to be converted into a fitted
    result object.

    Users will interact with the ``_fit`` method by calling the user-facing ``fit`` method.
    ``fit`` takes in a ``Dataset``, calls ``_collect_inputs``, then ``_preprocess_input``,
    then ``_fit``, and finally converts the dictionary returned by ``_fit`` into a result
    object via ``_make_result``.

    .. warning::
        Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed in
        NiMARE 1.0.0. Prefer :class:`~nimare.nimads.Studyset`.
    """

    #: The ``drop_invalid`` ``fit`` was called with, for the validity that can only be
    #: judged in ``_preprocess_input``, once the data are loaded.
    _drop_invalid = True

    def __init__(
        self,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        generate_description=True,
    ):
        self.memory = memory
        self.memory_level = memory_level
        self.generate_description = generate_description

    @abstractmethod
    def _generate_description(self):
        """Generate a text description of the Estimator."""
        pass

    def _description_text(self):
        """Return the estimator description, or an empty string when disabled."""
        if not self.generate_description:
            return ""
        return self._generate_description()

    @abstractmethod
    def _preprocess_input(self, dataset):
        """Perform any additional preprocessing steps on data in self.inputs_.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            The collection to preprocess.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in NiMARE 1.0.0. Prefer :class:`~nimare.nimads.Studyset`.
        """
        pass

    @abstractmethod
    def _fit(self, dataset):
        """Apply estimation to dataset and output results.

        Must return a dictionary of results, where keys are names of images
        and values are ndarrays.
        """
        pass

    def _make_result(self, dataset, maps=None, tables=None, description=""):
        """Construct the fitted result object for this estimator.

        Subclasses may override this to return a specialized ``MetaResult`` subclass.
        """
        masker = getattr(self, "masker", None) or dataset.masker
        return MetaResult(self, mask=masker, maps=maps, tables=tables, description=description)

    def fit(self, dataset, drop_invalid=True):
        """Fit Estimator to a collection.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection object to analyze.
        drop_invalid : :obj:`bool`, optional
            Whether to automatically ignore any studies without the required data, or with
            data that cannot be used, such as an all-zero image. Default is True.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Result of Estimator fitting. Subclasses may return a ``MetaResult`` subclass.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            Inputs used in _fit.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in NiMARE 1.0.0. Prefer :class:`~nimare.nimads.Studyset`.

        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Estimators' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.
        """
        dataset = normalize_collection(dataset)
        self._drop_invalid = drop_invalid
        self._collect_inputs(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        maps, tables, description = self._cache(self._fit, func_memory_level=1)(dataset)
        if not self.generate_description:
            description = ""

        return self._make_result(dataset, maps=maps, tables=tables, description=description)
