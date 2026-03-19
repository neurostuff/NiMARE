"""Base class for estimators."""

from abc import abstractmethod

from joblib import Memory

from nimare.base import NiMAREBase
from nimare.results import MetaResult
from nimare.studyset import StudysetView, ensure_studyset_view


class Estimator(NiMAREBase):
    """Estimators take in Datasets and return MetaResults.

    All Estimators must have a ``_fit`` method implemented, which applies algorithm-specific
    methods to a Dataset and returns a dictionary of arrays to be converted into a MetaResult.

    Users will interact with the ``_fit`` method by calling the user-facing ``fit`` method.
    ``fit`` takes in a ``Dataset``, calls ``_collect_inputs``, then ``_preprocess_input``,
    then ``_fit``, and finally converts the dictionary returned by ``_fit`` into a ``MetaResult``.

    .. warning::
        Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed in
        a future release. Prefer :class:`~nimare.nimads.Studyset`.
    """

    # Inputs that must be available in input Dataset. Keys are names of
    # attributes to set; values are strings indicating location in Dataset.
    _required_inputs = {}

    def __init__(
        self,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        generate_description=True,
    ):
        self.memory = memory
        self.memory_level = memory_level
        self.generate_description = generate_description

    def _collect_inputs(self, dataset, drop_invalid=True):
        """Search for, and validate, required inputs as necessary.

        This method populates the ``inputs_`` attribute.

        .. versionchanged:: 0.0.12

            Renamed from ``_validate_input``.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
        drop_invalid : :obj:`bool`, default=True
            Whether to automatically drop any studies in the Dataset without valid data or not.
            Default is True.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            A dictionary of required inputs for the Estimator, extracted from the Dataset.
            The actual inputs collected in this attribute are determined by the
            ``_required_inputs`` variable that should be specified in each child class.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        if not isinstance(dataset, StudysetView):
            dataset = ensure_studyset_view(dataset)

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
        dataset : :obj:`~nimare.dataset.Dataset`
            The Dataset

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
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
        """Fit Estimator to a Studyset-backed collection.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset`, :obj:`~nimare.studyset.StudysetView`, \
                or :obj:`~nimare.dataset.Dataset`
            Collection object to analyze.
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

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.

        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Estimators' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.
        """
        dataset = ensure_studyset_view(dataset)
        self._collect_inputs(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        maps, tables, description = self._cache(self._fit, func_memory_level=1)(dataset)
        if not self.generate_description:
            description = ""

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        return MetaResult(self, mask=masker, maps=maps, tables=tables, description=description)
