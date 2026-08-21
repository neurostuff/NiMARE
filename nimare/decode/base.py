"""Base classes for the decode module."""

import logging
from abc import abstractmethod

from nimare.base import NiMAREBase
from nimare.dataset import Dataset
from nimare.nimads import Studyset
from nimare.studyset import normalize_collection

LGR = logging.getLogger(__name__)


class Decoder(NiMAREBase):
    """Base class for decoders in :mod:`~nimare.decode`.

    .. versionchanged:: 0.0.12

        Moved from ``nimare.base`` to ``nimare.decode.base``.

    .. versionadded:: 0.0.3

    """

    __id_cols = ["id", "study_id", "contrast_id"]

    def _collect_inputs(self, dataset, drop_invalid=True):
        """Search for, and validate, required inputs as necessary."""
        dataset = normalize_collection(dataset)

        if self._required_inputs:
            from nimare.studyset.inputs import collect_inputs

            self.data, data = collect_inputs(
                dataset, self._required_inputs, drop_invalid=drop_invalid
            )
            if not hasattr(self, "inputs_"):
                self.inputs_ = {}
            self.inputs_.update(data)

    def _preprocess_input(self, dataset):
        """Select features for model based on requested features and feature_group.

        This also takes into account which features have at least one study in the
        Studyset/Dataset collection with the feature.
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
            raise Exception("No features identified in the input Studyset/Dataset collection!")
        elif len(features) < n_features_orig:
            LGR.info(f"Retaining {len(features)}/{n_features_orig} features.")

        self.features_ = features

    def fit(self, dataset, drop_invalid=True):
        """Fit Decoder to a Studyset/Dataset collection.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Studyset-backed collection object to analyze.
        drop_invalid : :obj:`bool`, default=True
            Whether to automatically ignore any studies without the required data or not.
            Default is True.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.

        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Decoders' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.

        Selection of features based on requested features and feature group is performed in
        `Decoder._preprocess_input`.
        """
        if not isinstance(dataset, (Dataset, Studyset)):
            dataset = normalize_collection(dataset)

        self._collect_inputs(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        self._fit(dataset)

    @abstractmethod
    def _fit(self, dataset):
        """Apply decoding to dataset and output results.

        Must return a DataFrame, with one row for each feature.
        """
