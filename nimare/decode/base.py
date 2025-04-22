"""Base classes for the decode module."""

import logging
from abc import abstractmethod

from nimare.base import NiMAREBase

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
            LGR.info(f"Retaining {len(features)}/{n_features_orig} features.")

        self.features_ = features

    def fit(self, dataset, drop_invalid=True):
        """Fit Decoder to Dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset object to analyze.
        drop_invalid : :obj:`bool`, default=True
            Whether to automatically ignore any studies without the required data or not.
            Default is True.

        Notes
        -----
        The `fit` method is a light wrapper that runs input validation and
        preprocessing before fitting the actual model. Decoders' individual
        "fitting" methods are implemented as `_fit`, although users should
        call `fit`.

        Selection of features based on requested features and feature group is performed in
        `Decoder._preprocess_input`.
        """
        self._collect_inputs(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        self._fit(dataset)

    @abstractmethod
    def _fit(self, dataset):
        """Apply decoding to dataset and output results.

        Must return a DataFrame, with one row for each feature.
        """
