"""The container :mod:`nimare.ml` hands to scikit-learn."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from nimare.base import NiMAREBase
from nimare.ml._helpers import (
    _check_lengths,
    _hstack,
    _jsonable,
    _preview,
    _take_rows,
    _to_dense,
)
from nimare.ml.reduce import _is_atlas_like, _resolve_map_reducer


def _dense_step(transformer):
    """Wrap a descriptor transformer so that it is handed dense columns.

    The block arrives sparse because it sits beside the voxels in one matrix,
    and ``StandardScaler`` refuses to centre sparse data at all.
    """
    if isinstance(transformer, str):
        return transformer
    return Pipeline(
        [
            ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("transform", transformer),
        ]
    )


def _is_passthrough(transformer):
    """Report whether a transformer argument asks for nothing to be done."""
    return isinstance(transformer, str) and transformer == "passthrough"


def _step_name(name, position, used):
    """Return a ColumnTransformer step name, which may not contain ``__``.

    Annotation fields often do -- ``Neurosynth_TFIDF__pain`` -- so the name is
    softened, and disambiguated by position if softening made it collide.
    """
    safe = name.replace("__", "_") or f"descriptor_{position}"
    if safe in used:
        safe = f"{safe}_{position}"
    used.add(safe)
    return safe


class FeatureSet(NiMAREBase):
    """Aligned modeled activation features, descriptors, target and provenance.

    Build one from a Studyset with :meth:`from_studyset`. The constructor below
    takes the blocks directly, which is how :meth:`split`,
    :meth:`select_analyses`, :meth:`copy` and the map-reduction methods build
    their results, and what to use for features generated some other way.

    Every array here shares one row order: row ``i`` is analysis ``ids[i]`` from
    study ``study_ids[i]``. That order is fixed when the feature set is built
    and is preserved by :meth:`split`, :meth:`select_analyses` and the
    map-reduction methods.

    Parameters
    ----------
    map_features : :obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`
        Analysis-by-voxel modeled activation features. Unreduced voxelwise
        features are sparse; a reduced representation may be dense.
    ids : array_like of :obj:`str`
        Full ``"<study_id>-<analysis_id>"`` identifier for each row.
    study_ids : array_like of :obj:`str`
        Study identifier for each row, used as scikit-learn ``groups``.
    descriptor_features : :obj:`numpy.ndarray`, optional
        Numeric descriptor columns appended to :attr:`features`, by default None.
    descriptor_names : :obj:`list` of :obj:`str`, optional
        Names of the descriptor columns, by default None.
    target : array_like, optional
        Row-aligned prediction target, by default None.
    provenance : :obj:`dict`, optional
        Conversion settings and source Studyset details, by default None.
    masker : :obj:`nilearn.maskers.NiftiMasker`, optional
        Masker defining the voxel order of unreduced map features, by default
        None.
    map_feature_names : :obj:`list` of :obj:`str`, optional
        Names of the map columns, by default None, which names them
        ``"voxel_<i>"`` on demand.

    Attributes
    ----------
    features : :obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`
        Map features and descriptor features side by side. Built on first
        access from the two blocks, which are the source of truth.
    feature_names : :obj:`list` of :obj:`str`
        Names for :attr:`features` in column order, built on first access.

    See Also
    --------
    AtlasAggregator : Reduce the voxelwise features over the regions of an atlas.
    """

    def __init__(
        self,
        map_features: Any,
        ids: Sequence[str],
        study_ids: Sequence[str],
        *,
        descriptor_features: Any | None = None,
        descriptor_names: Sequence[str] | None = None,
        target: Any | None = None,
        provenance: dict[str, Any] | None = None,
        masker: Any | None = None,
        map_feature_names: Sequence[str] | None = None,
    ) -> None:
        n_rows = map_features.shape[0]

        ids = np.asarray(ids, dtype=str)
        study_ids = np.asarray(study_ids, dtype=str)

        if descriptor_features is not None and not sparse.issparse(descriptor_features):
            descriptor_features = np.asarray(descriptor_features)
        if descriptor_features is not None and len(descriptor_features.shape) != 2:
            raise ValueError("descriptor_features must be two-dimensional.")

        _check_lengths(
            n_rows,
            rows={
                "ids": ids,
                "study_ids": study_ids,
                "target": target,
                "descriptor_features": descriptor_features,
            },
            columns={
                "map_feature_names": (map_feature_names, map_features.shape[1]),
                "descriptor_names": (
                    descriptor_names,
                    None if descriptor_features is None else descriptor_features.shape[1],
                ),
            },
        )

        self._map_features = map_features
        self._descriptor_features = descriptor_features
        self._descriptor_names = None if descriptor_names is None else list(descriptor_names)
        self._map_feature_names = None if map_feature_names is None else list(map_feature_names)
        self._features = None
        self._feature_names = None

        self.ids = ids
        self.study_ids = study_ids
        self.target = None if target is None else np.asarray(target)
        self.provenance = {} if provenance is None else provenance
        self.masker = masker

    @classmethod
    def from_studyset(
        cls,
        studyset,
        kernel_transformer,
        *,
        descriptor_fields=None,
        target_field=None,
        target_transformer=None,
        missing_coordinates="drop",
        missing_values="raise",
        memory=None,
        memory_level=2,
    ):
        """Build a feature set from a Studyset.

        Generates one modeled activation (MA) map per analysis through the
        kernel transformer, appends any numeric descriptor fields, extracts any
        target, and aligns them to the analyses they came from.

        Parameters
        ----------
        studyset : :class:`~nimare.nimads.Studyset`
            The Studyset to convert. One analysis becomes one row.
        kernel_transformer : :class:`~nimare.meta.kernel.KernelTransformer`
            Kernel transformer instance or class used to generate the MA maps.
            There is no default: the choice is scientific.
        descriptor_fields : :obj:`list`, optional
            Fields appended to the feature matrix as extra numeric columns, by
            default None. Each is a field name, a ``(source, field)`` tuple, or
            a mapping with ``source`` and ``field``; sources are
            ``"metadata"``, ``"annotations"`` and ``"texts"``. A field that
            reads as a glob pattern, such as ``"Neurosynth_TFIDF__*"``, selects
            every annotation label matching it. Non-numeric fields are refused.
        target_field : :obj:`str` or :obj:`tuple` or :obj:`dict`, optional
            Field exported as ``y``, by default None. Scalar numeric and scalar
            categorical fields are supported directly.
        target_transformer : :obj:`callable` or transformer, optional
            Applied to the raw target values before they become ``y``, by
            default None. Required for text fields, which have no scalar
            reading.
        missing_coordinates : {"drop", "include"}, default="drop"
            Whether analyses reporting no coordinates are removed before rows
            are built, or kept as all-zero sparse map rows.
        missing_values : {"raise", "drop", "keep"}, default="raise"
            What to do when a selected descriptor or target value is missing:
            report the analyses and fields, remove those analyses, or leave the
            gaps for a pipeline to impute.
        memory : :class:`joblib.Memory`, :obj:`str` or :class:`pathlib.Path`, optional
            Cache location for MA map generation, by default None. Used only
            when the kernel transformer does not define its own.
        memory_level : :obj:`int`, default=2
            How eagerly ``memory`` caches. Kernel transformers cache their maps
            at level 2, so a lower level asks for them not to be cached.

        Returns
        -------
        :class:`FeatureSet`
            One row per retained analysis. Call :meth:`to_sklearn` for the
            scikit-learn bundle.

        Raises
        ------
        :obj:`ValueError`
            If a field cannot be resolved or used, if a value is missing under
            ``missing_values="raise"``, if the target is constant over the
            analyses that were kept, or if the Studyset has no analyses or
            repeats an analysis id.

        See Also
        --------
        AtlasAggregator : Reduce the map features over the regions of an atlas.

        Examples
        --------
        >>> features = FeatureSet.from_studyset(  # doctest: +SKIP
        ...     studyset,
        ...     kernel_transformer=MKDAKernel(r=10),
        ...     target_field=("metadata", "comparison_task"),
        ... )
        """
        from nimare.ml.extract import _FeatureExtractor

        return _FeatureExtractor(
            kernel_transformer=kernel_transformer,
            descriptor_fields=descriptor_fields,
            target_field=target_field,
            target_transformer=target_transformer,
            missing_coordinates=missing_coordinates,
            missing_values=missing_values,
            memory=memory,
            memory_level=memory_level,
        ).transform(studyset, container=cls)

    def __repr__(self):
        """Show the shape, and whether any columns are descriptors."""
        n_rows, n_features = self.shape
        descriptors = ""
        if self._descriptor_features is not None:
            descriptors = f", n_descriptors={self._descriptor_features.shape[1]}"
        return (
            f"{self.__class__.__name__}(n_rows={n_rows}, n_features={n_features}, "
            f"n_studies={len(np.unique(self.study_ids))}{descriptors})"
        )

    def __len__(self):
        """Return the number of analysis rows."""
        return len(self.ids)

    # ----------------------------------------------------------- feature views

    @property
    def features(self):
        """:obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`: map and descriptor columns."""
        if self._features is None:
            self._features = _hstack(self._map_features, self._descriptor_features)
        return self._features

    @property
    def map_features(self):
        """:obj:`scipy.sparse.csr_matrix` or :obj:`numpy.ndarray`: the map block alone."""
        return self._map_features

    @property
    def descriptor_features(self):
        """:obj:`numpy.ndarray` or None: the descriptor block alone."""
        return self._descriptor_features

    @property
    def shape(self):
        """:obj:`tuple`: ``(n_rows, n_features)``."""
        n_descriptors = (
            0 if self._descriptor_features is None else (self._descriptor_features.shape[1])
        )
        return (self._map_features.shape[0], self._map_features.shape[1] + n_descriptors)

    @property
    def map_columns(self):
        """:obj:`slice`: the columns of :attr:`features` holding map features."""
        return slice(0, self._map_features.shape[1])

    @property
    def descriptor_columns(self):
        """:obj:`slice`: the columns of :attr:`features` holding descriptor features."""
        return slice(self._map_features.shape[1], self.shape[1])

    @property
    def descriptor_names(self):
        """:obj:`list` of :obj:`str`: the descriptor columns, in order."""
        return list(self._descriptor_names or [])

    @property
    def feature_names(self):
        """:obj:`list` of :obj:`str`: names for :attr:`features`, in column order.

        Built on first access. Naming 200,000 voxels eagerly costs more memory
        than the sparse matrix being named.
        """
        if self._feature_names is None:
            names = self._map_feature_names
            if names is None:
                names = [f"voxel_{idx}" for idx in range(self._map_features.shape[1])]
            self._feature_names = list(names) + list(self._descriptor_names or [])
        return self._feature_names

    # ----------------------------------------------------------------- export

    def to_sklearn(self, return_X_y=False):
        """Export the dataset for scikit-learn.

        Parameters
        ----------
        return_X_y : :obj:`bool`, default=False
            If True, return ``(data, target)`` instead of a Bunch, following
            the :mod:`sklearn.datasets` convention.

        Returns
        -------
        :class:`sklearn.utils.Bunch` or :obj:`tuple`
            A Bunch with ``data``, ``target``, ``groups``, ``feature_names``,
            ``ids``, ``provenance``, ``map_columns`` and ``descriptor_columns``;
            or ``(data, target)`` when ``return_X_y`` is True.

        Notes
        -----
        ``data`` stays sparse while the map features are unreduced. Pass
        ``groups`` straight to a scikit-learn group splitter.
        """
        if return_X_y:
            return self.features, self.target

        return Bunch(
            data=self.features,
            target=self.target,
            groups=self.study_ids,
            feature_names=self.feature_names,
            ids=self.ids,
            provenance=self.provenance,
            map_columns=self.map_columns,
            descriptor_columns=self.descriptor_columns,
        )

    # ------------------------------------------------------------- evaluation

    def split(self, test_size=0.25, random_state=None):
        """Split into train and test partitions without splitting a study.

        Parameters
        ----------
        test_size : :obj:`float` or :obj:`int`, default=0.25
            Proportion of *studies* held out, or the number of studies to hold
            out. Analyses are not split evenly when studies contribute
            different numbers of analyses.
        random_state : :obj:`int` or None, optional
            Seed for a reproducible split, by default None.

        Returns
        -------
        (:class:`FeatureSet`, :class:`FeatureSet`)
            Train and test datasets. No study appears in both.

        Raises
        ------
        :obj:`ValueError`
            If the available studies cannot serve the requested split. Nothing
            partial is returned.

        See Also
        --------
        sklearn.model_selection.GroupKFold : Cross-validation over the same groups.
        """
        n_studies = len(np.unique(self.study_ids))
        n_test = self._check_split(test_size, n_studies)

        splitter = GroupShuffleSplit(n_splits=1, test_size=n_test, random_state=random_state)
        train_rows, test_rows = next(splitter.split(np.empty(len(self)), groups=self.study_ids))

        return self.select_analyses(train_rows), self.select_analyses(test_rows)

    @staticmethod
    def _check_split(test_size, n_studies):
        """Return the number of held-out studies, or explain why there is none."""
        if n_studies < 2:
            raise ValueError(
                f"A grouped split needs at least 2 studies, but the dataset has {n_studies}. "
                "Analyses from one study are never split across partitions."
            )

        if isinstance(test_size, (int, np.integer)) and not isinstance(test_size, bool):
            n_test = int(test_size)
        elif isinstance(test_size, (float, np.floating)):
            if not 0.0 < test_size < 1.0:
                raise ValueError(
                    f"test_size={test_size!r} must be between 0 and 1 when given as a fraction."
                )
            n_test = int(np.ceil(test_size * n_studies))
        else:
            raise ValueError(f"test_size={test_size!r} must be a float or an int.")

        if not 1 <= n_test <= n_studies - 1:
            raise ValueError(
                f"test_size={test_size!r} holds out {n_test} of {n_studies} studies, which "
                "leaves one partition empty. Lower test_size or use more studies."
            )
        return n_test

    def make_preprocessor(
        self,
        map_reducer,
        descriptor_transformer="passthrough",
        **reducer_params,
    ):
        """Apply a reducer to the map columns and something else to the rest.

        A :class:`~sklearn.compose.ColumnTransformer` with the column boundary
        filled in, the masker bound into an atlas reducer, and
        ``sparse_threshold=1.0``. With no descriptor columns to keep the
        reducer away from, the reducer is returned as it is.

        Parameters
        ----------
        map_reducer : estimator, :obj:`type`, atlas or None
            A scikit-learn transformer, a transformer class built here from
            ``**reducer_params``, any atlas :class:`AtlasAggregator` accepts,
            or None to leave the map columns alone.
        descriptor_transformer : estimator, :obj:`str` or :obj:`dict`, default="passthrough"
            One transformer for every descriptor column, or a mapping from
            descriptor name to transformer. Descriptors the mapping does not
            name are passed through, and the column order is the one they came
            in with. Transformers are handed their columns dense.
        **reducer_params
            Passed to ``map_reducer`` when it is a class.

        Returns
        -------
        estimator
            A :class:`~sklearn.compose.ColumnTransformer` when there are
            descriptor columns, and the reducer itself when there are not.

        Raises
        ------
        :obj:`ValueError`
            If ``descriptor_transformer`` names something that is not a
            descriptor, if it is given for a feature set that has no descriptor
            columns, or if parameters are passed alongside a built transformer.

        Examples
        --------
        >>> pipeline = make_pipeline(  # doctest: +SKIP
        ...     features.make_preprocessor(TruncatedSVD(n_components=50)),
        ...     LogisticRegression(),
        ... )
        """
        if map_reducer is None or (isinstance(map_reducer, str) and map_reducer == "passthrough"):
            reducer = "passthrough"
        else:
            reducer = _resolve_map_reducer(map_reducer, masker=self.masker, **reducer_params)

        if self._descriptor_features is None:
            if not _is_passthrough(descriptor_transformer):
                raise ValueError(
                    "This feature set has no descriptor columns, so there is nothing for "
                    "descriptor_transformer to act on."
                )
            return reducer

        return ColumnTransformer(
            [
                ("maps", reducer, self.map_columns),
                (
                    "descriptors",
                    self._descriptor_step(descriptor_transformer),
                    self.descriptor_columns,
                ),
            ],
            sparse_threshold=1.0,
        )

    def _descriptor_step(self, descriptor_transformer):
        """Return the step applied to the descriptor block."""
        if not isinstance(descriptor_transformer, Mapping):
            return _dense_step(descriptor_transformer)

        names = self.descriptor_names
        unknown = [name for name in descriptor_transformer if name not in names]
        if unknown:
            raise ValueError(
                f"No descriptor called {_preview(unknown)}. This feature set has "
                f"{_preview(names)}."
            )

        # Column indices here are relative to the descriptor block, and every
        # descriptor gets a step of its own so the columns come out in the order they
        # went in rather than in the order the mapping happened to name them.
        used = set()
        steps = []
        for position, name in enumerate(names):
            steps.append(
                (
                    _step_name(name, position, used),
                    _dense_step(descriptor_transformer.get(name, "passthrough")),
                    [position],
                )
            )

        return ColumnTransformer(steps, sparse_threshold=1.0)

    # -------------------------------------------------------- map reduction

    def fit_transform_maps(self, reducer):
        """Fit a reducer on this dataset's map features and apply it.

        Fit it on the training dataset only, then pass the same, now fitted,
        reducer to :meth:`transform_maps` for held-out data.

        Parameters
        ----------
        reducer : estimator
            A scikit-learn transformer, or an :class:`AtlasAggregator` built
            with this feature set's masker.

        Returns
        -------
        :class:`FeatureSet`
            A feature set with reduced map features and everything else
            unchanged.
        """
        if _is_atlas_like(reducer):
            raise TypeError(
                "Build the aggregator first, as AtlasAggregator(atlas, "
                "masker=features.masker), so that the fitted one can be reused on "
                "held-out rows with transform_maps."
            )
        return self._with_map_features(reducer.fit_transform(self._map_features), reducer)

    def transform_maps(self, reducer):
        """Apply an already fitted reducer to this dataset's map features.

        Parameters
        ----------
        reducer : estimator
            A transformer already fitted by :meth:`fit_transform_maps` on the
            training dataset.

        Returns
        -------
        :class:`FeatureSet`
            A dataset with reduced map features and everything else unchanged.

        Raises
        ------
        :class:`sklearn.exceptions.NotFittedError`
            If the reducer has not been fitted, which would mean fitting it on
            held-out data.
        """
        try:
            check_is_fitted(reducer)
        except NotFittedError:
            raise NotFittedError(
                f"{type(reducer).__name__} is not fitted. Fit it on the training dataset "
                "with fit_transform_maps() first; fitting it here would use held-out data."
            ) from None
        except TypeError:
            # Not a scikit-learn estimator; let its own transform complain.
            pass

        return self._with_map_features(reducer.transform(self._map_features), reducer)

    def _with_map_features(self, reduced, reducer):
        """Return a copy carrying reduced map features."""
        if reduced.shape[0] != len(self):
            raise ValueError(
                f"{type(reducer).__name__} returned {reduced.shape[0]} rows for "
                f"{len(self)} analyses. A map reducer must preserve the analysis rows."
            )

        try:
            # str(), because numpy's str_ prints as np.str_('...') in numpy 2.
            names = [str(name) for name in reducer.get_feature_names_out()]
        except (AttributeError, NotFittedError, ValueError):
            names = [f"component_{idx}" for idx in range(reduced.shape[1])]
        if len(names) != reduced.shape[1]:
            names = [f"component_{idx}" for idx in range(reduced.shape[1])]

        provenance = copy.deepcopy(self.provenance)
        provenance.setdefault("map_reductions", []).append(
            {
                "reducer": type(reducer).__name__,
                "params": _jsonable(getattr(reducer, "get_params", dict)()),
                "n_features_out": int(reduced.shape[1]),
            }
        )

        return self._rebuild(map_features=reduced, map_feature_names=names, provenance=provenance)

    # -------------------------------------------------------------- plumbing

    def select_analyses(self, rows):
        """Return the dataset restricted to the analyses a mask or positions select.

        Parameters
        ----------
        rows : array_like
            A boolean mask over the rows, or an array of row positions.

        Returns
        -------
        :class:`FeatureSet`
            A dataset holding the selected rows, in the order they were given.
        """
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if len(rows) != len(self):
                raise ValueError(
                    f"A boolean mask must have one entry per analysis row ({len(self)})."
                )
            rows = np.flatnonzero(rows)
        rows = rows.astype(int, copy=False)

        return self._rebuild(
            map_features=_take_rows(self._map_features, rows),
            descriptor_features=_take_rows(self._descriptor_features, rows),
            ids=self.ids[rows],
            study_ids=self.study_ids[rows],
            target=None if self.target is None else self.target[rows],
        )

    def copy(self):
        """Return an independent copy of the dataset.

        Returns
        -------
        :class:`FeatureSet`
            A copy that shares no mutable state with this dataset.
        """
        return self._rebuild(
            map_features=self._map_features.copy(),
            descriptor_features=(
                None if self._descriptor_features is None else self._descriptor_features.copy()
            ),
            ids=self.ids.copy(),
            study_ids=self.study_ids.copy(),
            target=None if self.target is None else self.target.copy(),
        )

    def _rebuild(self, **changes):
        """Return a feature set of this type, keeping whatever ``changes`` omits."""
        kwargs = {
            "map_features": self._map_features,
            "ids": self.ids,
            "study_ids": self.study_ids,
            "descriptor_features": self._descriptor_features,
            "descriptor_names": self._descriptor_names,
            "target": self.target,
            "provenance": self.provenance,
            "masker": self.masker,
            "map_feature_names": self._map_feature_names,
        }
        kwargs.update(changes)
        positional = (kwargs.pop("map_features"), kwargs.pop("ids"), kwargs.pop("study_ids"))

        provenance = copy.deepcopy(kwargs.pop("provenance"))
        if "n_rows" in provenance:
            provenance["n_rows"] = len(positional[1])

        return type(self)(*positional, provenance=provenance, **kwargs)
