"""Splitting a bundle without letting a study span the split."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, check_cv


def study_folds(bunch, cv=5, rows=None):
    """Return a cross-validator that already knows which study each row came from.

    Analyses from one study are not independent, so a study belongs to exactly
    one side of any split. scikit-learn asks for that in two pieces -- a group
    splitter *and* ``groups=`` -- and only one of them is checked: a group
    splitter without ``groups`` raises, while ``groups`` without a group
    splitter is accepted and ignored. Binding the two together is what this is
    for, since there is then nothing to get half right. The result goes
    wherever a ``cv=`` goes.

    Parameters
    ----------
    bunch : :class:`sklearn.utils.Bunch`
        A bundle from :meth:`~nimare.studyset.Studyset.to_bunch`.
    cv : :obj:`int` or cross-validator, default=5
        How many folds, or any scikit-learn group splitter to bind, such as
        :class:`~sklearn.model_selection.LeaveOneGroupOut` or
        :class:`~sklearn.model_selection.GroupShuffleSplit`.
    rows : array_like, optional
        Row positions the splitter will be used over, by default None, meaning
        every row. Pass ``bunch.train`` for the inner loop of a nested
        cross-validation, along with the matching rows of ``data``.

    Returns
    -------
    cross-validator
        Usable as ``cv=`` for :func:`~sklearn.model_selection.cross_val_score`,
        :class:`~sklearn.model_selection.GridSearchCV` and the rest.

    See Also
    --------
    nimare.studyset.Studyset.to_bunch : Where ``groups`` comes from.

    Examples
    --------
    >>> cross_val_score(pipeline, bunch.data, bunch.target,  # doctest: +SKIP
    ...                 cv=study_folds(bunch))
    """
    groups = np.asarray(bunch.groups)
    if rows is not None:
        groups = groups[np.asarray(rows)]

    n_groups = len(np.unique(groups))
    if isinstance(cv, int) and cv > n_groups:
        raise ValueError(
            f"{cv} folds were asked for, but these rows come from {n_groups} "
            "studies and a study cannot be split across folds."
        )
    splitter = GroupKFold(n_splits=cv) if isinstance(cv, int) else check_cv(cv)
    return _BoundToStudies(splitter, groups)


class _BoundToStudies:
    """A cross-validator that supplies ``groups`` itself.

    Follows :class:`~sklearn.model_selection.PredefinedSplit`, which likewise
    carries its own assignment and ignores what it is passed.
    """

    def __init__(self, splitter, groups):
        self.splitter = splitter
        self.groups = groups

    def __repr__(self):
        """Return the splitter being bound and how many studies it holds."""
        return f"<{self.splitter!r} over {len(np.unique(self.groups))} studies>"

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return how many splits the bound splitter makes."""
        return self.splitter.get_n_splits(X, y, self.groups)

    def split(self, X, y=None, groups=None):
        """Yield ``(train, test)`` row positions, grouped by study.

        Raises
        ------
        :obj:`ValueError`
            If ``X`` has a different number of rows from the study labels,
            which means it is not the matrix the bundle described.
        """
        if _n_rows(X) != len(self.groups):
            raise ValueError(
                f"These study labels cover {len(self.groups)} rows, but X has "
                f"{_n_rows(X)}. Split the bundle before slicing it, or pass the same "
                "rows to study_folds as to the estimator."
            )
        return self.splitter.split(X, y, self.groups)


def _n_rows(X):
    """Return how many rows ``X`` has, frame, array or sparse matrix."""
    return X.shape[0] if hasattr(X, "shape") else len(X)
