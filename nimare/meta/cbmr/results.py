"""Meta-analytic result class for CBMR."""

import copy
import logging
import re

import numpy as np

from nimare.meta.cbmr._helpers import DEFAULT_GROUP_NAME
from nimare.meta.cbmr._torch import torch  # noqa: F401
from nimare.results import MetaResult

LGR = logging.getLogger(__name__)


class CBMRResult(MetaResult):
    """Result of a CBMR fit.

    Hypotheses are tested by name through :meth:`test`, over the terms and levels of the fitted
    design. The interface this replaced took contrast matrices positionally, which were
    unreadable and silently order-dependent: reorder the factor levels and the same matrix tests
    a different hypothesis.
    """

    def copy(self):
        """Return a copy of the result."""
        new = CBMRResult(
            estimator=self.estimator,
            corrector=self.corrector,
            diagnostics=self.diagnostics,
            mask=self.masker,
            maps=copy.deepcopy(self.maps),
            tables=copy.deepcopy(self.tables),
            description=self.description_,
        )
        new.metadata = copy.deepcopy(self.metadata)
        return new

    @property
    def design(self):
        """Return the fitted design."""
        return self.estimator.bound_design.design

    @property
    def terms(self):
        """Return the fitted terms, in formula order."""
        return self.estimator.bound_design.terms

    def test(
        self,
        hypotheses=None,
        name=None,
        inplace=False,
        *,
        term=None,
        method=None,
        **covariance_kwargs,
    ):
        """Test one or more hypotheses about the fitted coefficients.

        Parameters
        ----------
        hypotheses : :obj:`str` or :obj:`list` of :obj:`str`, optional
            Hypotheses written over level names, in the notation the result maps use::

                result.test("schizophrenia = depression")
                result.test(["a = b", "b = c"])       # tested jointly, as a GLH
                result.test("s(avg_age) = 0")
                result.test("2 * a = b + c")          # arithmetic

            Parsed by :meth:`patsy.DesignInfo.linear_constraint`, the same parser statsmodels
            uses, so arithmetic, bare difference expressions and non-zero right-hand sides all
            work. Give this or ``term`` and ``method``, not both.
        name : :obj:`str`, optional
            Overrides the label derived from the hypothesis. Rarely needed.
        term, method : :obj:`str`, optional
            Generate a family of contrasts over one term rather than naming them one at a time::

                result.test(term="diagnosis", method="pairwise")

            ``"pairwise"`` compares every pair, ``"reference"`` every level against the first,
            ``"consecutive"`` each level against the previous, and ``"zero"`` each against zero.
            Each contrast is emitted under its own label.
        inplace : :obj:`bool`, optional
            Whether to add the results to this object rather than to a copy. Default is False.
        **covariance_kwargs
            Passed to :meth:`~nimare.meta.cbmr.model.CBMRModel.covariance`. Pass
            ``cov_type="sandwich"`` for standard errors robust to overdispersion and to
            correlation among an experiment's own foci. Distinct from ``method`` above, which
            names a contrast family.

        Returns
        -------
        :class:`CBMRFormulaResult`
            The result carrying the new ``z_``, ``p_``, ``logp_`` and ``chiSquare_`` entries.
        """
        from nimare.meta.cbmr.contrasts import evaluate_hypotheses

        estimator = self.estimator
        model = getattr(estimator, "cbmr_model", None)
        if model is None:
            raise ValueError(
                "This result has no fitted model to test; it was not produced by CBMR.fit."
            )

        foci = estimator.inputs_["foci"]
        computed = evaluate_hypotheses(
            model,
            hypotheses,
            foci,
            name=name,
            term=term,
            method=method,
            **covariance_kwargs,
        )

        target = self if inplace else self.copy()
        target.maps.update(computed["maps"])
        target.tables.update(computed["tables"])
        return target

    def describe_terms(self):
        """Return the per-term parameter budget of the fitted design."""
        estimator = self.estimator
        return estimator.bound_design.describe(estimator.predictor.n_bases)

    def moderator_effect_maps(self, moderators=None, unit_change=1.0, inplace=False):
        """Add relative-intensity and intensity-difference maps for spatial moderators.

        A spatial moderator's coefficient map is a derivative of *log* intensity, which is hard
        to read directly. These express it on two interpretable scales, for a stated change in
        the moderator:

        Relative Intensity (RI)
            ``exp(unit_change * coefficient)`` -- the multiplicative factor applied to the
            intensity. RI of 1.2 means 20% more foci expected at that voxel.
        Intensity Difference (ID)
            ``baseline * (RI - 1)`` -- the same effect in foci per voxel, against each baseline
            map. Small where the baseline is small, which is what makes it the honest one to
            threshold: a large ratio in a region nobody reports is not a finding.

        Parameters
        ----------
        moderators : :obj:`str`, sequence of :obj:`str`, or None, optional
            Spatial moderator terms to diagnose, by term or column label. If None, every spatial
            term that is not the baseline is used.
        unit_change : :obj:`float`, optional
            Change in the moderator to visualize, in its own units -- so 1.0 on a standardized
            moderator is one standard deviation. Default is 1.0.
        inplace : :obj:`bool`, optional
            Whether to add the maps to this object rather than to a copy. Default is False.

        Returns
        -------
        :class:`CBMRFormulaResult`
            The result carrying the new ``relativeIntensity_`` and ``intensityDifference_`` maps.
        """
        estimator = self.estimator
        design = getattr(estimator, "bound_design", None)
        if design is None:
            raise ValueError("This result was not produced by CBMR.fit.")

        unit_change = float(unit_change)
        coefficients = estimator.cbmr_model.fitted_coefficients()
        bases = estimator.predictor.bases

        candidates = [b for b in design.blocks if b.term.spatial and not b.is_baseline]
        if not candidates:
            raise ValueError(
                f"The design {design.design} has no spatial moderator to diagnose. Mark a "
                "moderator with s() to give it a coefficient map."
            )
        if moderators is not None:
            wanted = {moderators} if isinstance(moderators, str) else set(moderators)
            candidates = [
                b
                for b in candidates
                if str(b.term) in wanted or b.term.expr in wanted or set(b.column_names) & wanted
            ]
            if not candidates:
                available = sorted(str(b.term) for b in design.blocks if b.term.spatial)
                raise ValueError(
                    f"No spatial moderator matched {sorted(wanted)}. Available: "
                    f"{', '.join(available)}."
                )

        # Taken from the published intensity maps rather than recomputed, so that
        # ID == baseline * (RI - 1) holds exactly against the map the user is looking at.
        # Results store maps in single precision, so a float64 recomputation would disagree in
        # the eighth digit and the identity would only hold approximately.
        baselines = {}
        for block in design.baseline_blocks:
            values = np.atleast_2d(coefficients[str(block.term)])
            for index, column in enumerate(block.column_names):
                label = _column_label(column)
                published = self.maps.get(f"spatialIntensity_group-{label}")
                baselines[label] = (
                    np.asarray(published, dtype=float)
                    if published is not None
                    else np.exp(values[index] @ bases.T)
                )

        unit_label = f"{unit_change:g}"
        target = self if inplace else self.copy()
        for block in candidates:
            values = np.atleast_2d(coefficients[str(block.term)])
            for index, column in enumerate(block.column_names):
                label = _column_label(column)
                # Clipped before exponentiating: a moderator coefficient can be large where the
                # data is thin, and exp() of it would overflow the stored map to infinity.
                relative = np.exp(np.clip(unit_change * (values[index] @ bases.T), -100, 100))
                target.maps[f"relativeIntensity_{label}_unit-{unit_label}"] = relative
                for group, baseline in baselines.items():
                    key = f"intensityDifference_{label}_unit-{unit_label}_group-{group}"
                    target.maps[key] = baseline * (relative - 1.0)
        return target

    def plot_moderator_effects(
        self, moderator=None, unit_change=1.0, group=None, threshold=None, **plot_kwargs
    ):
        """Plot a spatial moderator's relative intensity beside its intensity difference.

        The two answer different questions and are misleading apart: RI says how much the rate
        changes proportionally, ID says how much it changes in foci. A region with a large RI and
        a negligible ID is a large proportional change to almost nothing.

        Parameters
        ----------
        moderator : :obj:`str` or None, optional
            Moderator term or column label. Defaults to the only spatial moderator, and errors
            if there is more than one.
        unit_change : :obj:`float`, optional
            Change in the moderator to visualize. Default is 1.0.
        group : :obj:`str` or None, optional
            Baseline group for the intensity-difference panel. Defaults to the only baseline.
        threshold : :obj:`float` or None, optional
            Passed to :func:`nilearn.plotting.plot_stat_map`.
        **plot_kwargs
            Passed to :func:`nilearn.plotting.plot_stat_map`.

        Returns
        -------
        :class:`matplotlib.figure.Figure`
        """
        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_stat_map

        diagnosed = self.moderator_effect_maps(moderators=moderator, unit_change=unit_change)
        unit_label = f"{float(unit_change):g}"

        relative_names = [
            name
            for name in diagnosed.maps
            if name.startswith("relativeIntensity_") and name.endswith(f"unit-{unit_label}")
        ]
        if len(relative_names) != 1:
            raise ValueError(
                f"Expected one moderator to plot, got {len(relative_names)}. Name one with the "
                "moderator argument."
            )
        relative_name = relative_names[0]
        difference_names = [
            name
            for name in diagnosed.maps
            if name.startswith("intensityDifference_") and f"unit-{unit_label}" in name
        ]
        if group is not None:
            difference_names = [n for n in difference_names if n.endswith(f"group-{group}")]
        if len(difference_names) != 1:
            groups = sorted(name.split("group-")[-1] for name in difference_names)
            raise ValueError(
                f"Expected one baseline group to plot, got {groups}. Name one with the group "
                "argument."
            )

        figure, axes = plt.subplots(2, 1, figsize=(10, 6))
        for axis, name, title in (
            (axes[0], relative_name, f"Relative intensity, {unit_label} unit(s)"),
            (axes[1], difference_names[0], f"Intensity difference, {unit_label} unit(s)"),
        ):
            plot_stat_map(
                diagnosed.get_map(name),
                axes=axis,
                title=title,
                threshold=threshold,
                **plot_kwargs,
            )
        return figure


def _column_label(column):
    """Turn a patsy column name into a readable label, as the map names use."""
    parts = []
    for piece in column.split(":"):
        match = re.search(r"\[(?:T\.)?([^\]]+)\]", piece)
        parts.append(match.group(1) if match else piece)
    label = "-".join(parts)
    return DEFAULT_GROUP_NAME if label in ("1", "Intercept") else label
