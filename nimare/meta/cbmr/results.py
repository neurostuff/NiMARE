"""Meta-analytic result class for CBMR."""

import copy
import logging

from nimare.meta.cbmr._helpers import (
    DEFAULT_GROUP_NAME,
    _normalize_named_pairwise_contrasts,
)
from nimare.meta.cbmr._torch import torch  # noqa: F401
from nimare.results import MetaResult

LGR = logging.getLogger(__name__)


class CBMRResult(MetaResult):
    """Meta-analytic result for CBMR with result-centered inference helpers.

    The same result class is used for both standard global-moderator CBMR and voxelwise
    moderator-effect CBMR. Model-specific inference is selected from the fitted estimator's
    ``moderator_effect`` attribute.
    """

    @property
    def moderator_effect(self):
        """Return the fitted moderator-effect parameterization."""
        return getattr(self.estimator, "moderator_effect", "global")

    @property
    def groups(self):
        """Return fitted group names in display order."""
        return tuple(getattr(self.estimator, "groups", ()) or ())

    @property
    def moderators(self):
        """Return fitted moderator names in display order."""
        return tuple(getattr(self.estimator, "moderators", ()) or ())

    @property
    def global_moderators(self):
        """Return fitted global moderator names in display order."""
        return tuple(getattr(self.estimator, "global_moderators", ()) or ())

    @property
    def voxelwise_moderators(self):
        """Return fitted voxelwise moderator names in display order."""
        return tuple(getattr(self.estimator, "voxelwise_moderators", ()) or ())

    @property
    def voxelwise_moderator_effect_map_names(self):
        """Return voxelwise moderator-effect map names."""
        return tuple(name for name in self.maps if name.startswith("voxelwiseModeratorEffect_"))

    def copy(self):
        """Return a copy of the CBMR result object."""
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

    def describe_inference_inputs(self):
        """Summarize the fitted groups, moderators, and moderator-effect type."""
        return {
            "groups": self.groups,
            "moderators": self.moderators,
            "global_moderators": self.global_moderators,
            "voxelwise_moderators": self.voxelwise_moderators,
            "moderator_effect": self.moderator_effect,
        }

    def describe_voxelwise_moderator_effect_maps(self):
        """Return simple summaries for voxelwise moderator-effect maps."""
        return {
            name: (float(values.min()), float(values.mean()), float(values.max()))
            for name, values in self.maps.items()
            if name.startswith("voxelwiseModeratorEffect_")
        }

    def get_inference(self, device=None, method=None, **kwargs):
        """Return a fitted inference engine for advanced CBMR use cases.

        Parameters
        ----------
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        inference_device = device or getattr(self.estimator, "device", "cpu")
        kwargs.setdefault(
            "incidence_threshold",
            getattr(self.estimator, "incidence_threshold", None),
        )
        # Imported here rather than at module scope: inference imports the estimator, which
        # imports this module, so a top-level import would close the cycle.
        from nimare.meta.cbmr.inference import CBMRInference

        inference = CBMRInference(
            device=inference_device,
            moderator_effect=self.moderator_effect,
            method=method,
            **kwargs,
        )
        inference.fit(self)
        return inference

    def infer(
        self,
        group_contrasts=False,
        moderator_contrasts=False,
        device=None,
        method=None,
        **kwargs,
    ):
        """Run CBMR inference from a fitted result.

        Parameters
        ----------
        group_contrasts : bool, dict, list, tuple, str, or None, optional
            Group homogeneity or comparison specification. Use ``False`` to skip group inference.
        moderator_contrasts : bool, dict, list, tuple, str, or None, optional
            Moderator effect or comparison specification. Use ``False`` to skip moderator
            inference.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        inference = self.get_inference(device=device, method=method, **kwargs)
        return inference.transform(
            t_con_groups=group_contrasts,
            t_con_moderators=moderator_contrasts,
        )

    def _infer_named_effects(
        self,
        source,
        contrasts=None,
        pairwise=False,
        device=None,
        method=None,
        **kwargs,
    ):
        """Run inference for named group or moderator effects through one shared path."""
        if source == "groups":
            if contrasts is None:
                contrasts = list(self.groups)
            group_contrasts = (
                _normalize_named_pairwise_contrasts(contrasts) if pairwise else contrasts
            )
            moderator_contrasts = False
        elif source == "moderators":
            if not self.moderators:
                raise ValueError("This CBMR result does not include moderators.")
            if contrasts is None:
                contrasts = list(self.moderators)
            group_contrasts = False
            moderator_contrasts = (
                _normalize_named_pairwise_contrasts(contrasts) if pairwise else contrasts
            )
        else:
            raise ValueError("source must be either 'groups' or 'moderators'.")

        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=moderator_contrasts,
            device=device,
            method=method,
            **kwargs,
        )

    def test_groups(self, groups=None, device=None, method=None, **kwargs):
        """Run one-group spatial homogeneity tests for the requested groups.

        Parameters
        ----------
        groups : list, tuple, str, or None, optional
            Group name or names to test. Defaults to all fitted groups.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "groups",
            contrasts=groups,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_groups(self, contrasts, device=None, method=None, **kwargs):
        """Run pairwise group-comparison tests using names or ``(group_a, group_b)`` tuples.

        Parameters
        ----------
        contrasts : list, tuple, or str
            Group comparison specification or specifications.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "groups",
            contrasts=contrasts,
            pairwise=True,
            device=device,
            method=method,
            **kwargs,
        )

    def test_moderators(self, moderators=None, device=None, method=None, **kwargs):
        """Test whether the requested moderator effects differ from zero.

        Parameters
        ----------
        moderators : list, tuple, str, or None, optional
            Moderator name or names to test. Defaults to all fitted moderators.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "moderators",
            contrasts=moderators,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_moderators(self, contrasts, device=None, method=None, **kwargs):
        """Run pairwise moderator-comparison tests using names or tuples.

        Parameters
        ----------
        contrasts : list, tuple, or str
            Moderator comparison specification or specifications.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "moderators",
            contrasts=contrasts,
            pairwise=True,
            device=device,
            method=method,
            **kwargs,
        )


class CBMRFormulaResult(CBMRResult):
    """Result of a formula-specified CBMR fit.

    Adds :meth:`test`, which replaces the positional contrast matrices the older interface
    required. Those were unreadable and silently order-dependent -- reorder the factor levels
    and the same matrix tests a different hypothesis.
    """

    def test(self, hypotheses, name=None, inplace=False, **covariance_kwargs):
        """Test one or more hypotheses about the fitted coefficients.

        Parameters
        ----------
        hypotheses : :obj:`str` or :obj:`list` of :obj:`str`
            Hypotheses written over term and level names, such as
            ``"diagnosis[schizophrenia] = diagnosis[depression]"`` or ``"s(avg_age) = 0"``.
            A list is tested jointly, as a generalized linear hypothesis, rather than one at a
            time.
        name : :obj:`str`, optional
            Label for the emitted map and table keys. Defaults to the hypotheses joined by
            ``";"``.
        inplace : :obj:`bool`, optional
            Whether to add the results to this object rather than to a copy. Default is False.
        **covariance_kwargs
            Passed to :meth:`~nimare.meta.cbmr.model.CBMRModel.covariance`. Pass
            ``method="sandwich"`` for standard errors robust to overdispersion and to correlation
            among an experiment's own foci.

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

        foci = estimator.inputs_["foci_by_experiment"][DEFAULT_GROUP_NAME]
        computed = evaluate_hypotheses(model, hypotheses, foci, name=name, **covariance_kwargs)

        target = self if inplace else self.copy()
        target.maps.update(computed["maps"])
        target.tables.update(computed["tables"])
        return target

    def describe_terms(self):
        """Return the per-term parameter budget of the fitted design."""
        estimator = self.estimator
        return estimator.bound_design.describe(estimator.predictor.n_bases)
