"""Meta-analytic result class for CBMR."""

import copy
import logging

from nimare.meta.cbmr._helpers import _normalize_named_pairwise_contrasts
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
