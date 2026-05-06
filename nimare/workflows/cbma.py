"""Workflow for running an coordinates-based meta-analysis from a NiMARE database."""

import logging
import os.path as op

import numpy as np
import pandas as pd

from nimare.base import NiMAREBase
from nimare.correct import Corrector, FDRCorrector, FWECorrector
from nimare.diagnostics import Jackknife
from nimare.meta import ALE, ALESubtraction, MKDAChi2
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.transforms import threshold_image
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _check_ncores,
    _p_to_logp_values,
)
from nimare.workflows.base import Workflow, _check_input

LGR = logging.getLogger(__name__)


class CBMAWorkflow(Workflow):
    """Compose a coordinate-based meta-analysis workflow.

    .. versionchanged:: 0.1.2

        - `cbma_workflow` function was converted to CBMAWorkflow class.

    .. versionadded:: 0.0.14

    This workflow performs a coordinate-based meta-analysis, multiple comparison corrections,
    and diagnostics analyses on corrected meta-analytic z-score maps.

    Parameters
    ----------
    estimator : :class:`~nimare.meta.cbma.base.CBMAEstimator`, :obj:`str` {'ale', 'scale', \
    'mkdadensity', 'kda'}, optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.cbma.ale.ALE`.
    corrector : :class:`~nimare.correct.Corrector`, :obj:`str` {'montecarlo', 'fdr', \
    'bonferroni'}, optional
        Meta-analysis corrector. Default is :class:`~nimare.correct.FWECorrector`.
    diagnostics : :obj:`list` of :class:`~nimare.diagnostics.Diagnostics`, \
    :class:`~nimare.diagnostics.Diagnostics`, :obj:`str` {'jackknife', 'focuscounter'}, \
    optional
        List of meta-analysis diagnostic classes. A single diagnostic class can also be passed.
        Default is :class:`~nimare.diagnostics.FocusCounter`.
    voxel_thresh : :obj:`float` or None, optional
        An optional voxel-level threshold that may be applied to the ``target_image`` in the
        :class:`~nimare.diagnostics.Diagnostics` class to define clusters. This can be None or 0
        if the ``target_image`` is already thresholded (e.g., a cluster-level corrected map).
        If diagnostics are passed as initialized objects, this parameter will be ignored.
        Default is 1.65, which corresponds to p-value = .05, one-tailed.
    cluster_threshold : :obj:`int` or None, optional
        Cluster size threshold, in :term:`voxels<voxel>`.
        If None, then no cluster size threshold will be applied.
        If diagnostics are passed as initialized objects, this parameter will be ignored.
        Default is 10.
    output_dir : :obj:`str`, optional
        Output directory in which to save results. If the directory doesn't
        exist, it will be created. Default is None (the results are not saved).
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        If estimator, corrector, or diagnostics are passed as initialized objects, this parameter
        will be ignored.
        Default is 1.
    """

    # Options allows for string input
    _estm_base = CBMAEstimator
    _estm_options = ("ale", "scale", "mkdadensity", "kda")
    _corr_options = ("montecarlo", "fdr", "bonferroni")
    _diag_options = ("jackknife", "focuscounter")
    _mcc_method = "montecarlo"
    _estm_default = ALE
    _corr_default = FWECorrector
    _diag_default = Jackknife

    def fit(self, dataset, drop_invalid=True):
        """Fit Workflow to a Studyset-backed collection.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection to analyze.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Estimator fitting.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        LGR.info("Performing meta-analysis...")
        results = self.estimator.fit(dataset, drop_invalid=drop_invalid)

        return self._transform(results)


class PairwiseCBMAWorkflow(Workflow):
    """Base class for pairwise coordinate-based meta-analysis workflow methods.

    .. versionadded:: 0.1.2

    Parameters
    ----------
    estimator : :class:`~nimare.meta.cbma.base.PairwiseCBMAEstimator`, :obj:`str` \
    {'alesubtraction', 'mkdachi2'}, or optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.cbma.kda.MKDAChi2`.
    corrector : :class:`~nimare.correct.Corrector`, :obj:`str` {'montecarlo', 'fdr', \
    'bonferroni'} or optional
        Meta-analysis corrector. Default is :class:`~nimare.correct.FWECorrector`.
    diagnostics : :obj:`list` of :class:`~nimare.diagnostics.Diagnostics`, \
    :class:`~nimare.diagnostics.Diagnostics`, :obj:`str` {'jackknife', 'focuscounter'}, \
    or optional
        List of meta-analysis diagnostic classes. A single diagnostic class can also be passed.
        Default is :class:`~nimare.diagnostics.FocusCounter`.
    voxel_thresh : :obj:`float` or None, optional
        An optional voxel-level threshold that may be applied to the ``target_image`` in the
        :class:`~nimare.diagnostics.Diagnostics` class to define clusters. This can be None or 0
        if the ``target_image`` is already thresholded (e.g., a cluster-level corrected map).
        If diagnostics are passed as initialized objects, this parameter will be ignored.
        Default is 1.65, which corresponds to p-value = .05, one-tailed.
    cluster_threshold : :obj:`int` or None, optional
        Cluster size threshold, in :term:`voxels<voxel>`.
        If None, then no cluster size threshold will be applied.
        If diagnostics are passed as initialized objects, this parameter will be ignored.
        Default is 10.
    output_dir : :obj:`str`, optional
        Output directory in which to save results. If the directory doesn't
        exist, it will be created. Default is None (the results are not saved).
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        If estimator, corrector, or diagnostics are passed as initialized objects, this parameter
        will be ignored.
        Default is 1.
    """

    # Options allows for string input
    _estm_base = PairwiseCBMAEstimator
    _estm_options = ("alesubtraction", "mkdachi2")
    _corr_options = ("montecarlo", "fdr", "bonferroni")
    _diag_options = ("jackknife", "focuscounter")
    _mcc_method = "montecarlo"
    _estm_default = MKDAChi2
    _corr_default = FWECorrector
    _diag_default = Jackknife

    def fit(self, dataset1, dataset2, drop_invalid=True):
        """Fit Workflow to two Studyset-backed collections.

        Parameters
        ----------
        dataset1/dataset2 : :obj:`~nimare.nimads.Studyset` or \
                :obj:`~nimare.dataset.Dataset`
            Collection objects to analyze.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Estimator fitting.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        LGR.info("Performing meta-analysis...")
        results = self.estimator.fit(dataset1, dataset2, drop_invalid=drop_invalid)

        return self._transform(results)


def _corrected_map_priority(p_map_name):
    """Return a stable preference ordering for corrected main-effect p maps."""
    if "desc-size_level-cluster" in p_map_name or "Size_level-cluster" in p_map_name:
        scope_rank = 0
    elif "desc-mass_level-cluster" in p_map_name or "Mass_level-cluster" in p_map_name:
        scope_rank = 1
    elif "level-voxel" in p_map_name:
        scope_rank = 2
    else:
        scope_rank = 3

    if "_corr-FWE_method-montecarlo" in p_map_name:
        method_rank = 0
    elif "_corr-FWE_method-predictive" in p_map_name:
        method_rank = 1
    elif "_corr-FDR_method-indep" in p_map_name:
        method_rank = 2
    elif "_corr-FDR_method-negcorr" in p_map_name:
        method_rank = 3
    elif "_corr-FWE_method-bonferroni" in p_map_name:
        method_rank = 4
    else:
        method_rank = 5

    return (scope_rank, method_rank, p_map_name)


def _infer_corrected_main_effect_maps(result):
    """Infer the preferred corrected value/threshold map pair from a MetaResult."""
    candidates = []
    for z_map_name in result.maps:
        if not z_map_name.startswith("z_") or "_corr-" not in z_map_name:
            continue

        p_map_name = z_map_name.replace("z_", "p_", 1)
        logp_map_name = z_map_name.replace("z_", "logp_", 1)
        if p_map_name in result.maps:
            candidates.append((z_map_name, p_map_name, "p"))
        elif logp_map_name in result.maps:
            candidates.append((z_map_name, logp_map_name, "logp"))

    if candidates:
        return min(candidates, key=lambda triple: _corrected_map_priority(triple[1]))

    raise ValueError(
        "Unable to infer corrected main-effect value and threshold maps from the supplied "
        "result. "
        f"Available maps: {', '.join(sorted(result.maps))}"
    )


def _threshold_corrected_main_effect(
    result,
    z_map_name,
    threshold_map_name,
    threshold_kind,
    alpha,
):
    """Threshold a corrected main-effect map for workflow gating.

    This helper is estimator-agnostic. It works for ALE- or MKDA-family
    results and for either FWE- or FDR-corrected maps, as long as the
    corrected threshold map can be inferred from the supplied ``MetaResult``.
    """
    threshold_values = result.get_map(threshold_map_name, return_type="array")
    if threshold_kind == "p":
        threshold = alpha
        tail = "lower"
    else:
        threshold = -np.log10(alpha)
        tail = "upper"

    return threshold_image(
        result.get_map(z_map_name, return_type="array"),
        threshold=threshold,
        thresholding_values=threshold_values,
        tail=tail,
    ).astype(DEFAULT_FLOAT_DTYPE, copy=False)


class ContrastWorkflow(NiMAREBase):
    """Compose a masked contrast workflow for pairwise CBMA analyses.

    This workflow fits separate one-sample main-effect analyses for two groups, corrects
    and thresholds those main effects, and passes the thresholded maps to the pairwise
    estimator as directional inference masks. Positive group1 > group2 effects are
    evaluated only where group 1 has a surviving main effect, and negative group2 > group1
    effects are evaluated only where group 2 has a surviving main effect. This differs from
    a standard :class:`~nimare.meta.cbma.ale.ALESubtraction` run without inference maps,
    which evaluates differences across all voxels in the analysis mask.

    The workflow then thresholds the pairwise contrast p-values at ``alpha`` and stores the
    thresholded contrast z map. It also stores the thresholded group main-effect maps and a
    voxelwise-minimum conjunction of those maps. This mirrors the main-effect-gated ALE
    subtraction logic used in earlier ALE subtraction workflows :footcite:p:`laird2005ale`,
    while keeping the correction and gating steps explicit in NiMARE. This specific
    implementation was based on :footcite:t:`Frahm_Monimu_Hoffstaedter`.

    Parameters
    ----------
    main_estimator : :class:`~nimare.meta.cbma.base.CBMAEstimator`, \
:obj:`str` {'ale', 'scale', 'mkdadensity', 'kda'}, optional
        Estimator to use for computing main effects.
        Default is :class:`~nimare.meta.cbma.ale.ALE`.
    pairwise_estimator : :class:`~nimare.meta.cbma.base.PairwiseCBMAEstimator`, \
:obj:`str` {'alesubtraction', 'mkdachi2'}, optional
        Estimator to use for computing pairwise contrast.
        Default is :class:`~nimare.meta.cbma.ale.ALESubtraction`.
    corrector : :class:`~nimare.correct.Corrector`, \
:obj:`str` {'fdr', 'montecarlo', 'bonferroni'}, optional
        Corrector to use for correcting main effects.
        Default is :class:`~nimare.correct.FDRCorrector` with ``method="indep"``.
    alpha : :obj:`float`, optional
        Significance level to use for thresholding main effects and pairwise contrast.
        Default is 0.05.
    output_dir : :obj:`str`, optional
        Output directory in which to save results. If the directory doesn't
        exist, it will be created. Default is None (the results are not saved).
    generate_description : :obj:`bool`, optional
        Whether to generate workflow boilerplate and extract references for the
        returned result. Default is True.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If ``main_estimator``, ``pairwise_estimator``, or ``corrector`` are passed as
        already-initialized instances this parameter will be ignored for those objects.
        Default is 1.
    """

    _main_estm_options = ("ale", "scale", "mkdadensity", "kda")
    _pairwise_estm_options = ("alesubtraction", "mkdachi2")
    _corr_options = ("fdr", "montecarlo", "bonferroni")

    def __init__(
        self,
        main_estimator=ALE,
        pairwise_estimator=ALESubtraction,
        corrector=None,
        alpha=0.05,
        output_dir=None,
        generate_description=True,
        n_cores=1,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1; got {alpha}.")

        self.alpha = alpha
        self.output_dir = output_dir
        self.generate_description = generate_description
        self.n_cores = _check_ncores(n_cores)

        self.main_estimator = _check_input(
            main_estimator, CBMAEstimator, self._main_estm_options, n_cores=self.n_cores
        )

        self.pairwise_estimator = _check_input(
            pairwise_estimator,
            PairwiseCBMAEstimator,
            self._pairwise_estm_options,
            n_cores=self.n_cores,
        )

        if corrector is None:
            self.corrector = FDRCorrector(method="indep")
        else:
            self.corrector = _check_input(corrector, Corrector, self._corr_options)

    def _compute_main_effect_map(self, result, corr_result=None):
        """Compute the directional inference map used to gate pairwise contrast."""
        if corr_result is None:
            corr_result = self.corrector.transform(result)
        else:
            self._validate_cached_corrected_result(result, corr_result)
        z_map_name, threshold_map_name, threshold_kind = _infer_corrected_main_effect_maps(
            corr_result
        )
        thresholded_map = _threshold_corrected_main_effect(
            corr_result,
            z_map_name,
            threshold_map_name,
            threshold_kind,
            self.alpha,
        )
        return thresholded_map, corr_result

    def _validate_cached_corrected_result(self, result, corr_result):
        """Validate that a cached corrected result can be used with a main-effect result."""
        if corr_result.corrector is None:
            raise ValueError("corr_result must be a corrected MetaResult.")

        if type(corr_result.corrector) is not type(self.corrector):
            raise ValueError(
                "corr_result was generated with a different corrector type; "
                f"expected {type(self.corrector).__name__}, got "
                f"{type(corr_result.corrector).__name__}."
            )

        if corr_result.corrector.get_params(deep=False) != self.corrector.get_params(deep=False):
            raise ValueError("corr_result was generated with different corrector parameters.")

        result_ids = result.estimator.inputs_.get("id")
        corr_ids = corr_result.estimator.inputs_.get("id")
        if (
            result_ids is not None
            and corr_ids is not None
            and not np.array_equal(result_ids, corr_ids)
        ):
            raise ValueError("corr_result does not match the study ids in result.")

        result_n_voxels = next(
            (value.shape[0] for value in result.maps.values() if isinstance(value, np.ndarray)),
            None,
        )
        corr_n_voxels = next(
            (
                value.shape[0]
                for value in corr_result.maps.values()
                if isinstance(value, np.ndarray)
            ),
            None,
        )
        if (
            result_n_voxels is not None
            and corr_n_voxels is not None
            and result_n_voxels != corr_n_voxels
        ):
            raise ValueError(
                "corr_result maps are not aligned with result maps; "
                f"got {corr_n_voxels} and {result_n_voxels} voxels."
            )

    def _generate_description(self, group1_result, group2_result, pairwise_result):
        """Generate a workflow description that preserves component boilerplate."""
        sections = []
        if group1_result.description_:
            sections.append(f"Group 1 main-effect analysis: {group1_result.description_}")
        if group2_result.description_:
            sections.append(f"Group 2 main-effect analysis: {group2_result.description_}")
        if pairwise_result.description_:
            sections.append(f"Pairwise contrast analysis: {pairwise_result.description_}")

        sections.append(
            "Masked contrast workflow: The group 1 and group 2 main-effect maps were "
            f"corrected with {type(self.corrector).__name__} and thresholded at "
            f"alpha = {self.alpha}. The thresholded main-effect maps were passed to "
            f"{type(self.pairwise_estimator).__name__} as directional inference masks. "
            "Positive group1 > group2 effects were evaluated only where group 1 had a "
            "surviving main effect, and negative group2 > group1 effects were evaluated "
            "only where group 2 had a surviving main effect. The pairwise contrast "
            "p-values were then thresholded at the same alpha level. The output includes "
            "the thresholded group main-effect maps, a voxelwise-minimum conjunction map, "
            "and the thresholded pairwise contrast map. This follows the main-effect-gated "
            "ALE subtraction logic used in earlier ALE subtraction workflows "
            "\\citep{laird2005ale} and the JALE implementation "
            "\\citep{Frahm_Monimu_Hoffstaedter}."
        )
        return " ".join(sections)

    def _contrast_keys(self):
        if isinstance(self.pairwise_estimator, ALESubtraction):
            return "p_desc-group1MinusGroup2", "z_desc-group1MinusGroup2"
        elif isinstance(self.pairwise_estimator, MKDAChi2):
            return "p_desc-association", "z_desc-association"
        else:
            raise ValueError(
                f"Unable to determine contrast map keys for pairwise estimator of type "
                f"{type(self.pairwise_estimator).__name__}."
            )

    def _save_result(self, result):
        if self.output_dir is None:
            return

        LGR.info(f"Saving contrast workflow outputs to {self.output_dir}...")
        result.save_maps(output_dir=self.output_dir)
        result.save_tables(output_dir=self.output_dir)
        with open(op.join(self.output_dir, "boilerplate.txt"), "w") as fo:
            fo.write(result.description_)
        with open(op.join(self.output_dir, "references.bib"), "w") as fo:
            fo.write(result.bibtex_)

    def fit(
        self,
        dataset1,
        dataset2,
        drop_invalid=True,
        result1=None,
        result2=None,
        corr_result1=None,
        corr_result2=None,
    ):
        """Fit the masked contrast workflow to two Studyset-backed collections.

        Parameters
        ----------
        dataset1, dataset2 : :class:`~nimare.dataset.Dataset`
            The two groups to contrast.
        drop_invalid : bool, optional
            Passed through to the main estimator when fitting from scratch.
        result1, result2 : :class:`~nimare.results.MetaResult`, optional
            Pre-fitted main-effect results for each group.  When supplied the
            per-group ``main_estimator.fit()`` call is skipped and the cached
            MA maps are passed directly to the pairwise estimator, avoiding
            redundant kernel convolution.
        corr_result1, corr_result2 : :class:`~nimare.results.MetaResult`, optional
            Pre-computed corrected main-effect results (i.e. the output of
            ``corrector.transform(result)``).  When supplied the per-group
            ``corrector.transform()`` call inside ``_compute_main_effect_map``
            is skipped.  If corresponding ``result1``/``result2`` values are
            not provided, the workflow fits them before validating and using
            the cached corrected results.
        """
        if result1 is None:
            LGR.info("Fitting group 1 main effect...")
            result1 = self.main_estimator.fit(dataset1, drop_invalid=drop_invalid)
        if result2 is None:
            LGR.info("Fitting group 2 main effect...")
            result2 = self.main_estimator.fit(dataset2, drop_invalid=drop_invalid)

        group1_result = result1
        group2_result = result2
        group1_map, group1_corr_result = self._compute_main_effect_map(
            group1_result, corr_result=corr_result1
        )
        group2_map, group2_corr_result = self._compute_main_effect_map(
            group2_result, corr_result=corr_result2
        )

        LGR.info("Fitting masked pairwise contrast...")
        ma_maps1 = group1_result.estimator.inputs_.get("ma_maps")
        ma_maps2 = group2_result.estimator.inputs_.get("ma_maps")
        pairwise_result = self.pairwise_estimator.fit(
            dataset1,
            dataset2,
            drop_invalid=drop_invalid,
            ma_maps1=ma_maps1,
            ma_maps2=ma_maps2,
            inference_map1=group1_map,
            inference_map2=group2_map,
        )

        contrast_p_key, contrast_z_key = self._contrast_keys()
        contrast_p = pairwise_result.get_map(contrast_p_key, return_type="array")
        contrast_z = pairwise_result.get_map(contrast_z_key, return_type="array")
        thresholded_z = np.where(contrast_p <= self.alpha, contrast_z, 0).astype(
            DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        thresholded_p = np.where(thresholded_z != 0, contrast_p, 1).astype(
            DEFAULT_FLOAT_DTYPE,
            copy=False,
        )

        result = pairwise_result.copy()
        result.maps["z_desc-group1MainEffect"] = group1_map
        result.maps["z_desc-group2MainEffect"] = group2_map
        result.maps["z_desc-conjunction"] = np.minimum(group1_map, group2_map).astype(
            DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        result.maps["p_desc-contrast"] = thresholded_p
        result.maps["z_desc-contrast"] = thresholded_z
        result.maps["logp_desc-contrast"] = _p_to_logp_values(
            thresholded_p,
            dtype=DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        result.tables["contrast_tab-metadata"] = pd.DataFrame(
            [
                {
                    "alpha": self.alpha,
                    "main_estimator": type(self.main_estimator).__name__,
                    "pairwise_estimator": type(self.pairwise_estimator).__name__,
                    "corrector": type(self.corrector).__name__,
                }
            ]
        )
        if self.generate_description:
            result.description_ = self._generate_description(
                group1_corr_result,
                group2_corr_result,
                pairwise_result,
            )
        else:
            result.description_ = ""
        self._save_result(result)
        return result
