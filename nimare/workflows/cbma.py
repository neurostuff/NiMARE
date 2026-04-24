"""Workflow for running an coordinates-based meta-analysis from a NiMARE database."""

import logging
import os.path as op

import numpy as np
import pandas as pd

from nimare.base import NiMAREBase
from nimare.correct import Corrector, FWECorrector
from nimare.diagnostics import Jackknife
from nimare.meta import ALE, ALESubtraction, MKDAChi2, MKDADensity
from nimare.meta.cbma.ale import _jale_threshold_z_clusters
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.stats import safe_logp
from nimare.transforms import threshold_image
from nimare.utils import DEFAULT_FLOAT_DTYPE, _check_ncores
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


def _threshold_ale_montecarlo_main_effect(result, z_map_name, alpha):
    """Threshold ALE Monte Carlo FWE results using cached null distributions."""
    estimator = result.estimator
    percentile = 100.0 * (1.0 - alpha)

    if (
        z_map_name == "z_desc-size_level-cluster_corr-FWE_method-montecarlo"
        and "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
        in estimator.null_distributions_
    ):
        cluster_threshold = np.percentile(
            estimator.null_distributions_[
                "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
            ],
            percentile,
        )
        voxel_thresh = result.corrector.parameters.get("voxel_thresh", 0.001)
        thresholded_z, _ = _jale_threshold_z_clusters(
            result.get_map("z", return_type="array"),
            estimator.masker,
            voxel_thresh=voxel_thresh,
            cluster_size_threshold=cluster_threshold,
        )
        return thresholded_z.astype(DEFAULT_FLOAT_DTYPE, copy=False)

    if (
        z_map_name == "z_level-voxel_corr-FWE_method-montecarlo"
        and "values_level-voxel_corr-fwe_method-montecarlo" in estimator.null_distributions_
    ):
        stat_threshold = np.percentile(
            estimator.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"],
            percentile,
        )
        return threshold_image(
            result.get_map("z", return_type="array"),
            threshold=stat_threshold,
            thresholding_values=result.get_map("stat", return_type="array"),
            tail="upper",
        ).astype(DEFAULT_FLOAT_DTYPE, copy=False)

    return None


class ContrastWorkflow(NiMAREBase):
    """Compose a masked contrast workflow for pairwise CBMA analyses.

    Notes
    -----
    When ``mode="ALE"``, corrected main-effect maps are used to gate
    directional pairwise inference. This mirrors the masked-contrast pattern
    used by related ALE software; see that project's README for background and
    references.
    """

    def __init__(
        self,
        mode="ALE",
        main_estimator=None,
        pairwise_estimator=None,
        corrector=None,
        alpha=0.05,
        output_dir=None,
        n_cores=1,
    ):
        mode = mode.upper()
        if mode not in ("ALE", "MKDA"):
            raise ValueError("mode must be 'ALE' or 'MKDA'.")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1; got {alpha}.")

        self.mode = mode
        self.alpha = alpha
        self.output_dir = output_dir
        self.n_cores = _check_ncores(n_cores)

        if mode == "ALE":
            default_main = ALE
            default_pairwise = ALESubtraction
            main_options = ("ale",)
            pairwise_options = ("alesubtraction",)
        else:
            default_main = MKDADensity
            default_pairwise = MKDAChi2
            main_options = ("mkdadensity",)
            pairwise_options = ("mkdachi2",)

        if main_estimator is None:
            main_estimator = default_main(n_cores=self.n_cores)
        else:
            main_estimator = _check_input(
                main_estimator, CBMAEstimator, main_options, n_cores=self.n_cores
            )

        if pairwise_estimator is None:
            pairwise_estimator = default_pairwise(n_cores=self.n_cores)
        else:
            pairwise_estimator = _check_input(
                pairwise_estimator,
                PairwiseCBMAEstimator,
                pairwise_options,
                n_cores=self.n_cores,
            )

        if corrector is None:
            corrector = FWECorrector(method="montecarlo", n_cores=self.n_cores)
        else:
            corrector = _check_input(
                corrector,
                Corrector,
                ("montecarlo", "fdr", "bonferroni"),
                n_cores=self.n_cores,
            )

        self.main_estimator = main_estimator
        self.pairwise_estimator = pairwise_estimator
        self.corrector = corrector

    def _compute_main_effect_map(self, result):
        """Compute the directional inference map used to gate pairwise contrast."""
        corr_result = self.corrector.transform(result)
        z_map_name, threshold_map_name, threshold_kind = _infer_corrected_main_effect_maps(
            corr_result
        )

        if (
            self.mode == "ALE"
            and isinstance(corr_result.estimator, ALE)
            and isinstance(corr_result.corrector, FWECorrector)
            and corr_result.corrector.method == "montecarlo"
        ):
            thresholded = _threshold_ale_montecarlo_main_effect(
                corr_result, z_map_name, self.alpha
            )
            if thresholded is not None:
                return thresholded

        threshold_values = corr_result.get_map(threshold_map_name, return_type="array")
        if threshold_kind == "p":
            threshold = self.alpha
            tail = "lower"
        else:
            threshold = -np.log10(self.alpha)
            tail = "upper"

        return threshold_image(
            corr_result.get_map(z_map_name, return_type="array"),
            threshold=threshold,
            thresholding_values=threshold_values,
            tail=tail,
        ).astype(DEFAULT_FLOAT_DTYPE, copy=False)

    def _contrast_keys(self):
        if self.mode == "ALE":
            return "p_desc-group1MinusGroup2", "z_desc-group1MinusGroup2"
        return "p_desc-association", "z_desc-association"

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

    def fit(self, dataset1, dataset2, drop_invalid=True):
        """Fit the masked contrast workflow to two Studyset-backed collections."""
        LGR.info("Fitting group 1 main effect...")
        group1_result = self.main_estimator.fit(dataset1, drop_invalid=drop_invalid)
        LGR.info("Fitting group 2 main effect...")
        group2_result = self.main_estimator.fit(dataset2, drop_invalid=drop_invalid)

        group1_map = self._compute_main_effect_map(group1_result)
        group2_map = self._compute_main_effect_map(group2_result)

        LGR.info("Fitting masked pairwise contrast...")
        pairwise_result = self.pairwise_estimator.fit(
            dataset1,
            dataset2,
            drop_invalid=drop_invalid,
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
        result.maps["logp_desc-contrast"] = safe_logp(thresholded_p).astype(
            DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        result.tables["contrast_tab-metadata"] = pd.DataFrame(
            [
                {
                    "mode": self.mode,
                    "alpha": self.alpha,
                    "main_estimator": type(self.main_estimator).__name__,
                    "pairwise_estimator": type(self.pairwise_estimator).__name__,
                    "corrector": type(self.corrector).__name__,
                }
            ]
        )
        result.description_ = (
            f"A masked {self.mode} contrast workflow was run in NiMARE using "
            f"{type(self.main_estimator).__name__} main effects, "
            f"{type(self.corrector).__name__} main-effect correction, and "
            f"{type(self.pairwise_estimator).__name__} pairwise inference."
        )
        self._save_result(result)
        return result
