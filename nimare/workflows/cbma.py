"""Workflow for running an coordinates-based meta-analysis from a NiMARE database."""

import logging

from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.diagnostics import Jackknife
from nimare.meta import ALE, MKDAChi2
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.utils import _check_type
from nimare.workflows.base import Workflow

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
        """Fit Workflow to a Dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Estimator fitting.
        """
        # Check dataset type
        dataset = _check_type(dataset, Dataset)

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
        """Fit Workflow to two Datasets.

        Parameters
        ----------
        dataset1/dataset2 : :obj:`~nimare.dataset.Dataset`
            Dataset objects to analyze.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Estimator fitting.
        """
        # Check dataset type
        dataset1 = _check_type(dataset1, Dataset)
        dataset2 = _check_type(dataset2, Dataset)

        LGR.info("Performing meta-analysis...")
        results = self.estimator.fit(dataset1, dataset2, drop_invalid=drop_invalid)

        return self._transform(results)
