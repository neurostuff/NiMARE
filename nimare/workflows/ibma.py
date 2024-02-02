"""Workflow for running an image-based meta-analysis from a NiMARE database."""

import logging

from nimare.correct import FDRCorrector
from nimare.dataset import Dataset
from nimare.diagnostics import Jackknife
from nimare.meta.ibma import IBMAEstimator, Stouffers
from nimare.transforms import ImageTransformer
from nimare.utils import _check_type
from nimare.workflows.base import Workflow

LGR = logging.getLogger(__name__)


class IBMAWorkflow(Workflow):
    """Compose a coordinate-based meta-analysis workflow.

    .. versionadded:: 0.2.0

    This workflow performs a coordinate-based meta-analysis, multiple comparison corrections,
    and diagnostics analyses on corrected meta-analytic z-score maps.

    Parameters
    ----------
    estimator : :class:`~nimare.meta.ibma.IBMAEstimator`, :obj:`str` {'stouffers', 'fishers', \
    'hedges', 'permutedols', 'wleastsquares', 'dersimonianlaird', 'samplesizebl'. 'variancebl'}, \
    or optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.cbma.ale.ALE`.
    corrector : :class:`~nimare.correct.Corrector`, :obj:`str` {'montecarlo', 'fdr', \
    'bonferroni'} or optional
        Meta-analysis corrector. Default is :class:`~nimare.correct.FDRCorrector`.
    diagnostics : :obj:`list` of :class:`~nimare.diagnostics.Diagnostics`, \
    :class:`~nimare.diagnostics.Diagnostics`, :obj:`str` {'jackknife'}, \
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
    _estm_base = IBMAEstimator
    _estm_options = (
        "stouffers",
        "fishers",
        "permutedols",
        "wleastsquares",
        "dersimonianlaird",
        "hedges",
        "samplesizebl",
        "variancebl",
    )
    _corr_options = ("montecarlo", "fdr", "bonferroni")
    _diag_options = "jackknife"
    _mcc_method = "indep"
    _estm_default = Stouffers
    _corr_default = FDRCorrector
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

        # Calculate missing images. Possible targets: {"z", "p", "beta", "varcope"}.
        # Infer from self.estimator._required_inputs
        targets = [
            target
            for _, (type_, target) in self.estimator._required_inputs.items()
            if type_ == "image"
        ]
        xformer = ImageTransformer(target=targets)
        dataset = xformer.transform(dataset)

        LGR.info("Performing meta-analysis...")
        results = self.estimator.fit(dataset, drop_invalid=drop_invalid)

        return self._transform(results)
