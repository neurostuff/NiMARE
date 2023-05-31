"""Base class for workflow."""
import copy
import itertools
import logging
import os.path as op
from abc import abstractmethod

from nimare.base import NiMAREBase
from nimare.utils import _check_ncores

LGR = logging.getLogger(__name__)


class Workflow(NiMAREBase):
    """Base class for workflow methods.

    .. versionadded:: 0.1.1

    Parameters
    ----------
    estimator : :class:`~nimare.base.CBMAEstimator`, :obj:`str` {'ale', 'scale', 'mkdadensity', \
    'kda'}, or optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.cbma.ale.ALE`.
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

    def __init__(
        self,
        estimator=None,
        corrector=None,
        diagnostics=None,
        voxel_thresh=1.65,
        cluster_threshold=10,
        output_dir=None,
        n_cores=1,
    ):
        self.voxel_thresh = voxel_thresh
        self.cluster_threshold = cluster_threshold
        self.output_dir = output_dir
        self.n_cores = _check_ncores(n_cores)
        self._preprocess_input(estimator, corrector, diagnostics)

    @abstractmethod
    def fit(self, dataset):
        """Apply estimation to dataset and output results."""

    @abstractmethod
    def _preprocess_input(self, estimator, corrector, diagnostics):
        """Check estimator, corrector, diagnostics classes."""

    def _transform(self, result):
        """Implement the correction procedure and perform diagnostics.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            MetaResult object from which to extract the p value map and Estimator.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Estimator, Corrector, and Diagnostics fitting with label maps,
            cluster and diagnostic tables.
        """
        LGR.info("Performing correction on meta-analysis...")
        corr_result = self.corrector.transform(result)

        LGR.info("Performing diagnostics on corrected meta-analyses...")
        # Perform diagnostic only on desc-mass when using montecarlo correction
        corr_method = corr_result.get_params()["corrector__method"]
        modalities = ["_desc-mass", "_corr-"] if corr_method == "montecarlo" else ["_corr-"]
        img_keys = [
            img_key
            for img_key in corr_result.maps.keys()
            if img_key.startswith("z_") and all(mod in img_key for mod in modalities)
        ]

        for img_key, diagnostic in itertools.product(img_keys, self.diagnostics):
            # Work on copy of diagnostic:
            diagnostic_cp = copy.deepcopy(diagnostic)
            diagnostic_cp = diagnostic_cp.set_params(target_image=img_key)
            corr_result = diagnostic_cp.transform(corr_result)

        if self.output_dir is not None:
            LGR.info(f"Saving meta-analytic maps, tables and boilerplate to {self.output_dir}...")
            corr_result.save_maps(output_dir=self.output_dir)
            corr_result.save_tables(output_dir=self.output_dir)

            boilerplate = corr_result.description_
            with open(op.join(self.output_dir, "boilerplate.txt"), "w") as fo:
                fo.write(boilerplate)

            bibtex = corr_result.bibtex_
            with open(op.join(self.output_dir, "references.bib"), "w") as fo:
                fo.write(bibtex)

        LGR.info("Workflow completed.")
        return corr_result
