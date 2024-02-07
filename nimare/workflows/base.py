"""Base class for workflow."""

import copy
import itertools
import logging
import os.path as op
from abc import abstractmethod

from nimare.base import NiMAREBase
from nimare.correct import Corrector, FDRCorrector, FWECorrector
from nimare.diagnostics import Diagnostics, FocusCounter, Jackknife
from nimare.meta import ALE, KDA, SCALE, ALESubtraction, MKDAChi2, MKDADensity
from nimare.meta.cbma.base import PairwiseCBMAEstimator
from nimare.meta.ibma import (
    DerSimonianLaird,
    Fishers,
    Hedges,
    PermutedOLS,
    SampleSizeBasedLikelihood,
    Stouffers,
    VarianceBasedLikelihood,
    WeightedLeastSquares,
)
from nimare.utils import _check_ncores, _check_type

LGR = logging.getLogger(__name__)

# Match a string to a class name without initializing the class.
STR_TO_CLASS = {
    "ale": ALE,
    "scale": SCALE,
    "mkdadensity": MKDADensity,
    "kda": KDA,
    "mkdachi2": MKDAChi2,
    "alesubtraction": ALESubtraction,
    "stouffers": Stouffers,
    "fishers": Fishers,
    "permutedols": PermutedOLS,
    "wleastsquares": WeightedLeastSquares,
    "dersimonianlaird": DerSimonianLaird,
    "hedges": Hedges,
    "samplesizebl": SampleSizeBasedLikelihood,
    "variancebl": VarianceBasedLikelihood,
    "montecarlo": FWECorrector,
    "fdr": FDRCorrector,
    "bonferroni": FWECorrector,
    "jackknife": Jackknife,
    "focuscounter": FocusCounter,
}


def _check_input(obj, clss, options, **kwargs):
    """Check input for workflow functions."""
    if isinstance(obj, str):
        if obj not in options:
            raise ValueError(f'"{obj}" of kind string must be {", ".join(options)}')

        # Get the class from the string
        obj_str = obj
        obj = STR_TO_CLASS[obj_str]

        # Add the method to the kwargs if it's a FWECorrector
        if obj == FWECorrector:
            kwargs["method"] = obj_str

    return _check_type(obj, clss, **kwargs)


class Workflow(NiMAREBase):
    """Base class for workflow methods.

    .. versionadded:: 0.1.2
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

    def _preprocess_input(self, estimator, corrector, diagnostics):
        if not isinstance(diagnostics, list) and diagnostics is not None:
            diagnostics = [diagnostics]

        # Check inputs and set defaults if input is None
        estimator = (
            self._estm_default(n_cores=self.n_cores)
            if estimator is None
            else _check_input(estimator, self._estm_base, self._estm_options, n_cores=self.n_cores)
        )

        corrector = (
            self._corr_default(method=self._mcc_method, n_cores=self.n_cores)
            if corrector is None
            else _check_input(corrector, Corrector, self._corr_options, n_cores=self.n_cores)
        )

        diag_kwargs = {
            "voxel_thresh": self.voxel_thresh,
            "cluster_threshold": self.cluster_threshold,
            "n_cores": self.n_cores,
        }
        if diagnostics is None:
            diagnostics = [self._diag_default(**diag_kwargs)]
        else:
            diagnostics = [
                _check_input(diagnostic, Diagnostics, self._diag_options, **diag_kwargs)
                for diagnostic in diagnostics
            ]

        pairwaise_workflow = self.__class__.__name__ == "PairwiseCBMAWorkflow"
        if (not pairwaise_workflow) and isinstance(estimator, PairwiseCBMAEstimator):
            raise AttributeError('"CBMAWorkflow" does not work with pairwise Estimators.')

        self.estimator = estimator
        self.corrector = corrector
        self.diagnostics = diagnostics

    @abstractmethod
    def fit(self, dataset):
        """Apply estimation to dataset and output results."""

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

        if issubclass(type(result.estimator), PairwiseCBMAEstimator):
            modalities = (
                ["_desc-associationMass", "_corr-"]
                if corr_method == "montecarlo"
                else ["_desc-", "_corr-"]
            )
        else:
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
