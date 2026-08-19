"""Base class for workflow."""

import copy
import inspect
import itertools
import logging
import os.path as op
from abc import abstractmethod

import numpy as np

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


def _init_defaults(clss):
    """Collect the keyword arguments a class names anywhere in its constructor chain.

    Subclasses commonly forward ``**kwargs`` to a parent, so the most-derived ``__init__``
    signature alone does not say what the class accepts. Walk the MRO from base to derived
    so that an override wins, and skip ``*args``/``**kwargs`` since they name nothing.
    """
    defaults = {}
    for klass in reversed(inspect.getmro(clss)):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue

        for name, param in inspect.signature(init).parameters.items():
            if name == "self" or param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            defaults[name] = param.default

    return defaults


def _supported_kwargs(clss, kwargs):
    """Drop the workflow settings a class has no constructor argument for.

    The workflow fans one ``n_cores`` out to its estimator, corrector, and diagnostics, but
    not every component has that knob. Passing it anyway is at best ignored and at worst an
    error, so filter instead.
    """
    accepted = _init_defaults(clss)
    supported = {key: value for key, value in kwargs.items() if key in accepted}

    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        LGR.debug(f"{clss.__name__} does not accept {', '.join(dropped)}; not passing it on.")

    return supported


def _holds_default(current, default):
    """Whether an attribute still holds the value its ``__init__`` signature named.

    ``==`` is not enough on its own: an array-valued parameter returns an array of
    comparisons, whose truth value is ambiguous, and a type that refuses the comparison
    altogether raises. Treat both as "the caller set this" rather than guessing.
    """
    if current is default:
        return True

    try:
        return bool(np.array_equal(current, default))
    except (TypeError, ValueError):
        return False


def _unset_params(obj, kwargs):
    """Select the ``kwargs`` an already-initialized object left at its own defaults.

    A parameter counts as unset when the instance still holds the default from its
    ``__init__`` signature, so an explicitly configured object keeps its own settings. A
    caller who passed the default value explicitly is indistinguishable from one who left it
    alone, and is treated as the latter.
    """
    defaults = _init_defaults(type(obj))
    # set_params only accepts what the most-derived __init__ names, while _init_defaults
    # walks the whole MRO. Intersect the two: a parameter that a base class names but a
    # subclass only forwards through **kwargs has to be skipped, or set_params rejects it.
    settable = set(obj.get_params(deep=False))

    unset = {}
    for key, value in kwargs.items():
        if key not in defaults or key not in settable or not hasattr(obj, key):
            continue

        if _holds_default(getattr(obj, key), defaults[key]):
            unset[key] = value

    return unset


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

    # Apply kwargs (including n_cores) when the caller named a class rather than an instance
    if isinstance(obj, type):
        return _check_type(obj, clss, **_supported_kwargs(obj, kwargs))

    # The object is already instantiated. Fill in the workflow's settings for any parameter
    # the caller left at its default, so that e.g. `diagnostics=Jackknife()` and
    # `diagnostics="jackknife"` define clusters the same way. Parameters the caller set
    # explicitly are left alone.
    obj = _check_type(obj, clss)
    unset = _unset_params(obj, kwargs)
    if unset:
        LGR.info(
            f"Applying workflow settings to the already-initialized {type(obj).__name__}: "
            f"{', '.join(f'{k}={v}' for k, v in unset.items())}."
        )
        # Copy first, so the caller's object is not modified behind their back.
        obj = copy.deepcopy(obj).set_params(**unset)

    return obj


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
        if estimator is None:
            estimator = self._estm_default(
                **_supported_kwargs(self._estm_default, {"n_cores": self.n_cores})
            )
        else:
            estimator = _check_input(
                estimator, self._estm_base, self._estm_options, n_cores=self.n_cores
            )

        if corrector is None:
            corrector = self._corr_default(
                method=self._mcc_method,
                **_supported_kwargs(self._corr_default, {"n_cores": self.n_cores}),
            )
        else:
            corrector = _check_input(
                corrector, Corrector, self._corr_options, n_cores=self.n_cores
            )

        diag_kwargs = {
            # The workflow's voxel_thresh thresholds the diagnostics' target image, which
            # diagnostics call target_threshold. Passing it as voxel_thresh would hit the
            # deprecated alias and warn about a parameter the caller never set.
            "target_threshold": self.voxel_thresh,
            "cluster_threshold": self.cluster_threshold,
        }
        diag_kwargs["n_cores"] = self.n_cores
        if diagnostics is None:
            diagnostics = [
                self._diag_default(**_supported_kwargs(self._diag_default, diag_kwargs))
            ]
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
            if corr_method == "montecarlo":
                # Pairwise Monte Carlo estimators use different descriptor labels:
                # MKDAChi2 -> associationMass, ALESubtraction -> group1MinusGroup2Mass.
                mass_labels = ("_desc-associationMass", "_desc-group1MinusGroup2Mass")
                img_keys = [
                    img_key
                    for img_key in corr_result.maps.keys()
                    if img_key.startswith("z_")
                    and "_corr-" in img_key
                    and any(label in img_key for label in mass_labels)
                ]

                # Fall back to voxel-level pairwise maps when only voxel-level FWE is available.
                if not img_keys:
                    voxel_labels = (
                        "_desc-association_level-voxel",
                        "_desc-group1MinusGroup2_level-voxel",
                    )
                    img_keys = [
                        img_key
                        for img_key in corr_result.maps.keys()
                        if img_key.startswith("z_")
                        and "_corr-" in img_key
                        and any(label in img_key for label in voxel_labels)
                    ]
            else:
                img_keys = [
                    img_key
                    for img_key in corr_result.maps.keys()
                    if img_key.startswith("z_") and "_corr-" in img_key and "_desc-" in img_key
                ]
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
