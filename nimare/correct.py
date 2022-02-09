"""Multiple comparisons correction methods."""
import inspect
import logging
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import statsmodels.stats.multitest as mc

from .results import MetaResult
from .transforms import p_to_z

LGR = logging.getLogger(__name__)


class Corrector(metaclass=ABCMeta):
    """Base class for multiple comparison correction methods in :mod:`~nimare.correct`.

    .. versionadded:: 0.0.3

    """

    # The name of the method that must be implemented in an Estimator class
    # in order to override the default correction method.
    _correction_method = None

    # A list of valid method values for the Corrector that *aren't* Estimator-specific
    _native_methods = []

    # Maps that must be available in the MetaResult instance
    _required_maps = ("p",)

    def __init__(self):
        pass

    @abstractproperty
    def _name_suffix(self):
        pass

    def _validate_input(self, result):
        if not isinstance(result, MetaResult):
            raise ValueError(
                "First argument to transform() must be an "
                f"instance of class MetaResult, not {type(result)}."
            )

        # Get Estimator correction methods
        pattern = f"correct_{self._correction_method}_"
        est_methods = inspect.getmembers(result.estimator, predicate=inspect.ismethod)
        est_methods = [meth[0] for meth in est_methods]
        est_methods = [meth for meth in est_methods if meth.startswith(pattern)]
        est_methods = [meth.replace(pattern, "") for meth in est_methods]

        # Check requested method against available methods
        if self.method not in self._native_methods + est_methods:
            raise ValueError(
                f"Unsupported {self._correction_method} correction method '{self.method}'\n"
                f"\tAvailable native methods: {', '.join(self._native_methods)}\n"
                f"\tAvailable estimator methods: {', '.join(est_methods)}"
            )

        # Check required maps
        for rm in self._required_maps:
            if result.maps.get(rm) is None:
                raise ValueError(
                    f"{type(self)} requires '{rm}' maps to be present in the MetaResult, "
                    "but none were found."
                )

    def _generate_secondary_maps(self, result, corr_maps):
        # Generates corrected version of z and log-p maps if they exist
        p = corr_maps["p"]
        if "z" in result.maps:
            corr_maps["z"] = p_to_z(p) * np.sign(result.maps["z"])
        if "logp" in result.maps:
            corr_maps["logp"] = -np.log10(p)
        return corr_maps

    @classmethod
    def inspect(cls, result):
        """Identify valid 'method' values for a MetaResult object.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Object for which valid correction methods (i.e., 'method' values) will be identified.

        Returns
        -------
        :obj:`list`
            List of valid 'method' values for the Corrector+Estimator combination, including
            both non-specific methods and Estimator-specific ones.
        """
        est = result.estimator
        correction_str = f"correct_{cls._correction_method}_"
        est_methods = inspect.getmembers(est, predicate=inspect.ismethod)
        valid_methods = [em[0] for em in est_methods if em[0].startswith(correction_str)]
        method_names = [vm.split(correction_str)[1] for vm in valid_methods]
        LGR.info(
            f"Available non-specific methods: {', '.join(cls._native_methods)}\n"
            f"Available Estimator-specific methods: {', '.join(method_names)}"
        )
        return cls._native_methods + method_names

    def transform(self, result):
        """Apply the multiple comparisons correction method to a MetaResult object.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            MetaResult generated by an Estimator to be corrected for multiple
            comparisons.

        Returns
        -------
        result : :obj:`~nimare.results.MetaResult`
            MetaResult with new corrected maps added.
        """
        est = result.estimator
        correction_method = f"correct_{self._correction_method}_{self.method}"

        # Make sure we return a copy of the MetaResult
        result = result.copy()

        # If a correction method with the same name exists in the current
        # MetaEstimator, use it. Otherwise fall back on _transform.
        if correction_method is not None and hasattr(est, correction_method):
            LGR.info(
                "Using correction method implemented in Estimator: "
                f"{est.__class__.__module__}.{est.__class__.__name__}.{correction_method}."
            )
            corr_maps = getattr(est, correction_method)(result, **self.parameters)
        else:
            self._validate_input(result)
            corr_maps = self._transform(result)

        # Update corrected map names and add them to maps dict
        corr_maps = {(k + self._name_suffix): v for k, v in corr_maps.items()}
        result.maps.update(corr_maps)

        # Update the estimator as well, in order to retain updated null distributions
        result.estimator = est

        return result

    @abstractmethod
    def _transform(self, result, **kwargs):
        # Must return a dictionary of new maps to add to .maps, where keys are
        # map names and values are the maps. Names must _not_ include
        # the _name_suffix:, as that will be added in transform() (i.e.,
        # return "p" not "p_corr-FDR_q-0.05_method-indep").
        pass


class FWECorrector(Corrector):
    """Perform family-wise error rate correction on a meta-analysis.

    Parameters
    ----------
    method : :obj:`str`
        The FWE correction to use. Available internal methods are 'bonferroni'.
        Additional methods may be implemented within the provided Estimator.
    **kwargs
        Keyword arguments to be used by the FWE correction implementation.

    Notes
    -----
    This corrector supports a small number of internal FWE correction methods,
    but can also use special methods implemented within individual Estimators.
    To determine what methods are available for the Estimator you're using,
    check the Estimator's documentation. Estimators have special methods
    following the naming convention correct_[correction-type]_[method]
    (e.g., :func:`~nimare.meta.cbma.ale.ALE.correct_fwe_montecarlo`).
    """

    _correction_method = "fwe"
    _native_methods = ["bonferroni"]

    def __init__(self, method="bonferroni", **kwargs):
        self.method = method
        self.parameters = kwargs

    @property
    def _name_suffix(self):
        return f"_corr-FWE_method-{self.method}"

    def _transform(self, result):
        p = result.maps["p"]
        _, p_corr, _, _ = mc.multipletests(p, method=self.method, is_sorted=False)
        corr_maps = {"p": p_corr}
        self._generate_secondary_maps(result, corr_maps)
        return corr_maps


class FDRCorrector(Corrector):
    """Perform false discovery rate correction on a meta-analysis.

    Parameters
    ----------
    alpha : :obj:`float`
        The FDR correction rate to use.
    method : :obj:`str`
        The FDR correction to use. Either 'indep' (for independent or
        positively correlated values) or 'negcorr' (for general or negatively
        correlated tests).

    Notes
    -----
    This corrector supports a small number of internal FDR correction methods,
    but can also use special methods implemented within individual Estimators.
    To determine what methods are available for the Estimator you're using,
    check the Estimator's documentation. Estimators have special methods
    following the naming convention correct_[correction-type]_[method]
    (e.g., :class:`~nimare.meta.mkda.MKDAChi2.correct_fdr_bh`).
    """

    _correction_method = "fdr"
    _native_methods = ["indep", "negcorr"]

    def __init__(self, alpha=0.05, method="indep", **kwargs):
        self.alpha = alpha
        self.method = method
        self.parameters = kwargs

    @property
    def _name_suffix(self):
        return f"_corr-FDR_method-{self.method}"

    def _transform(self, result):
        p = result.maps["p"]
        _, p_corr = mc.fdrcorrection(p, alpha=self.alpha, method=self.method, is_sorted=False)
        corr_maps = {"p": p_corr}
        self._generate_secondary_maps(result, corr_maps)
        return corr_maps
