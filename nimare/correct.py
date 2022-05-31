"""Multiple comparisons correction methods."""
import inspect
import logging
from abc import ABCMeta, abstractproperty

import numpy as np
from pymare.stats import bonferroni, fdr

from nimare.results import MetaResult
from nimare.transforms import p_to_z

LGR = logging.getLogger(__name__)


class Corrector(metaclass=ABCMeta):
    """Base class for multiple comparison correction methods in :mod:`~nimare.correct`.

    .. versionadded:: 0.0.3

    """

    # The name of the method that must be implemented in an Estimator class
    # in order to override the default correction method.
    _correction_method = None

    # Maps that must be available in the MetaResult instance
    _required_maps = ("p",)

    def __init__(self):
        pass

    @abstractproperty
    def _name_suffix(self):
        """Identify parameters in a string, to be added to generated filenames."""
        pass

    @classmethod
    def _get_corrector_methods(cls):
        """List correction methods implemented within the Corrector."""
        method_name_str = f"correct_{cls._correction_method}_"
        corr_methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        corr_methods = [meth[0] for meth in corr_methods if meth[0].startswith(method_name_str)]
        corr_methods = [meth.replace(method_name_str, "") for meth in corr_methods]
        return corr_methods

    @classmethod
    def _get_estimator_methods(cls, estimator):
        """List correction methods implemented in an Estimator."""
        method_name_str = f"correct_{cls._correction_method}_"
        est_methods = inspect.getmembers(estimator, predicate=inspect.ismethod)
        est_methods = [meth[0] for meth in est_methods]
        est_methods = [meth for meth in est_methods if meth.startswith(method_name_str)]
        est_methods = [meth.replace(method_name_str, "") for meth in est_methods]
        return est_methods

    def _collect_inputs(self, result):
        """Check that inputs and options are valid.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            The MetaResult to validate.
        """
        if not isinstance(result, MetaResult):
            raise ValueError(
                "First argument to transform() must be an instance of class MetaResult, not "
                f"{type(result)}."
            )

        # Get generic Corrector methods
        corr_methods = self._get_corrector_methods()

        # Get Estimator correction methods
        est_methods = self._get_estimator_methods(result.estimator)

        # Check requested method against available methods
        if self.method not in corr_methods + est_methods:
            raise ValueError(
                f"Unsupported {self._correction_method} correction method '{self.method}'\n"
                f"\tAvailable native methods: {', '.join(corr_methods)}\n"
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
        """Generate corrected version of z and log-p maps if they exist."""
        p = corr_maps["p"]

        if "z" in result.maps:
            corr_maps["z"] = p_to_z(p) * np.sign(result.maps["z"])

        if "logp" in result.maps:
            corr_maps["logp"] = -np.log10(p)

        return corr_maps

    @classmethod
    def inspect(cls, result):
        """Identify valid 'method' values for a MetaResult object.

        In addition to returning a list of valid values, this method will also print out those
        values, divided by the value type (Estimator or generic).

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
        # Get generic Corrector methods
        corr_methods = cls._get_corrector_methods()

        # Get Estimator correction methods
        est_methods = cls._get_estimator_methods(result.estimator)

        all_methods = sorted(list(set(corr_methods + est_methods)))

        # Flag any methods implemented in both.
        # The Estimator method takes priority and the Corrector method is overridden.
        duplicate_methods = list(set(corr_methods) & set(est_methods))
        for duplicate_method in duplicate_methods:
            if duplicate_method in corr_methods:
                corr_methods[
                    corr_methods.index(duplicate_method)
                ] = f"{duplicate_method} (overridden)"

        LGR.info(
            f"Available non-specific methods: {', '.join(corr_methods)}\n"
            f"Available Estimator-specific methods: {', '.join(est_methods)}"
        )
        return all_methods

    def transform(self, result):
        """Apply the multiple comparisons correction method to a MetaResult object.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            MetaResult generated by an Estimator to be corrected for multiple comparisons.

        Returns
        -------
        result : :obj:`~nimare.results.MetaResult`
            MetaResult with new corrected maps added.
        """
        correction_method = f"correct_{self._correction_method}_{self.method}"

        # Make sure we return a copy of the MetaResult
        result = result.copy()

        # Also operate on a copy of the estimator
        est = result.estimator

        # If a correction method with the same name exists in the current MetaEstimator, use it.
        # Otherwise fall back on _transform, and the Corrector methods.
        # In case a method is present in both the Estimator and the Corrector, the Estimator's
        # implementation takes precedence.
        if hasattr(est, correction_method):
            LGR.info(
                "Using correction method implemented in Estimator: "
                f"{est.__class__.__module__}.{est.__class__.__name__}.{correction_method}."
            )
            corr_maps = getattr(est, correction_method)(result, **self.parameters)
        else:
            self._collect_inputs(result)
            corr_maps = self._transform(result, method=correction_method)

        # Update corrected map names and add them to maps dict
        corr_maps = {(k + self._name_suffix): v for k, v in corr_maps.items()}
        result.maps.update(corr_maps)

        # Update the estimator as well, in order to retain updated null distributions
        result.estimator = est

        return result

    def _transform(self, result, method):
        """Implement the correction procedure and return a dictionary of arrays.

        This was originally an abstract method, with FWECorrector and FDRCorrector having their
        own implementations, but those implementations were exactly the same.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            MetaResult object from which to extract the p value map and Estimator.
        method : :obj:`str`
            The correction method to use. This name must match a method in the Corrector,
            according to the pattern "correct_[FWE|FDR]_[method]".

        Returns
        -------
        corr_maps : :obj:`dict`
            A dictionary of new maps that will be added to the MetaResult's ``maps`` attribute,
            where keys are map names and values are the arrays.

            The map names must _not_ include the ``_name_suffix``:, as that will be added in
            ``transform()`` (i.e., return "p" not "p_corr-FDR_q-0.05_method-indep").
        """
        p = result.maps["p"]

        # Find NaNs in the p value map, and mask them out
        nonnan_mask = ~np.isnan(p)
        p_corr = np.empty_like(p)
        p_no_nans = p[nonnan_mask]

        # Call the correction method
        p_corr_no_nans = getattr(self, method)(p_no_nans)

        # Unmask the corrected p values based on the NaN mask
        p_corr[nonnan_mask] = p_corr_no_nans

        # Create a dictionary of the corrected results
        corr_maps = {"p": p_corr}
        self._generate_secondary_maps(result, corr_maps)
        return corr_maps


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
    This corrector supports a small number of internal FWE correction methods, but can also use
    special methods implemented within individual Estimators.
    To determine what methods are available for the Estimator you're using, use :meth:`inspect`.
    Estimators have special methods following the naming convention
    ``correct_[correction-type]_[method]``
    (e.g., :func:`~nimare.meta.cbma.ale.ALE.correct_fwe_montecarlo`).
    """

    _correction_method = "fwe"

    def __init__(self, method="bonferroni", **kwargs):
        self.method = method
        self.parameters = kwargs

    @property
    def _name_suffix(self):
        return f"_corr-FWE_method-{self.method}"

    def correct_fwe_bonferroni(self, p):
        """Perform Bonferroni FWE correction.

        This correction is based on the one described in :footcite:t:`bonferroni1936teoria` and
        :footcite:t:`shaffer1995multiple`.

        .. warning::
            Do not call this method directly. Call :meth:`transform` with ``method='bonferroni'``
            instead.

        .. versionadded:: 0.0.12

        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            A 1D array of p values.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 1D array of adjusted p values.

        References
        ----------
        .. footbibliography::

        See Also
        --------
        nimare.stats.bonferroni
        """
        return bonferroni(p)


class FDRCorrector(Corrector):
    """Perform false discovery rate correction on a meta-analysis.

    Parameters
    ----------
    method : :obj:`str`, optional
        The FDR correction to use.
        Either 'indep' (for independent or positively correlated values) or 'negcorr'
        (for general or negatively correlated tests).
        Default is 'indep'.
    alpha : :obj:`float`, optional
        The FDR correction rate to use. Default is 0.05.

    Notes
    -----
    This corrector supports a small number of internal FDR correction methods, but can also use
    special methods implemented within individual Estimators.
    To determine what methods are available for the Estimator you're using, use :meth:`inspect`.
    Estimators have special methods following the naming convention
    ``correct_[correction-type]_[method]``
    (e.g., :class:`~nimare.meta.mkda.MKDAChi2.correct_fdr_indep`).
    """

    _correction_method = "fdr"

    def __init__(self, method="indep", alpha=0.05, **kwargs):
        self.alpha = alpha
        self.method = method
        self.parameters = kwargs

    @property
    def _name_suffix(self):
        return f"_corr-FDR_method-{self.method}"

    def correct_fdr_indep(self, p):
        """Perform Benjamini-Hochberg FDR correction.

        This correction is based on the one described in :footcite:t:`benjamini1995controlling`.
        This method is not universally appropriate. It works well for tests that are independent,
        or which are positively correlated.

        .. warning::
            Do not call this method directly. Call :meth:`transform` with ``method='indep'``
            instead.

        .. versionadded:: 0.0.12

        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            A 1D array of p values.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 1D array of adjusted p values.

        References
        ----------
        .. footbibliography::

        See Also
        --------
        pymare.stats.fdr
        """
        return fdr(p, q=self.alpha, method="bh")

    def correct_fdr_negcorr(self, p):
        """Perform Benjamini-Yekutieli FDR correction.

        This correction is based on the one described in :footcite:t:`benjamini2001control`.
        It is most appropriate for tests that are negatively correlated.

        .. warning::
            Do not call this method directly. Call :meth:`transform` with ``method='negcorr'``
            instead.

        .. versionadded:: 0.0.12

        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            A 1D array of p values.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 1D array of adjusted p values.

        Notes
        -----
        The difference between the Benjamini-Yekutieli and Benjamini-Hochberg methods is that
        Benjamini-Yekutieli includes an additional term, ``c(m)``.
        When the tests are independent or positively correlated, ``c(m)`` is 1 (and thus has no
        effect).
        In cases of other forms of dependence, ``c(m)`` has an effect.

        References
        ----------
        .. footbibliography::

        See Also
        --------
        pymare.stats.fdr
        """
        return fdr(p, q=self.alpha, method="by")
