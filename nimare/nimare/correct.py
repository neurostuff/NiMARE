"""Multiple comparisons correction methods."""

import inspect
import logging
from abc import abstractproperty

import numpy as np

from nimare.base import NiMAREBase
from nimare.results import MetaResult
from nimare.stats import nlogp_bonferroni, nlogp_fdr
from nimare.transforms import nlogp_to_z
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _acceptable_kwargs,
    _clip_p_values,
    _minimum_positive_float,
    _nlogp_to_logp_values,
)

LGR = logging.getLogger(__name__)


class Corrector(NiMAREBase):
    """Base class for multiple comparison correction methods in :mod:`~nimare.correct`.

    .. versionadded:: 0.0.3

    .. versionchanged:: 0.21.0

        ``None``-valued keyword arguments are dropped, and the rest are validated in
        :meth:`transform` against the correction method that will receive them.

    Parameters
    ----------
    **kwargs
        Keyword arguments for the correction method. Ones set to ``None`` are dropped, so an
        unset parameter defers to that method's own default.
    """

    # The name of the method that must be implemented in an Estimator class
    # in order to override the default correction method.
    _correction_method = None

    # Named ``__init__`` parameters to forward to an Estimator-implemented correction method,
    # which cannot read them off ``self`` the way the Corrector's own methods do.
    _estimator_parameters = ()

    # Maps that must be available in the MetaResult instance
    _required_maps = ("p",)

    def __init__(self, **kwargs):
        # A ``None`` is an unset parameter, not a request to override a default with nothing.
        self.parameters = {k: v for k, v in kwargs.items() if v is not None}

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
        # for cbmr approach, we have customized name for groupwise p maps
        p_map_cbmr = tuple(
            [m for m in result.maps.keys() if m.startswith("p_") and "_corr-" not in m]
        )
        if len(p_map_cbmr) > 0:
            self._required_maps = p_map_cbmr
        for rm in self._required_maps:
            if result.maps.get(rm) is None:
                raise ValueError(
                    f"{type(self)} requires '{rm}' maps to be present in the MetaResult, "
                    "but none were found."
                )

    def _estimator_kwargs(self):
        """Return the keywords for an Estimator-implemented correction method.

        The stored ``**kwargs``, plus ``_estimator_parameters``. The latter are absorbed by
        the constructor signature, so they never reach ``self.parameters`` and an Estimator
        would otherwise run at its own default however the caller set them.
        """
        kwargs = dict(self.parameters)
        for name in self._estimator_parameters:
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value

        return kwargs

    def _validate_parameters(self, correction_method, parameters):
        """Raise if ``correction_method`` cannot accept ``parameters``.

        The correction method is only known once ``transform`` has a MetaResult, so this is
        the first point at which keywords collected by ``__init__`` can be checked.

        Raises
        ------
        TypeError
            Naming ``correction_method`` and the keywords it does accept. A method
            declaring ``**kwargs`` accepts anything and is not checked.
        """
        # The result is passed positionally, so the first parameter is not a keyword.
        accepted = _acceptable_kwargs(correction_method, n_positional=1)
        if accepted is None:
            return

        unexpected = sorted(set(parameters) - set(accepted))
        if unexpected:
            method_name = getattr(correction_method, "__qualname__", str(correction_method))
            raise TypeError(
                f"{type(self).__name__}(method='{self.method}') was given "
                f"{', '.join(repr(kwarg) for kwarg in unexpected)}, which "
                f"{method_name} does not accept. "
                f"Accepted keyword arguments: {', '.join(accepted) if accepted else '(none)'}."
            )

    @staticmethod
    def _secondary_map_names(rm):
        """Return the z and logp map names that go with a p map name."""
        if rm == "p":
            return "z", "logp"
        return rm.replace("p_", "z_"), rm.replace("p_", "logp_")

    def _uncorrected_nlogp(self, result, rm):
        """Return the ``nlogp`` values to correct.

        Taken from the ``p`` map while it holds one, and from the ``logp`` map past its
        floor. Neither alone will do: ``p`` is stored as a float32 and bottoms out at 1e-45,
        which would cap every corrected statistic derived from it however deep the real tail
        went, while a float32 *logarithm* carries coarser relative precision on p than a
        float32 p does, which would cost a digit everywhere else.
        """
        p = np.asarray(result.maps[rm])
        nlogp = np.log(_clip_p_values(p, dtype=np.float64))

        logp_map_name = self._secondary_map_names(rm)[1]
        if logp_map_name in result.maps:
            floored = p <= _minimum_positive_float(p.dtype)
            if floored.any():
                logp = np.asarray(result.maps[logp_map_name], dtype=np.float64)
                nlogp = np.where(floored, -np.log(10.0) * logp, nlogp)

        return nlogp

    def _generate_secondary_maps(self, result, corr_maps, rm, nlogp):
        """Generate corrected version of z and logp maps if they exist."""
        corr_maps[rm] = _clip_p_values(corr_maps[rm], dtype=DEFAULT_FLOAT_DTYPE, copy=False)

        z_map_name, logp_map_name = self._secondary_map_names(rm)
        if z_map_name in result.maps:
            corr_maps[z_map_name] = nlogp_to_z(nlogp) * np.sign(result.maps[z_map_name])

        if logp_map_name in result.maps:
            corr_maps[logp_map_name] = _nlogp_to_logp_values(nlogp, dtype=DEFAULT_FLOAT_DTYPE)

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
                corr_methods[corr_methods.index(duplicate_method)] = (
                    f"{duplicate_method} (overridden)"
                )

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
            MetaResult with new corrected maps, tables, and description added.
        """
        correction_method = f"correct_{self._correction_method}_{self.method}"

        # Make sure we return a copy of the MetaResult
        result = result.copy()

        # Also operate on a copy of the estimator
        est = result.estimator

        # If a correction method with the same name exists in the current Estimator, use it.
        # Otherwise fall back on _transform, and the Corrector methods.
        # In case a method is present in both the Estimator and the Corrector, the Estimator's
        # implementation takes precedence.
        if hasattr(est, correction_method):
            LGR.info(
                "Using correction method implemented in Estimator: "
                f"{est.__class__.__module__}.{est.__class__.__name__}.{correction_method}."
            )
            estimator_method = getattr(est, correction_method)
            parameters = self._estimator_kwargs()
            self._validate_parameters(estimator_method, parameters)
            corr_maps, corr_tables, description = estimator_method(result, **parameters)
        else:
            self._collect_inputs(result)
            self._validate_parameters(getattr(self, correction_method), self.parameters)
            corr_maps, corr_tables, description = self._transform(result, method=correction_method)

        # Update corrected map names and enforce float32 outputs for map arrays.
        corr_maps = {
            k
            + self._name_suffix: (
                v.astype(DEFAULT_FLOAT_DTYPE)
                if isinstance(v, np.ndarray)
                and np.issubdtype(v.dtype, np.floating)
                and v.dtype != DEFAULT_FLOAT_DTYPE
                else v
            )
            for k, v in corr_maps.items()
        }
        result.maps.update(corr_maps)
        result.description_ += " " + description

        corr_tables = {(k + self._name_suffix): v for k, v in corr_tables.items()}
        result.tables.update(corr_tables)

        # Update the estimator as well, in order to retain updated null distributions
        result.estimator = est

        # Save the corrected maps
        result.corrector = self

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
        corr_tables : :obj:`dict`
            An empty dictionary meant to contain any tables (pandas DataFrames) produced by the
            correction procedure.
        description_ : :obj:`str`
            A description of the correction procedure.
        """
        # Create a dictionary of the corrected results
        corr_maps = {}
        for rm in self._required_maps:
            nlogp = self._uncorrected_nlogp(result, rm)

            # Find NaNs in the p value map, and mask them out. Prefilled with NaN, since
            # nothing is written back at those positions.
            nonnan_mask = ~np.isnan(nlogp)
            nlogp_corr = np.full_like(nlogp, np.nan)

            # Call the correction method
            nlogp_corr_no_nans, tables, description = getattr(self, method)(nlogp[nonnan_mask])

            # Unmask the corrected p values based on the NaN mask
            nlogp_corr[nonnan_mask] = nlogp_corr_no_nans

            # Create a dictionary of the corrected results
            corr_maps[rm] = np.exp(nlogp_corr)
            self._generate_secondary_maps(result, corr_maps, rm, nlogp_corr)

        return corr_maps, tables, description


class FWECorrector(Corrector):
    """Perform family-wise error rate correction on a meta-analysis.

    Parameters
    ----------
    method : {'bonferoni', 'montecarlo', 'predictive'}
        The FWE correction to use. Note that the 'montecarlo' method is only available for
        a subset of Estimators. To determine what methods are available for the Estimator you're
        using, use :meth:`inspect`.
    voxel_thresh : :obj:`float`, optional
        Only used if ``method='montecarlo'``. The uncorrected voxel-level threshold to use.
    n_iters : :obj:`int`, default=5000
        Number of iterations to use for Monte Carlo correction.
        Default varies by Estimator.
        For publication-quality results, 5000 or more iterations are recommended.
    n_cores : :obj:`int`, default=1
        Number of cores to use for Monte Carlo correction. Default is 1.
    **kwargs
        Keyword arguments to be used by the FWE correction implementation.
        Ones set to ``None`` are dropped; the rest must be accepted by that implementation,
        or :meth:`transform` raises :obj:`TypeError`.
    """

    _correction_method = "fwe"

    def __init__(self, method="bonferroni", n_iters=None, n_cores=1, **kwargs):
        if method not in ("bonferroni", "montecarlo", "predictive"):
            raise ValueError(f"Unsupported FWE correction method '{method}'")

        if method == "montecarlo":
            # ``None`` is dropped by ``Corrector.__init__``, deferring to the estimator's own
            # default, which varies (e.g., MKDAChi2 vs. CBMAEstimator).
            kwargs["n_iters"] = n_iters
            kwargs["n_cores"] = n_cores

        self.method = method
        super().__init__(**kwargs)

    @property
    def _name_suffix(self):
        return f"_corr-FWE_method-{self.method}"

    def correct_fwe_bonferroni(self, nlogp):
        """Perform Bonferroni FWE correction.

        This correction is based on the one described in :footcite:t:`bonferroni1936teoria` and
        :footcite:t:`shaffer1995multiple`.

        .. warning::
            Do not call this method directly. Call :meth:`transform` with ``method='bonferroni'``
            instead.

        .. versionadded:: 0.0.12

        Parameters
        ----------
        nlogp : :obj:`numpy.ndarray`
            A 1D array of ``nlogp`` values.

        Returns
        -------
        nlogp_corr : :obj:`numpy.ndarray`
            A 1D array of adjusted ``nlogp`` values.
        tables : :obj:`dict`
            A dictionary of DataFrames with summary information from the correction.
            This correction method does not produce any tables, so it will be an empty dict.
        description_ : :obj:`str`
            A description of the correction procedure.

        References
        ----------
        .. footbibliography::

        See Also
        --------
        nimare.stats.nlogp_bonferroni
        """
        description = (
            "Family-wise error rate correction was performed with the Bonferroni correction "
            "procedure \\citep{bonferroni1936teoria,shaffer1995multiple}."
        )
        return nlogp_bonferroni(nlogp), {}, description


class FDRCorrector(Corrector):
    """Perform false discovery rate correction on a meta-analysis.

    Parameters
    ----------
    method : :obj:`str`, default='indep'
        The FDR correction to use.
        Either 'indep' (for independent or positively correlated values) or 'negcorr'
        (for general or negatively correlated tests).
        Default is 'indep'.
    alpha : :obj:`float`, default=0.05
        The FDR correction rate to use. Default is 0.05. Note that the step-up procedure
        only rescales the p-values, so this does not affect the corrected maps; it is the
        rate they are meant to be thresholded at.
        Forwarded to an Estimator-implemented correction method, which cannot read it off
        the Corrector; one with no ``alpha`` parameter raises rather than quietly using its
        own rate.
    **kwargs
        Keyword arguments to be used by the FDR correction implementation.
        Ones set to ``None`` are dropped; the rest must be accepted by that implementation,
        or :meth:`transform` raises :obj:`TypeError`.

    Notes
    -----
    .. versionchanged:: 0.21.0

        ``alpha`` now reaches an Estimator-implemented correction method.

    This corrector supports a small number of internal FDR correction methods, but can also use
    special methods implemented within individual Estimators.
    To determine what methods are available for the Estimator you're using, use :meth:`inspect`.
    Estimators have special methods following the naming convention
    ``correct_[correction-type]_[method]``
    (e.g., :class:`~nimare.meta.mkda.MKDAChi2.correct_fdr_indep`).
    """

    _correction_method = "fdr"
    _estimator_parameters = ("alpha",)

    def __init__(self, method="indep", alpha=0.05, **kwargs):
        self.alpha = alpha
        self.method = method
        super().__init__(**kwargs)

    @property
    def _name_suffix(self):
        return f"_corr-FDR_method-{self.method}"

    def correct_fdr_indep(self, nlogp):
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
        nlogp : :obj:`numpy.ndarray`
            A 1D array of ``nlogp`` values.

        Returns
        -------
        nlogp_corr : :obj:`numpy.ndarray`
            A 1D array of adjusted ``nlogp`` values.
        tables : :obj:`dict`
            A dictionary of DataFrames with summary information from the correction.
            This correction method does not produce any tables, so it will be an empty dict.
        description_ : :obj:`str`
            A description of the correction procedure.

        References
        ----------
        .. footbibliography::

        See Also
        --------
        nimare.stats.nlogp_fdr
        """
        description = (
            "False discovery rate correction was performed with the Benjamini-Hochberg procedure "
            "\\citep{benjamini1995controlling}."
        )
        return nlogp_fdr(nlogp, method="bh"), {}, description

    def correct_fdr_negcorr(self, nlogp):
        """Perform Benjamini-Yekutieli FDR correction.

        This correction is based on the one described in :footcite:t:`benjamini2001control`.
        It is most appropriate for tests that are negatively correlated.

        .. warning::
            Do not call this method directly. Call :meth:`transform` with ``method='negcorr'``
            instead.

        .. versionadded:: 0.0.12

        Parameters
        ----------
        nlogp : :obj:`numpy.ndarray`
            A 1D array of ``nlogp`` values.

        Returns
        -------
        nlogp_corr : :obj:`numpy.ndarray`
            A 1D array of adjusted ``nlogp`` values.
        tables : :obj:`dict`
            A dictionary of DataFrames with summary information from the correction.
            This correction method does not produce any tables, so it will be an empty dict.
        description_ : :obj:`str`
            A description of the correction procedure.

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
        nimare.stats.nlogp_fdr
        """
        description = (
            "False discovery rate correction was performed with the Benjamini-Yekutieli procedure "
            "\\citep{benjamini2001control}."
        )
        return nlogp_fdr(nlogp, method="by"), {}, description
