"""Fitting and covariance for a term-based CBMR model.

Holds one flat coefficient vector, sliced according to the design's parameter layout.


The cost is a dense ``(n_parameters, n_parameters)`` Hessian: 3.3M entries for a
four-group model at spacing 10, which is fine, but it grows quadratically, so
``BoundDesign.describe`` is worth reading before adding another ``s()`` term.
"""

import logging

import numpy as np

from nimare.meta.cbmr._torch import torch
from nimare.meta.cbmr.distributions import resolve_distribution
from nimare.meta.cbmr.information import closed_form_information

LGR = logging.getLogger(__name__)


class CBMRModel(torch.nn.Module):
    """A CBMR model over a term-based design.

    Parameters
    ----------
    predictor : :class:`~nimare.meta.cbmr.predictor.CBMRPredictor`
        Assembled linear predictor.
    distribution : :obj:`str`, :class:`~nimare.meta.cbmr.distributions.Distribution`, or class
        Observation distribution. Resolved by
        :func:`~nimare.meta.cbmr.distributions.resolve_distribution`.
    device : :obj:`str`, optional
        Torch device. Default is ``"cpu"``.
    """

    def __init__(self, predictor, distribution="poisson", device="cpu"):
        super().__init__()
        self.predictor = predictor
        self.distribution = resolve_distribution(distribution)
        self.distribution.check_design(predictor)
        self.device = device

        self.n_spatial = predictor.n_spatial_columns * predictor.n_bases
        self.n_global = predictor.n_global_columns
        self.n_nuisance = self.distribution.n_nuisance_parameters(predictor.patterns.n_patterns)

        # Small nonzero starting values, matching the historical uniform(-0.01, 0.01) init.
        generator = torch.Generator().manual_seed(0)
        start = (
            torch.rand(self.n_spatial + self.n_global, generator=generator, dtype=torch.float64)
            * 0.02
            - 0.01
        )
        self.coefficients = torch.nn.Parameter(start.to(device))

        nuisance = self.distribution.initial_nuisance(predictor.patterns.n_patterns)
        self.nuisance = (
            torch.nn.Parameter(nuisance.detach().to(device)) if nuisance is not None else None
        )
        self._iterations = 0
        self._foci = None
        self._candidate_foci = None
        self._covariance_cache = None

    @property
    def n_parameters(self):
        """Number of regression coefficients, excluding nuisance parameters."""
        return self.n_spatial + self.n_global

    @property
    def foci(self):
        """Foci counts the model was fit to."""
        return self._require_foci()

    @staticmethod
    def _copy_foci(foci):
        """Return a copy of foci that external mutation cannot change in place."""
        copied = foci.copy() if hasattr(foci, "copy") else np.array(foci, copy=True)
        if hasattr(copied, "setflags"):
            copied.setflags(write=False)
        for array_name in ("data", "indices", "indptr", "row", "col"):
            array = getattr(copied, array_name, None)
            if hasattr(array, "setflags"):
                array.setflags(write=False)
        return copied

    def unpack(self, flat):
        """Split a flat coefficient vector into spatial and non-spatial parts."""
        spatial = flat[: self.n_spatial].reshape(
            self.predictor.n_spatial_columns, self.predictor.n_bases
        )
        global_coef = flat[self.n_spatial :] if self.n_global else None
        return spatial, global_coef

    def _require_foci(self):
        """Return the fitted foci matrix, or fail if the model has not been fit."""
        if self._foci is None:
            raise ValueError(
                "CBMRModel must be fit before foci-dependent quantities are available."
            )
        return self._foci

    def _active_foci(self):
        """Return the candidate foci during fitting, otherwise the fitted foci."""
        return self._candidate_foci if self._candidate_foci is not None else self._require_foci()

    def log_likelihood(self, flat=None, nuisance=None):
        """Return the log-likelihood at ``flat``, defaulting to the current coefficients."""
        foci = self._active_foci()
        flat = self.coefficients if flat is None else flat
        raw_nuisance = self.nuisance if nuisance is None else nuisance
        spatial, global_coef = self.unpack(flat)
        transformed = (
            None if raw_nuisance is None else self.distribution.transform_nuisance(raw_nuisance)
        )
        return self.distribution.log_likelihood(
            self.predictor, spatial, global_coef, transformed, foci
        )

    def forward(self):
        """Return the negative log-likelihood, the quantity being minimized."""
        return -self.log_likelihood()

    def fit(self, foci, n_iter=1000, lr=1.0, tol=1e-8):
        """Fit by L-BFGS.

        Parameters
        ----------
        foci : :obj:`scipy.sparse.spmatrix` or :obj:`numpy.ndarray`
            Foci counts, of shape ``(n_experiments, n_voxels)``.
        n_iter : :obj:`int`, optional
            Maximum L-BFGS iterations. Default is 1000.
        lr : :obj:`float`, optional
            Learning rate. Default is 1.0.
        tol : :obj:`float`, optional
            Stopping tolerance on the change in the objective. Default is 1e-8.
        """
        parameters = [self.coefficients] + ([] if self.nuisance is None else [self.nuisance])
        previous_values = [parameter.detach().clone() for parameter in parameters]
        previous_iterations = self._iterations
        candidate_foci = self._copy_foci(foci)
        self._candidate_foci = candidate_foci
        optimizer = torch.optim.LBFGS(
            params=parameters,
            lr=lr,
            max_iter=n_iter,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        try:

            def closure():
                optimizer.zero_grad()
                loss = self()
                loss.backward()
                return loss

            optimizer.step(closure)
            state = optimizer.state.get(parameters[0], {})
            iterations = int(state.get("n_iter", 0))

            loss = self()
            if not torch.isfinite(loss):
                raise ValueError(
                    f"The {self.distribution.name} log-likelihood became "
                    f"{float(loss.detach())} during optimization. Try a smaller lr, a coarser "
                    "spline_spacing, or the Poisson distribution."
                )
        except Exception:
            with torch.no_grad():
                for parameter, previous_value in zip(parameters, previous_values):
                    parameter.copy_(previous_value)
            self._iterations = previous_iterations
            raise
        finally:
            self._candidate_foci = None

        self._foci = candidate_foci
        self._covariance_cache = None
        self._iterations = iterations
        return self

    def information_matrix(self):
        """Return the observed Fisher information over the flat coefficient vector.

        One matrix over all coefficients, so the cross blocks between terms are present rather
        than assumed away. Nuisance parameters are held fixed at their fitted values.

        Computed in closed form for every distribution with automatic differentiation as the
        fallback for distributions added without a derivation.
        """
        self._require_foci()
        closed_form = closed_form_information(self.distribution)
        if closed_form is not None:
            return closed_form(self)

        # No derivation, so differentiate. jacfwd(jacrev(.)) builds an intermediate of shape
        # (n_parameters, n_patterns, n_voxels), which is tens of GB on a many-group model.
        flat = self.coefficients.detach().clone()
        nuisance = None if self.nuisance is None else self.nuisance.detach().clone()

        def negative_log_likelihood(vector):
            return -self.log_likelihood(flat=vector, nuisance=nuisance)

        hessian = torch.func.hessian(negative_log_likelihood)(flat)
        return hessian.reshape(self.n_parameters, self.n_parameters).detach().cpu().numpy()

    def covariance(self, cov_type="fisher", meat="cluster", correction="hc1", ridge=0.0):
        """Return the coefficient covariance.

        Parameters
        ----------
        cov_type : {"fisher", "sandwich"}, optional
            ``"fisher"`` inverts the observed information, which is correct only if the Poisson
            mean-variance relationship holds. ``"sandwich"`` replaces the model-based variance
            with an empirical one, which is the safer default for foci that are overdispersed and
            clustered within experiments. Default is ``"fisher"``, matching what CBMR has always
            reported for regression coefficients.
        meat, correction, ridge
            Passed to :func:`~nimare.meta.cbmr.covariance.sandwich_covariance` when
            ``cov_type="sandwich"``.

        Notes
        -----
        The result is cached. It is a function of the converged coefficients and the fitted foci,
        so every hypothesis tested against one fit asks for the same matrix, and
        :meth:`~nimare.meta.cbmr.results.CBMRResult.test` would otherwise rebuild it each time.
        The cache holds one matrix, is keyed on the covariance options, and is cleared by
        :meth:`fit`. At many groups that matrix is large; could be well over 500mb, but it is
        shared by every result derived from the fit, since ``CBMRResult.copy`` passes the
        estimator by reference.
        """
        from nimare.meta.cbmr.covariance import fisher_covariance, sandwich_covariance

        self._require_foci()
        key = (cov_type, meat, correction, ridge)
        cached = self._covariance_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        if cov_type == "sandwich":
            value = sandwich_covariance(self, meat=meat, correction=correction, ridge=ridge)
        elif cov_type == "fisher":
            value = fisher_covariance(self, self.information_matrix())
        else:
            raise ValueError(f"cov_type must be 'fisher' or 'sandwich', got {cov_type!r}.")

        self._covariance_cache = (key, value)
        return value

    def standard_errors(self, **covariance_kwargs):
        """Return coefficient standard errors, keyed by term.

        Parameters
        ----------
        **covariance_kwargs
            Passed to :meth:`covariance`, so ``cov_type="sandwich"`` gives robust errors.

        Returns
        -------
        :obj:`dict`
            Maps the rendered term to an array of standard errors, shaped
            ``(n_columns, n_bases)`` for a spatial term and ``(n_columns,)`` otherwise.
        """
        covariance = self.covariance(**covariance_kwargs)
        errors = np.sqrt(np.diag(covariance))
        result = {}
        for name, term_slice in self.predictor.design.parameter_slices(
            self.predictor.n_bases
        ).items():
            block = errors[term_slice]
            term = next(t for t in self.predictor.design.terms if str(t) == name)
            columns = next(
                b.n_columns for b in self.predictor.design.blocks if str(b.term) == name
            )
            result[name] = block.reshape(columns, -1) if term.spatial else block
        return result

    def overdispersion(self):
        """Return the fitted overdispersion per spatial pattern, or None if there is none.

        Reported on the statistical scale, not the unconstrained scale the optimizer works on.
        """
        if self.nuisance is None:
            return None
        with torch.no_grad():
            return self.distribution.transform_nuisance(self.nuisance).cpu().numpy()

    def fitted_coefficients(self):
        """Return the fitted coefficients, keyed by term, in the design's layout."""
        flat = self.coefficients.detach().cpu().numpy()
        result = {}
        for name, term_slice in self.predictor.design.parameter_slices(
            self.predictor.n_bases
        ).items():
            term = next(t for t in self.predictor.design.terms if str(t) == name)
            columns = next(
                b.n_columns for b in self.predictor.design.blocks if str(b.term) == name
            )
            block = flat[term_slice]
            result[name] = block.reshape(columns, -1) if term.spatial else block
        return result

    def log_intensity(self):
        """Return the fitted log spatial intensity, one row per spatial pattern."""
        spatial, _ = self.unpack(self.coefficients.detach())
        return self.predictor.log_intensity_by_pattern(spatial).cpu().numpy()
