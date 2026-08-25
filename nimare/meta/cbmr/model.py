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

    @property
    def n_parameters(self):
        """Number of regression coefficients, excluding nuisance parameters."""
        return self.n_spatial + self.n_global

    def unpack(self, flat):
        """Split a flat coefficient vector into spatial and non-spatial parts."""
        spatial = flat[: self.n_spatial].reshape(
            self.predictor.n_spatial_columns, self.predictor.n_bases
        )
        global_coef = flat[self.n_spatial :] if self.n_global else None
        return spatial, global_coef

    def log_likelihood(self, foci, flat=None, nuisance=None):
        """Return the log-likelihood at ``flat``, defaulting to the current coefficients."""
        flat = self.coefficients if flat is None else flat
        raw_nuisance = self.nuisance if nuisance is None else nuisance
        spatial, global_coef = self.unpack(flat)
        transformed = (
            None if raw_nuisance is None else self.distribution.transform_nuisance(raw_nuisance)
        )
        return self.distribution.log_likelihood(
            self.predictor, spatial, global_coef, transformed, foci
        )

    def forward(self, foci):
        """Return the negative log-likelihood, the quantity being minimized."""
        return -self.log_likelihood(foci)

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
        optimizer = torch.optim.LBFGS(
            params=parameters,
            lr=lr,
            max_iter=n_iter,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self(foci)
            loss.backward()
            return loss

        optimizer.step(closure)
        state = optimizer.state.get(parameters[0], {})
        self._iterations = int(state.get("n_iter", 0))

        loss = self(foci)
        if not torch.isfinite(loss):
            raise ValueError(
                f"The {self.distribution.name} log-likelihood became "
                f"{float(loss.detach())} during optimization. Try a smaller lr, a coarser "
                "spline_spacing, or the Poisson distribution."
            )
        return self

    def information_matrix(self, foci):
        """Return the observed Fisher information over the flat coefficient vector.

        One matrix over all coefficients, so the cross blocks between terms are present rather
        than assumed away. Nuisance parameters are held fixed at their fitted values, matching
        how CBMR has always reported regression standard errors.

        Computed in closed form for every distribution NiMARE ships; see
        :mod:`nimare.meta.cbmr.information`. Automatic differentiation is the fallback for one
        added without a derivation.
        """
        closed_form = closed_form_information(self.distribution)
        if closed_form is not None:
            return closed_form(self, foci)

        # No derivation, so differentiate. jacfwd(jacrev(.)) builds an intermediate of shape
        # (n_parameters, n_patterns, n_voxels), which is tens of GB on a many-group model.
        flat = self.coefficients.detach().clone()
        nuisance = None if self.nuisance is None else self.nuisance.detach().clone()

        def negative_log_likelihood(vector):
            return -self.log_likelihood(foci, flat=vector, nuisance=nuisance)

        hessian = torch.func.hessian(negative_log_likelihood)(flat)
        return hessian.reshape(self.n_parameters, self.n_parameters).detach().cpu().numpy()

    def covariance(self, foci, cov_type="fisher", meat="cluster", correction="hc1", ridge=0.0):
        """Return the coefficient covariance.

        Parameters
        ----------
        foci : array_like
            Foci counts the model was fitted to.
        cov_type : {"fisher", "sandwich"}, optional
            ``"fisher"`` inverts the observed information, which is correct only if the Poisson
            mean-variance relationship holds. ``"sandwich"`` replaces the model-based variance
            with an empirical one, which is the safer default for foci that are overdispersed and
            clustered within experiments. Default is ``"fisher"``, matching what CBMR has always
            reported for regression coefficients.
        meat, correction, ridge
            Passed to :func:`~nimare.meta.cbmr.covariance.sandwich_covariance` when
            ``cov_type="sandwich"``.
        """
        if cov_type == "sandwich":
            from nimare.meta.cbmr.covariance import sandwich_covariance

            return sandwich_covariance(self, foci, meat=meat, correction=correction, ridge=ridge)
        if cov_type != "fisher":
            raise ValueError(f"cov_type must be 'fisher' or 'sandwich', got {cov_type!r}.")

        from nimare.meta.cbmr.covariance import fisher_covariance

        return fisher_covariance(self, self.information_matrix(foci))

    def standard_errors(self, foci, **covariance_kwargs):
        """Return coefficient standard errors, keyed by term.

        Parameters
        ----------
        foci : array_like
            Foci counts the model was fitted to.
        **covariance_kwargs
            Passed to :meth:`covariance`, so ``cov_type="sandwich"`` gives robust errors.

        Returns
        -------
        :obj:`dict`
            Maps the rendered term to an array of standard errors, shaped
            ``(n_columns, n_bases)`` for a spatial term and ``(n_columns,)`` otherwise.
        """
        covariance = self.covariance(foci, **covariance_kwargs)
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
