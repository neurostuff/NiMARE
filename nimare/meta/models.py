"""CBMR Models."""

import abc
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Torch is required to use `CBMR` models. Install with `pip install 'nimare[cbmr]'`."
    ) from e

LGR = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CBMRTensorInputs:
    """Container for tensorized CBMR design and response data."""

    coef_spline_bases: torch.Tensor
    moderators_by_group: object
    foci_per_voxel: dict
    foci_per_experiment: dict

    def subset(self, groups):
        """Return a group-restricted view of the tensor inputs."""
        moderators_by_group = None
        if self.moderators_by_group is not None:
            moderators_by_group = {group: self.moderators_by_group[group] for group in groups}

        return _CBMRTensorInputs(
            coef_spline_bases=self.coef_spline_bases,
            moderators_by_group=moderators_by_group,
            foci_per_voxel={group: self.foci_per_voxel[group] for group in groups},
            foci_per_experiment={group: self.foci_per_experiment[group] for group in groups},
        )


class GeneralLinearModelEstimator(torch.nn.Module):
    """Base class for GLM estimators.

    Parameters
    ----------
    spatial_coef_dim : :obj:`int`
        Number of spatial B-spline bases. Default is None.
    moderators_coef_dim : :obj:`int`, optional
        Number of experiment-level moderators. Default is None.
    penalty : :obj:`bool`
        Whether to Firth-type regularization term. Default is False.
    lr : :obj:`float`
        Learning rate. Default is 0.1.
    lr_decay : :obj:`float`
        Learning rate decay for each iteration. Default is 0.999.
    n_iter : :obj:`int`
        Maximum number of iterations. Default is 1000.
    tol : :obj:`float`
        Tolerance for convergence. Default is 1e-2.
    device : :obj:`str`
        Device to use for computations. Default is "cpu".
    """

    _hessian_kwargs = {
        "create_graph": False,
        "vectorize": True,
        "outer_jacobian_strategy": "forward-mode",
    }

    def __init__(
        self,
        spatial_coef_dim=None,
        moderators_coef_dim=None,
        penalty=False,
        lr=1,
        lr_decay=0.999,
        n_iter=2000,
        tol=1e-9,
        device="cpu",
    ):
        super().__init__()
        self.spatial_coef_dim = spatial_coef_dim
        self.moderators_coef_dim = moderators_coef_dim
        self.penalty = penalty
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_iter = n_iter
        self.tol = tol
        self.device = device

        # initialization for iteration set up
        self.iter = 0

        # after fitting, the following attributes will be created
        self.spatial_regression_coef = None
        self.spatial_intensity_estimation = None
        self.moderators_coef = None
        self.moderators_effect = None
        self.spatial_regression_coef_se = None
        self.log_spatial_intensity_se = None
        self.spatial_intensity_se = None
        self.se_moderators = None
        self._tensor_inputs_cache = None
        self._tensor_inputs_cache_keys = None

    def _invalidate_tensor_inputs_cache(self):
        """Drop cached tensor inputs when model state or device changes."""
        self._tensor_inputs_cache = None
        self._tensor_inputs_cache_keys = None

    @staticmethod
    def _to_numpy_array(array_like):
        """Convert tensors or array-likes to NumPy arrays on the host."""
        if torch.is_tensor(array_like):
            return array_like.detach().cpu().numpy()
        return np.asarray(array_like)

    @staticmethod
    def _flatten_tensor(tensor):
        """Return a 1D view of the provided tensor."""
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            return torch.as_tensor(tensor).reshape(-1)
        return tensor.reshape(-1)

    def _as_float_tensor(self, array_like):
        """Convert array-like inputs to float64 tensors on the estimator device."""
        if array_like is None:
            return None
        return torch.as_tensor(array_like, dtype=torch.float64, device=self.device)

    def _prepare_tensor_inputs(
        self,
        coef_spline_bases,
        moderators_by_group=None,
        foci_per_voxel=None,
        foci_per_experiment=None,
    ):
        """Normalize CBMR inputs into a reusable tensor container."""
        if isinstance(coef_spline_bases, _CBMRTensorInputs):
            return coef_spline_bases

        if foci_per_voxel is None or foci_per_experiment is None:
            raise ValueError(
                "foci_per_voxel and foci_per_experiment are required for CBMR fitting."
            )

        if moderators_by_group is not None:
            moderators_by_group = {
                group: self._as_float_tensor(moderators_by_group[group]) for group in self.groups
            }

        return _CBMRTensorInputs(
            coef_spline_bases=self._as_float_tensor(coef_spline_bases),
            moderators_by_group=moderators_by_group,
            foci_per_voxel={
                group: self._as_float_tensor(foci_per_voxel[group]) for group in self.groups
            },
            foci_per_experiment={
                group: self._as_float_tensor(foci_per_experiment[group]) for group in self.groups
            },
        )

    def _cache_tensor_inputs(
        self,
        coef_spline_bases,
        moderators_by_group,
        foci_per_voxel,
        foci_per_experiment,
        tensor_inputs,
    ):
        """Cache tensorized inputs so repeated inference can reuse them."""
        self._tensor_inputs_cache_keys = (
            id(coef_spline_bases),
            id(moderators_by_group),
            id(foci_per_voxel),
            id(foci_per_experiment),
        )
        self._tensor_inputs_cache = tensor_inputs

    def _resolve_tensor_inputs(
        self,
        coef_spline_bases,
        moderators_by_group=None,
        foci_per_voxel=None,
        foci_per_experiment=None,
    ):
        """Return cached tensor inputs when the current call matches the last fit inputs."""
        if isinstance(coef_spline_bases, _CBMRTensorInputs):
            return coef_spline_bases

        input_keys = (
            id(coef_spline_bases),
            id(moderators_by_group),
            id(foci_per_voxel),
            id(foci_per_experiment),
        )
        if input_keys == self._tensor_inputs_cache_keys and self._tensor_inputs_cache is not None:
            return self._tensor_inputs_cache

        return self._prepare_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        )

    @staticmethod
    def _frame_from_uniform_group_dict(group_values):
        """Construct a DataFrame from same-length group vectors with minimal Python overhead."""
        if not group_values:
            return pd.DataFrame()

        group_names = list(group_values.keys())
        rows = [np.ravel(np.asarray(group_values[group])) for group in group_names]
        n_columns = rows[0].size
        if any(row.size != n_columns for row in rows[1:]):
            return pd.DataFrame.from_dict(group_values, orient="index")

        return pd.DataFrame(np.vstack(rows), index=group_names)

    @abc.abstractmethod
    def _log_likelihood_single_group(self, **kwargs):
        """Log-likelihood of a single group.

        Returns
        -------
        torch.Tensor
            Value of the log-likelihood of a single group.
        """
        pass

    @abc.abstractmethod
    def _log_likelihood_mult_group(self, **kwargs):
        """Total log-likelihood of all groups in the dataset.

        Returns
        -------
        torch.Tensor
            Value of total log-likelihood of all groups in the dataset.
        """
        pass

    @abc.abstractmethod
    def forward(self, **kwargs):
        """Define the loss function (nagetive log-likelihood function) for each model.

        Returns
        -------
        torch.Tensor
            Value of the log-likelihood of a single group.
        """
        pass

    def init_spatial_weights(self):
        """Initialize spatial regression coefficients.

        Default is uniform distribution between -0.01 and 0.01.
        """
        # initialization for spatial regression coefficients
        spatial_coef_linears = dict()
        for group in self.groups:
            spatial_coef_linear_group = torch.nn.Linear(
                self.spatial_coef_dim, 1, bias=False
            ).double()
            torch.nn.init.uniform_(spatial_coef_linear_group.weight, a=-0.01, b=0.01)
            spatial_coef_linears[group] = spatial_coef_linear_group
        self.spatial_coef_linears = torch.nn.ModuleDict(spatial_coef_linears)

    def init_moderator_weights(self):
        """Initialize the intercept and regression coefficients for moderators.

        Default is uniform distribution between -0.01 and 0.01.
        """
        self.moderators_linear = torch.nn.Linear(
            self.moderators_coef_dim,
            1,
            bias=False,
        ).double()
        torch.nn.init.uniform_(self.moderators_linear.weight, a=-0.01, b=0.01)
        return

    def init_weights(self, groups, moderators, spatial_coef_dim, moderators_coef_dim):
        """Initialize spatial and experiment-level moderator coefficients."""
        self.groups = groups
        self.moderators = moderators
        self.spatial_coef_dim = spatial_coef_dim
        self.moderators_coef_dim = moderators_coef_dim
        self.init_spatial_weights()
        if moderators_coef_dim:
            self.init_moderator_weights()
        self.to(self.device)
        self._invalidate_tensor_inputs_cache()

    def _update(
        self,
        optimizer,
        coef_spline_bases,
        moderators,
        foci_per_voxel,
        foci_per_experiment,
        prev_loss,
    ):
        """One iteration in optimization with L-BFGS.

        Adjust learning rate based on the number of iteration (with learning rate decay parameter
        `lr_decay`, default value is 0.999). Reset L-BFGS optimizer (as params in the previous
        iteration) if NaN occurs.

        Parameters
        ----------
        optimizer : :obj:`torch.optim.lbfgs.LBFGS`
            L-BFGS optimizer.
        coef_spline_bases : :obj:`torch.Tensor`
            Coefficient of B-spline bases evaluated at each voxel.
        moderators : :obj:`dict`, optional
            Dictionary of group-wise experiment-level moderators. Default is None.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_experiment : :obj:`dict`
            Dictionary of group-wise number of foci per experiment.
        prev_loss : :obj:`torch.Tensor`
            Value of the loss function of the previous iteration.

        Returns
        -------
        torch.Tensor
            Updated value of the loss (negative log-likelihood) function.
        """
        self.iter += 1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.lr_decay
        )  # learning rate decay

        def closure():
            optimizer.zero_grad()
            loss = self(coef_spline_bases, moderators, foci_per_voxel, foci_per_experiment)
            loss.backward()
            return loss

        optimizer.step(closure)
        scheduler.step()
        # recalculate the loss function
        loss = self(coef_spline_bases, moderators, foci_per_voxel, foci_per_experiment)

        if torch.isnan(loss):
            raise ValueError(
                f"""The current learing rate {str(self.lr)} or choice of model gives rise to
                NaN log-likelihood, please try Poisson model or adjust learning rate to a smaller
                value."""
            )

        return loss

    def _optimizer(
        self,
        coef_spline_bases,
        moderators_by_group=None,
        foci_per_voxel=None,
        foci_per_experiment=None,
    ):
        """
        Optimize the loss (negative log-likelihood) function with L-BFGS.

        Parameters
        ----------
        coef_spline_bases : :obj:`numpy.ndarray`
            Coefficient of B-spline bases evaluated at each voxel.
        moderators_by_group : :obj:`dict`, optional
            Dictionary of group-wise experiment-level moderators.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_experiment : :obj:`dict`
            Dictionary of group-wise number of foci per experiment.
        """
        torch.manual_seed(100)
        tensor_inputs = self._prepare_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        )
        optimizer = torch.optim.LBFGS(
            params=self.parameters(),
            lr=self.lr,
            max_iter=self.n_iter,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe",
        )
        prev_loss = torch.tensor(float("inf"), dtype=torch.float64, device=self.device)

        self._update(
            optimizer,
            tensor_inputs.coef_spline_bases,
            tensor_inputs.moderators_by_group,
            tensor_inputs.foci_per_voxel,
            tensor_inputs.foci_per_experiment,
            prev_loss,
        )

        return

    def fit(self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_experiment):
        """Fit the model and estimate standard error of estimates."""
        tensor_inputs = self._prepare_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        )
        self._cache_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
            tensor_inputs,
        )
        self._optimizer(tensor_inputs)
        self.extract_optimized_params(coef_spline_bases, moderators_by_group)
        self.standard_error_estimation(tensor_inputs)

        return

    def extract_optimized_params(self, coef_spline_bases, moderators_by_group):
        """Extract optimized experiment-level moderator coefficients from the model."""
        coef_spline_bases = np.asarray(coef_spline_bases)
        spatial_regression_coef, spatial_intensity_estimation = dict(), dict()
        for group in self.groups:
            # Extract optimized spatial regression coefficients from the model
            group_spatial_coef_linear_weight = self.spatial_coef_linears[group].weight
            group_spatial_coef_linear_weight = self._to_numpy_array(
                group_spatial_coef_linear_weight
            ).flatten()
            spatial_regression_coef[group] = group_spatial_coef_linear_weight
            # Estimate group-specific spatial intensity
            group_spatial_intensity_estimation = np.exp(
                np.matmul(coef_spline_bases, group_spatial_coef_linear_weight)
            )
            spatial_intensity_estimation["spatialIntensity_group-" + group] = (
                group_spatial_intensity_estimation
            )

        # Extract optimized regression coefficient of experiment-level moderators from the model
        if self.moderators_coef_dim:
            moderators_effect = dict()
            moderators_coef = self.moderators_linear.weight
            moderators_coef = self._to_numpy_array(moderators_coef)
            for group in self.groups:
                group_moderators = moderators_by_group[group]
                group_moderators_effect = np.exp(np.matmul(group_moderators, moderators_coef.T))
                moderators_effect[group] = group_moderators_effect.flatten()
        else:
            moderators_coef, moderators_effect = None, None

        self.spatial_regression_coef = spatial_regression_coef
        self.spatial_intensity_estimation = spatial_intensity_estimation
        self.moderators_coef = moderators_coef
        self.moderators_effect = moderators_effect

    def standard_error_estimation(
        self,
        coef_spline_bases,
        moderators_by_group=None,
        foci_per_voxel=None,
        foci_per_experiment=None,
    ):
        """Estimate standard error of estimates.

        For spatial regression coefficients, we estimate its covariance matrix using Fisher
        Information Matrix and then take the square root of the diagonal elements.
        For log spatial intensity, we use the delta method to estimate its standard error.
        For models with over-dispersion parameter, we also estimate its standard error.
        """
        tensor_inputs = self._prepare_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        )
        coef_spline_bases_array = self._to_numpy_array(tensor_inputs.coef_spline_bases)
        spatial_regression_coef_se, log_spatial_intensity_se, spatial_intensity_se = (
            dict(),
            dict(),
            dict(),
        )
        for group in self.groups:
            group_foci_per_voxel = tensor_inputs.foci_per_voxel[group]
            group_foci_per_experiment = tensor_inputs.foci_per_experiment[group]
            group_spatial_coef = self.spatial_coef_linears[group].weight
            if self.moderators_coef_dim:
                group_moderators = tensor_inputs.moderators_by_group[group]
                moderators_coef = self.moderators_linear.weight
            else:
                group_moderators, moderators_coef = None, None

            ll_single_group_kwargs = {
                "moderators_coef": moderators_coef if self.moderators_coef_dim else None,
                "coef_spline_bases": tensor_inputs.coef_spline_bases,
                "group_moderators": group_moderators if self.moderators_coef_dim else None,
                "group_foci_per_voxel": group_foci_per_voxel,
                "group_foci_per_experiment": group_foci_per_experiment,
                "device": self.device,
            }

            if hasattr(self, "overdispersion"):
                ll_single_group_kwargs["group_overdispersion"] = self.overdispersion[group]

            # create a negative log-likelihood function
            def nll_spatial_coef(group_spatial_coef):
                return -self._log_likelihood_single_group(
                    group_spatial_coef=group_spatial_coef,
                    **ll_single_group_kwargs,
                )

            f_spatial_coef = torch.func.hessian(nll_spatial_coef)(group_spatial_coef)
            f_spatial_coef = f_spatial_coef.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
            cov_spatial_coef = np.linalg.inv(self._to_numpy_array(f_spatial_coef))
            var_spatial_coef = np.diag(cov_spatial_coef)
            se_spatial_coef = np.sqrt(var_spatial_coef)
            spatial_regression_coef_se[group] = se_spatial_coef

            var_log_spatial_intensity = np.einsum(
                "ij,ji->i",
                coef_spline_bases_array,
                cov_spatial_coef @ coef_spline_bases_array.T,
            )
            se_log_spatial_intensity = np.sqrt(var_log_spatial_intensity)
            log_spatial_intensity_se[group] = se_log_spatial_intensity

            group_spatial_intensity = np.exp(
                np.matmul(coef_spline_bases_array, self._to_numpy_array(group_spatial_coef).T)
            ).flatten()
            se_spatial_intensity = group_spatial_intensity * se_log_spatial_intensity
            spatial_intensity_se[group] = se_spatial_intensity

        # Inference on regression coefficient of moderators
        if self.moderators_coef_dim:
            # modify ll_single_group_kwargs so that spatial_coef is fixed
            # and moderators_coef can vary
            del ll_single_group_kwargs["moderators_coef"]
            ll_single_group_kwargs["group_spatial_coef"] = group_spatial_coef

            def nll_moderators_coef(moderators_coef):
                return -self._log_likelihood_single_group(
                    moderators_coef=moderators_coef,
                    **ll_single_group_kwargs,
                )

            f_moderators_coef = torch.func.hessian(nll_moderators_coef)(moderators_coef)
            f_moderators_coef = f_moderators_coef.reshape(
                (self.moderators_coef_dim, self.moderators_coef_dim)
            )
            cov_moderators_coef = np.linalg.inv(self._to_numpy_array(f_moderators_coef))
            var_moderators = np.diag(cov_moderators_coef).reshape((1, self.moderators_coef_dim))
            se_moderators = np.sqrt(var_moderators)
        else:
            se_moderators = None

        self.spatial_regression_coef_se = spatial_regression_coef_se
        self.log_spatial_intensity_se = log_spatial_intensity_se
        self.spatial_intensity_se = spatial_intensity_se
        self.se_moderators = se_moderators

    def summary(self):
        """Summarize the main results of the fitted model.

        Summarize optimized regression coefficients from model and store in `tables`,
        summarize standard error of regression coefficient and (Log-)spatial intensity
        and store in `results`.
        """
        params = (
            self.spatial_regression_coef,
            self.spatial_intensity_estimation,
            self.spatial_regression_coef_se,
            self.log_spatial_intensity_se,
            self.spatial_intensity_se,
        )
        if any([param is None for param in params]):
            raise ValueError("Run fit first")
        tables = dict()
        # Extract optimized regression coefficients from model and store them in 'tables'
        tables["spatial_regression_coef"] = self._frame_from_uniform_group_dict(
            self.spatial_regression_coef
        )
        maps = self.spatial_intensity_estimation
        if self.moderators_coef_dim:
            tables["moderators_regression_coef"] = pd.DataFrame(
                data=self.moderators_coef, columns=self.moderators
            )
            tables["moderators_effect"] = self._frame_from_uniform_group_dict(
                self.moderators_effect
            )

        # Estimate standard error of regression coefficient and (Log-)spatial intensity and store
        # them in 'tables'
        tables["spatial_regression_coef_se"] = self._frame_from_uniform_group_dict(
            self.spatial_regression_coef_se
        )
        tables["log_spatial_intensity_se"] = self._frame_from_uniform_group_dict(
            self.log_spatial_intensity_se
        )
        tables["spatial_intensity_se"] = self._frame_from_uniform_group_dict(
            self.spatial_intensity_se
        )
        if self.moderators_coef_dim:
            tables["moderators_regression_se"] = pd.DataFrame(
                data=self.se_moderators, columns=self.moderators
            )
        return maps, tables

    def fisher_info_multiple_group_spatial(
        self,
        involved_groups,
        coef_spline_bases,
        moderators_by_group,
        foci_per_voxel,
        foci_per_experiment,
    ):
        """Estimate the Fisher information matrix of spatial regression coeffcients.

        Fisher information matrix is estimated by negative Hessian of the log-likelihood.

        Parameters
        ----------
        involved_groups : :obj:`list`
            Group names involved in generalized linear hypothesis (GLH) testing in `CBMRInference`.
        coef_spline_bases : :obj:`numpy.ndarray`
            Coefficient of B-spline bases evaluated at each voxel.
        moderators_by_group : :obj:`dict`, optional
            Dictionary of group-wise experiment-level moderators. Default is None.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_experiment : :obj:`dict`
            Dictionary of group-wise number of foci per experiment.

        Returns
        -------
        numpy.ndarray
            Fisher information matrix of spatial regression coefficients (for involved groups).
        """
        tensor_inputs = self._resolve_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        ).subset(involved_groups)
        n_involved_groups = len(involved_groups)
        involved_foci_per_voxel = [
            tensor_inputs.foci_per_voxel[group] for group in involved_groups
        ]
        involved_foci_per_experiment = [
            tensor_inputs.foci_per_experiment[group] for group in involved_groups
        ]
        spatial_coef = [self.spatial_coef_linears[group].weight.T for group in involved_groups]
        spatial_coef = torch.stack(spatial_coef, dim=0)
        if self.moderators_coef_dim:
            involved_moderators_by_group = [
                tensor_inputs.moderators_by_group[group] for group in involved_groups
            ]
            moderators_coef = self._as_float_tensor(self.moderators_coef.T)
        else:
            involved_moderators_by_group, moderators_coef = None, None

        ll_mult_group_kwargs = {
            "moderator_coef": moderators_coef,
            "coef_spline_bases": tensor_inputs.coef_spline_bases,
            "foci_per_voxel": involved_foci_per_voxel,
            "foci_per_experiment": involved_foci_per_experiment,
            "moderators": involved_moderators_by_group,
            "device": self.device,
        }

        if hasattr(self, "overdispersion"):
            ll_mult_group_kwargs["overdispersion_coef"] = [
                self.overdispersion[group] for group in involved_groups
            ]

        # create a negative log-likelihood function
        def nll_spatial_coef(spatial_coef):
            return -self._log_likelihood_mult_group(
                spatial_coef=spatial_coef,
                **ll_mult_group_kwargs,
            )

        h = torch.func.hessian(nll_spatial_coef)(spatial_coef)
        h = h.view(n_involved_groups * self.spatial_coef_dim, -1)

        return h.detach().cpu().numpy()

    def fisher_info_multiple_group_moderator(
        self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_experiment
    ):
        """Estimate the Fisher information matrix of regression coefficients of moderators.

        Fisher information matrix is estimated by negative Hessian of the log-likelihood.

        Parameters
        ----------
        coef_spline_bases : :obj:`numpy.ndarray`
            Coefficient of B-spline bases evaluated at each voxel.
        moderators_by_group : :obj:`dict`, optional
            Dictionary of group-wise experiment-level moderators. Default is None.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_experiment : :obj:`dict`
            Dictionary of group-wise number of foci per experiment.

        Returns
        -------
        numpy.ndarray
            Fisher information matrix of experiment-level moderator regressors.
        """
        tensor_inputs = self._resolve_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        )
        foci_per_voxel = [tensor_inputs.foci_per_voxel[group] for group in self.groups]
        foci_per_experiment = [tensor_inputs.foci_per_experiment[group] for group in self.groups]
        spatial_coef = [self.spatial_coef_linears[group].weight.T for group in self.groups]
        spatial_coef = torch.stack(spatial_coef, dim=0)

        if self.moderators_coef_dim:
            moderators_by_group = [
                tensor_inputs.moderators_by_group[group] for group in self.groups
            ]
            moderator_coef = self._as_float_tensor(self.moderators_coef.T)
        else:
            moderators_by_group, moderator_coef = None, None

        ll_mult_group_kwargs = {
            "spatial_coef": spatial_coef,
            "coef_spline_bases": tensor_inputs.coef_spline_bases,
            "foci_per_voxel": foci_per_voxel,
            "foci_per_experiment": foci_per_experiment,
            "moderators": moderators_by_group,
            "device": self.device,
        }
        if hasattr(self, "overdispersion"):
            ll_mult_group_kwargs["overdispersion_coef"] = [
                self.overdispersion[group] for group in self.groups
            ]

        # create a negative log-likelihood function w.r.t moderator coefficients
        def nll_moderator_coef(moderator_coef):
            return -self._log_likelihood_mult_group(
                moderator_coef=moderator_coef,
                **ll_mult_group_kwargs,
            )

        h = torch.func.hessian(nll_moderator_coef)(moderator_coef)
        h = h.view(self.moderators_coef_dim, self.moderators_coef_dim)

        return h.detach().cpu().numpy()

    def firth_penalty(
        self,
        foci_per_voxel,
        foci_per_experiment,
        moderators,
        coef_spline_bases,
        overdispersion=False,
    ):
        """Compute Firth's penalized log-likelihood.

        Parameters
        ----------
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_experiment : :obj:`dict`
            Dictionary of group-wise number of foci per experiment.
        moderators : :obj:`dict`, optional
            Dictionary of group-wise experiment-level moderators. Default is None.
        coef_spline_bases : :obj:`torch.Tensor`
            Coefficient of B-spline bases evaluated at each voxel.
        overdispersion : :obj:`bool`
            Whether the model contains overdispersion parameter. Default is False.

        Returns
        -------
        torch.Tensor
            Firth-type regularization term.
        """
        group_firth_penalty = 0
        for group in self.groups:
            partial_kwargs = {"coef_spline_bases": coef_spline_bases}
            if overdispersion:
                partial_kwargs["group_overdispersion"] = self.overdispersion[group]
            if getattr(self, "square_root", False):
                partial_kwargs["group_overdispersion"] = (
                    partial_kwargs["group_overdispersion"] ** 2
                )
            partial_kwargs["group_foci_per_voxel"] = foci_per_voxel[group]
            partial_kwargs["group_foci_per_experiment"] = foci_per_experiment[group]
            if self.moderators_coef_dim:
                moderators_coef = self.moderators_linear.weight
                group_moderators = moderators[group]
            else:
                moderators_coef, group_moderators = None, None
            partial_kwargs["moderators_coef"] = moderators_coef
            partial_kwargs["group_moderators"] = group_moderators

            # create a negative log-likelihood function w.r.t spatial coefficients
            def nll_spatial_coef(group_spatial_coef):
                return -self._log_likelihood_single_group(
                    group_spatial_coef=group_spatial_coef,
                    **partial_kwargs,
                )

            group_spatial_coef = self.spatial_coef_linears[group].weight
            group_f = torch.autograd.functional.hessian(
                nll_spatial_coef,
                group_spatial_coef,
                **self._hessian_kwargs,
            )

            group_f = group_f.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
            group_eig_vals = torch.real(torch.linalg.eigvals(group_f))
            del group_f
            group_firth_penalty = 0.5 * torch.sum(torch.log(group_eig_vals))
            del group_eig_vals
            group_firth_penalty += group_firth_penalty

        return group_firth_penalty


class OverdispersionModelEstimator(GeneralLinearModelEstimator):
    """Base class for CBMR models with over-dispersion parameter."""

    def __init__(self, **kwargs):
        self.square_root = kwargs.pop("square_root", False)
        super().__init__(**kwargs)

    def init_overdispersion_weights(self):
        """Initialize weights for overdispersion parameters.

        Default is 1e-2.
        """
        overdispersion = dict()
        for group in self.groups:
            # initialization for alpha
            overdispersion_init_group = torch.tensor(1e-2, dtype=torch.float64, device=self.device)
            if self.square_root:
                overdispersion_init_group = torch.sqrt(overdispersion_init_group)
            overdispersion[group] = torch.nn.Parameter(
                overdispersion_init_group, requires_grad=True
            )
        self.overdispersion = torch.nn.ParameterDict(overdispersion)

    def init_weights(self, groups, moderators, spatial_coef_dim, moderators_coef_dim):
        """Initialize weights for spatial and experiment-level moderator coefficients."""
        super().init_weights(groups, moderators, spatial_coef_dim, moderators_coef_dim)
        self.init_overdispersion_weights()
        self.to(self.device)
        self._invalidate_tensor_inputs_cache()

    def inference_outcome(
        self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_experiment
    ):
        """Summarize inference outcome into `maps` and `tables`.

        Add optimized overdispersion parameter to the tables.
        """
        maps, tables = super().inference_outcome(
            coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_experiment
        )
        overdispersion_param = dict()
        for group in self.groups:
            group_overdispersion = self.overdispersion[group]
            group_overdispersion = group_overdispersion.cpu().detach().numpy()
            overdispersion_param[group] = group_overdispersion
        tables["overdispersion_coef"] = self._frame_from_uniform_group_dict(overdispersion_param)
        tables["overdispersion_coef"].columns = ["overdispersion"]

        return maps, tables


class PoissonEstimator(GeneralLinearModelEstimator):
    """CBMR framework with Poisson model.

    Poisson model is the most basic model for Coordinate-based Meta-regression (CBMR).
    It's based on the assumption that foci arise from a realisation of a (continues)
    inhomogeneous Poisson process, so that the (discrete) voxel-wise foci counts will
    be independently distributed as Poisson random variables, with rate equal to the
    integral of the (true, unobserved, continous) intensity function over each voxels.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fisher_info_multiple_group_spatial(
        self,
        involved_groups,
        coef_spline_bases,
        moderators_by_group,
        foci_per_voxel,
        foci_per_experiment,
    ):
        """Estimate the spatial Fisher information analytically for the Poisson model."""
        tensor_inputs = self._resolve_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        ).subset(involved_groups)
        coef_spline_bases = tensor_inputs.coef_spline_bases

        group_fisher = []
        for group in involved_groups:
            group_spatial_coef = self.spatial_coef_linears[group].weight.T
            log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef).reshape(-1)
            mu_spatial = torch.exp(log_mu_spatial)

            if self.moderators_coef_dim:
                moderators_coef = self.moderators_linear.weight.T
                group_moderators = tensor_inputs.moderators_by_group[group]
                sum_moderator_effect = torch.exp(
                    torch.matmul(group_moderators, moderators_coef).reshape(-1)
                ).sum()
            else:
                sum_moderator_effect = torch.tensor(
                    float(tensor_inputs.foci_per_experiment[group].shape[0]),
                    dtype=torch.float64,
                    device=coef_spline_bases.device,
                )

            weighted_design = coef_spline_bases * mu_spatial[:, None]
            group_fisher.append(
                sum_moderator_effect * torch.matmul(coef_spline_bases.T, weighted_design)
            )

        return torch.block_diag(*group_fisher).detach().cpu().numpy()

    def fisher_info_multiple_group_moderator(
        self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_experiment
    ):
        """Estimate the moderator Fisher information analytically for the Poisson model."""
        if not self.moderators_coef_dim:
            return np.zeros((0, 0), dtype=np.float64)

        tensor_inputs = self._resolve_tensor_inputs(
            coef_spline_bases,
            moderators_by_group,
            foci_per_voxel,
            foci_per_experiment,
        )
        coef_spline_bases = tensor_inputs.coef_spline_bases
        moderator_coef = self.moderators_linear.weight.T

        fisher_info = torch.zeros(
            (self.moderators_coef_dim, self.moderators_coef_dim),
            dtype=torch.float64,
            device=coef_spline_bases.device,
        )
        for group in self.groups:
            group_spatial_coef = self.spatial_coef_linears[group].weight.T
            spatial_sum = torch.exp(
                torch.matmul(coef_spline_bases, group_spatial_coef).reshape(-1)
            ).sum()
            group_moderators = tensor_inputs.moderators_by_group[group]
            moderator_effect = torch.exp(
                torch.matmul(group_moderators, moderator_coef).reshape(-1)
            )
            weighted_design = group_moderators * moderator_effect[:, None]
            fisher_info += spatial_sum * torch.matmul(group_moderators.T, weighted_design)

        return fisher_info.detach().cpu().numpy()

    def _log_likelihood_single_group(
        self,
        group_spatial_coef,
        moderators_coef,
        coef_spline_bases,
        group_moderators,
        group_foci_per_voxel,
        group_foci_per_experiment,
        device="cpu",
    ):
        log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef.T).reshape(-1)
        group_foci_per_voxel = self._flatten_tensor(group_foci_per_voxel)
        mu_spatial = torch.exp(log_mu_spatial)
        group_foci_per_experiment = self._flatten_tensor(group_foci_per_experiment)
        if moderators_coef is None:
            log_mu_moderators = torch.zeros_like(group_foci_per_experiment, dtype=torch.float64)
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            log_mu_moderators = torch.matmul(group_moderators, moderators_coef.T).reshape(-1)
            mu_moderators = torch.exp(log_mu_moderators)
        log_l = (
            torch.dot(group_foci_per_voxel, log_mu_spatial)
            + torch.dot(group_foci_per_experiment, log_mu_moderators)
            - torch.sum(mu_spatial) * torch.sum(mu_moderators)
        )
        return log_l

    def _log_likelihood_mult_group(
        self,
        spatial_coef,
        moderator_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_experiment,
        moderators,
        device="cpu",
    ):
        log_spatial_intensity = torch.matmul(
            spatial_coef.transpose(1, 2), coef_spline_bases.T
        ).squeeze(1)
        foci_per_voxel_tensor = torch.stack(
            [
                self._flatten_tensor(group_foci_per_voxel)
                for group_foci_per_voxel in foci_per_voxel
            ],
            dim=0,
        )
        spatial_term = torch.sum(foci_per_voxel_tensor * log_spatial_intensity, dim=1)
        spatial_intensity_sum = torch.exp(log_spatial_intensity).sum(dim=1)

        moderator_term = []
        moderator_effect_sum = []
        if moderator_coef is not None:
            for group_moderator, group_foci_per_experiment in zip(moderators, foci_per_experiment):
                foci_per_experiment_flat = self._flatten_tensor(group_foci_per_experiment)
                log_moderator_effect = torch.matmul(group_moderator, moderator_coef).reshape(-1)
                moderator_term.append(torch.dot(foci_per_experiment_flat, log_moderator_effect))
                moderator_effect_sum.append(torch.exp(log_moderator_effect).sum())
        else:
            for foci_per_experiment_i in foci_per_experiment:
                foci_per_experiment_flat = self._flatten_tensor(foci_per_experiment_i)
                moderator_term.append(
                    torch.zeros((), dtype=torch.float64, device=foci_per_experiment_flat.device)
                )
                moderator_effect_sum.append(
                    torch.tensor(
                        float(foci_per_experiment_flat.numel()),
                        dtype=torch.float64,
                        device=foci_per_experiment_flat.device,
                    )
                )

        moderator_term = torch.stack(moderator_term)
        moderator_effect_sum = torch.stack(moderator_effect_sum)
        return torch.sum(
            spatial_term + moderator_term - spatial_intensity_sum * moderator_effect_sum
        )

    def forward(self, coef_spline_bases, moderators, foci_per_voxel, foci_per_experiment):
        """Define the loss function (nagetive log-likelihood function) for Poisson model.

        Model refactorization is applied to reduce the dimensionality of variables.

        Returns
        -------
        torch.Tensor
            Loss (nagative log-likelihood) of Poisson model at current iteration.
        """
        log_l = 0
        for group in self.groups:
            group_spatial_coef = self.spatial_coef_linears[group].weight
            group_foci_per_voxel = foci_per_voxel[group]
            group_foci_per_experiment = foci_per_experiment[group]
            if isinstance(moderators, dict):
                moderators_coef = self.moderators_linear.weight
                group_moderators = moderators[group]
            else:
                moderators_coef, group_moderators = None, None
            group_log_l = self._log_likelihood_single_group(
                group_spatial_coef,
                moderators_coef,
                coef_spline_bases,
                group_moderators,
                group_foci_per_voxel,
                group_foci_per_experiment,
            )
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            log_l += self.firth_penalty(
                foci_per_voxel,
                foci_per_experiment,
                moderators,
                coef_spline_bases,
                overdispersion=False,
            )
        return -log_l


class NegativeBinomialEstimator(OverdispersionModelEstimator):
    """CBMR framework with Negative Binomial (NB) model.

    Negative Binomial (NB) model is a generalized Poisson model with overdispersion.
    It's a more flexible model, but more difficult to estimate. In practice, foci
    counts often display over-dispersion (the variance of response variable
    substantially exceeeds the mean), which is not captured by Poisson model.
    """

    def __init__(self, **kwargs):
        kwargs["square_root"] = True
        super().__init__(**kwargs)

    def _three_term(self, y, r):
        max_foci = torch.max(y).to(dtype=torch.int64, device=self.device)
        sum_three_term = 0
        for k in range(max_foci):
            foci_index = (y == k + 1).nonzero()[:, 0]
            r_j = r[foci_index]
            n_voxel = list(foci_index.shape)[0]
            y_j = torch.tensor([k + 1] * n_voxel, device=self.device).double()
            y_j = y_j.reshape((n_voxel, 1))
            # y=0 => sum_three_term = 0
            sum_three_term += torch.sum(
                torch.lgamma(y_j + r_j) - torch.lgamma(y_j + 1) - torch.lgamma(r_j)
            )

        return sum_three_term

    def _log_likelihood_single_group(
        self,
        group_overdispersion,
        group_spatial_coef,
        moderators_coef,
        coef_spline_bases,
        group_moderators,
        group_foci_per_voxel,
        group_foci_per_experiment,
        device="cpu",
    ):
        log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef.T).reshape(-1)
        group_foci_per_voxel = self._flatten_tensor(group_foci_per_voxel)
        mu_spatial = torch.exp(log_mu_spatial)
        group_foci_per_experiment = self._flatten_tensor(group_foci_per_experiment)
        if moderators_coef is not None:
            log_mu_moderators = torch.matmul(group_moderators, moderators_coef.T).reshape(-1)
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            log_mu_moderators = torch.zeros_like(group_foci_per_experiment, dtype=torch.float64)
            mu_moderators = torch.exp(log_mu_moderators)
        # parameter of a NB variable to approximate a sum of NB variables
        r = 1 / group_overdispersion * torch.sum(mu_moderators) ** 2 / torch.sum(mu_moderators**2)
        p = 1 / (
            1
            + torch.sum(mu_moderators)
            / (group_overdispersion * mu_spatial * torch.sum(mu_moderators**2))
        )
        # log-likelihood (moment matching approach)
        log_l = torch.sum(
            torch.lgamma(group_foci_per_voxel + r)
            - torch.lgamma(group_foci_per_voxel + 1)
            - torch.lgamma(r)
            + r * torch.log(1 - p)
            + group_foci_per_voxel * torch.log(p)
        )

        return log_l

    def _log_likelihood_mult_group(
        self,
        overdispersion_coef,
        spatial_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_experiment,
        moderator_coef=None,
        moderators=None,
        device="cpu",
    ):
        n_groups = len(foci_per_voxel)
        flattened_foci_per_voxel = [self._flatten_tensor(f) for f in foci_per_voxel]
        flattened_foci_per_experiment = [self._flatten_tensor(f) for f in foci_per_experiment]
        log_spatial_intensity = []
        spatial_intensity = []
        for i in range(n_groups):
            log_si = torch.matmul(coef_spline_bases, spatial_coef[i, :, :]).reshape(-1)
            log_spatial_intensity.append(log_si)
            spatial_intensity.append(torch.exp(log_si))
        if moderator_coef is not None:
            log_moderator_effect = []
            moderator_effect = []
            for group_moderator in moderators:
                log_me = torch.matmul(group_moderator, moderator_coef).reshape(-1)
                log_moderator_effect.append(log_me)
                moderator_effect.append(torch.exp(log_me))
        else:
            log_moderator_effect = []
            moderator_effect = []
            for fps_flat in flattened_foci_per_experiment:
                log_me = torch.zeros_like(fps_flat, dtype=torch.float64)
                log_moderator_effect.append(log_me)
                moderator_effect.append(torch.exp(log_me))
        # After similification, we have:
        # r' = 1/alpha * sum(mu^Z_i)^2 / sum((mu^Z_i)^2)
        # p'_j = 1 / (1 + sum(mu^Z_i) / (alpha * mu^X_j * sum((mu^Z_i)^2)
        r = [
            1
            / overdispersion_coef[i]
            * torch.sum(moderator_effect[i]) ** 2
            / torch.sum(moderator_effect[i] ** 2)
            for i in range(n_groups)
        ]
        p_frac = [
            torch.sum(moderator_effect[i])
            / (overdispersion_coef[i] * spatial_intensity[i] * torch.sum(moderator_effect[i] ** 2))
            for i in range(n_groups)
        ]
        p = [1 / (1 + p_frac[i]) for i in range(n_groups)]

        log_l = 0
        for i in range(n_groups):
            group_log_l = torch.sum(
                torch.lgamma(flattened_foci_per_voxel[i] + r[i])
                - torch.lgamma(flattened_foci_per_voxel[i] + 1)
                - torch.lgamma(r[i])
                + r[i] * torch.log(1 - p[i])
                + flattened_foci_per_voxel[i] * torch.log(p[i])
            )
            log_l += group_log_l

        return log_l

    def forward(self, coef_spline_bases, moderators, foci_per_voxel, foci_per_experiment):
        """Define the loss function (nagetive log-likelihood function) for NB model.

        Model refactorization is applied to reduce the dimensionality of variables.

        Returns
        -------
        torch.Tensor
            Loss (nagative log-likelihood) of NB model at current iteration.
        """
        log_l = 0
        for group in self.groups:
            group_overdispersion = self.overdispersion[group] ** 2
            group_spatial_coef = self.spatial_coef_linears[group].weight
            group_foci_per_voxel = foci_per_voxel[group]
            group_foci_per_experiment = foci_per_experiment[group]
            if isinstance(moderators, dict):
                moderators_coef = self.moderators_linear.weight
                group_moderators = moderators[group]
            else:
                moderators_coef, group_moderators = None, None
            group_log_l = self._log_likelihood_single_group(
                group_overdispersion,
                group_spatial_coef,
                moderators_coef,
                coef_spline_bases,
                group_moderators,
                group_foci_per_voxel,
                group_foci_per_experiment,
            )

            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            log_l += self.firth_penalty(
                foci_per_voxel,
                foci_per_experiment,
                moderators,
                coef_spline_bases,
                overdispersion=True,
            )

        return -log_l


class ClusteredNegativeBinomialEstimator(OverdispersionModelEstimator):
    """CBMR framework with Clustered Negative Binomial (Clustered NB) model.

    Clustered NB model can also accommodate over-dispersion in foci counts.
    In NB model, the latent Gamma random variable introduces indepdentent variation
    at each voxel. While in Clustered NB model, we assert the random effects are not
    independent voxelwise effects, but rather latent characteristics of each experiment,
    and represent a shared effect over the entire brain for a given experiment.
    """

    def __init__(self, **kwargs):
        kwargs["square_root"] = False
        super().__init__(**kwargs)

    def _log_likelihood_single_group(
        self,
        group_overdispersion,
        group_spatial_coef,
        moderators_coef,
        coef_spline_bases,
        group_moderators,
        group_foci_per_voxel,
        group_foci_per_experiment,
        device="cpu",
    ):
        v = 1 / group_overdispersion
        log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef.T).reshape(-1)
        group_foci_per_voxel = self._flatten_tensor(group_foci_per_voxel)
        mu_spatial = torch.exp(log_mu_spatial)
        group_foci_per_experiment = self._flatten_tensor(group_foci_per_experiment)
        if moderators_coef is not None:
            log_mu_moderators = torch.matmul(group_moderators, moderators_coef.T).reshape(-1)
        else:
            log_mu_moderators = torch.zeros_like(group_foci_per_experiment, dtype=torch.float64)
        mu_moderators = torch.exp(log_mu_moderators)
        mu_sum_per_experiment = torch.sum(mu_spatial) * mu_moderators
        group_n_experiment = group_foci_per_experiment.shape[0]

        log_l = (
            group_n_experiment * v * torch.log(v)
            - group_n_experiment * torch.lgamma(v)
            + torch.sum(torch.lgamma(group_foci_per_experiment + v))
            - torch.sum((group_foci_per_experiment + v) * torch.log(mu_sum_per_experiment + v))
            + torch.dot(group_foci_per_voxel, log_mu_spatial)
            + torch.dot(group_foci_per_experiment, log_mu_moderators)
        )

        return log_l

    def _log_likelihood_mult_group(
        self,
        overdispersion_coef,
        spatial_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_experiment,
        moderator_coef=None,
        moderators=None,
        device="cpu",
    ):
        n_groups = len(foci_per_voxel)
        v = [1 / group_overdispersion_coef for group_overdispersion_coef in overdispersion_coef]
        # estimated intensity and log estimated intensity
        flattened_foci_per_voxel = [self._flatten_tensor(f) for f in foci_per_voxel]
        flattened_foci_per_experiment = [self._flatten_tensor(f) for f in foci_per_experiment]
        log_spatial_intensity = []
        spatial_intensity = []
        for i in range(n_groups):
            log_si = torch.matmul(coef_spline_bases, spatial_coef[i, :, :]).reshape(-1)
            log_spatial_intensity.append(log_si)
            spatial_intensity.append(torch.exp(log_si))
        if moderator_coef is not None:
            log_moderator_effect = []
            moderator_effect = []
            for group_moderator in moderators:
                log_me = torch.matmul(group_moderator, moderator_coef).reshape(-1)
                log_moderator_effect.append(log_me)
                moderator_effect.append(torch.exp(log_me))
        else:
            log_moderator_effect = []
            moderator_effect = []
            for fps_flat in flattened_foci_per_experiment:
                log_me = torch.zeros_like(fps_flat, dtype=torch.float64)
                log_moderator_effect.append(log_me)
                moderator_effect.append(torch.exp(log_me))
        mu_sum_per_experiment = [
            torch.sum(spatial_intensity[i]) * moderator_effect[i] for i in range(n_groups)
        ]
        n_experiment_list = [
            group_foci_per_experiment.shape[0]
            for group_foci_per_experiment in flattened_foci_per_experiment
        ]

        log_l = 0
        for i in range(n_groups):
            log_l += (
                n_experiment_list[i] * v[i] * torch.log(v[i])
                - n_experiment_list[i] * torch.lgamma(v[i])
                + torch.sum(torch.lgamma(flattened_foci_per_experiment[i] + v[i]))
                - torch.sum(
                    (flattened_foci_per_experiment[i] + v[i])
                    * torch.log(mu_sum_per_experiment[i] + v[i])
                )
                + torch.dot(flattened_foci_per_voxel[i], log_spatial_intensity[i])
                + torch.dot(flattened_foci_per_experiment[i], log_moderator_effect[i])
            )

        return log_l

    def forward(self, coef_spline_bases, moderators, foci_per_voxel, foci_per_experiment):
        """Define the loss function (nagetive log-likelihood function) for Clustered NB model.

        Model refactorization is applied to reduce the dimensionality of variables.

        Returns
        -------
        torch.Tensor
            Loss (nagative log-likelihood) of Poisson model at current iteration.
        """
        log_l = 0
        for group in self.groups:
            group_overdispersion = self.overdispersion[group]
            group_spatial_coef = self.spatial_coef_linears[group].weight
            group_foci_per_voxel = foci_per_voxel[group]
            group_foci_per_experiment = foci_per_experiment[group]
            if isinstance(moderators, dict):
                moderators_coef = self.moderators_linear.weight
                group_moderators = moderators[group]
            else:
                moderators_coef, group_moderators = None, None
            group_log_l = self._log_likelihood_single_group(
                group_overdispersion,
                group_spatial_coef,
                moderators_coef,
                coef_spline_bases,
                group_moderators,
                group_foci_per_voxel,
                group_foci_per_experiment,
            )
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            log_l += self.firth_penalty(
                foci_per_voxel,
                foci_per_experiment,
                moderators,
                coef_spline_bases,
                overdispersion=True,
            )

        return -log_l
