"""CBMR Models."""

import abc
import logging

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Torch is required to use `CBMR` models. Install with `pip install 'nimare[cbmr]'`."
    ) from e

LGR = logging.getLogger(__name__)


class GeneralLinearModelEstimator(torch.nn.Module):
    """Base class for GLM estimators.

    Parameters
    ----------
    spatial_coef_dim : :obj:`int`
        Number of spatial B-spline bases. Default is None.
    moderators_coef_dim : :obj:`int`, optional
        Number of study-level moderators. Default is None.
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
        self.moderators_linear = torch.nn.Linear(self.moderators_coef_dim, 1, bias=False).double()
        torch.nn.init.uniform_(self.moderators_linear.weight, a=-0.01, b=0.01)
        return

    def init_weights(self, groups, moderators, spatial_coef_dim, moderators_coef_dim):
        """Initialize regression coefficients of spatial struture and study-level moderators."""
        self.groups = groups
        self.moderators = moderators
        self.spatial_coef_dim = spatial_coef_dim
        self.moderators_coef_dim = moderators_coef_dim
        self.init_spatial_weights()
        if moderators_coef_dim:
            self.init_moderator_weights()

    def _update(
        self,
        optimizer,
        coef_spline_bases,
        moderators,
        foci_per_voxel,
        foci_per_study,
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
            Dictionary of group-wise study-level moderators. Default is None.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_study : :obj:`dict`
            Dictionary of group-wise number of foci per study.
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
            loss = self(coef_spline_bases, moderators, foci_per_voxel, foci_per_study)
            loss.backward()
            return loss

        optimizer.step(closure)
        scheduler.step()
        # recalculate the loss function
        loss = self(coef_spline_bases, moderators, foci_per_voxel, foci_per_study)

        if torch.isnan(loss):
            raise ValueError(
                f"""The current learing rate {str(self.lr)} or choice of model gives rise to
                NaN log-likelihood, please try Poisson model or adjust learning rate to a smaller
                value."""
            )

        return loss

    def _optimizer(self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study):
        """
        Optimize the loss (negative log-likelihood) function with L-BFGS.

        Parameters
        ----------
        coef_spline_bases : :obj:`numpy.ndarray`
            Coefficient of B-spline bases evaluated at each voxel.
        moderators_by_group : :obj:`dict`, optional
            Dictionary of group-wise study-level moderators.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_study : :obj:`dict`
            Dictionary of group-wise number of foci per study.
        """
        torch.manual_seed(100)
        optimizer = torch.optim.LBFGS(
            params=self.parameters(),
            lr=self.lr,
            max_iter=self.n_iter,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe",
        )
        # load dataset info to torch.tensor
        coef_spline_bases = torch.tensor(
            coef_spline_bases, dtype=torch.float64, device=self.device
        )
        if moderators_by_group:
            moderators_by_group_tensor = dict()
            for group in self.groups:
                moderators_tensor = torch.tensor(
                    moderators_by_group[group], dtype=torch.float64, device=self.device
                )
                moderators_by_group_tensor[group] = moderators_tensor
        else:
            moderators_by_group_tensor = None
        foci_per_voxel_tensor, foci_per_study_tensor = dict(), dict()
        for group in self.groups:
            group_foci_per_voxel_tensor = torch.tensor(
                foci_per_voxel[group], dtype=torch.float64, device=self.device
            )
            group_foci_per_study_tensor = torch.tensor(
                foci_per_study[group], dtype=torch.float64, device=self.device
            )
            foci_per_voxel_tensor[group] = group_foci_per_voxel_tensor
            foci_per_study_tensor[group] = group_foci_per_study_tensor

        if self.iter == 0:
            prev_loss = torch.tensor(float("inf"))  # initialization loss difference

        self._update(
            optimizer,
            coef_spline_bases,
            moderators_by_group_tensor,
            foci_per_voxel_tensor,
            foci_per_study_tensor,
            prev_loss,
        )

        return

    def fit(self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study):
        """Fit the model and estimate standard error of estimates."""
        self._optimizer(coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study)
        self.extract_optimized_params(coef_spline_bases, moderators_by_group)
        self.standard_error_estimation(
            coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study
        )

        return

    def extract_optimized_params(self, coef_spline_bases, moderators_by_group):
        """Extract optimized regression coefficient of study-level moderators from the model."""
        spatial_regression_coef, spatial_intensity_estimation = dict(), dict()
        for group in self.groups:
            # Extract optimized spatial regression coefficients from the model
            group_spatial_coef_linear_weight = self.spatial_coef_linears[group].weight
            group_spatial_coef_linear_weight = (
                group_spatial_coef_linear_weight.cpu().detach().numpy().flatten()
            )
            spatial_regression_coef[group] = group_spatial_coef_linear_weight
            # Estimate group-specific spatial intensity
            group_spatial_intensity_estimation = np.exp(
                np.matmul(coef_spline_bases, group_spatial_coef_linear_weight)
            )
            spatial_intensity_estimation["spatialIntensity_group-" + group] = (
                group_spatial_intensity_estimation
            )

        # Extract optimized regression coefficient of study-level moderators from the model
        if self.moderators_coef_dim:
            moderators_effect = dict()
            moderators_coef = self.moderators_linear.weight
            moderators_coef = moderators_coef.cpu().detach().numpy()
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
        self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study
    ):
        """Estimate standard error of estimates.

        For spatial regression coefficients, we estimate its covariance matrix using Fisher
        Information Matrix and then take the square root of the diagonal elements.
        For log spatial intensity, we use the delta method to estimate its standard error.
        For models with over-dispersion parameter, we also estimate its standard error.
        """
        spatial_regression_coef_se, log_spatial_intensity_se, spatial_intensity_se = (
            dict(),
            dict(),
            dict(),
        )
        for group in self.groups:
            group_foci_per_voxel = torch.tensor(
                foci_per_voxel[group], dtype=torch.float64, device=self.device
            )
            group_foci_per_study = torch.tensor(
                foci_per_study[group], dtype=torch.float64, device=self.device
            )
            group_spatial_coef = self.spatial_coef_linears[group].weight
            if self.moderators_coef_dim:
                group_moderators = torch.tensor(
                    moderators_by_group[group], dtype=torch.float64, device=self.device
                )
                moderators_coef = self.moderators_linear.weight
            else:
                group_moderators, moderators_coef = None, None

            ll_single_group_kwargs = {
                "moderators_coef": moderators_coef if self.moderators_coef_dim else None,
                "coef_spline_bases": torch.tensor(
                    coef_spline_bases, dtype=torch.float64, device=self.device
                ),
                "group_moderators": group_moderators if self.moderators_coef_dim else None,
                "group_foci_per_voxel": group_foci_per_voxel,
                "group_foci_per_study": group_foci_per_study,
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
            cov_spatial_coef = np.linalg.inv(f_spatial_coef.detach().numpy())
            var_spatial_coef = np.diag(cov_spatial_coef)
            se_spatial_coef = np.sqrt(var_spatial_coef)
            spatial_regression_coef_se[group] = se_spatial_coef

            var_log_spatial_intensity = np.einsum(
                "ij,ji->i",
                coef_spline_bases,
                cov_spatial_coef @ coef_spline_bases.T,
            )
            se_log_spatial_intensity = np.sqrt(var_log_spatial_intensity)
            log_spatial_intensity_se[group] = se_log_spatial_intensity

            group_studywise_spatial_intensity = np.exp(
                np.matmul(coef_spline_bases, group_spatial_coef.detach().cpu().numpy().T)
            ).flatten()
            se_spatial_intensity = group_studywise_spatial_intensity * se_log_spatial_intensity
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
            cov_moderators_coef = np.linalg.inv(f_moderators_coef.detach().numpy())
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
        tables["spatial_regression_coef"] = pd.DataFrame.from_dict(
            self.spatial_regression_coef, orient="index"
        )
        maps = self.spatial_intensity_estimation
        if self.moderators_coef_dim:
            tables["moderators_regression_coef"] = pd.DataFrame(
                data=self.moderators_coef, columns=self.moderators
            )
            tables["moderators_effect"] = pd.DataFrame.from_dict(
                data=self.moderators_effect, orient="index"
            )

        # Estimate standard error of regression coefficient and (Log-)spatial intensity and store
        # them in 'tables'
        tables["spatial_regression_coef_se"] = pd.DataFrame.from_dict(
            self.spatial_regression_coef_se, orient="index"
        )
        tables["log_spatial_intensity_se"] = pd.DataFrame.from_dict(
            self.log_spatial_intensity_se, orient="index"
        )
        tables["spatial_intensity_se"] = pd.DataFrame.from_dict(
            self.spatial_intensity_se, orient="index"
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
        foci_per_study,
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
            Dictionary of group-wise study-level moderators. Default is None.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_study : :obj:`dict`
            Dictionary of group-wise number of foci per study.

        Returns
        -------
        numpy.ndarray
            Fisher information matrix of spatial regression coefficients (for involved groups).
        """
        n_involved_groups = len(involved_groups)
        involved_foci_per_voxel = [
            torch.tensor(foci_per_voxel[group], dtype=torch.float64, device=self.device)
            for group in involved_groups
        ]
        involved_foci_per_study = [
            torch.tensor(foci_per_study[group], dtype=torch.float64, device=self.device)
            for group in involved_groups
        ]
        spatial_coef = [self.spatial_coef_linears[group].weight.T for group in involved_groups]
        spatial_coef = torch.stack(spatial_coef, dim=0)
        if self.moderators_coef_dim:
            involved_moderators_by_group = [
                torch.tensor(moderators_by_group[group], dtype=torch.float64, device=self.device)
                for group in involved_groups
            ]
            moderators_coef = torch.tensor(
                self.moderators_coef.T, dtype=torch.float64, device=self.device
            )
        else:
            involved_moderators_by_group, moderators_coef = None, None

        ll_mult_group_kwargs = {
            "moderator_coef": moderators_coef,
            "coef_spline_bases": torch.tensor(
                coef_spline_bases, dtype=torch.float64, device=self.device
            ),
            "foci_per_voxel": involved_foci_per_voxel,
            "foci_per_study": involved_foci_per_study,
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
        self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study
    ):
        """Estimate the Fisher information matrix of regression coefficients of moderators.

        Fisher information matrix is estimated by negative Hessian of the log-likelihood.

        Parameters
        ----------
        coef_spline_bases : :obj:`numpy.ndarray`
            Coefficient of B-spline bases evaluated at each voxel.
        moderators_by_group : :obj:`dict`, optional
            Dictionary of group-wise study-level moderators. Default is None.
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_study : :obj:`dict`
            Dictionary of group-wise number of foci per study.

        Returns
        -------
        numpy.ndarray
            Fisher information matrix of study-level moderator regressors.
        """
        foci_per_voxel = [
            torch.tensor(foci_per_voxel[group], dtype=torch.float64, device=self.device)
            for group in self.groups
        ]
        foci_per_study = [
            torch.tensor(foci_per_study[group], dtype=torch.float64, device=self.device)
            for group in self.groups
        ]
        spatial_coef = [self.spatial_coef_linears[group].weight.T for group in self.groups]
        spatial_coef = torch.stack(spatial_coef, dim=0)

        if self.moderators_coef_dim:
            moderators_by_group = [
                torch.tensor(moderators_by_group[group], dtype=torch.float64, device=self.device)
                for group in self.groups
            ]
            moderator_coef = torch.tensor(
                self.moderators_coef.T, dtype=torch.float64, device=self.device
            )
        else:
            moderators_by_group, moderator_coef = None, None

        ll_mult_group_kwargs = {
            "spatial_coef": spatial_coef,
            "coef_spline_bases": torch.tensor(
                coef_spline_bases, dtype=torch.float64, device=self.device
            ),
            "foci_per_voxel": foci_per_voxel,
            "foci_per_study": foci_per_study,
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
        foci_per_study,
        moderators,
        coef_spline_bases,
        overdispersion=False,
    ):
        """Compute Firth's penalized log-likelihood.

        Parameters
        ----------
        foci_per_voxel : :obj:`dict`
            Dictionary of group-wise number of foci per voxel.
        foci_per_study : :obj:`dict`
            Dictionary of group-wise number of foci per study.
        moderators : :obj:`dict`, optional
            Dictionary of group-wise study-level moderators. Default is None.
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
            partial_kwargs["group_foci_per_study"] = foci_per_study[group]
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
            overdispersion_init_group = torch.tensor(1e-2).double()
            if self.square_root:
                overdispersion_init_group = torch.sqrt(overdispersion_init_group)
            overdispersion[group] = torch.nn.Parameter(
                overdispersion_init_group, requires_grad=True
            )
        self.overdispersion = torch.nn.ParameterDict(overdispersion)

    def init_weights(self, groups, moderators, spatial_coef_dim, moderators_coef_dim):
        """Initialize weights for spatial and study-level moderator coefficients."""
        super().init_weights(groups, moderators, spatial_coef_dim, moderators_coef_dim)
        self.init_overdispersion_weights()

    def inference_outcome(
        self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study
    ):
        """Summarize inference outcome into `maps` and `tables`.

        Add optimized overdispersion parameter to the tables.
        """
        maps, tables = super().inference_outcome(
            coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study
        )
        overdispersion_param = dict()
        for group in self.groups:
            group_overdispersion = self.overdispersion[group]
            group_overdispersion = group_overdispersion.cpu().detach().numpy()
            overdispersion_param[group] = group_overdispersion
        tables["overdispersion_coef"] = pd.DataFrame.from_dict(
            overdispersion_param, orient="index", columns=["overdispersion"]
        )

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

    def _log_likelihood_single_group(
        self,
        group_spatial_coef,
        moderators_coef,
        coef_spline_bases,
        group_moderators,
        group_foci_per_voxel,
        group_foci_per_study,
        device="cpu",
    ):
        log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if moderators_coef is None:
            n_study, _ = group_foci_per_study.shape
            log_mu_moderators = torch.tensor(
                [0] * n_study, dtype=torch.float64, device=device
            ).reshape((-1, 1))
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            log_mu_moderators = torch.matmul(group_moderators, moderators_coef.T)
            mu_moderators = torch.exp(log_mu_moderators)
        log_l = (
            torch.sum(torch.mul(group_foci_per_voxel, log_mu_spatial))
            + torch.sum(torch.mul(group_foci_per_study, log_mu_moderators))
            - torch.sum(mu_spatial) * torch.sum(mu_moderators)
        )
        return log_l

    def _log_likelihood_mult_group(
        self,
        spatial_coef,
        moderator_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_study,
        moderators,
        device="cpu",
    ):
        n_groups = len(spatial_coef)
        log_spatial_intensity = [
            torch.matmul(coef_spline_bases, spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        spatial_intensity = [
            torch.exp(group_log_spatial_intensity)
            for group_log_spatial_intensity in log_spatial_intensity
        ]
        if moderator_coef is not None:
            log_moderator_effect = [
                torch.matmul(group_moderator, moderator_coef) for group_moderator in moderators
            ]
            moderator_effect = [
                torch.exp(group_log_moderator_effect)
                for group_log_moderator_effect in log_moderator_effect
            ]
        else:
            log_moderator_effect = [
                torch.tensor(
                    [0] * foci_per_study_i.shape[0], dtype=torch.float64, device=device
                ).reshape((-1, 1))
                for foci_per_study_i in foci_per_study
            ]
            moderator_effect = [
                torch.exp(group_log_moderator_effect)
                for group_log_moderator_effect in log_moderator_effect
            ]
        log_l = 0
        for i in range(n_groups):
            log_l += (
                torch.sum(foci_per_voxel[i] * log_spatial_intensity[i])
                + torch.sum(foci_per_study[i] * log_moderator_effect[i])
                - torch.sum(spatial_intensity[i]) * torch.sum(moderator_effect[i])
            )
        return log_l

    def forward(self, coef_spline_bases, moderators, foci_per_voxel, foci_per_study):
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
            group_foci_per_study = foci_per_study[group]
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
                group_foci_per_study,
            )
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            log_l += self.firth_penalty(
                foci_per_voxel,
                foci_per_study,
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
        group_foci_per_study,
        device="cpu",
    ):
        log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if moderators_coef is not None:
            log_mu_moderators = torch.matmul(group_moderators, moderators_coef.T)
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            n_study, _ = group_foci_per_study.shape
            log_mu_moderators = torch.tensor(
                [0] * n_study, dtype=torch.float64, device=device
            ).reshape((-1, 1))
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
        foci_per_study,
        moderator_coef=None,
        moderators=None,
        device="cpu",
    ):
        n_groups = len(foci_per_voxel)
        log_spatial_intensity = [
            torch.matmul(coef_spline_bases, spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        spatial_intensity = [
            torch.exp(group_log_spatial_intensity)
            for group_log_spatial_intensity in log_spatial_intensity
        ]
        if moderator_coef is not None:
            log_moderator_effect = [
                torch.matmul(group_moderator, moderator_coef) for group_moderator in moderators
            ]
            moderator_effect = [
                torch.exp(group_log_moderator_effect)
                for group_log_moderator_effect in log_moderator_effect
            ]
        else:
            log_moderator_effect = [
                torch.tensor(
                    [0] * foci_per_study.shape[0], dtype=torch.float64, device=device
                ).reshape((-1, 1))
                for foci_per_study in foci_per_study
            ]
            moderator_effect = [
                torch.exp(group_log_moderator_effect)
                for group_log_moderator_effect in log_moderator_effect
            ]
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
                torch.lgamma(foci_per_voxel[i] + r[i])
                - torch.lgamma(foci_per_voxel[i] + 1)
                - torch.lgamma(r[i])
                + r[i] * torch.log(1 - p[i])
                + foci_per_voxel[i] * torch.log(p[i])
            )
            log_l += group_log_l

        return log_l

    def forward(self, coef_spline_bases, moderators, foci_per_voxel, foci_per_study):
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
            group_foci_per_study = foci_per_study[group]
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
                group_foci_per_study,
            )

            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            log_l += self.firth_penalty(
                foci_per_voxel,
                foci_per_study,
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
    independent voxelwise effects, but rather latent characteristics of each study,
    and represent a shared effect over the entire brain for a given study.
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
        group_foci_per_study,
        device="cpu",
    ):
        v = 1 / group_overdispersion
        log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if moderators_coef is not None:
            log_mu_moderators = torch.matmul(group_moderators, moderators_coef.T)
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            n_study, _ = group_foci_per_study.shape
            log_mu_moderators = torch.tensor(
                [0] * n_study, dtype=torch.float64, device=device
            ).reshape((-1, 1))
            mu_moderators = torch.exp(log_mu_moderators)
        mu_sum_per_study = torch.sum(mu_spatial) * mu_moderators
        group_n_study, _ = group_foci_per_study.shape

        log_l = (
            group_n_study * v * torch.log(v)
            - group_n_study * torch.lgamma(v)
            + torch.sum(torch.lgamma(group_foci_per_study + v))
            - torch.sum((group_foci_per_study + v) * torch.log(mu_sum_per_study + v))
            + torch.sum(group_foci_per_voxel * log_mu_spatial)
            + torch.sum(group_foci_per_study * log_mu_moderators)
        )

        return log_l

    def _log_likelihood_mult_group(
        self,
        overdispersion_coef,
        spatial_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_study,
        moderator_coef=None,
        moderators=None,
        device="cpu",
    ):
        n_groups = len(foci_per_voxel)
        v = [1 / group_overdispersion_coef for group_overdispersion_coef in overdispersion_coef]
        # estimated intensity and log estimated intensity
        log_spatial_intensity = [
            torch.matmul(coef_spline_bases, spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        spatial_intensity = [
            torch.exp(group_log_spatial_intensity)
            for group_log_spatial_intensity in log_spatial_intensity
        ]
        if moderator_coef is not None:
            log_moderator_effect = [
                torch.matmul(group_moderator, moderator_coef) for group_moderator in moderators
            ]
            moderator_effect = [
                torch.exp(group_log_moderator_effect)
                for group_log_moderator_effect in log_moderator_effect
            ]
        else:
            log_moderator_effect = [
                torch.tensor(
                    [0] * foci_per_study.shape[0], dtype=torch.float64, device=device
                ).reshape((-1, 1))
                for foci_per_study in foci_per_study
            ]
            moderator_effect = [
                torch.exp(group_log_moderator_effect)
                for group_log_moderator_effect in log_moderator_effect
            ]
        mu_sum_per_study = [
            torch.sum(spatial_intensity[i]) * moderator_effect[i] for i in range(n_groups)
        ]
        n_study_list = [group_foci_per_study.shape[0] for group_foci_per_study in foci_per_study]

        log_l = 0
        for i in range(n_groups):
            log_l += (
                n_study_list[i] * v[i] * torch.log(v[i])
                - n_study_list[i] * torch.lgamma(v[i])
                + torch.sum(torch.lgamma(foci_per_study[i] + v[i]))
                - torch.sum((foci_per_study[i] + v[i]) * torch.log(mu_sum_per_study[i] + v[i]))
                + torch.sum(foci_per_voxel[i] * log_spatial_intensity[i])
                + torch.sum(foci_per_study[i] * log_moderator_effect[i])
            )

        return log_l

    def forward(self, coef_spline_bases, moderators, foci_per_voxel, foci_per_study):
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
            group_foci_per_study = foci_per_study[group]
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
                group_foci_per_study,
            )
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            log_l += self.firth_penalty(
                foci_per_voxel,
                foci_per_study,
                moderators,
                coef_spline_bases,
                overdispersion=True,
            )

        return -log_l
