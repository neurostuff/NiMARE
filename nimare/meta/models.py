
import abc
import torch
import numpy as np
import pandas as pd
import functorch
import logging
import copy

LGR = logging.getLogger(__name__)
class GeneralLinearModelEstimator(torch.nn.Module):
    def __init__(
        self,
        spatial_coef_dim=None,
        moderators_coef_dim=None,
        groups=None,
        penalty=False,
        lr = 0.1,
        lr_decay=0.999,
        n_iter=1000,
        tol=1e-2,
        device="cpu",
    ):  
        super().__init__()
        self.spatial_coef_dim = spatial_coef_dim
        self.moderators_coef_dim = moderators_coef_dim
        self.groups = groups
        self.penalty = penalty
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_iter = n_iter
        self.tol = tol
        self.device = device

        # initialization for spatial regression coefficients
        if self.spatial_coef_dim and self.groups:
            self.init_spatial_weights()
        # initialization for regression coefficients of moderators
        if self.moderators_coef_dim:
            self.init_moderator_weights()
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
        """Document this."""
        return

    @abc.abstractmethod
    def _log_likelihood_mult_group(self, **kwargs):
        """Document this."""
        return

    @abc.abstractmethod
    def forward(self, **kwargs):
        """Document this."""
        return

    def init_spatial_weights(self):
        """Document this."""
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
        """Document this."""
        self.moderators_linear = torch.nn.Linear(
            self.moderators_coef_dim, 1, bias=False
        ).double()
        torch.nn.init.uniform_(self.moderators_linear.weight, a=-0.01, b=0.01)

    def init_weights(self, groups, spatial_coef_dim, moderators_coef_dim):
        """Document this."""
        self.groups = groups
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

        loss = optimizer.step(closure)
        scheduler.step()
        # reset the L-BFGS params if NaN appears in coefficient of regression
        if any(
            [
                torch.any(torch.isnan(self.spatial_coef_linears[group].weight))
                for group in self.groups
            ]
        ):
            if self.iter == 1:  # NaN occurs in the first iteration
                raise ValueError(
                    """The current learing rate {str(self.lr)} gives rise to NaN values, adjust
                    to a smaller value."""
                )
            spatial_coef_linears, overdispersion = dict(), dict()
            for group in self.groups:
                group_spatial_linear = torch.nn.Linear(self.spatial_coef_dim, 1, bias=False).double()
                group_spatial_linear.weight = torch.nn.Parameter(
                    self.last_state["spatial_coef_linears." + group + ".weight"]
                )
                spatial_coef_linears[group] = group_spatial_linear

                if hasattr(self, "overdispersion"):
                    group_overdispersion = torch.nn.Parameter(
                        self.last_state["overdispersion." + group]
                    )
                    overdispersion[group] = group_overdispersion
            self.spatial_coef_linears = torch.nn.ModuleDict(spatial_coef_linears)
            if hasattr(self, "overdispersion"):
                self.overdispersion = torch.nn.ParameterDict(overdispersion)
            LGR.debug("Reset L-BFGS optimizer......")
        else:
            self.last_state = copy.deepcopy(
                self.state_dict()
            ) 

        return loss
    
    def _optimizer(self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study):
        optimizer = torch.optim.LBFGS(self.parameters(), self.lr)
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

        for i in range(self.n_iter):
            loss = self._update(
                optimizer,
                coef_spline_bases,
                moderators_by_group_tensor,
                foci_per_voxel_tensor,
                foci_per_study_tensor,
                prev_loss,
            )
            loss_diff = loss - prev_loss
            LGR.debug(f"Iter {self.iter:04d}: log-likelihood {loss:.4f}")
            if torch.abs(loss_diff) < self.tol:
                break
            prev_loss = loss

        return

    def fit(self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study):
        """Fit the model."""
        self._optimizer(coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study)
        self.extract_optimized_params(coef_spline_bases, moderators_by_group)
        self.standard_error_estimation(coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study)

        return

    def extract_optimized_params(self, coef_spline_bases, moderators_by_group):
        """Document this."""
        spatial_regression_coef, spatial_intensity_estimation = dict(), dict()
        for group in self.groups:
            # Extract optimized spatial regression coefficients from the model
            group_spatial_coef_linear_weight = self.spatial_coef_linears[group].weight
            group_spatial_coef_linear_weight = group_spatial_coef_linear_weight.cpu().detach().numpy().flatten()
            spatial_regression_coef[group] = group_spatial_coef_linear_weight
            # Estimate group-specific spatial intensity
            group_spatial_intensity_estimation = np.exp(np.matmul(coef_spline_bases, group_spatial_coef_linear_weight))
            spatial_intensity_estimation["Group_" + group + "_Studywise_Spatial_Intensity"] = group_spatial_intensity_estimation
            
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

    def standard_error_estimation(self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study):
        """Document this."""
        spatial_regression_coef_se, log_spatial_intensity_se, spatial_intensity_se = dict(), dict(), dict()
        for group in self.groups:
            group_foci_per_voxel = torch.tensor(
                foci_per_voxel[group], dtype=torch.float64, device=self.device)
            group_foci_per_study = torch.tensor(
                foci_per_study[group], dtype=torch.float64, device=self.device
            )
            group_spatial_coef = torch.tensor(self.spatial_coef_linears[group].weight,
                                              dtype=torch.float64, device=self.device)
            
            if self.moderators_coef_dim:
                group_moderators = torch.tensor(
                    moderators_by_group[group], dtype=torch.float64, device=self.device
                )
                moderators_coef = torch.tensor(self.moderators_linear.weight, dtype=torch.float64, device=self.device)
            else:
                group_moderators, moderators_coef = None, None
            
            ll_single_group_kwargs = {
                "moderators_coef": moderators_coef if self.moderators_coef_dim else None,
                "coef_spline_bases": torch.tensor(coef_spline_bases, dtype=torch.float64, device=self.device),
                "group_moderators": group_moderators if self.moderators_coef_dim else None,
                "group_foci_per_voxel": group_foci_per_voxel,
                "group_foci_per_study": group_foci_per_study,
                "device": self.device,
            }

            if hasattr(self, "overdispersion"):
                ll_single_group_kwargs['group_overdispersion'] = self.overdispersion[group]
            # create a negative log-likelihood function
            def nll_spatial_coef(group_spatial_coef):
                return -self._log_likelihood_single_group(
                    group_spatial_coef=group_spatial_coef, **ll_single_group_kwargs,
                )
            F_spatial_coef = functorch.hessian(nll_spatial_coef)(group_spatial_coef)
            F_spatial_coef = F_spatial_coef.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
            cov_spatial_coef = np.linalg.inv(F_spatial_coef.detach().numpy())
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
                    moderators_coef=moderators_coef, **ll_single_group_kwargs,
                )
            F_moderators_coef = functorch.hessian(nll_moderators_coef)(moderators_coef)
            F_moderators_coef = F_moderators_coef.reshape((self.moderators_coef_dim, self.moderators_coef_dim))
            cov_moderators_coef = np.linalg.inv(F_moderators_coef.detach().numpy())
            var_moderators = np.diag(cov_moderators_coef).reshape((1, self.moderators_coef_dim))
            se_moderators = np.sqrt(var_moderators)
        else: 
            se_moderators = None

        self.spatial_regression_coef_se = spatial_regression_coef_se
        self.log_spatial_intensity_se = log_spatial_intensity_se
        self.spatial_intensity_se = spatial_intensity_se
        self.se_moderators = se_moderators

    def summary(self):
        """Document this."""
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
        tables["Spatial_Regression_Coef"] = pd.DataFrame.from_dict(self.spatial_regression_coef, orient="index")
        maps = self.spatial_intensity_estimation
        if self.moderators_coef_dim:
            tables["Moderators_Regression_Coef"] = pd.DataFrame(self.moderators_coef)
            tables["Moderators_Effect"] = pd.DataFrame.from_dict(self.moderators_effect, orient="index")
        
        # Estimate standard error of regression coefficient and (Log-)spatial intensity and store them in 'tables'
        # spatial_regression_coef_se, log_spatial_intensity_se, spatial_intensity_se, se_moderators = self.standard_error_estimation(coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study)
        tables["Spatial_Regression_Coef_SE"] = pd.DataFrame.from_dict(
            self.spatial_regression_coef_se, orient="index"
        )
        tables["Log_Spatial_Intensity_SE"] = pd.DataFrame.from_dict(
            self.log_spatial_intensity_se, orient="index"
        )
        tables["Spatial_Intensity_SE"] = pd.DataFrame.from_dict(
            self.spatial_intensity_se, orient="index"
        )
        if self.moderators_coef_dim:
            tables["Moderators_Regression_SE"] = pd.DataFrame(self.se_moderators)
        return maps, tables

class OverdispersionModelEstimator(GeneralLinearModelEstimator):
    def __init__(self, **kwargs):
        self.square_root = kwargs.pop("square_root", False)
        super().__init__(**kwargs)
        if self.groups:
            self.init_overdispersion_weights()

    def init_overdispersion_weights(self):
        """Document this."""
        overdispersion = dict()
        for group in self.groups:
            # initialization for alpha
            overdispersion_init_group = torch.tensor(1e-2).double()
            if self.square_root:
                overdispersion_init_group = torch.sqrt(overdispersion_init_group)
            overdispersion[group] = torch.nn.Parameter(overdispersion_init_group, requires_grad=True)
        self.overdispersion = torch.nn.ParameterDict(overdispersion)

    def init_weights(self, groups, spatial_coef_dim, moderators_coef_dim):
        """Document this."""
        super().init_weights(groups, spatial_coef_dim, moderators_coef_dim)
        self.init_overdispersion_weights()

    def inference_outcome(self, coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study):
        """Document this."""
        maps, tables = super().inference_outcome(coef_spline_bases, moderators_by_group, foci_per_voxel, foci_per_study)
        overdispersion_param = dict()
        for group in self.groups:
            group_overdispersion = self.overdispersion[group]
            group_overdispersion = group_overdispersion.cpu().detach().numpy()
            overdispersion_param[group] = group_overdispersion
        tables["Overdispersion_Coef"] = pd.DataFrame.from_dict(
            overdispersion_param, orient="index", columns=["overdispersion"])
        
        return maps, tables

class PoissonEstimator(GeneralLinearModelEstimator):
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
        device="cpu"
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
        coef_spline_bases,
        foci_per_voxel,
        foci_per_study,
        moderator_coef=None,
        moderators=None,
        device="cpu",
    ):
        n_groups = len(spatial_coef)
        log_spatial_intensity = [
            torch.matmul(coef_spline_bases, spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        spatial_intensity = [
            torch.exp(group_log_spatial_intensity) for group_log_spatial_intensity in log_spatial_intensity
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
                        group_foci_per_study)
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            for group in self.groups:
                group_spatial_coef = self.spatial_coef_linears[group].weight
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                if self.moderators_coef_dim:
                    moderators_coef = self.moderators_linear.weight
                    group_moderators = moderators[group]
                else:
                    moderators_coef, group_moderators = None, None

                nll = lambda group_spatial_coef: -self._log_likelihood_single_group(
                    group_spatial_coef,
                    moderators_coef,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                group_F = torch.autograd.functional.hessian(
                    nll,
                    group_spatial_coef,
                    create_graph=False,
                    vectorize=True,
                    outer_jacobian_strategy="forward-mode",
                )
                group_F = group_F.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
                group_eig_vals = torch.real(
                    torch.linalg.eigvals(group_F)
                )  
                del group_F
                group_firth_penalty = 0.5 * torch.sum(torch.log(group_eig_vals))
                del group_eig_vals
                log_l += group_firth_penalty
        return -log_l


class NegativeBinomialEstimator(OverdispersionModelEstimator):
    def __init__(self, **kwargs):
        kwargs['square_root'] = True
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
        numerator = mu_spatial**2 * torch.sum(mu_moderators**2)
        denominator = mu_spatial**2 * torch.sum(mu_moderators) ** 2
        # estimated_sum_alpha = alpha * numerator / denominator

        p = numerator / (v * mu_spatial * torch.sum(mu_moderators) + numerator)
        r = v * denominator / numerator

        log_l = self._three_term(group_foci_per_voxel, r) + torch.sum(
            r * torch.log(1 - p) + group_foci_per_voxel * torch.log(p)
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
        v = 1 / overdispersion_coef
        n_groups = len(foci_per_voxel)
        log_spatial_intensity = [
            torch.matmul(coef_spline_bases, spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        spatial_intensity = [
            torch.exp(group_log_spatial_intensity) for group_log_spatial_intensity in log_spatial_intensity
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

        numerators = [
            spatial_intensity[i] ** 2 * torch.sum(moderator_effect[i] ** 2)
            for i in range(n_groups)
        ]
        denominators = [
            spatial_intensity[i] ** 2 * torch.sum(moderator_effect[i]) ** 2
            for i in range(n_groups)
        ]
        p = [
            numerators[i]
            / (
                v[i] * spatial_intensity[i] * torch.sum(moderator_effect[i])
                + denominators[i]
            )
            for i in range(n_groups)
        ]
        r = [v[i] * denominators[i] / numerators[i] for i in range(n_groups)]

        log_l = 0
        for i in range(n_groups):
            log_l += NegativeBinomial._three_term(foci_per_voxel[i], r[i], device=device) + torch.sum(
                r[i] * torch.log(1 - p[i]) + foci_per_voxel[i] * torch.log(p[i])
            )

        return log_l

    def forward(self, coef_spline_bases, moderators, foci_per_voxel, foci_per_study):
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
                group_foci_per_study
            )

            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            for group in self.groups:
                group_overdispersion = self.overdispersion[group] ** 2
                group_spatial_coef = self.spatial_coef_linears[group].weight
                if self.moderators_coef_dim:
                    moderators_coef = self.moderators_linear.weight
                    group_moderators = moderators[group]
                else:
                    moderators_coef, group_moderators = None, None
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
            
                nll = lambda group_spatial_coef: -self._log_likelihood_single_group(
                    group_overdispersion,
                    group_spatial_coef,
                    moderators_coef,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                group_F = torch.autograd.functional.hessian(nll, group_spatial_coef, create_graph=True)
                group_F = group_F.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
                group_eig_vals = torch.real(torch.linalg.eigvals(group_F))
                del group_F
                group_firth_penalty = 0.5 * torch.sum(torch.log(group_eig_vals))
                del group_eig_vals
                log_l += group_firth_penalty

        return -log_l


class ClusteredNegativeBinomialEstimator(OverdispersionModelEstimator):
    def __init__(self, **kwargs):
        kwargs['square_root'] = False
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
        v = [1 / group_overdispersion_coef  for group_overdispersion_coef in overdispersion_coef]
        # estimated intensity and log estimated intensity
        log_spatial_intensity = [
            torch.matmul(coef_spline_bases, spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        spatial_intensity = [
            torch.exp(group_log_spatial_intensity) for group_log_spatial_intensity in log_spatial_intensity
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
        mu_sum_per_study = [torch.sum(spatial_intensity[i]) * moderator_effect[i] for i in range(n_groups)]
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
                        group_foci_per_study)
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            for group in self.groups:
                group_overdispersion = self.overdispersion[group]
                group_spatial_coef = self.spatial_coef_linears[group].weight
                if self.moderators_coef_dim:
                    moderators_coef = self.moderators_linear.weight
                    group_moderators = moderators[group]
                else:
                    moderators_coef, group_moderators = None, None
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                group_moderators = moderators[group]
                
                nll = lambda group_spatial_coef: -self._log_likelihood_single_group(
                    group_overdispersion,
                    group_spatial_coef,
                    moderators_coef,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                group_F = torch.autograd.functional.hessian(
                    nll, group_spatial_coef, create_graph=True
                ) 
                group_F = group_F.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
                group_eig_vals = torch.real(torch.linalg.eigvals(group_F))
                del group_F
                group_firth_penalty = 0.5 * torch.sum(torch.log(group_eig_vals))
                del group_eig_vals
                log_l += group_firth_penalty

        return -log_l
