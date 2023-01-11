
import abc
import torch


class GeneralLinearModel(torch.nn.Module):
    def __init__(
        self,
        spatial_coef_dim=None,
        moderators_coef_dim=None,
        groups=None,
        penalty=False,
        device="cpu",
    ):  
        super().__init__()
        self.spatial_coef_dim = spatial_coef_dim
        self.moderators_coef_dim = moderators_coef_dim
        self.groups = groups
        self.penalty = penalty
        self.device = device
    
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


class Poisson(GeneralLinearModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialization for spatial regression coefficients
        all_spatial_coef_linears = dict()
        for group in self.groups:
            spatial_coef_linear_group = torch.nn.Linear(self.spatial_coef_dim, 1, bias=False).double()
            torch.nn.init.uniform_(spatial_coef_linear_group.weight, a=-0.01, b=0.01)
            all_spatial_coef_linears[group] = spatial_coef_linear_group
        self.all_spatial_coef_linears = torch.nn.ModuleDict(all_spatial_coef_linears)
        # initialization for regression coefficients of moderators
        if self.moderators_coef_dim:
            self.moderators_linear = torch.nn.Linear(self.moderators_coef_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.moderators_linear.weight, a=-0.01, b=0.01)

    def _log_likelihood_single_group(
        group_spatial_coef,
        moderators_coef, 
        coef_spline_bases, 
        moderators, 
        foci_per_voxel, 
        foci_per_study, 
        device="cpu"
    ):
        log_mu_spatial = torch.matmul(coef_spline_bases, group_spatial_coef.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if moderators_coef is None:
            n_study, _ = foci_per_study.shape
            log_mu_moderators = torch.tensor(
                [0] * n_study, dtype=torch.float64, device=device
            ).reshape((-1, 1))
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            log_mu_moderators = torch.matmul(moderators, moderators_coef.T)
            mu_moderators = torch.exp(log_mu_moderators)
        log_l = (
            torch.sum(torch.mul(foci_per_voxel, log_mu_spatial))
            + torch.sum(torch.mul(foci_per_study, log_mu_moderators))
            - torch.sum(mu_spatial) * torch.sum(mu_moderators)
        )
        return log_l

    def _log_likelihood_mult_group(
        all_spatial_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_study,
        moderator_coef=None,
        all_moderators=None,
        device="cpu",
    ):
        n_groups = len(all_spatial_coef)
        all_log_spatial_intensity = [
            torch.matmul(coef_spline_bases, all_spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        all_spatial_intensity = [
            torch.exp(log_spatial_intensity) for log_spatial_intensity in all_log_spatial_intensity
        ]
        if moderator_coef is not None:
            all_log_moderator_effect = [
                torch.matmul(moderator, moderator_coef) for moderator in all_moderators
            ]
            all_moderator_effect = [
                torch.exp(log_moderator_effect)
                for log_moderator_effect in all_log_moderator_effect
            ]
        else:
            all_log_moderator_effect = [
                torch.tensor(
                    [0] * foci_per_study_i.shape[0], dtype=torch.float64, device=device
                ).reshape((-1, 1))
                for foci_per_study_i in foci_per_study
            ]
            all_moderator_effect = [
                torch.exp(log_moderator_effect)
                for log_moderator_effect in all_log_moderator_effect
            ]
        log_l = 0
        for i in range(n_groups):
            log_l += (
                torch.sum(foci_per_voxel[i] * all_log_spatial_intensity[i])
                + torch.sum(foci_per_study[i] * all_log_moderator_effect[i])
                - torch.sum(all_spatial_intensity[i]) * torch.sum(all_moderator_effect[i])
            )
        return log_l

    def forward(self, coef_spline_bases, all_moderators, foci_per_voxel, foci_per_study):
        if isinstance(all_moderators, dict):
            all_log_mu_moderators = dict()
            for group in all_moderators.keys():
                group_moderators = all_moderators[group]
                log_mu_moderators = self.moderators_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        # spatial effect
        for group in foci_per_voxel.keys():
            log_mu_spatial = self.all_spatial_coef_linears[group](coef_spline_bases)
            mu_spatial = torch.exp(log_mu_spatial)
            group_foci_per_voxel = foci_per_voxel[group]
            group_foci_per_study = foci_per_study[group]
            if self.moderators_coef_dim:
                log_mu_moderators = all_log_mu_moderators[group]
                mu_moderators = torch.exp(log_mu_moderators)
            else:
                n_group_study, _ = group_foci_per_study.shape
                log_mu_moderators = torch.tensor([0] * n_group_study, device=self.device).reshape(
                    (-1, 1)
                )
                mu_moderators = torch.exp(log_mu_moderators)
            # Under the assumption that Y_ij is either 0 or 1
            # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
            group_log_l = (
                torch.sum(torch.mul(group_foci_per_voxel, log_mu_spatial))
                + torch.sum(torch.mul(group_foci_per_study, log_mu_moderators))
                - torch.sum(mu_spatial) * torch.sum(mu_moderators)
            )
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            for group in foci_per_voxel.keys():
                group_spatial_coef = self.all_spatial_coef_linears[group].weight
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                if self.moderators_coef_dim:
                    moderators_coef = self.moderators_linear.weight
                    group_moderators = all_moderators[group]
                else:
                    moderators_coef, group_moderators = None, None

                nll = lambda group_spatial_coef: -Poisson._log_likelihood_single_group(
                    group_spatial_coef,
                    moderators_coef,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                F = torch.autograd.functional.hessian(
                    nll,
                    group_spatial_coef,
                    create_graph=False,
                    vectorize=True,
                    outer_jacobian_strategy="forward-mode",
                )
                F = F.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
                eig_vals = torch.real(
                    torch.linalg.eigvals(F)
                )  # torch.eig(F, eigenvectors=False)[0][:,0]
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty
        return -log_l


class NegativeBinomial(GeneralLinearModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialization for group-wise spatial coefficient of regression
        all_spatial_coef_linears, all_overdispersion_sqrt = dict(), dict()
        for group in self.groups:
            spatial_coef_linear_group = torch.nn.Linear(self.spatial_coef_dim, 1, bias=False).double()
            torch.nn.init.uniform_(spatial_coef_linear_group.weight, a=-0.01, b=0.01)
            all_spatial_coef_linears[group] = spatial_coef_linear_group
            # initialization for alpha
            overdispersion_init_group = torch.tensor(1e-2).double()
            all_overdispersion_sqrt[group] = torch.nn.Parameter(
                torch.sqrt(overdispersion_init_group), requires_grad=True
            )
        self.all_spatial_coef_linears = torch.nn.ModuleDict(all_spatial_coef_linears)
        self.all_overdispersion_sqrt = torch.nn.ParameterDict(all_overdispersion_sqrt)
        if self.moderators_coef_dim:
            self.moderators_linear = torch.nn.Linear(self.moderators_coef_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.moderators_linear.weight, a=-0.01, b=0.01)

    def _three_term(y, r, device):
        max_foci = torch.max(y).to(dtype=torch.int64, device=device)
        sum_three_term = 0
        for k in range(max_foci):
            foci_index = (y == k + 1).nonzero()[:, 0]
            r_j = r[foci_index]
            n_voxel = list(foci_index.shape)[0]
            y_j = torch.tensor([k + 1] * n_voxel, device=device).double()
            y_j = y_j.reshape((n_voxel, 1))
            # y=0 => sum_three_term = 0
            sum_three_term += torch.sum(
                torch.lgamma(y_j + r_j) - torch.lgamma(y_j + 1) - torch.lgamma(r_j)
            )

        return sum_three_term

    def _log_likelihood_single_group(
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

        log_l = NegativeBinomial._three_term(group_foci_per_voxel, r, device=device) + torch.sum(
            r * torch.log(1 - p) + group_foci_per_voxel * torch.log(p)
        )

        return log_l

    def _log_likelihood_mult_group(
        all_overdispersion_coef,
        all_spatial_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_study,
        moderator_coef=None,
        all_moderators=None,
        device="cpu",
    ):
        all_v = 1 / all_overdispersion_coef
        n_groups = len(foci_per_voxel)
        all_log_spatial_intensity = [
            torch.matmul(coef_spline_bases, all_spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        all_spatial_intensity = [
            torch.exp(log_spatial_intensity) for log_spatial_intensity in all_log_spatial_intensity
        ]
        if moderator_coef is not None:
            all_log_moderator_effect = [
                torch.matmul(moderator, moderator_coef) for moderator in all_moderators
            ]
            all_moderator_effect = [
                torch.exp(log_moderator_effect)
                for log_moderator_effect in all_log_moderator_effect
            ]
        else:
            all_log_moderator_effect = [
                torch.tensor(
                    [0] * foci_per_study.shape[0], dtype=torch.float64, device=device
                ).reshape((-1, 1))
                for foci_per_study in foci_per_study
            ]
            all_moderator_effect = [
                torch.exp(log_moderator_effect)
                for log_moderator_effect in all_log_moderator_effect
            ]

        all_numerator = [
            all_spatial_intensity[i] ** 2 * torch.sum(all_moderator_effect[i] ** 2)
            for i in range(n_groups)
        ]
        all_denominator = [
            all_spatial_intensity[i] ** 2 * torch.sum(all_moderator_effect[i]) ** 2
            for i in range(n_groups)
        ]
        # all_estimated_sum_alpha = [
        #     all_overdispersion_coef[i, :] * all_numerator[i] / all_denominator[i]
        #     for i in range(n_groups)
        # ]

        p = [
            all_numerator[i]
            / (
                all_v[i] * all_spatial_intensity[i] * torch.sum(all_moderator_effect[i])
                + all_denominator[i]
            )
            for i in range(n_groups)
        ]
        r = [all_v[i] * all_denominator[i] / all_numerator[i] for i in range(n_groups)]

        log_l = 0
        for i in range(n_groups):
            log_l += NegativeBinomial._three_term(foci_per_voxel[i], r[i], device=device) + torch.sum(
                r[i] * torch.log(1 - p[i]) + foci_per_voxel[i] * torch.log(p[i])
            )

        return log_l

    def forward(self, coef_spline_bases, all_moderators, foci_per_voxel, foci_per_study):
        if isinstance(all_moderators, dict):
            all_log_mu_moderators = dict()
            for group in all_moderators.keys():
                group_moderators = all_moderators[group]
                log_mu_moderators = self.moderators_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        # spatial effect
        for group in foci_per_voxel.keys():
            overdispersion = self.all_overdispersion_sqrt[group] ** 2
            v = 1 / overdispersion
            log_mu_spatial = self.all_spatial_coef_linears[group](coef_spline_bases)
            mu_spatial = torch.exp(log_mu_spatial)
            if self.moderators_coef_dim:
                log_mu_moderators = all_log_mu_moderators[group]
                mu_moderators = torch.exp(log_mu_moderators)
            else:
                n_group_study, _ = foci_per_study[group].shape
                log_mu_moderators = torch.tensor([0] * n_group_study, device=self.device).reshape(
                    (-1, 1)
                )
                mu_moderators = torch.exp(log_mu_moderators)
            # Now the sum of NB variates are no long NB distributed (since mu_ij != mu_i'j),
            # Therefore, we use moment matching approach,
            # create a new NB approximation to the mixture of NB distributions:
            # alpha' = sum_i mu_{ij}^2 / (sum_i mu_{ij})^2 * alpha
            numerator = mu_spatial**2 * torch.sum(mu_moderators**2)
            denominator = mu_spatial**2 * torch.sum(mu_moderators) ** 2
            # estimated_sum_alpha = alpha * numerator / denominator
            # moment matching NB distribution
            p = numerator / (v * mu_spatial * torch.sum(mu_moderators) + numerator)
            r = v * denominator / numerator

            group_foci_per_voxel = foci_per_voxel[group]
            # group_foci_per_study = foci_per_study[group]
            group_log_l = NegativeBinomial._three_term(
                group_foci_per_voxel, r, device=self.device
            ) + torch.sum(r * torch.log(1 - p) + group_foci_per_voxel * torch.log(p))
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            for group in foci_per_voxel.keys():
                group_overdispersion = self.all_overdispersion_sqrt[group] ** 2
                group_spatial_coef = self.all_spatial_coef_linears[group].weight
                moderators_coef = self.moderators_linear.weight.detach()
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                group_moderators = all_moderators[group]
            
                nll = lambda group_spatial_coef: -NegativeBinomial._log_likelihood_single_group(
                    group_overdispersion,
                    group_spatial_coef,
                    moderators_coef,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                F = torch.autograd.functional.hessian(nll, group_spatial_coef, create_graph=True)
                F = F.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
                eig_vals = eig_vals = torch.real(torch.linalg.eigvals(F))
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty

        return -log_l


class ClusteredNegativeBinomial(GeneralLinearModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialization for spatial regression coefficient 
        all_spatial_coef_linears, all_overdispersion = dict(), dict()
        for group in self.groups:
            spatial_coef_linear_group = torch.nn.Linear(self.spatial_coef_dim, 1, bias=False).double()
            torch.nn.init.uniform_(spatial_coef_linear_group.weight, a=-0.01, b=0.01)
            all_spatial_coef_linears[group] = spatial_coef_linear_group
            # initialization for overdispersion parameter
            overdispersion_init_group = torch.tensor(1e-2).double()
            all_overdispersion[group] = torch.nn.Parameter(overdispersion_init_group, requires_grad=True)
        self.all_spatial_coef_linears = torch.nn.ModuleDict(all_spatial_coef_linears)
        self.all_overdispersion = torch.nn.ParameterDict(all_overdispersion)
        # regression coefficient for moderators
        if self.moderators_coef_dim:
            self.moderators_linear = torch.nn.Linear(self.moderators_coef_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.moderators_linear.weight, a=-0.01, b=0.01)

    def _log_likelihood_single_group(
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
        all_overdispersion_coef,
        all_spatial_coef,
        coef_spline_bases,
        foci_per_voxel,
        foci_per_study,
        moderator_coef=None,
        all_moderators=None,
        device="cpu",
    ):
        n_groups = len(foci_per_voxel)
        all_v = [1 / overdispersion_coef  for overdispersion_coef in all_overdispersion_coef]
        # estimated intensity and log estimated intensity
        all_log_spatial_intensity = [
            torch.matmul(coef_spline_bases, all_spatial_coef[i, :, :]) for i in range(n_groups)
        ]
        all_spatial_intensity = [
            torch.exp(log_spatial_intensity) for log_spatial_intensity in all_log_spatial_intensity
        ]
        if moderator_coef is not None:
            all_log_moderator_effect = [
                torch.matmul(moderator, moderator_coef) for moderator in all_moderators
            ]
            all_moderator_effect = [
                torch.exp(log_moderator_effect)
                for log_moderator_effect in all_log_moderator_effect
            ]
        else:
            all_log_moderator_effect = [
                torch.tensor(
                    [0] * foci_per_study.shape[0], dtype=torch.float64, device=device
                ).reshape((-1, 1))
                for foci_per_study in foci_per_study
            ]
            all_moderator_effect = [
                torch.exp(log_moderator_effect)
                for log_moderator_effect in all_log_moderator_effect
            ]
        all_mu_sum_per_study = [torch.sum(all_spatial_intensity[i]) * all_moderator_effect[i] for i in range(n_groups)]
        all_group_n_study = [group_foci_per_study.shape[0] for group_foci_per_study in foci_per_study]

        log_l = 0
        for i in range(n_groups):
            log_l += (
                    all_group_n_study[i] * all_v[i] * torch.log(all_v[i])
                    - all_group_n_study[i] * torch.lgamma(all_v[i])
                    + torch.sum(torch.lgamma(foci_per_study[i] + all_v[i]))
                    - torch.sum((foci_per_study[i] + all_v[i]) * torch.log(all_mu_sum_per_study[i] + all_v[i]))
                    + torch.sum(foci_per_voxel[i] * all_log_spatial_intensity[i])
                    + torch.sum(foci_per_study[i] * all_log_moderator_effect[i])
                    )
            
        return log_l

    def forward(self, coef_spline_bases, all_moderators, foci_per_voxel, foci_per_study):
        if isinstance(all_moderators, dict):
            all_log_mu_moderators = dict()
            for group in all_moderators.keys():
                group_moderators = all_moderators[group]
                log_mu_moderators = self.moderators_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        for group in foci_per_voxel.keys():
            group_overdispersion = self.all_overdispersion[group]
            v = 1 / group_overdispersion
            log_mu_spatial = self.all_spatial_coef_linears[group](coef_spline_bases)
            mu_spatial = torch.exp(log_mu_spatial)
            group_foci_per_voxel = foci_per_voxel[group]
            group_foci_per_study = foci_per_study[group]
            if self.study_level_moderators:
                log_mu_moderators = all_log_mu_moderators[group]
                mu_moderators = torch.exp(log_mu_moderators)
            else:
                n_group_study, _ = group_foci_per_study.shape
                log_mu_moderators = torch.tensor([0] * n_group_study, device=self.device).reshape(
                    (-1, 1)
                )
                mu_moderators = torch.exp(log_mu_moderators)
            group_n_study, _ = group_foci_per_study.shape
            mu_sum_per_study = torch.sum(mu_spatial) * mu_moderators
            group_log_l = (
                group_n_study * v * torch.log(v)
                - group_n_study * torch.lgamma(v)
                + torch.sum(torch.lgamma(group_foci_per_study + v))
                - torch.sum((group_foci_per_study + v) * torch.log(mu_sum_per_study + v))
                + torch.sum(group_foci_per_voxel * log_mu_spatial)
                + torch.sum(group_foci_per_study * log_mu_moderators)
            )
            log_l += group_log_l

        if self.penalty:
            # Firth-type penalty
            for group in foci_per_voxel.keys():
                group_overdispersion = self.all_overdispersion[group]
                group_spatial_coef = self.all_spatial_coef_linears[group].weight
                moderators_coef = self.moderators_linear.weight
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                group_moderators = all_moderators[group]
                
                nll = lambda group_spatial_coef: -ClusteredNegativeBinomial._log_likelihood_single_group(
                    group_overdispersion,
                    group_spatial_coef,
                    moderators_coef,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                F = torch.autograd.functional.hessian(
                    nll, group_spatial_coef, create_graph=True
                ) 
                F = F.reshape((self.spatial_coef_dim, self.spatial_coef_dim))
                eig_vals = torch.real(torch.linalg.eigvals(F))
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty

        return -log_l
