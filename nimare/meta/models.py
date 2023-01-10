
import abc
import torch


class GeneralLinearModel(torch.nn.Module):
    def __init__(
        self,
        beta_dim=None,
        gamma_dim=None,
        groups=None,
        study_level_moderators=False,
        penalty=False,
        device="cpu",
    ):  
        super().__init__()
        self.beta_dim = beta_dim
        self.gamma_dim = gamma_dim
        self.groups = groups
        self.study_level_moderators = study_level_moderators
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
        # initialization for beta
        all_beta_linears = dict()
        for group in self.groups:
            beta_linear_group = torch.nn.Linear(self.beta_dim, 1, bias=False).double()
            torch.nn.init.uniform_(beta_linear_group.weight, a=-0.01, b=0.01)
            all_beta_linears[group] = beta_linear_group
        self.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
        # gamma
        if self.study_level_moderators:
            self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)

    def _log_likelihood_single_group(
        beta, gamma, coef_spline_bases, moderators, foci_per_voxel, foci_per_study, device="cpu"
    ):
        log_mu_spatial = torch.matmul(coef_spline_bases, beta.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if gamma is not None:
            log_mu_moderators = torch.matmul(moderators, gamma.T)
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            n_study, _ = foci_per_study.shape
            log_mu_moderators = torch.tensor(
                [0] * n_study, dtype=torch.float64, device=device
            ).reshape((-1, 1))
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
                # mu^Z = exp(Z * gamma)
                log_mu_moderators = self.gamma_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        # spatial effect: mu^X = exp(X * beta)
        for group in foci_per_voxel.keys():
            log_mu_spatial = self.all_beta_linears[group](coef_spline_bases)
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
                beta = self.all_beta_linears[group].weight.T
                beta_dim = beta.shape[0]
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                if self.study_level_moderators:
                    gamma = self.gamma_linear.weight.T
                    group_moderators = all_moderators[group]
                    gamma, group_moderators = [gamma], [group_moderators]
                else:
                    gamma, group_moderators = None, None

                # all_spatial_coef = torch.stack([beta])
                foci_per_voxel, foci_per_study = torch.stack(
                    [group_foci_per_voxel]
                ), torch.stack([group_foci_per_study])
                nll = lambda beta: -self._log_likelihood(
                    beta,
                    gamma,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                params = beta
                F = torch.autograd.functional.hessian(
                    nll,
                    params,
                    create_graph=False,
                    vectorize=True,
                    outer_jacobian_strategy="forward-mode",
                )
                F = F.reshape((beta_dim, beta_dim))
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
        # initialization for beta
        all_beta_linears, all_alpha_sqrt = dict(), dict()
        for group in self.groups:
            beta_linear_group = torch.nn.Linear(self.beta_dim, 1, bias=False).double()
            torch.nn.init.uniform_(beta_linear_group.weight, a=-0.01, b=0.01)
            all_beta_linears[group] = beta_linear_group
            # initialization for alpha
            alpha_init_group = torch.tensor(1e-2).double()
            all_alpha_sqrt[group] = torch.nn.Parameter(
                torch.sqrt(alpha_init_group), requires_grad=True
            )
        self.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
        self.all_alpha_sqrt = torch.nn.ParameterDict(all_alpha_sqrt)
        # gamma
        if self.study_level_moderators:
            self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)

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
        alpha,
        beta,
        gamma,
        coef_spline_bases,
        group_moderators,
        group_foci_per_voxel,
        group_foci_per_study,
        device="cpu",
    ):
        v = 1 / alpha
        log_mu_spatial = torch.matmul(coef_spline_bases, beta.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if gamma is not None:
            log_mu_moderators = torch.matmul(group_moderators, gamma.T)
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
                # mu^Z = exp(Z * gamma)
                log_mu_moderators = self.gamma_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        # spatial effect: mu^X = exp(X * beta)
        for group in foci_per_voxel.keys():
            alpha = self.all_alpha_sqrt[group] ** 2
            v = 1 / alpha
            log_mu_spatial = self.all_beta_linears[group](coef_spline_bases)
            mu_spatial = torch.exp(log_mu_spatial)
            if self.study_level_moderators:
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
                alpha = self.all_alpha_sqrt[group] ** 2
                beta = self.all_beta_linears[group].weight.T
                beta_dim = beta.shape[0]
                gamma = self.gamma_linear.weight.detach().T
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                group_moderators = all_moderators[group]
                nll = lambda beta: -self._log_likelihood(
                    alpha,
                    beta,
                    gamma,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                params = beta
                F = torch.autograd.functional.hessian(nll, params, create_graph=True)
                F = F.reshape((beta_dim, beta_dim))
                eig_vals = eig_vals = torch.real(torch.linalg.eigvals(F))
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty

        return -log_l


class ClusteredNegativeBinomial(GeneralLinearModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialization for beta
        all_beta_linears, all_alpha = dict(), dict()
        for group in self.groups:
            beta_linear_group = torch.nn.Linear(self.beta_dim, 1, bias=False).double()
            torch.nn.init.uniform_(beta_linear_group.weight, a=-0.01, b=0.01)
            all_beta_linears[group] = beta_linear_group
            # initialization for alpha
            alpha_init_group = torch.tensor(1e-2).double()
            all_alpha[group] = torch.nn.Parameter(alpha_init_group, requires_grad=True)
        self.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
        self.all_alpha = torch.nn.ParameterDict(all_alpha)
        # gamma
        if self.study_level_moderators:
            self.gamma_linear = torch.nn.Linear(self.gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)

    def _log_likelihood_single_group(
        alpha,
        beta,
        gamma,
        coef_spline_bases,
        group_moderators,
        group_foci_per_voxel,
        group_foci_per_study,
        device="cpu",
    ):
        v = 1 / alpha
        log_mu_spatial = torch.matmul(coef_spline_bases, beta.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if gamma is not None:
            log_mu_moderators = torch.matmul(group_moderators, gamma.T)
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
                # mu^Z = exp(Z * gamma)
                log_mu_moderators = self.gamma_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        for group in foci_per_voxel.keys():
            alpha = self.all_alpha[group]
            v = 1 / alpha
            log_mu_spatial = self.all_beta_linears[group](coef_spline_bases)
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
                alpha = self.all_alpha[group]
                beta = self.all_beta_linears[group].weight.T
                beta_dim = beta.shape[0]
                gamma = self.gamma_linear.weight.T
                group_foci_per_voxel = foci_per_voxel[group]
                group_foci_per_study = foci_per_study[group]
                group_moderators = all_moderators[group]
                nll = lambda beta: -self._log_likelihood(
                    alpha,
                    beta,
                    gamma,
                    coef_spline_bases,
                    group_moderators,
                    group_foci_per_voxel,
                    group_foci_per_study,
                )
                params = beta
                F = torch.autograd.functional.hessian(
                    nll, params, create_graph=True
                )  # vectorize=True, outer_jacobian_strategy='forward-mode'
                # F = hessian(nll)(beta)
                F = F.reshape((beta_dim, beta_dim))
                eig_vals = torch.real(torch.linalg.eigvals(F))
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty

        return -log_l
