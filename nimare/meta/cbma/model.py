"""
Model-based coordinate-based meta-analysis estimators
"""
import numpy as np
import pandas as pd
from scipy import stats

from ...base import CBMAEstimator
from ...due import due, Doi, BibTeX


@due.dcite(Doi('10.1198/jasa.2011.ap09735'),
           description='Introduces the BHICP model.')
class BHICP(CBMAEstimator):
    """
    Bayesian hierarchical cluster process model
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, sample):
        pass


@due.dcite(BibTeX("""
           @article{kang2014bayesian,
             title={A Bayesian hierarchical spatial point process model for
                    multi-type neuroimaging meta-analysis},
             author={Kang, Jian and Nichols, Thomas E and Wager, Tor D and
                     Johnson, Timothy D},
             journal={The annals of applied statistics},
             volume={8},
             number={3},
             pages={1800},
             year={2014},
             publisher={NIH Public Access}
             }
           """),
           description='Introduces the HPGRF model.')
class HPGRF(CBMAEstimator):
    """
    Hierarchical Poisson/Gamma random field model
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, sample):
        pass


@due.dcite(Doi('10.1111/biom.12713'),
           description='Introduces the SBLFR model.')
class SBLFR(CBMAEstimator):
    """
    Spatial Bayesian latent factor regression model

    This uses the HamiltonianMC and the truncnorm.rvs method.
    """
    def __init__(self, dataset, ids):
        from scipy.stats import truncnorm
        from pymc3.step_methods.hmc.hmc import HamiltonianMC
        self.dataset = dataset
        self.ids = ids

    def _hmc(self, epsilon_hmc, L_hmc, theta, A, Bpred, tBpred, Lambda, eta,
             sig, sum_B):
        """
        """
        q = theta
        N, nbasis = theta.shape

        # Independent standard normal variates
        kinetic = stats.norm.rvs(0, 1, size=(N, nbasis))
        current_kinetic = kinetic

        # Pre-compute matrices to speed up computation
        Leta = np.dot(Lambda, eta)
        AtBpred = np.dot(A, tBpred)

        # Make a half step momentum at the beginning
        add_grad_q = -1 * np.diag(sig) * Leta - sum_B.T
        ttheta = theta.T
        grad_q = (AtBpred * np.exp(Bpred * ttheta)) + np.diag(sig) * ttheta + add_grad_q;
        kinetic = kinetic - epsilon_hmc * 0.5 * grad_q.T

        # Alternate full steps for position and momentum
        for e in range(L_hmc):
            # Make a full step for the position
            q = q + epsilon_hmc * kinetic
            tq = q.T

            # Make a full step for the momentum, except at end of trajectory
            if e < L_hmc - 1:
                grad_q = (AtBpred * exp(Bpred * tq)) + diag(sig) * (tq) + add_grad_q
                kinetic = kinetic - epsilon_hmc * grad_q.T

        # Make a half step momentum at the end
        grad_q = (AtBpred * exp(Bpred * (tq))) + diag(sig) * (tq) + add_grad_q;
        kinetic = kinetic - epsilon_hmc * 0.5 * grad_q.T

        # Negate momentum at end of trajectory to make the proposal symmetric
        kinetic = -1 * kinetic

        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U = (A * np.sum(np.exp(np.dot(Bpred, ttheta)), axis=0).T -
                     np.diag(np.dot(sum_B, ttheta)) +
                     0.5 * np.diag((theta - Leta.T) *
                     np.diag(sig) * (theta - Leta.T).T))
        current_K = np.sum((current_kinetic ** 2).T, axis=0).T / 2
        proposed_U = (A * np.sum(np.exp(np.dot(Bpred, tq)), axis=0).T -
                      np.diag(np.dot(sum_B, tq)) +
                      0.5 * np.diag((q - Leta.T) *
                      np.diag(sig) * (q - Leta.T).T))
        proposed_K = np.sum((kinetic ** 2).T, axis=0).T / 2

        # Accept or reject the state at end of trajectory, returning either the position
        # at the end of the trajectory or the initial position
        pratio = current_U - proposed_U + current_K - proposed_K  # Log acceptance ratio
        u = np.log(np.random.random([N, 1]))
        acc = (u < pratio).astype(int)
        ind = np.where(acc == 1)[0]
        theta_hmc = theta
        theta_hmc[ind, :] = q[ind, :]  # theta_hmc = old, excpet for those proposal which get accepted

        pot_energy = (A * np.sum(np.exp(np.dot(Bpred, theta_hmc.T)), axis=0).T -
                      np.diag(np.dot(sum_B, theta_hmc.T)) +
                      0.5 * np.diag((theta_hmc - Leta.T) *
                      np.diag(sig) * (theta_hmc - Leta.T).T))
        return theta_hmc, acc, pot_energy

    def fit(self, voxel_thresh=0.01, q=0.05, corr='FDR', covariates=None):
        """
        Looks good through line 99 of the MATLAB script main_3D.m.

        Parameters
        ----------
        covariates : :obj:`list` or None, optional
            List of column names for covariates to include in model.
        """
        if isinstance(covariates, list):
            n_cov = len(covariates)  # int
            covariate_data = df[covariates]  # (n_foci, n_covariates)
        else:
            n_cov = 0
            covariate_data = None

        if isinstance(studytype_col, str):
            study_types = df[studytype_col]  # (n_foci,)
            n_study_types = len(np.unique(study_types))  # int
        else:
            study_types = []
            n_study_types = 0

        ijk_id = df[['i', 'j', 'k', 'id']].values  # (n_foci, 4)
        ids = np.unique(ijk_id[:, -1])  # (n_studies,)
        n_dims = 3  # TODO: remove
        n_studies = len(np.unique(ijk_id[:, -1]))  # int
        n_foci = ijk_id.shape[0]  # int
        n_foci_per_study = df.groupby('id').count()  # (n_studies,)

        # Get voxel volume
        voxel_vol = np.prod(mask_img.header.get_zooms())  # float

        # Define bases: Gaussian kernels
        xx = np.linspace(40, 145, 9)
        yy = np.linspace(40, 180, 8)
        zz = np.linspace(38, 90, 8)
        temp = [np.ravel(o) for o in np.meshgrid(yy, xx, zz)] # [(576,)] * 3
        knots = np.vstack((temp[1], temp[0], temp[2])).T  # (576, 3)

        # Sort for easy comparison to MATLAB code
        temp_knots = pd.DataFrame(data=knots, columns=['a', 'b', 'c'])
        temp_knots = temp_knots.sort_values(by=['c', 'b', 'a'])
        knots = temp_knots.values

        # At every *even* numbered axial slice, shift the x coordinate to
        # re-create a chess-like kernel grid
        # Is *odd* slices in MATLAB, but due to indexing differences...
        for n in range(int(np.floor(len(zz) / 2))):
            knots[knots[:, -1] == zz[n*2 + 1], 0] += 10
        # Remove knots whose x-coordinate is above 145 mm (falling outside of
        # the mask)
        knots = knots[knots[:, 0] <= 145, :]

        bf_bandwidth = 1. / (2 * 256)  # float

        # Observed matrix of basis functions
        B = np.zeros((ijk_id.shape[0], knots.shape[0]))
        temp_ijk_id = ijk_id - 1  # offset to match matlab
        for i in range(ijk_id.shape[0]):
            obs_knot = np.tile(temp_ijk_id[i, :-1], (knots.shape[0], 1)) - knots
            for j in range(knots.shape[0]):
                B[i, j] = np.exp(-bf_bandwidth * np.linalg.norm(obs_knot[j, :]) ** 2)

        B = np.hstack((np.ones((B.shape[0], 1)), B))  # (n_foci, 545)
        n_basis = B.shape[1]

        # Insert into HMC function the sum of bases
        sum_B = np.zeros((n_studies, B.shape[1]))
        for h, id_ in enumerate(ids):
            if (B[ijk_id[:, -1] == id_, :].shape[0] *
                    B[ijk_id[:, -1] == id_, :].shape[1]) == n_basis:
                sum_B[h, :] = B[ijk_id[:, -1] == id_, :]
            else:
                sum_B[h, :] = np.sum(B[ijk_id[:, -1] == id_, :], axis=0)

        # -- % Loading mask % -- %
        mask_img = nib.load(mask_file)
        mask_full = mask_img.get_data() > 0.5
        # NOTE: thresholding done slice-wise below in matlab, but naw

        # -- % Grid used to evaluate the integral at HMC step (the finer the
        # grid, the better the approximation) % -- %
        # Get voxel volume
        voxel_vol = np.prod(mask_img.header.get_zooms())

        # Also get grids in mm for 4mm mask
        Ix = np.arange(2, 180, 4)
        Iy = np.arange(2, 216, 4)
        Iz = np.arange(2, 180, 4)

        temp = [np.ravel(o) for o in np.meshgrid(Iy, Ix)]
        mesh_grid = np.vstack((temp[1], temp[0])).T  # (2430, 2)
        # Axial slices to use (9:24 restricts computation to the amygdalae;
        # for a full 3D use 1:91)
        for i_slice, slice_idx in enumerate(range(8, 10)):
            arr = mask_full[:, :, slice_idx]
            msk = np.ravel(arr)  # NOTE: are we sure this matches up?
            n_voxels_in_mask = mesh_grid[msk, :].shape[0]  # was sg
            # Get X, Y (from mesh_grid), and Z (from Iz) coords for mask voxels
            temp = np.hstack((mesh_grid[msk, :],
                              np.tile(Iz[slice_idx], (n_voxels_in_mask, 1))))

            if i_slice == 0:
                Grid = temp
            else:
                Grid = np.vstack((Grid, temp))
        # Sort for easy comparison to MATLAB
        temp_grid = pd.DataFrame(columns=['x', 'y', 'z'], data=Grid)
        temp_grid = temp_grid.sort_values(by=['z', 'y', 'x'])
        Grid = temp_grid.values

        # -- % Matrix of basis functions to evaluate HMC % -- %
        Bpred = np.zeros((Grid.shape[0], knots.shape[0]))
        for i in range(Grid.shape[0]):
            obs_knot = np.tile(Grid[i, :], (knots.shape[0], 1)) - knots
            for j in range(knots.shape[0]):
                Bpred[i, j] = np.exp(-bf_bandwidth * np.linalg.norm(obs_knot[j, :]) ** 2)

        Bpred = np.hstack((np.ones((Grid.shape[0], 1)), Bpred))
        Bpred[Bpred < 1e-35] = 0
        tBpred = Bpred.T
        V = Bpred.shape[0]  # Number of grid points

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	    Define global constants      % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nrun = 1  # Number of MCMC iterations
        burn = 25000  # Burn-in
        thin = 50  # Thinning MCMC samples
        every = 100  # Number of previous samples consider to update the HMC stepsize
        start = 250  # Starting iteration to update the stepsize
        sp = (nrun - burn) / thin  # Number of posterior samples
        epsilon = 1e-4  # Threshold limit (update of Lambda)
        prop = 1.00  # Proportion of redundant elements within columns

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	 Define hyperparameter values    % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Gamma hyperprior params on a_1 and a_2
        b0 = 2.
        b1 = 0.0001
        # Gamma hyperparameters for residual precision
        as_ = 1.  # as is a keyword
        bs = 0.3
        # Gamma hyperparameters for t_{ij}
        df_ = 3.
        # Gamma hyperparameters for delta_1
        ad1 = 2.1
        bd1 = 1.
        # Gamma hyperparameters for delta_h, h >=2
        ad2 = 3.1
        bd2 = 1.
        # Gamma hyperparameters for ad1 and ad2 or df_
        adf = 1.
        bdf = 1.
        # Starting valye for Leapfrog stepsize
        epsilon_hmc = 0.0001
        # Leapfrog trajectory length
        L_hmc_def = 30.

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	 		Initial values    		 % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        k = int(np.floor(np.log(n_basis) * 2))  # Number of latent factors to start with
        sig = np.tile(1, (n_basis, 1))  # Residual variance (diagonal of Sigma^-1)
        phiih = stats.gamma.rvs(df_ / 2, scale=(2 / df_), size=(n_basis, k))  # Local shrinkage coefficients
        delta = np.vstack((  # Global shrinkage coefficients multipliers
            stats.gamma.rvs(ad1, scale=bd1),
            stats.gamma.rvs(ad2, scale=bd2, size=(k-1, 1))))
        tauh = np.cumprod(delta)  # Global shrinkage coefficients (ex tau)
        Plam = phiih * np.tile(tauh.T, (n_basis, 1))  # Precision of loadings rows (ex Ptht)
        Lambda = np.zeros((n_basis, k))  # Matrix of factor loading
        eta = stats.norm.rvs(0, 1, size=(k, n_studies))  # Matrix of latent factors
        # multivariate_normal only takes 1D arrays for mu,
        # but in MATLAB when mu is 2D, it iterates over rows in mu
        temp = np.dot(Lambda, eta).T
        theta = np.zeros((temp.shape))
        for i_row in range(temp.shape[0]):
            theta[i_row, :] = np.random.multivariate_normal(  # Matrix of basis function coefficients
                temp[i_row, :], np.diag(1. / np.squeeze(sig)))
        iota = stats.uniform.rvs(0, 1, size=(n_cov, k))  # Initialise matrix of covariates' coefficients
        Omega = np.zeros((n_cov, k))  # Initialise Matrix of iota variances

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	DO mult probit extension: intial values & hyper.  % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # NOTE: If you change mu_gamma or inv_sigma_gamma, modify
        # lines 416 and 417, 428, 429 accordingly (adaptation of k block)
        latent = np.zeros((n_studies, n_study_types))  # Latent variable indicators
        # Hyperparameters for normal prior on alpha
        mu_alpha = np.zeros((n_study_types, 1))
        sigma_alpha = np.eye(n_study_types)
        alpha = stats.multivariate_normal.rvs(mu_alpha, sigma_alpha)  # Type-specific random intercept
        post_var_alpha = np.diag(np.tile(1 / (n_studies+1), (n_study_types, 1)))  # Post cov matrix for update of alpha

        gamma = np.ones((k, n_study_types))  # Gamma coefficients for DO probit model
        # Hyperparameters for normal prior on gamma
        mu_gamma = np.zeros((k, 1))
        Sigma_gamma = np.eye(k)
        inv_sigma_gamma = np.linalg.inv(Sigma_gamma)  # Invert cov matrix for posterior computation
        # But since Sigma_gamma is an identity matrix, the inverse is the same, so why do this?

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	Setting train & test sets  % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Test set (20% of data points)
        test = sorted(np.random.choice(n_studies, int(np.ceil(n_studies / 5))))
        train = setdiff(np.arange(n_studies), test)  # Test set
        ltrain = len(train)  # Length of train set
        np.savetxt('train.txt', train)
        # Looks good up to here! Though all of the random arrays make it hard
        # to tell for sure...

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	     Define output files      % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        maxk = 50  # Max expected number of factors
        Acc = np.zeros((N, nrun))  # Acceptance probabilities

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	  Start Gibbs sampling 	  % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        for i in range(nrun):
            # -- Update Lambda -- %
            Lambda = np.zeros((nbasis, k))
            for j in range(nbasis):
                Vlam1 = np.diag(Plam[j, :]) + sig[j] * np.dot(eta, eta.T)
                T = cholcov(Vlam1)
                Q, R = qr[T]
                S = np.inv(R)
                Vlam = np.dot(S, S.T)
                Elam = Vlam * sig[j] * eta * theta[:, j]
                Lambda[j, :] = (Elam + S * stats.norm.rvs(np.zeros((k, 1)), 1)).T
            k = Lambda.shape[1]

            # -- Update phi_{ih}'s -- %
            phiih = stats.gamma.rvs(
                df_/2 + 0.5, scale=1./(df_/2 + bsxfun(@times, Lambda**2, tauh.T)))

            # -- Update delta -- %
            ad = ad1 + nbasis * k / 2
            bd = bd1 + 0.5 * (1 / delta[1]) * np.sum(tauh.T * np.sum(phiih * Lambda ** 2))
            delta[1] = stats.gamma.rvs(ad, 1/bd)

            tauh = np.cumprod(delta)
            for h in range(2, k):
                ad = ad2 + nbasis * (k - h + 1) / 2
                temp1 = tauh.T * np.sum(phiih * Lambda ** 2)
                bd = bd2 + 0.5 * (1/delta[h])*sum(temp1[h:k])
                delta[h] = stats.gamma.rvs(ad, scale=1/bd)
                tauh = np.cumprod(delta)

            # -- Update precision parameters -- %
            Plam = bsxfun(@times, phiih, tauh.T)

            # -- Update Sigma precisions -- %
            thetatil = theta.T - np.dot(Lambda, eta)
            sig = stats.gamma.rvs(as_ + N/2, scale=1/(bs + 0.5 * np.sum(thetatil**2, axis=1)))

            # -- Update linear model on latent factors -- %
            for l in range(1, k):
                Omega[:, l] = stats.gamma.rvs(1, scale=1./(0.5 * (1 + iota[:, l]**2)))
                Veta1 = np.dot(cova.T, cova) + np.diag(Omega[:,l])
                T = cholcov(Veta1)
                [Q, R] = qr(T)
                S = inv(R)
                Vlam = np.dot(S, S.T)
                Meta = Vlam * np.dot(eta[l, :], cova).T
                iota[:, l] = Meta + np.dot(S, stats.norm.rvs(0, 1, size=(n_cov, 1)))

            # -- Update of eta (probit model extension with covariates) -- %
            Lmsg = Lambda.T * np.diag(sig)
            Veta1 = Lmsg * Lambda + eye(k) + np.dot(gamma, gamma.T)
            T = cholcov(Veta1)
            Q, R = qr[T]
            S = np.inv(R)
            Vlam = np.dot(S, S.T)
            Meta = Vlam * (Lmsg * theta.T + gamma * (latent - np.tile(alpha, [N, 1])).T + (cova * iota).T);
            eta = Meta + np.dot(S, stats.norm.rvs(0, 1, size=(k, N)))

            # -- Update alpha intercept -- %
            post_mean_alpha = sigma_alpha * mu_alpha + sum(latent - eta.T*gamma).T
            alpha = mvnrnd(post_var_alpha * post_mean_alpha, post_var_alpha)

            # -- Update gamma coefficients -- %
            Veta1 = inv_sigma_gamma + np.dot(eta, eta.T)
            T = cholcov(Veta1)
            [Q, R] = qr(T)
            S = inv(R)
            Vlam = np.dot(S, S.T)
            Meta = Vlam * (np.tile(inv_sigma_gamma * mu_gamma, [1, n_study_types]) + eta * (latent - np.tile(alpha, [N, 1])))
            gamma = Meta + S * stats.norm.rvs(np.zeros((k, n_study_types)), 1)

            # -- Update latent indicators -- %
            mean_latent = repmat(alpha, [N, 1]) + np.dot(eta.T, gamma)
            latent = np.zeros((N, n_study_types))

            # % % % % % % % % % % % % % % % % % % % % % % %
            # Take care of studies in the train set first %
            # % % % % % % % % % % % % % % % % % % % % % % %
            for j in range(ltrain):
                quale = train[j]
                ind = Y[quale]
                latent[quale, ind] = randraw('normaltrunc', [0, Inf, mean_latent(quale, ind), 1], 1)
                diffind = np.setdiff1d(np.arange(n_study_types), ind)
                for l in range(len(diffind)):
                    m = diffind[l]
                    latent[quale, m] = randraw('normaltrunc', [-Inf, 0, mean_latent(quale, m), 1], 1);

            # % % % % % % % % % % % % % % % % % % % %
            # Test set via predictive probabilities % --> DO mult probit model
            # % % % % % % % % % % % % % % % % % % % %
            m1 = 1 - normcdf(-mean_latent[test, :], 0, 1)  # [1 - F(-wj)]
            m2 = normcdf(-mean_latent[test, :], 0, 1)  # F(-wk)
            # NOTE: change lines 337-348 to adapt to the correct number of study-types one is working with
            # E.G.: with 3 study-types, delete lines 342-345 and remove p4 and p5
            # E.G.: with 7 study-types, add computation of p6 and p7 and modify lines 347 & 348
            # Predictive probability for anger
            m3 = np.cumprod([m1[:, 1], m2[:, 2:5]], 2)
            p1 = m3[:, n_study_types]
            # Predictive probability for disgust
            m3 = np.cumprod([m1[:, 2], m2[:, 1], m2[:, 3:5]], 2)
            p2 = m3[:, n_study_types]
            # Predictive probability for fear
            m3 = np.cumprod([m1[:, 3], m2[:, 1:2], m2[:, 4:5]], 2)
            p3 = m3[:, n_study_types]
            # Predictive probability for happy
            m3 = np.cumprod([m1[:, 4], m2[:, 1:3], m2[:, 5]], 2)
            p4 = m3[:, n_study_types]
            # Predictive probability for sad
            m3 = np.cumprod([m1[:, 5], m2[:, 1:4]], 2)
            p5 = m3[:, n_study_types]

            ptot = p1 + p2 + p3 + p4 + p5
            pred_prob = [p1./ptot, p2./ptot, p3./ptot, p4./ptot, p5./ptot]
            cdf = np.cumsum(pred_prob, 2)
            u = stats.uniform.rvs(0, 1, [length(test), 1])
            pred_cat = sum(bsxfun(@gt, u, cdf),2)+1

            for j in range(len(test)):
                quale = test(j)
                ind = pred_cat(j)
                latent(quale, ind) = randraw('normaltrunc', [0, Inf, mean_latent(quale, ind), 1], 1)
                diffind = setdiff(1:n_study_types, ind)
                for l in range(len(diffind)):
                    m = diffind[l]
                    latent[quale, m] = randraw('normaltrunc', [-Inf, 0, mean_latent(quale, m), 1], 1);

            if mod[i, thin] == 0:
                dlmwrite('pred_prob.txt', pred_prob[:].T, 'delimiter', ' ', '-append')
                dlmwrite('pred_cat.txt', pred_cat.T, 'delimiter', ' ', '-append')

            # -- Update of theta -- #
            L_hmc = stats.poisson.rvs(L_hmc_def)
            dlmwrite('HMC_nsteps.txt', L_hmc, 'delimiter', ' ', '-append')

            [theta_hmc, acc, pot_energy] = self._hmc(
                epsilon_hmc, L_hmc, theta, A, Bpred, tBpred, Lambda, eta, sig,
                sum_B)

            if sum(isnan(acc)) != 0:
                acc[isnan(acc), 1] = 0

            theta = theta_hmc
            Acc[:, i] = acc

         	dlmwrite('Acc_while.txt', mean(Acc(:, i)), '-append');

            if mod(i,every) == 0 and i <= burn:
                dlmwrite('thetacbma_pre.txt', (theta(:)).T, 'delimiter', ' ', '-append')

            if mod(i,thin) == 0 and i > burn:
                dlmwrite('thetacbma_post.txt', (theta(:)).T, 'delimiter', ' ', '-append')

        	if mod(i,thin) == 0 and i <= burn and i >= start:
        	    avr = mean(mean(Acc(:, ((i - start) + 1) : i)))
        		epsilon_hmc = (avr >= 0.65) * 1.05 * epsilon_hmc + (avr < 0.65) * 0.95 * epsilon_hmc
        		dlmwrite('Eps.txt', epsilon_hmc, 'delimiter', ' ', '-append')

            # -- Adapt number of latent factors -- %
        	prob = 1/exp(b0 + b1*i)  # Probability of adapting
        	uu = rand
        	lind = sum(abs(Lambda) < epsilon)/nbasis  # Proportion of elements in each column less than eps in magnitude
        	vec = lind >= prop; num = sum(vec)  # number of redundant columns

            if uu < prob:
                if i > 20 && num == 0 && all(lind < 0.995):
                    k = k + 1
                    Lambda(:,k) = zeros(nbasis,1)
                    eta(k, :) = normrnd(0,1,[1, N])
                    Omega(:, k) = gamrnd(.5, 2, [n_cov,1])
                    iota(:, k) = mvnrnd(zeros(1,n_cov), diag(1./(Omega(:,k)))).T
                    phiih(:,k) = gamrnd(df_/2, 2/df_,[nbasis,1])
                    delta(k) = gamrnd(ad2, 1/bd2)
                    tauh = cumprod(delta)
        			Plam = bsxfun(@times, phiih, tauh.T)
                    gamma(k,:) = normrnd(0,1,[n_study_types, 1]).T
                    mu_gamma = zeros(k, 1)
        			inv_sigma_gamma = eye(k)  # Covariance for normal prior on gamma
                elif num > 0:
                    nonred = setdiff(1:k, find(vec))
                    k = max(k - num,1)
                    Lambda = Lambda(:,nonred)
                    eta = eta(nonred,:)
                    phiih = phiih(:,nonred)
                    delta = delta(nonred)
                    tauh = cumprod(delta)
                    Plam = bsxfun(@times, phiih, tauh.T)
                    gamma = gamma(nonred,:)
                    mu_gamma = zeros(k, 1)
                    inv_sigma_gamma = eye(k)  # Covariance for normal prior on gamma
                    iota = iota(:, nonred)
                    Omega = Omega(:, nonred)

            # -- Save sampled values (after thinning) -- %
            if mod(i,every) == 0 and i <= burn:
                dlmwrite('sigma.txt', sig.T, 'delimiter', ' ', '-append')
                dlmwrite('alpha.txt', alpha, 'delimiter', ' ', '-append')
                dlmwrite('Factor.txt', k, 'delimiter', ' ', '-append')

                Etaout_PreBIN = zeros(N*maxk, 1)
                teta = eta.T
                Etaout_PreBIN(1:(N*k), 1) = teta(:)
                clear teta
                dlmwrite('Eta_PreBIN.txt', Etaout_PreBIN.T, 'delimiter', ' ', '-append')
                clear Etaout_PreBIN

                Gammaout_PreBIN = zeros(n_study_types*maxk, 1)
                tgamma = gamma.T
                Gammaout_PreBIN(1:(n_study_types*k), 1) = tgamma(:)
                clear tgamma
                dlmwrite('Gamma_PreBIN.txt', Gammaout_PreBIN.T, 'delimiter', ' ', '-append')
                clear Gammaout_PreBIN

                Iotaout_PreBIN = zeros(n_cov*maxk, 1)
                Iotaout_PreBIN(1:(n_cov*k), 1) = iota(:)
                dlmwrite('Iota_PreBIN.txt', Iotaout_PreBIN.T, 'delimiter', ' ', '-append')
                clear Iotaout_PreBIN

                Omegaout_PreBIN = zeros(n_cov*maxk, 1)
                Omegaout_PreBIN(1:(n_cov*k), 1) = Omega(:)
         	    dlmwrite('Omega_PreBIN.txt', Omegaout_PreBIN.T, 'delimiter', ' ', '-append'); clear Omegaout_PreBIN;
                # dlmwrite('mgpshyper.txt', [ad1, bd1, ad2, bd2, df_], 'delimiter', ' ', '-append');
                phiihout = zeros(nbasis * maxk, 1)
                phiihout(1:(nbasis * k), 1) = phiih(:)
                dlmwrite('Phiih.txt', phiihout.T, 'delimiter', ' ', '-append')
                clear phiihout
                deltaout = zeros(maxk, 1)
                deltaout(1:k, 1) = delta
                dlmwrite('Delta.txt', deltaout.T, 'delimiter', ' ', '-append')
                clear deltaout
            elif mod(i, thin) == 0 and i > burn:
                Lambdaout = zeros(nbasis*maxk, 1)
                Lambdaout(1:(nbasis*k), 1) = Lambda(:).T
                dlmwrite('Lambda.txt', Lambdaout.T, 'delimiter', ' ', '-append')
                clear Lambdaout

                Etaout = zeros(N*maxk, 1)
                teta = eta.T
                Etaout(1:(N*k), 1) = eta(:)
                clear teta
                dlmwrite('Eta_PBIN.txt', Etaout.T, 'delimiter', ' ', '-append')
                clear Etaout

                dlmwrite('HMC_Energy.txt', pot_energy.T, 'delimiter', ' ', '-append')

                Gammaout = zeros(n_study_types*maxk, 1)
                tgamma = gamma.T
                Gammaout(1:(n_study_types*k), 1) = tgamma(:)
                clear tgamma
         	    dlmwrite('Gamma_PBIN.txt', Gammaout.T, 'delimiter', ' ', '-append')
                clear Gammaout

                Iotaout = zeros(n_cov*maxk, 1)
                Iotaout(1:(n_cov*k), 1) = iota(:)
                dlmwrite('Iota_PBIN.txt', Iotaout.T, 'delimiter', ' ', '-append')
                clear Iotaout

                Omegaout_PBIN = zeros(n_cov*maxk, 1)
                Omegaout_PBIN(1:(n_cov*k), 1) = Omega(:)
                dlmwrite('Omega.txt', Omegaout_PBIN.T, 'delimiter', ' ')
                clear Omegaout_PBIN

                dlmwrite('sigma.txt', sig.T, 'delimiter', ' ', '-append')
                dlmwrite('alpha.txt', alpha, 'delimiter', ' ', '-append')
                dlmwrite('Factor.txt', k, 'delimiter', ' ', '-append')
                dlmwrite('latent.txt', latent(:).T, 'delimiter', ' ', '-append')
                % dlmwrite('mgpshyper.txt', [ad1, bd1, ad2, bd2, df_], 'delimiter', ' ', '-append')
                phiihout = zeros(nbasis * maxk, 1)
                phiihout(1:(nbasis * k), 1) = phiih(:)
                dlmwrite('Phiih.txt', phiihout.T, 'delimiter', ' ', '-append')
                clear phiihout
                deltaout = zeros(maxk, 1)
                deltaout(1:k, 1) = delta
                dlmwrite('Delta.txt', deltaout.T, 'delimiter', ' ', '-append')
                clear deltaout
        np.savetxt('HMC_Acc.txt', Acc.T)
        """


@due.dcite(Doi('10.1214/11-AOAS523'),
           description='Introduces the SBR model.')
class SBR(CBMAEstimator):
    """
    Spatial binary regression model
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, sample):
        pass
