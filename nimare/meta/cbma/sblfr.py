"""
Spatial Bayesian latent factor regression model
"""
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

from .base import CBMAEstimator
from ...due import due, Doi


@due.dcite(references.SBLFR, description='Introduces the SBLFR model.')
class SBLFR(CBMAEstimator):
    """
    Spatial Bayesian latent factor regression model

    This uses the HamiltonianMC and the truncnorm.rvs method.
    """
    def __init__(self, df, mask_file):
        self.df = df
        self.mask_img = nib.load(mask_file)

    def _hmc(self, epsilon_hmc, L_hmc, theta, A, Bpred, tBpred, Lambda, eta,
             sig, sum_B):
        """
        Parameters
        ----------
        A : float
            voxel volume
        """
        q = theta
        n_studies, n_basis = theta.shape

        # Independent standard normal variates
        kinetic = stats.norm.rvs(0, 1, size=(n_studies, n_basis))
        current_kinetic = kinetic.copy()

        # Pre-compute matrices to speed up computation
        Leta = np.dot(Lambda, eta)
        AtBpred = np.dot(A, tBpred)

        # Make a half step momentum at the beginning
        add_grad_q = -1 * np.dot(np.diag(sig), Leta) - sum_B.T
        ttheta = theta.T
        grad_q = (np.dot(AtBpred, np.exp(np.dot(Bpred, ttheta))) +
                  np.dot(np.diag(sig), ttheta) + add_grad_q)
        kinetic = kinetic - epsilon_hmc * 0.5 * grad_q.T

        # Alternate full steps for position and momentum
        for e in range(L_hmc):
            # Make a full step for the position
            q = q + epsilon_hmc * kinetic
            tq = q.T

            # Make a full step for the momentum, except at end of trajectory
            if e < L_hmc - 1:
                grad_q = (np.dot(AtBpred, np.exp(np.dot(Bpred, tq))) +
                          np.dot(np.diag(sig), tq) + add_grad_q)
                kinetic = kinetic - epsilon_hmc * grad_q.T

        # Make a half step momentum at the end
        grad_q = (np.dot(AtBpred, np.exp(np.dot(Bpred, tq))) +
                  np.dot(np.diag(sig), tq) + add_grad_q)
        kinetic = kinetic - epsilon_hmc * 0.5 * grad_q.T

        # Negate momentum at end of trajectory to make the proposal symmetric
        kinetic = -1 * kinetic

        # Evaluate potential and kinetic energies at start and end of trajectory
        current_U = (
            A * np.sum(np.exp(np.dot(Bpred, ttheta)), axis=0).T -
            np.diag(np.dot(sum_B, ttheta)) + 0.5 *
            np.diag(np.dot(np.dot(theta - Leta.T, np.diag(sig)),
                           (theta - Leta.T).T))
        )
        current_K = np.sum((current_kinetic ** 2).T, axis=0).T / 2.

        proposed_U = (
            A * np.sum(np.exp(np.dot(Bpred, ttheta)), axis=0).T -
            np.diag(np.dot(sum_B, tq)) + 0.5 *
            np.diag(np.dot(np.dot(q - Leta.T, np.diag(sig)), (q - Leta.T).T))
        )
        proposed_K = np.sum((kinetic ** 2).T, axis=0).T / 2.

        # Accept or reject the state at end of trajectory, returning either the position
        # at the end of the trajectory or the initial position
        pratio = current_U - proposed_U + current_K - proposed_K  # Log acceptance ratio
        u = np.log(np.random.random(n_studies))
        assert u.shape == pratio.shape
        acc = (u < pratio).astype(int)
        ind = np.where(acc == 1)[0]
        theta_hmc = theta
        theta_hmc[ind, :] = q[ind, :]  # theta_hmc = old, excpet for those proposal which get accepted

        pot_energy = (
            A * np.sum(np.exp(np.dot(Bpred, theta_hmc.T)), axis=0).T -
            np.diag(np.dot(sum_B, theta_hmc.T)) + 0.5 *
            np.diag(np.dot(np.dot(theta_hmc - Leta.T, np.diag(sig)),
                           (theta_hmc - Leta.T).T))
        )
        return theta_hmc, acc, pot_energy

    def fit(self, studytype_col=None, covariates=None):
        """
        Looks good through line 99 of the MATLAB script main_3D.m.

        Parameters
        ----------
        covariates : :obj:`list` or None, optional
            List of column names for covariates to include in model.
        """
        out_dir = 'sblfr/'
        df = self.df
        mask_img = self.mask_img
        if isinstance(covariates, list):
            n_cov = len(covariates)  # int
            covariate_data = df.groupby('id').mean()[covariates]  # (n_studies, n_covariates)
        else:
            n_cov = 0
            covariate_data = None

        if isinstance(studytype_col, str):
            study_types = df[studytype_col]  # (n_foci,)
            n_study_types = len(np.unique(study_types))  # int
        else:
            study_types = []
            n_study_types = 0

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	    Define global constants      % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nrun = 50000  # Number of MCMC iterations
        burn = 25000  # Burn-in
        thin = 50  # Thinning MCMC samples
        every = 100  # Number of previous samples consider to update the HMC stepsize
        start = 250  # Starting iteration to update the stepsize
        sp = (nrun - burn) / thin  # Number of posterior samples
        epsilon = 1e-4  # Threshold limit (update of Lambda)
        prop = 1.  # Proportion of redundant elements within columns
        slices = [8, 9]

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
        for i_slice, slice_idx in enumerate(slices):
            arr = mask_full[:, :, slice_idx]
            msk = np.ravel(arr)  # NOTE: are we sure this matches up?
            n_voxels_in_mask = mesh_grid[msk, :].shape[0]  # was sg
            # Get X, study_types (from mesh_grid), and Z (from Iz) coords for mask voxels
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
        alpha = np.random.multivariate_normal(np.squeeze(mu_alpha), sigma_alpha)  # Type-specific random intercept
        post_var_alpha = np.diag(np.tile(1 / (n_studies+1), (n_study_types)))  # Post cov matrix for update of alpha

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
        test = sorted(np.random.choice(ids, int(np.ceil(n_studies / 5)),
                                       replace=False))
        train = np.setdiff1d(ids, test)  # Train set
        np.savetxt(op.join(out_dir, 'train.txt'), train)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	     Define output files      % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        maxk = 50  # Max expected number of factors
        Acc = np.zeros((n_studies, nrun))  # Acceptance probabilities

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	  Start Gibbs sampling 	  % %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i in range(nrun):
            print(i)
            # -- Update Lambda -- %
            Lambda = np.zeros((n_basis, k))
            for j in range(n_basis):
                Vlam1 = np.diag(Plam[j, :]) + sig[j] * np.dot(eta, eta.T)
                T = np.linalg.cholesky(Vlam1).T
                Q, R = np.linalg.qr(T)
                S = np.linalg.inv(R)
                Vlam = np.dot(S, S.T)
                Elam = np.dot(np.dot(Vlam * sig[j], eta), theta[:, j][:, None])
                Lambda[j, :] = (Elam + np.dot(S, stats.norm.rvs(np.zeros((k, 1)), 1))).T
            k = Lambda.shape[1]

            # -- Update phi_{ih}'s -- %
            phiih = stats.gamma.rvs(
                df_ / 2. + 0.5, scale=1. / ((df_ / 2) + (Lambda**2 * tauh)))

            # -- Update delta -- %
            ad = ad1 + n_basis * k / 2
            bd = bd1 + 0.5 * (1 / delta[1]) * np.sum(tauh.T * np.sum(phiih * Lambda ** 2))
            delta[0] = stats.gamma.rvs(ad, 1/bd)

            tauh = np.cumprod(delta)
            for h in range(1, k):
                ad = ad2 + n_basis * (k - h) / 2.
                temp1 = tauh.T * np.sum(phiih * Lambda ** 2, axis=0)
                bd = bd2 + 0.5 * (1. / delta[h]) * np.sum(temp1[h:k])
                delta[h] = stats.gamma.rvs(ad, scale=1/bd)
                tauh = np.cumprod(delta)

            # -- Update precision parameters -- %
            Plam = phiih * tauh

            # -- Update Sigma precisions -- %
            thetatil = theta.T - np.dot(Lambda, eta)
            sig = stats.gamma.rvs(
                as_ + n_studies/2.,
                scale=1/(bs + 0.5 * np.sum(thetatil**2, axis=1)))

            # -- Update linear model on latent factors -- %
            for l in range(k):
                Omega[:, l] = stats.gamma.rvs(
                    1, scale=1 / (0.5 * (1 + iota[:, l]**2)))
                Veta1 = np.dot(covariate_data.T, covariate_data) + np.diag(Omega[:, l])
                T = np.linalg.cholesky(Veta1).T
                Q, R = np.linalg.qr(T)
                S = np.linalg.inv(R)
                Vlam = np.dot(S, S.T)
                Meta = np.dot(Vlam, np.dot(eta[l, :], covariate_data).T)
                iota[:, l] = Meta + np.dot(S, stats.norm.rvs(0, 1, size=(n_cov)))

            # -- Update of eta (probit model extension with covariates) -- %
            Lmsg = np.dot(Lambda.T, np.diag(sig))
            Veta1 = np.dot(Lmsg, Lambda) + np.eye(k) + np.dot(gamma, gamma.T)
            T = np.linalg.cholesky(Veta1).T
            Q, R = np.linalg.qr(T)
            S = np.linalg.inv(R)
            Vlam = np.dot(S, S.T)
            temp0 = np.dot(Lmsg, theta.T)
            temp1 = np.dot(covariate_data, iota).T
            temp2 = (latent - np.tile(alpha, (n_studies, 1))).T
            temp3 = np.dot(gamma, temp2)
            temp4 = temp0 + temp3 + temp1
            Meta = np.dot(Vlam, temp4)
            eta = Meta + np.dot(S, stats.norm.rvs(0, 1, size=(k, n_studies)))

            # -- Update alpha intercept -- %
            post_mean_alpha = (np.dot(sigma_alpha, mu_alpha) +
                               np.sum(latent - np.dot(eta.T, gamma), axis=0, keepdims=True).T)
            alpha = np.random.multivariate_normal(
                np.squeeze(np.dot(post_var_alpha, post_mean_alpha)), post_var_alpha)

            # -- Update gamma coefficients -- %
            Veta1 = inv_sigma_gamma + np.dot(eta, eta.T)
            T = np.linalg.cholesky(Veta1).T
            Q, R = np.linalg.qr(T)
            S = np.linalg.inv(R)
            Vlam = np.dot(S, S.T)
            temp0 = np.tile(np.dot(inv_sigma_gamma, mu_gamma), (1, n_study_types))
            temp1 = np.tile(alpha, (n_studies, 1))
            temp2 = temp0 + np.dot(eta, (latent - temp1))
            Meta = np.dot(Vlam, temp2)
            gamma = Meta + np.dot(
                S, stats.norm.rvs(np.zeros((k, n_study_types)), 1))

            # -- Update latent indicators -- %
            # make sure to use transpose of cholesky or else scale will be weird
            mean_latent = np.tile(alpha, (n_studies, 1)) + np.dot(eta.T, gamma)
            latent = np.zeros((n_studies, n_study_types))

            # % % % % % % % % % % % % % % % % % % % % % % %
            # Take care of studies in the train set first %
            # % % % % % % % % % % % % % % % % % % % % % % %
            for j in range(len(train)):
                j_id = train[j]
                id_idx = np.where(ids == j_id)[0][0]
                st_idx = int(study_types[id_idx]) - 1
                latent[id_idx, st_idx] = stats.truncnorm.rvs(
                    0, np.Inf, loc=mean_latent[id_idx, st_idx], scale=1, size=1)[0]
                diffind = np.setdiff1d(np.arange(n_study_types), st_idx)
                for l in range(len(diffind)):
                    m = diffind[l]
                    latent[id_idx, m] = stats.truncnorm.rvs(
                        -np.Inf, 0, loc=mean_latent[id_idx, m], scale=1, size=1)[0]

            # % % % % % % % % % % % % % % % % % % % %
            # Test set via predictive probabilities % --> DO mult probit model
            # % % % % % % % % % % % % % % % % % % % %
            test_idx = np.where(np.isin(ids, test))[0]
            m1 = 1 - stats.norm.cdf(-mean_latent[test_idx, :], 0, 1)  # [1 - F(-wj)]
            m2 = stats.norm.cdf(-mean_latent[test_idx, :], 0, 1)  # F(-wk)
            # NOTE: change lines 337-348 to adapt to the correct number of study-types one is working with
            # E.G.: with 3 study-types, delete lines 342-345 and remove p4 and p5
            # E.G.: with 7 study-types, add computation of p6 and p7 and modify lines 347 & 348
            # Predictive probability for anger
            m3 = np.cumprod(np.hstack((m1[:, 0:1], m2[:, 1:5])), 1)
            p1 = m3[:, -1]
            # Predictive probability for disgust
            m3 = np.cumprod(np.hstack((m1[:, 1:2], m2[:, 0:1], m2[:, 2:5])), 1)
            p2 = m3[:, -1]
            # Predictive probability for fear
            m3 = np.cumprod(np.hstack((m1[:, 2:3], m2[:, 0:2], m2[:, 3:5])), 1)
            p3 = m3[:, -1]
            # Predictive probability for happy
            m3 = np.cumprod(np.hstack((m1[:, 3:4], m2[:, 0:3], m2[:, 4:5])), 1)
            p4 = m3[:, -1]
            # Predictive probability for sad
            m3 = np.cumprod(np.hstack((m1[:, 4:5], m2[:, 0:4])), 1)
            p5 = m3[:, -1]

            ptot = p1 + p2 + p3 + p4 + p5
            pred_prob = np.stack((p1/ptot, p2/ptot, p3/ptot, p4/ptot, p5/ptot)).T
            cdf = np.cumsum(pred_prob, axis=1)
            u = stats.uniform.rvs(0, 1, (len(test_idx), 1))
            pred_cat = np.sum(u > cdf, axis=1)
            # Looks good up to here!

            for j in range(len(test)):
                j_id = test[j]
                id_idx = np.where(ids == j_id)[0][0]
                st_idx = pred_cat[j]
                latent[id_idx, st_idx] = stats.truncnorm.rvs(
                    0, np.Inf, loc=mean_latent[id_idx, st_idx], scale=1, size=1)[0]
                diffind = np.setdiff1d(np.arange(n_study_types), st_idx)
                for l in range(len(diffind)):
                    m = diffind[l]
                    latent[id_idx, m] = stats.truncnorm.rvs(
                        -np.Inf, 0, loc=mean_latent[id_idx, m], scale=1, size=1)[0]

            if i % thin == 0:
                with open(op.join(out_dir, 'pred_prob.txt'), 'ab') as fo:
                    np.savetxt(fo, np.ravel(pred_prob)[None, :], delimiter='\t')

                with open(op.join(out_dir, 'pred_cat.txt'), 'ab') as fo:
                    np.savetxt(fo, pred_cat[None, :], delimiter='\t')

            # -- Update of theta -- #
            L_hmc = stats.poisson.rvs(L_hmc_def)
            with open(op.join(out_dir, 'HMC_nsteps.txt'), 'a') as fo:
                fo.write('{}\n'.format(L_hmc))

            theta_hmc, acc, pot_energy = self._hmc(
                epsilon_hmc, L_hmc, theta, voxel_vol, Bpred, tBpred, Lambda,
                eta, sig, sum_B)

            if np.sum(np.isnan(acc)) != 0:
                acc[np.isnan(acc), 1] = 0

            theta = theta_hmc
            Acc[:, i] = acc
            with open(op.join(out_dir, 'Acc_while.txt'), 'a') as fo:
                fo.write('{}\n'.format(np.mean(Acc[:, i])))

            if i % every == 0 and i <= burn:
                with open(op.join(out_dir, 'thetacbma_pre.txt'), 'ab') as fo:
                    np.savetxt(fo, np.ravel(theta)[None, :], delimiter='\t')

            if i % thin == 0 and i > burn:
                with open(op.join(out_dir, 'thetacbma_post.txt'), 'ab') as fo:
                    np.savetxt(fo, np.ravel(theta)[None, :], delimiter='\t')

            if i % thin == 0 and i <= burn and i >= start:
                slice_rng = np.arange(i-start+1, i)
                avr = np.mean(Acc[:, slice_rng])
                epsilon_hmc = (avr >= 0.65) * 1.05 * epsilon_hmc + (avr < 0.65) * 0.95 * epsilon_hmc
                with open(op.join(out_dir, 'Eps.txt'), 'a') as fo:
                    fo.write('{}\n'.format(epsilon_hmc))

            # -- Adapt number of latent factors -- %
            prob = 1. / np.exp(b0 + b1*i)  # Probability of adapting
            uu = np.random.random()
            # Proportion of elements in each column less than eps in magnitude
            lind = np.sum(np.abs(Lambda) < epsilon, axis=0) / n_basis
            vec = lind >= prop
            num = np.sum(vec)  # number of redundant columns

            if uu < prob:
                if i > 20 and num == 0 and np.all(lind < 0.995):
                    k += 1
                    # We need to expand some matrices here
                    if k > Lambda.shape[1]:
                        temp = np.zeros((n_basis, 1))
                        Lambda = np.hstack((Lambda, temp))
                    else:
                        Lambda[:, k] = np.zeros((n_basis, 1))

                    if k > eta.shape[0]:
                        temp = stats.norm.rvs(0, 1, size=(1, n_studies))
                        eta = np.vstack((eta, temp))
                    else:
                        eta[k, :] = stats.norm.rvs(0, 1, size=(1, n_studies))

                    if k > Omega.shape[1]:
                        temp = stats.gamma.rvs(.5, 2, size=(n_cov, 1))
                        Omega = np.hstack((Omega, temp))
                    else:
                        Omega[:, k] = stats.gamma.rvs(.5, 2, size=(n_cov, 1))

                    if k > iota.shape[1]:
                        temp = np.random.multivariate_normal(
                            np.zeros(n_cov), np.diag(1. / (Omega[:, k-1]))).T
                        iota = np.hstack((iota, temp[:, None]))
                    else:
                        iota[:, k] = np.random.multivariate_normal(
                            np.zeros(n_cov), np.diag(1. / (Omega[:, k-1]))).T

                    if k > phiih.shape[1]:
                        temp = stats.gamma.rvs(df_/2, 2/df_, size=(n_basis, 1))
                        phiih = np.hstack((phiih, temp))
                    else:
                        phiih[:, k] = stats.gamma.rvs(df_/2, 2/df_, size=(n_basis, 1))

                    if k > delta.shape[0]:
                        temp = stats.gamma.rvs(ad2, 1/bd2)
                        delta = np.vstack((delta, temp))
                    else:
                        delta[k] = stats.gamma.rvs(ad2, 1/bd2)

                    tauh = np.cumprod(delta)
                    Plam = phiih * tauh
                    if k > gamma.shape[0]:
                        temp = stats.norm.rvs(0, 1, size=(1, n_study_types))
                        gamma = np.vstack((gamma, temp))
                    else:
                        gamma[k, :] = stats.norm.rvs(0, 1, size=(1, n_study_types))

                    mu_gamma = np.zeros((k, 1))
                    inv_sigma_gamma = np.eye(k)  # Covariance for normal prior on gamma
                elif num > 0:
                    nonred = np.setdiff1d(np.arange(1, k), np.where(vec))
                    k = np.max((k - num, 1))
                    Lambda = Lambda[:, nonred]
                    eta = eta[nonred, :]
                    phiih = phiih[:, nonred]
                    delta = delta[nonred]
                    tauh = np.cumprod(delta)
                    Plam = phiih * tauh
                    gamma = gamma[nonred, :]
                    mu_gamma = np.zeros((k, 1))
                    inv_sigma_gamma = np.eye(k)  # Covariance for normal prior on gamma
                    iota = iota[:, nonred]
                    Omega = Omega[:, nonred]

            # -- Save sampled values (after thinning) -- %
            if i % every == 0 and i <= burn:
                with open(op.join(out_dir, 'sigma.txt'), 'ab') as fo:
                    np.savetxt(fo, sig[None, :], delimiter='\t')

                with open(op.join(out_dir, 'alpha.txt'), 'ab') as fo:
                    np.savetxt(fo, alpha[None, :], delimiter='\t')

                with open(op.join(out_dir, 'Factor.txt'), 'a') as fo:
                    fo.write('{}\n'.format(k))

                Etaout_PreBIN = np.zeros(n_studies*maxk)
                Etaout_PreBIN[:n_studies*k] = np.ravel(eta.T)
                with open(op.join(out_dir, 'Eta_PreBIN.txt'), 'ab') as fo:
                    np.savetxt(fo, Etaout_PreBIN[None, :], delimiter='\t')
                del Etaout_PreBIN

                Gammaout_PreBIN = np.zeros(n_study_types*maxk)
                Gammaout_PreBIN[:n_study_types*k] = np.ravel(gamma.T)
                with open(op.join(out_dir, 'Gamma_PreBIN.txt'), 'ab') as fo:
                    np.savetxt(fo, Gammaout_PreBIN[None, :], delimiter='\t')
                del Gammaout_PreBIN

                Iotaout_PreBIN = np.zeros(n_cov*maxk)
                Iotaout_PreBIN[:n_cov*k] = np.ravel(iota)
                with open(op.join(out_dir, 'Iotaout_PreBIN.txt'), 'ab') as fo:
                    np.savetxt(fo, Iotaout_PreBIN[None, :], delimiter='\t')
                del Iotaout_PreBIN

                Omegaout_PreBIN = np.zeros(n_cov*maxk)
                Omegaout_PreBIN[:n_cov*k] = np.ravel(Omega)
                with open(op.join(out_dir, 'Omegaout_PreBIN.txt'), 'ab') as fo:
                    np.savetxt(fo, Omegaout_PreBIN[None, :], delimiter='\t')
                del Omegaout_PreBIN

                phiihout = np.zeros(n_basis*maxk)
                phiihout[:n_basis*k] = np.ravel(phiih)
                with open(op.join(out_dir, 'Phiih.txt'), 'ab') as fo:
                    np.savetxt(fo, phiihout[None, :], delimiter='\t')
                del phiihout

                deltaout = np.zeros(maxk)
                deltaout[:k] = np.squeeze(delta)
                with open(op.join(out_dir, 'Delta.txt'), 'ab') as fo:
                    np.savetxt(fo, deltaout[None, :], delimiter='\t')
                del deltaout
            elif i % thin == 0 and i > burn:
                Lambdaout = np.zeros(n_basis*maxk)
                Lambdaout[:n_basis*k] = np.ravel(Lambda)
                with open(op.join(out_dir, 'Lambda.txt'), 'ab') as fo:
                    np.savetxt(fo, Lambdaout[None, :], delimiter='\t')
                del Lambdaout

                Etaout = np.zeros(n_studies*maxk)
                Etaout[:n_studies*k] = np.ravel(eta)
                with open(op.join(out_dir, 'Eta_PBIN.txt'), 'ab') as fo:
                    np.savetxt(fo, Etaout[None, :], delimiter='\t')
                del Etaout

                with open(op.join(out_dir, 'HMC_Energy.txt'), 'ab') as fo:
                    np.savetxt(fo, pot_energy[None, :], delimiter='\t')

                Gammaout = np.zeros(n_study_types*maxk)
                Gammaout[:n_study_types*k] = np.ravel(gamma.T)
                with open(op.join(out_dir, 'Gamma_PBIN.txt'), 'ab') as fo:
                    np.savetxt(fo, Gammaout[None, :], delimiter='\t')
                del Gammaout

                Iotaout = np.zeros(n_cov*maxk)
                Iotaout[:n_cov*k] = np.ravel(iota)
                with open(op.join(out_dir, 'Iota_PBIN.txt'), 'ab') as fo:
                    np.savetxt(fo, Iotaout[None, :], delimiter='\t')
                del Iotaout

                Omegaout_PBIN = np.zeros(n_cov*maxk)
                Omegaout_PBIN[:n_cov*k] = np.ravel(Omega)
                with open(op.join(out_dir, 'Omega.txt'), 'ab') as fo:
                    np.savetxt(fo, Omegaout_PBIN[None, :], delimiter='\t')
                del Omegaout_PBIN

                with open(op.join(out_dir, 'sigma.txt'), 'ab') as fo:
                    np.savetxt(fo, sig[None, :], delimiter='\t')

                with open(op.join(out_dir, 'alpha.txt'), 'ab') as fo:
                    np.savetxt(fo, alpha[None, :], delimiter='\t')

                with open(op.join(out_dir, 'Factor.txt'), 'a') as fo:
                    fo.write('{}\n'.format(k))

                with open(op.join(out_dir, 'latent.txt'), 'ab') as fo:
                    np.savetxt(fo, np.ravel(latent)[None, :], delimiter='\t')

                phiihout = np.zeros(n_basis*maxk)
                phiihout[:n_basis*k] = np.ravel(phiih)
                with open(op.join(out_dir, 'Phiih.txt'), 'ab') as fo:
                    np.savetxt(fo, phiihout[None, :], delimiter='\t')
                del phiihout

                deltaout = np.zeros(maxk)
                deltaout[:k] = np.squeeze(delta)
                with open(op.join(out_dir, 'Delta.txt'), 'ab') as fo:
                    np.savetxt(fo, deltaout[None, :], delimiter='\t')
                del deltaout
        np.savetxt(op.join(out_dir, 'HMC_Acc.txt'), Acc.T)
