"""
Model-based coordinate-based meta-analysis estimators
"""

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

    def fit(self, voxel_thresh=0.01, q=0.05, corr='FDR', covariates=None):
        """
        Looks good through line 99 of the MATLAB script main_3D.m.
        """
        if isinstance(covariates, list):
            n_cov = len(covariates)
            covariate_data = df[covariates]
        else:
            n_cov = 0
            covariate_data = None

        if isinstance(studytype_col, str):
            study_types = df[studytype_col]
            n_study_types = len(np.unique(study_types))
        else:
            study_types = []
            n_study_types = 0

        ijk_id = df[['i', 'j', 'k', 'id']].values
        ids = np.unique(ijk_id[:, -1])
        n_dims = 3  # to remove
        n_studies = len(np.unique(ijk_id[:, -1]))
        n_foci = ijk_id.shape[0]
        n_foci_per_study = df.groupby('id').count()

        # Get voxel volume
        voxel_vol = np.prod(mask_img.header.get_zooms())

        # Define bases: Gaussian kernels
        xx = np.linspace(40, 145, 9)
        yy = np.linspace(40, 180, 8)
        zz = np.linspace(38, 90, 8)
        temp = [np.ravel(o) for o in np.meshgrid(yy, xx, zz)]
        knots = np.vstack((temp[1], temp[0], temp[2])).T

        # Resort for easy comparison to MATLAB code
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

        bf_bandwidth = 1. / (2 * 256)

        # Observed matrix of basis functions
        B = np.zeros((ijk_id.shape[0], knots.shape[0]))
        temp_ijk_id = ijk_id - 1  # offset to match matlab
        for i in range(ijk_id.shape[0]):
            obs_knot = np.tile(temp_ijk_id[i, :-1], (knots.shape[0], 1)) - knots
            for j in range(knots.shape[0]):
                B[i, j] = np.exp(-bf_bandwidth * np.linalg.norm(obs_knot[j, :]) ** 2)

        B = np.hstack((np.ones((B.shape[0], 1)), B))
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
        # Note: thresholding done slice-wise below, but naw

        # -- % Grid used to evaluate the integral at HMC step (the finer the grid, the better the approximation) % -- %
        # Get voxel volume
        voxel_vol = np.prod(mask_img.header.get_zooms())

        # Also get grids in mm for 4mm mask
        Ix = np.arange(2, 180, 4)
        Iy = np.arange(2, 216, 4)
        Iz = np.arange(2, 180, 4)
        # Looks good up to here!

        Grid = []
        temp = [np.ravel(o) for o in np.meshgrid(Iy, Ix)]
        mesh_grid = np.vstack((temp[0], temp[1])).T
        # Axial slices to use (9:24 restricts computation to the amygdalae; for a full 3D use 1:91)
        for i_slice in range(9, 25):
            arr = mask_full[:, :, i_slice]
            msk = arr[:]
            sg = mesh_grid[msk, :].shape[0]
            temp = np.vstack(mesh_grid[msk, :], np.tile(Iz[i_slice], (sg, 1)))
            Grid = np.vstack(Grid, temp)

        # -- % Matrix of basis functions to evaluate HMC % -- %
        Bpred = []
        for i in range(Grid.shape[0]):
            obs_knot = np.tile(Grid[i, :], (knots.shape[0], 1)) - knots
            for j in range(knots.shape[0]):
                Bpred[i, j] = np.exp(-bf_bandwidth * np.linalg.norm(obs_knot[j, :]) ** 2);

        Bpred = np.vstack(np.ones(Grid.shape[0], 1), Bpred)
        Bpred[Bpred < 1e-35] = 0

        tBpred = Bpred.T
        V = Bpred.shape[0]  # Number of grid points

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	    Define global constants      % %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nrun = 50000  # Number of MCMC iterations
        burn = 25000  # Burn-in
        thin = 50  # Thinning MCMC samples
        every = 100  # Number of previous samples consider to update the HMC stepsize
        start = 250  # Starting iteration to update the stepsize
        sp = (nrun - burn) / thin  # Number of posterior samples
        epsilon = 1e-4  # Threshold limit (update of Lambda)
        prop = 1.00  # Proportion of redundant elements within columns

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	 Define hyperparameter values    % %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Gamma hyperprior params on a_1 and a_2
        b0 = 2
        b1 = 0.0001
        # Gamma hyperparameters for residual precision
        as_ = 1  # as is a keyword
        bs = 0.3
        # Gamma hyperparameters for t_{ij}
        df = 3
        # Gamma hyperparameters for delta_1
        ad1 = 2.1
        bd1 = 1
        # Gamma hyperparameters for delta_h, h >=2
        ad2 = 3.1
        bd2 = 1
        # Gamma hyperparameters for ad1 and ad2 or df
        adf = 1
        bdf = 1
        # Starting valye for Leapfrog stepsize
        epsilon_hmc = 0.0001
        # Leapfrog trajectory length
        L_hmc_def = 30

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %	 		Initial values    		 % %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
