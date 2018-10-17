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
        if isinstance(covariates, list):
            n_cov = len(covariates)
            covariate_data = self.dataset.coordinates[covariates]
        else:
            n_cov = 0
            covariate_data = None

        if isinstance(studytype_col, str):
            study_types = self.dataset.coordinates[study_types]
            n_study_types = len(np.unique(study_types))
        else:
            study_types = []
            n_study_types = 0

        ijk_id = self.dataset.coordinates[['i', 'j', 'k', 'id']].values
        n_dims = 3  # to remove
        n_studies = len(np.unique(ijk_id[:, -1]))
        n_foci = ijk_id.shape[0]
        n_foci_per_study = self.dataset.coordinates.groupby('id').count()

        # Get voxel volume
        voxel_vol = np.prod(self.dataset.mask.header.get_zooms())

        # Define bases: Gaussian kernels
        xx = np.linspace(40, 145, 9)
        yy = np.linspace(40, 180, 8)
        zz = np.linspace(38, 90, 8)
        temp = [np.ravel(o) for o in np.meshgrid(yy, xx, zz)]
        knots = np.vstack((temp[1], temp[0], temp[2])).T

        # At every odd numbered axial slice, shift the x coordinate to
        # re-create a chess-like kernel grid
        for n in range(int(np.floor(len(zz) / 2))):
            knots[knots[:, -1] == zz[n*2], 0] += 10
        # Remove knots whose x-coordinate is above 145 mm (falling outside of
        # the mask)
        knots = knots[knots[:, 0] <= 145, :]

        bf_bandwidth = 1. / (2 * 256)

        # Observed matrix of basis functions
        B = np.zeros((ijk_id.shape[0], knots.shape[0]))
        for i in range(ijk_id.shape[0]):
            obs_knot = repmat(ijk_id[i, :-1], (knots.shape[0], 1)) - knots
            for j in range(knots.shape[0]):
                B[i, j] = np.exp(-bf_bandwidth *
                                 np.sqrt(np.sum(obs_knot[j, :])) ** 2)

        B = np.hstack((np.ones(B.shape[0]), B))
        n_basis = B.shape[1]

        # Insert into HMC function the sum of bases
        sum_B = np.zeros((n_studies, B.shape[1]))
        for h in range(n_studies):
            if B[ids == h, :].shape[0] * B[ids == h, :].shape[1] == n_basis:
                sum_B[h, :] = B[ID == h, :]
            else:
                sum_B[h, :] = np.sum(B[ids == h, :])

        # Down to line 99
        return None


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
