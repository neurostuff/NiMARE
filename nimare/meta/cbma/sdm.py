"""Seed-based d-mapping-related methods."""
import nibabel as nib
import numpy as np
import pymare
from scipy import ndimage, stats

# from tqdm import tqdm, trange
from tqdm.autonotebook import tqdm, trange

from nimare.meta.utils import _calculate_cluster_measures, compute_sdm_ma
from nimare.stats import hedges_g, hedges_g_var


def mle_estimation(lower_bound_imgs, upper_bound_imgs):
    """Estimate the most likely effect size and variance maps across studies.

    .. todo::

        Implement.

    Parameters
    ----------
    lower_bound_imgs : :obj:`list` of :obj:`~nibabel.nifti1.Nifti1Image`
        Lower-bound effect size maps estimated from coordinates.
        One map for each study.
    upper_bound_imgs : :obj:`list` of :obj:`~nibabel.nifti1.Nifti1Image`
        Upper-bound effect size maps estimated from coordinates.
        One map for each study.

    Returns
    -------
    mle_coeff_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic coefficients map.
    mle_tau2_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic tau2 (variance) map.
    """
    mle_coeff_img, mle_tau2_img = lower_bound_imgs, upper_bound_imgs
    return mle_coeff_img, mle_tau2_img


def simulate_voxel_with_no_neighbors(n_subjects):
    """Simulate the data for a voxel with no neighbors.

    Parameters
    ----------
    n_subjects : int
        Number of subjects in the dataset.

    Returns
    -------
    y : numpy.ndarray of shape (n_subjects,)
        Data for the voxel across subjects.

    Notes
    -----
    For imputing subject values (Y) in a voxel that has no neighboring voxels imputed yet,
    SDM-PSI simply creates random normal values and standardizes them to have null mean and
    unit variance (R):

        -   Y = R
    """
    y = np.random.normal(loc=0, scale=1, size=n_subjects)
    # Enforce null mean and unit variance
    y = stats.zscore(y)
    return y


def simulate_voxel_with_one_neighbor(A, r_ay):
    """Simulate the data for a voxel with one neighbor that has already been simulated.

    Parameters
    ----------
    A : :obj:`~numpy.ndarray` of shape (n_subjects,)
        Subject values for the neighboring voxel.
    r_ay : :obj:`float`
        Desired correlation between A and Y.

    Returns
    -------
    y : :obj:`~numpy.ndarray` of shape (n_subjects,)
        Data for the voxel across subjects.

    Notes
    -----
    For imputing subject values in a voxel that has one neighboring voxel already imputed,
    SDM-PSI conducts a weighted average of the subject values of the neighboring voxel (A) and
    new standardized random normal values:

    -   Y = (w_a * A) + (w_r * R), where w are the weights that ensure that the resulting
        subject values have unit variance and the desired correlation.
    -   w_a = r_ay - (w_r * r_ar * w_r) = np.sqrt((1 - (r_ay ** 2)) / (1 - (r_ar ** 2)))
    """
    # Random normal values with null mean and unit variance.
    R = simulate_voxel_with_no_neighbors(A.size)
    # Correlation between A and R.
    r_ar = np.corrcoef((A, R))[1, 0]

    w_r = np.sqrt((1 - (r_ay**2)) / (1 - (r_ar**2)))
    w_a = r_ay - (w_r * r_ar)

    y = (w_a * A) + (w_r * R)
    y = stats.zscore(y)  # NOTE: The zscore isn't part of the math. It should be z-scores by w*
    return y


def simulate_voxel_with_two_neighbors(A, B, r_ay, r_by):
    """Simulate the data for a voxel with two neighbors that have already been simulated.

    Parameters
    ----------
    A, B : :obj:`~numpy.ndarray` of shape (n_subjects,)
        Subject values for the neighboring voxels.
    r_ay, r_by : :obj:`float`
        Desired correlation between A and Y, and between B and Y.

    Returns
    -------
    y : :obj:`~numpy.ndarray` of shape (n_subjects,)
        Data for the voxel across subjects.

    Notes
    -----
    For imputing subject values in a voxel that has two neighboring voxels already imputed,
    SDM-PSI conducts again a weighted average of the subject values of the neighboring voxels
    (A and B) and new standardized random normal values:

    -   Y = (w_a * A) + (w_b * B) + (w_r * R), where again w are the weights that ensure that
        the resulting subject values have unit variance and the desired correlations.
    -   w_a = r_ay - (w_b * r_ab) - (w_r * r_ar)
    -   w_b = ((r_by - (r_ab * r_ay)) - (w_r * (r_br - (r_ab * r_ar))) / (1 - (r_ab ** 2)))
    -   w_r = np.sqrt((1 - (r_ab ** 2) - (r_ay ** 2) - (r_by ** 2) + (2 * r_ab * r_ay * r_by))
        / (1 - (r_ab ** 2) - (r_ar ** 2) - (r_br ** 2) + (2 * r_ab * r_ar * r_br))
    """
    R = simulate_voxel_with_no_neighbors(A.size)
    r_ab = np.corrcoef((A, B))[1, 0]
    r_ar = np.corrcoef((A, R))[1, 0]
    r_br = np.corrcoef((B, R))[1, 0]

    w_r = (1 - (r_ab**2) - (r_ay**2) - (r_by**2) + (2 * r_ab * r_ay * r_by)) / (
        1 - (r_ab**2) - (r_ar**2) - (r_br**2) + (2 * r_ab * r_ar * r_br)
    )
    w_r = np.sqrt(np.abs(w_r))  # NOTE: The abs isn't part of the math
    w_b = ((r_by - (r_ab * r_ay)) - (w_r * (r_br - (r_ab * r_ar)))) / (1 - (r_ab**2))
    w_a = r_ay - (w_b * r_ab) - (w_r * r_ar)

    y = (w_a * A) + (w_b * B) + (w_r * R)
    y = stats.zscore(y)  # NOTE: The zscore isn't part of the math. It should be z-scores by w*
    return y


def simulate_voxel_with_three_neighbors(A, B, C, r_ay, r_by, r_cy):
    """Simulate the data for a voxel with three neighbors that have already been simulated.

    Parameters
    ----------
    A, B, C : :obj:`~numpy.ndarray` of shape (n_subjects,)
        Subject values for the neighboring voxels.
    r_ay, r_by, r_cy : :obj:`float`
        Desired correlation between A and Y, B and Y, and C and Y.

    Returns
    -------
    y : :obj:`~numpy.ndarray` of shape (n_subjects,)
        Data for the voxel across subjects.
    """
    R = simulate_voxel_with_no_neighbors(A.size)
    r_ab = np.corrcoef((A, B))[1, 0]
    r_ac = np.corrcoef((A, C))[1, 0]
    r_bc = np.corrcoef((B, C))[1, 0]
    r_ar = np.corrcoef((A, R))[1, 0]
    r_br = np.corrcoef((B, R))[1, 0]
    r_cr = np.corrcoef((C, R))[1, 0]

    w_r_num1 = (
        1
        - (r_ab**2)
        - (r_ac**2)
        - (r_bc**2)
        + (2 * r_ab * r_ac * r_bc)
        - ((r_ay**2) * (1 - (r_bc**2)))
        - ((r_by**2) * (1 - (r_ac**2)))
        - ((r_cy**2) * (1 - (r_ab**2)))
    )
    w_r_num2 = (
        (2 * r_ay * r_by * (r_ab - (r_ac * r_bc)))
        + (2 * r_ay * r_cy * (r_ac - (r_ab * r_bc)))
        + (2 * r_by * r_cy * (r_bc - (r_ab * r_ac)))
    )
    w_r_num = w_r_num1 + w_r_num2

    w_r_den1 = (
        1
        - (r_ab**2)
        - (r_ac**2)
        - (r_bc**2)
        + (2 * r_ab * r_ac * r_bc)
        - ((r_ar**2) * (1 - (r_bc**2)))
        - ((r_br**2) * (1 - (r_ac**2)))
        - ((r_cr**2) * (1 - (r_ab**2)))
    )
    w_r_den2 = (
        (2 * r_ar * r_br * (r_ab - (r_ac * r_bc)))
        + (2 * r_ar * r_cr * (r_ac - (r_ab * r_bc)))
        + (2 * r_br * r_cr * (r_bc - (r_ab * r_ac)))
    )
    w_r_den = w_r_den1 + w_r_den2
    w_r = w_r_num / w_r_den

    w_c_num1 = (
        (r_cy * (1 - (r_ab**2)))
        - (r_ac * (r_ay - (r_ab * r_by)))
        - (r_bc * (r_by - (r_ab * r_ay)))
    )
    w_c_num2 = w_r * (
        (r_cr * (1 - (r_ab**2)))
        - (r_ac * (r_ar - (r_ab * r_br)))
        - (r_bc * (r_br - (r_ab * r_ar)))
    )
    w_c_num = w_c_num1 - w_c_num2
    w_c_den = 1 - (r_ab**2) - (r_ac**2) - (r_bc**2) + (2 * r_ab * r_ac * r_bc)
    w_c = w_c_num / w_c_den

    w_b = (
        (r_by - (r_ab * r_ay)) - (w_c * (r_bc - (r_ab * r_ac))) - (w_r * (r_br - (r_ab * r_ar)))
    ) / (1 - (r_ab**2))
    w_a = r_ay - (w_b * r_ab) - (w_c * r_ac) - (w_r * r_ar)

    y = (w_a * A) + (w_b * B) + (w_c * C) + (w_r * R)
    y = stats.zscore(y)  # NOTE: The zscore isn't part of the math. It should be z-scores by w*
    return y


def simulate_subject_maps(n_subjects, masker, correlation_maps):
    """Simulate the preliminary set of subject maps.

    Parameters
    ----------
    n_subjects : :obj:`list` of :obj:`int`
        Number of subjects in each study in the meta-analytic dataset.
        The list has one element for each study.
    masker : :obj:`nilearn.maskers.NiftiMasker`
        Masker object.
    correlation_maps : dict of 3 91x109x91 arrays
        right, posterior, and inferior

    Returns
    -------
    subject_maps : :obj:`~numpy.ndarray` of shape (sum(n_subjects), n_voxels)
        The base subject-wise statistical maps.

    Notes
    -----
    -   For simplicity, the imputation function in SDM-PSI is a generation of random normal numbers
        with their mean equal to the sample effect size of the voxel in the (unpermuted) study
        image and unit variance.
    -   Note that the sample effect size is the effect size of the study after removing
        (i.e., dividing by) the J Hedge correction factor.
    -   In the common preliminary set, the values of any voxel have null mean, and adjacent voxels
        show the expected correlations. For instance, if the correlation observed in humans
        between voxels A and B is 0.67, the correlation between these two voxels in the imputed
        subject images must be 0.67.
        SDM-PSI uses the correlation templates created for AES-SDM to know the correlation between
        every pair of two voxels (i.e., to take the irregular spatial covariance of the brain into
        account), but other approaches are possible.
    -   Note that as far as the imputation of the voxels follows a simple order and the software
        only accounts for correlations between voxels sharing a face, a voxel cannot have more than
        three neighbor voxels already imputed. For example, imagine that the imputation follows a
        left/posterior/inferior to right/anterior/superior direction. When the software imputes a
        given voxel, it will have already imputed the three neighbors in the left,
        behind and below, while it will impute later the three neighbors in the right,
        in front and above.
        The number of neighbor voxels imputed or to impute will be lower if some of them are
        outside the mask.
    -   For two-sample studies, SDM-PSI imputes subject values separately for each sample, and it
        only adds the effect size to the patient (or non-control) subject images.
    """
    mask_arr = masker.mask_img.get_fdata()
    n_x, n_y, n_z = masker.mask_img.shape
    xyz = np.vstack(np.where(mask_arr)).T
    # Sort xyz coordinates by X, then Y, then Z
    # From https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/
    xyz = xyz[xyz[:, 2].argsort()]  # sort by z
    xyz = xyz[xyz[:, 1].argsort(kind="mergesort")]  # sort by y
    xyz = xyz[xyz[:, 0].argsort(kind="mergesort")]  # sort by x

    subject_maps = np.empty((np.sum(n_subjects), xyz.shape[0]))
    subject_counter = 0

    # NOTE: We can use joblib here.
    for i_study, n_study_subjects in enumerate(tqdm(n_subjects, desc="Studies")):
        study_subject_maps = np.full((n_x, n_y, n_z, n_study_subjects), fill_value=np.nan)
        for coord in trange(xyz.shape[0], leave=False, desc="Voxel"):
            x, y, z = xyz[coord, :]

            if mask_arr[x - 1, y, z] == 1:
                use_right = True
            else:
                use_right = False

            if mask_arr[x, y - 1, z] == 1:
                use_posterior = True
            else:
                use_posterior = False

            if mask_arr[x, y, z - 1] == 1:
                use_inferior = True
            else:
                use_inferior = False

            n_directions = sum((use_right, use_posterior, use_inferior))
            if n_directions == 0:
                voxel_values = simulate_voxel_with_no_neighbors(n_study_subjects)

            elif n_directions == 1:
                if use_right:
                    A_data = study_subject_maps[x - 1, y, z, :]
                    r_ay = correlation_maps["right"][x, y, z]

                elif use_posterior:
                    A_data = study_subject_maps[x, y - 1, z, :]
                    r_ay = correlation_maps["posterior"][x, y, z]

                else:
                    A_data = study_subject_maps[x, y, z - 1, :]
                    r_ay = correlation_maps["inferior"][x, y, z]

                voxel_values = simulate_voxel_with_one_neighbor(A_data, r_ay)

            elif n_directions == 2:
                if use_right:
                    A_data = study_subject_maps[x - 1, y, z, :]
                    r_ay = correlation_maps["right"][x, y, z]

                    if use_posterior:
                        B_data = study_subject_maps[x, y - 1, z, :]
                        r_by = correlation_maps["posterior"][x, y, z]

                    else:
                        B_data = study_subject_maps[x, y, z - 1, :]
                        r_by = correlation_maps["inferior"][x, y, z]

                elif use_posterior:
                    A_data = study_subject_maps[x, y - 1, z, :]
                    r_ay = correlation_maps["posterior"][x, y, z]

                    B_data = study_subject_maps[x, y, z - 1, :]
                    r_by = correlation_maps["inferior"][x, y, z]

                voxel_values = simulate_voxel_with_two_neighbors(A_data, B_data, r_ay, r_by)

            else:
                A_data = study_subject_maps[x - 1, y, z, :]
                B_data = study_subject_maps[x, y - 1, z, :]
                C_data = study_subject_maps[x, y, z - 1, :]
                r_ay = correlation_maps["right"][x, y, z]
                r_by = correlation_maps["posterior"][x, y, z]
                r_cy = correlation_maps["inferior"][x, y, z]
                voxel_values = simulate_voxel_with_three_neighbors(
                    A_data,
                    B_data,
                    C_data,
                    r_ay,
                    r_by,
                    r_cy,
                )

            study_subject_maps[x, y, z, :] = voxel_values

        study_subject_img = nib.Nifti1Image(
            study_subject_maps, masker.mask_img.affine, masker.mask_img.header
        )
        study_subject_data = masker.transform(study_subject_img)
        subject_maps[subject_counter : subject_counter + n_study_subjects, :] = study_subject_data
        subject_counter += n_study_subjects

    return subject_maps


def impute_study_img(mle_coeff_img, mle_tau2_img, lower_bound_img, upper_bound_img, seed=0):
    """Impute study's image.

    .. todo::

        Implement.

    Use the MLE map and each study's upper- and lower-bound effect size maps to impute
    study-wise effect size and variance images that meet specific requirements.

    Parameters
    ----------
    mle_coeff_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic coefficients map.
    mle_tau2_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic variance map.
    lower_bound_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Study's lower-bound effect size map.
    upper_bound_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Study's upper-bound effect size map.
    seed : :obj:`int`, optional
        Random seed. Default is 0.

    Returns
    -------
    effect_size_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Study's effect size map.
    var_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Study's effect variance map.

    Notes
    -----
    -   This step, separately conducted for each study, consists in imputing many times study
        images that meet the general PSI conditions adapted to SDM:

            1.  the effect sizes imputed for a voxel must follow a truncated normal distribution
                with the MLE estimates and the effect-size bounds as parameters; and
            2.  the effect sizes of adjacent voxels must show positive correlations

    -   First, it assigns each voxel a uniformly distributed value between zero and one.
    -   Second, and separately for each voxel, it applies a threshold, spatial smoothing and
        scaling that ensures that the voxel has the expected value and variance of the truncated
        normal distribution and, simultaneously, has strong correlations with the neighboring
        voxels.
    -   To ensure that the voxel has the expected value of the truncated normal distribution,
        the threshold applied to the voxels laying within the smoothing kernel is the expected
        value of the truncated normal distribution scaled to 0-1, and the number (between 0 and 1)
        resulting from the smoothing is rescaled to the bounds of the truncated normal
        distribution.
        To ensure that the voxel has the expected variance of the truncated normal distribution,
        SDM-PSI selects an anisotropic smoothing kernel that follows the spatial covariance of the
        voxel and makes the variance of the resulting value in the voxel coincide with that
        variance of the truncated normal distribution.
        Please note that each voxel must follow a different truncated normal distribution,
        and thus this thresholding/smoothing/rescaling process is different for each voxel.
    """
    # Nonsense for now
    effect_size_img = lower_bound_img.copy()
    var_img = lower_bound_img.copy()
    return effect_size_img, var_img


def scale_subject_maps(study_effect_size_map, study_variance_map, prelim_subject_maps):
    """Scale the "preliminary" set of subject-level maps for each dataset for a given imputation.

    This is redundant for studies for which the original statistical image is available.

    Parameters
    ----------
    study_effect_size_map : :obj:`~numpy.ndarray` of shape (V,)
        Imputed study-level effect size map.
        V is voxel.
    study_variance_map : :obj:`~numpy.ndarray` of shape (V,)
    prelim_subject_maps : :obj:`~numpy.ndarray` of shape (N, V)
        Preliminary (unscaled) subject-level maps.
        An array that is subjects (N) by voxels (V).

    Returns
    -------
    scaled_subject_maps : :obj:`~numpy.ndarray` of shape (N, V)
        An array that is subjects (N) by voxels (V).

    Notes
    -----
    -   The mean across subject-level images, for each voxel, must equal the value from the
        study-level effect size map.
    -   Values for each voxel, across subjects, must correlate with the values for the same
        voxel at 1 in all other imputations.
    -   Values of adjacent voxels must show "realistic" correlations as well.
        SDM uses tissue-type masks for this.
    -   SDM simplifies the simulation process by creating a single "preliminary" set of
        subject-level maps for each dataset (across imputations), and scaling it across
        imputations.
    """
    scaled_subject_maps = prelim_subject_maps * study_variance_map[None, :]
    scaled_subject_maps += study_effect_size_map[None, :]

    return scaled_subject_maps


def calculate_hedges_maps(subject_effect_size_imgs):
    """Run study-wise group-level analyses and calculate study-level Hedges' g maps.

    .. todo::

        Simplify calculation based on paper's description.

    .. todo::

        Support multiple sample sizes.

    Parameters
    ----------
    subject_effect_size_imgs : S-length :obj:`list` of :obj:`~numpy.ndarray` of shape (N, V)
        Subject-level effect size data.

        -   S = studies
        -   N = study sample sizes
        -   V = voxels

    Returns
    -------
    out_g_arr : :obj:`~numpy.ndarray` of shape (S, V)
    out_g_var_arr : :obj:`~numpy.ndarray` of shape (S, V)

    Notes
    -----
    From :footcite:t:`albajes2019voxel`:

    -   In SDM-PSI, the group analysis is the estimation of Hedge-corrected effect sizes.
        In practice, this estimation simply consists of calculating the mean
        (or the difference of means in two-sample studies) and multiplying by J,
        given that imputed subject values have unit variance.
    """
    n_studies = len(subject_effect_size_imgs)
    n_voxels = subject_effect_size_imgs[0].shape[1]
    sample_sizes = [arr.shape[0] for arr in subject_effect_size_imgs]
    max_sample_size = np.max(sample_sizes)

    out_g_arr = np.empty((n_studies, n_voxels))
    out_g_var_arr = np.empty((n_studies, n_voxels))

    for i_voxel in range(n_voxels):
        effect_size_arr = np.full((n_studies, max_sample_size), np.nan)

        # Reorganize data into numpy.ndarray of shape (S, max(N))
        for j_study in range(n_studies):
            study_effect_size_arr = subject_effect_size_imgs[j_study][:, i_voxel]

            effect_size_arr[j_study, : sample_sizes[j_study]] = study_effect_size_arr

        g_arr = hedges_g(y=effect_size_arr, n_subjects1=sample_sizes)
        g_var_arr = hedges_g_var(g=g_arr, n_subjects1=sample_sizes)
        out_g_arr[:, i_voxel] = g_arr
        out_g_var_arr[:, i_voxel] = g_var_arr

    return out_g_arr, out_g_var_arr


def run_variance_meta(study_hedges_imgs, study_hedges_var_imgs, design):
    """Meta-analyze imputed effect size and variance maps.

    .. todo::

        Determine if DerSimonian-Laird is appropriate estimator and that it can handle g maps.

    .. todo::

        Implement linear contrast calculation.

    Parameters
    ----------
    study_hedges_imgs : :obj:`~numpy.ndarray` of shape (S, V)
        Study-wise Hedges g maps.

        -   S = studies
        -   V = voxels
    study_hedges_var_imgs : :obj:`~numpy.ndarray` of shape (S, V)
        Study-wise Hedges g variance maps.
    design : :obj:`~numpy.ndarray` of shape (S, X)
        Study-level predictors.

    Returns
    -------
    meta_effect_size_img : :obj:`~numpy.ndarray` of shape (V,)
        Meta-analytic effect size map.
    meta_tau2_img : :obj:`~numpy.ndarray` of shape (V,)
        Meta-analytic variance map.
    cochrans_q_img : :obj:`~numpy.ndarray` of shape (V,)
        Cochran's Q map.
    h2_img : :obj:`~numpy.ndarray` of shape (V,)
        H^2 map.
    i2_img : :obj:`~numpy.ndarray` of shape (V,)
        I^2 map.

    Notes
    -----
    From :footcite:t:`albajes2019voxel`:

    -   The meta-analysis consists of the fitting of a standard random-effects model.
        The design matrix includes any covariate used in the MLE step, and the weight of a study is
        the inverse of the sum of its variance and the between-study heterogeneity τ2,
        which in SDM-PSI may be estimated using either the DerSimonian-Laird or the slightly more
        accurate restricted-maximum likelihood (REML) method.
    -   After fitting the model, SDM conducts a standard linear hypothesis contrast and derives
        standard heterogeneity statistics H2, I2 and [Cochran's] Q.
    """
    n_studies, n_voxels = study_hedges_imgs.shape

    est = pymare.estimators.DerSimonianLaird()
    pymare_dset = pymare.Dataset(y=study_hedges_imgs, v=study_hedges_var_imgs, X=design)
    est.fit_dataset(pymare_dset)
    est_summary = est.summary()
    fe_stats = est_summary.get_fe_stats()
    hg_stats = est_summary.get_heterogeneity_stats()
    meta_tau2_img = est_summary.tau2.squeeze()
    meta_effect_size_img = fe_stats["est"].squeeze()
    cochrans_q_img = hg_stats["Q"].squeeze()
    i2_img = hg_stats["I^2"].squeeze()

    # NOTE: Should the linear hypothesis contrast per conducted here?

    return meta_effect_size_img, meta_tau2_img, cochrans_q_img, i2_img


def combine_imputation_results(coefficient_maps, variance_maps, cochrans_q_maps, i2_maps):
    """Use Rubin's rules to combine meta-analysis results across imputations.

    .. todo::

        Implement.

    .. todo::

        Extend to support multiple regressors. This will complicate Rubin's rules.

    Parameters
    ----------
    coefficient_maps : :obj:`~numpy.ndarray` of shape (I, V)
        Imputation-wise coefficient maps.

        -   I = imputations
        -   V = Voxels
    variance_maps : :obj:`~numpy.ndarray` of shape (I, V)
        Imputation-wise variance maps.
    cochrans_q_maps : :obj:`~numpy.ndarray` of shape (I, V)
        Imputation-wise Cochran's Q maps.
    i2_maps : :obj:`~numpy.ndarray` of shape (I, V)
        Imputation-wise I^2 maps.

    Returns
    -------
    meta_d_img : :obj:`~numpy.ndarray` of shape (V,)
    meta_var_img : :obj:`~numpy.ndarray` of shape (V,)
    meta_z_img : :obj:`~numpy.ndarray` of shape (V,)

    Notes
    -----
    This function uses Cochran's Q :footcite:p:`cochran1954combination`,
    I^2 :footcite:p:`higgins2002quantifying`, and H^2 :footcite:p:`higgins2002quantifying`.

    Clues from https://stats.stackexchange.com/a/476849 and :footcite:t:`marshall2009combining`.

    From :footcite:t:`albajes2019voxel`:

    -   Finally, SDM-PSI uses Rubin's rules to combine the coefficients of the model,
        their covariance and the heterogeneity statistics I and [Cochran's] Q of the different
        imputed datasets.
    -   Note that Q follows a χ2 distribution, but its combined statistic follows an F
        distribution.
        For convenience, SDM-PSI converts FQ back into a Q (i.e. converts an F statistic to a χ2
        statistic with the same p-value). It also derives H-combined from I-combined.

    References
    ----------
    .. footbibliography::
    """
    n_imputations = coefficient_maps.shape[0]  # m
    point_estimates = np.mean(coefficient_maps, axis=0)  # Q
    within_imputation_var = np.mean(variance_maps, axis=0)  # U
    sum_of_squared_diffs = (coefficient_maps - point_estimates[None, :]) ** 2
    between_imputation_var = sum_of_squared_diffs / (n_imputations - 1)  # B

    total_variance = within_imputation_var + ((1 + (1 / n_imputations)) * between_imputation_var)

    # Calculate covariance (only for multiple regressors?)

    # Combine Cochran's Q

    # Convert FQ back into Q (X2 statistic)

    # Combine I^2 into I-combined

    # Derive H-combined from I-combined

    # Delete the variables for linting purposes
    del total_variance

    # NOTE: Currently nonsense
    meta_d_img = point_estimates.copy()
    meta_var_img = point_estimates.copy()
    meta_z_img = point_estimates.copy()

    return meta_d_img, meta_var_img, meta_z_img


def permute_assignments(subject_imgs, design_type="one", seed=0):
    """Permute subject-level maps.

    I decided to permute the maps directly, rather than the design matrix.
    I'm not sure if that's the best approach though.

    Parameters
    ----------
    subject_imgs : numpy.ndarray of shape (N, V)
        N = study's sample size
        V = voxels
    design_type : {"one", "two", "correlation"}, optional
        "one" refers to a one-sample test, in which the data's signs are randomly flipped.
        "two" refers to a two-sample test, in which subjects' data are randomly shuffled.
        "correlation" refers to a correlation meta-analysis, in which subjects' data are randomly
        shuffled.
        Default is "one".
    seed : :obj:`int`, optional
        Random seed. Default is 0.

    Returns
    -------
    permuted_subject_imgs : numpy.ndarray of shape (N, V)

    Notes
    -----
    From :footcite:t:`albajes2019voxel`:

    -   PSI methods must randomly assign "1" or "-1" to each subject of a one-sample study,
        or randomly reassign each of the subjects of a two-sample study to one of the two groups...
    -   PSI methods must also swap subjects in a correlation meta-analysis.
    -   The assignments must be the same across imputations.
    """
    permuted_subject_imgs = subject_imgs.copy()
    gen = np.random.default_rng(seed=seed)
    n_subjects = permuted_subject_imgs.shape[0]

    if design_type == "one":
        # Randomly sign-flip ~50% of the subjects.
        code = gen.randint(0, 2, size=n_subjects).astype(bool)
        permuted_subject_imgs[code] *= -1

    elif design_type in ("two", "correlation"):
        # Shuffle rows randomly.
        # Assumes that group assignment is based on row index and occurs outside this function.
        id_idx = np.arange(n_subjects)
        gen.shuffle(id_idx)
        permuted_subject_imgs = permuted_subject_imgs[id_idx, :]

    return permuted_subject_imgs


def run_sdm(
    coords,
    masker,
    correlation_maps,
    design=None,
    threshold=0.001,
    n_imputations=50,
    n_iters=1000,
):
    """Run the SDM-PSI algorithm.

    The algorithm is implemented as described in :footcite:t:`albajes2019meta,albajes2019voxel`.

    Parameters
    ----------
    coords
        Coordinates.
    masker
    correlation_maps
    design : None or :obj:`~numpy.ndarray` of shape (S, X)
        Study-level predictors.
        Default is None.
    threshold : :obj:`float`, optional
            Cluster-defining p-value threshold. Default is 0.001.
    n_imputations : int
        Number of imputations. Default is 50, based on the SDM software.
    n_iters : int
        Number of iterations for the Monte Carlo FWE correction procedure.
        Default is 1000, based on the SDM software.

    Returns
    -------
    meta_d_img
        Meta-analytic effect-size map.
    meta_var_img
        Meta-analytic variance map.
    meta_z_img
        Meta-analytic z-statistic map.
    max_stats : :obj:`dict`
        Dictionary of maximum statistics from Monte Carlo permutations.

    Notes
    -----
    1.  Use anisotropic Gaussian kernels, plus effect size estimates and metadata,
        to produce lower-bound and upper-bound effect size maps from the coordinates.
        See :func:`~nimare.meta.utils.compute_sdm_ma`.

        -   We need generic inter-voxel correlation maps for this.
            NOTE: Correlation maps are unidirectional. There are 26 directions for each voxel,
            but the flipped versions (e.g., right and left) use equivalent maps so there are only
            13 correlation maps.
        -   We also need a fast implementation of Dijkstra's algorithm to estimate the shortest
            path (i.e., "virtual distance") between two voxels based on the map of correlations
            between each voxel and its neighbors. I think ``dijkstra3d`` might be useful here.

    2.  Use maximum likelihood estimation to estimate the most likely effect size and variance
        maps across studies, from the lower- and upper-bound maps. See: :func:`mle_estimation`.

    3.  Use the MLE map and each study's upper- and lower-bound effect size maps to impute
        study-wise effect size and variance images that meet specific requirements.
        See :func:`impute_study_img`.
    4.  For each imputed pair of effect size and variance images, simulate subject-level images.
        See :func:`simulate_subject_maps` and :func:`scale_subject_maps`.

        -   The mean across subject-level images, for each voxel, must equal the value from the
            study-level effect size map.
        -   Values for each voxel, across subjects, must correlate with the values for the same
            voxel at 1 in all other imputations.
        -   Values of adjacent voxels must show "realistic" correlations as well.
            SDM uses tissue-type masks for this.
        -   SDM simplifies the simulation process by creating a single "preliminary" set of
            subject-level maps for each dataset (:func:`simulate_subject_maps`),
            and scaling it to each imputation (:func:`scale_subject_maps`).

    5.  Subject-based permutation test on all of the pre-generated imputations.
        See :func:`permute_assignments`.

        -   Create one random permutation of the subjects and apply it to the subject images of
            the different imputed datasets.
        -   For one-sample tests, randomly assign "1" or "-1" to each subject.
        -   For two-sample tests, randomly reassign each of the subjects.
        -   For correlation meta-analyses, randomly swap subjects.

    6.  Separately for each imputed dataset, conduct a group analysis of the permuted subject
        images to obtain one study image per study, and then conduct a meta-analysis of the
        study images to obtain one meta-analysis image.
        See :func:`calculate_hedges_maps`.

        -   "In SDM-PSI, the group analysis is the estimation of Hedge-corrected effect sizes.
            In practice, this estimation simply consists of calculating the mean
            (or the difference of means in two-sample studies) and multiplying by J,
            given that imputed subject values have unit variance."

    7.  Perform meta-analysis across study-level effect size maps using random effects model.
        Performed separately for each imputation.
        See :func:`run_variance_meta`.

        -   "The meta-analysis consists of the fitting of a standard random-effects model.
            The design matrix includes any covariate used in the MLE step,
            and the weight of a study is the inverse of the sum of its variance and the
            between-study heterogeneity τ2, which in SDM-PSI may be estimated using either the
            DerSimonian-Laird or the slightly more accurate restricted-maximum likelihood (REML)
            method. After fitting the model, SDM conducts a standard linear hypothesis contrast
            and derives standard heterogeneity statistics H2, I2 and Q."
        -   One of NiMARE's IBMA interfaces should be able to handle the meta-analysis part.
            Either DerSimonianLaird or VarianceBasedLikelihood.
            We'll need to add heterogeneity statistic calculation either to PyMARE or NiMARE.

    8.  Compute imputation-wise heterogeneity statistics. See :func:`run_variance_meta`.
    9.  Use "Rubin's rules" to combine heterogeneity statistics, coefficients, and variance for
        each imputed dataset. This should result in a combined meta-analysis image.
        See :func:`combine_imputation_results`.
    10. Save a maximum statistic from the combined meta-analysis image (e.g., the largest z-value).
        This is the last step within the iteration loop.
        See ``_calculate_cluster_measures()``.
    11. Perform Monte Carlo-like maximum statistic procedure to get null distributions for vFWE or
        cFWE. Or do TFCE.

        -   "After enough iterations of steps a) to d), use the distribution of the maximum
            statistic to threshold the combined meta-analysis image obtained from unpermuted data."
        -   This approach is used for the main analysis (i.e., the mean), but for meta-regressions
            we need to use the Freedman-Lane permutation method on the study-level maps.

    References
    ----------
    .. footbibliography::
    """
    # Extract sample size information from the coordinates DataFrame.
    n_subjects = coords.groupby("id")["sample_sizes"].values
    n_studies = len(n_subjects)

    if design is None:
        design = np.ones((n_studies, 1))
        design_type = "one"
    elif np.std(design[:, 0]) == 0:
        design_type = "one"
    else:
        design_type = "two"

    # Step 1: Estimate lower- and upper-bound effect size maps from coordinates.
    lower_bound_imgs, upper_bound_imgs = compute_sdm_ma(coords)

    # Step 2: Estimate the most likely effect size and variance maps across studies.
    mle_coeff_img, mle_tau2_img = mle_estimation(lower_bound_imgs, upper_bound_imgs)

    # Step 4a: Create base set of simulated subject maps.
    raw_subject_effect_size_imgs = simulate_subject_maps(n_subjects, masker, correlation_maps)

    all_subject_effect_size_imgs = []
    imp_meta_effect_size_imgs, imp_meta_tau2_imgs = [], []
    imp_cochrans_q_imgs, imp_i2_imgs = [], []
    for i_imp in range(n_imputations):

        imp_subject_effect_size_imgs = []
        for j_study in range(n_studies):
            # Step 3: Impute study-wise effect size and variance maps.
            imp_study_effect_size_img, imp_study_var_img = impute_study_img(
                mle_coeff_img,
                mle_tau2_img,
                lower_bound_imgs[j_study],
                upper_bound_imgs[j_study],
                seed=i_imp,
            )

            # Step 4: Simulate subject-wise effect size(?) maps.
            study_imp_subject_effect_size_imgs = scale_subject_maps(
                imp_study_effect_size_img,
                imp_study_var_img,
                raw_subject_effect_size_imgs[j_study],
            )
            imp_subject_effect_size_imgs.append(study_imp_subject_effect_size_imgs)

        # Step ???: Create imputed study-wise effect size and variance maps from subject maps.
        imp_hedges_imgs, imp_hedges_var_imgs = calculate_hedges_maps(
            imp_subject_effect_size_imgs,
        )

        # Step ???: Estimate imputation-wise meta-analytic maps.
        (
            imp_meta_effect_size_img,
            imp_meta_tau2_img,
            imp_cochrans_q_img,
            imp_i2_img,
        ) = run_variance_meta(
            imp_hedges_imgs,
            imp_hedges_var_imgs,
            design=design,
        )
        imp_meta_effect_size_imgs.append(imp_meta_effect_size_img)
        imp_meta_tau2_imgs.append(imp_meta_tau2_img)
        imp_cochrans_q_imgs.append(imp_cochrans_q_img)
        imp_i2_imgs.append(imp_i2_img)

    # Step ???: Combine meta-analytic maps across imputations with Rubin's rules.
    # These should be the meta-analysis maps we care about.
    meta_d_img, meta_var_img, meta_z_img = combine_imputation_results(
        imp_meta_effect_size_imgs,
        imp_meta_tau2_imgs,
        imp_cochrans_q_imgs,
        imp_i2_imgs,
    )

    # Step 5: Permutations...
    max_stats = {
        "z": [],
        "mass": [],
        "size": [],
    }
    # Define connectivity matrix for cluster labeling
    conn = ndimage.generate_binary_structure(3, 2)

    seed_counter = 0  # Each permutation/study's random seed must be the same for all imputations
    for i_iter in range(n_iters):
        # Step 6: Calculate study-level Hedges-corrected effect size maps.
        perm_meta_effect_size_imgs, perm_meta_tau2_imgs = [], []
        perm_cochrans_q_imgs, perm_i2_imgs = [], []
        for j_imp in range(n_imputations):
            # Permute subject maps
            imp_subject_imgs = all_subject_effect_size_imgs[j_imp]

            perm_imp_subject_imgs = []
            seed_counter_2 = seed_counter
            for k_study in range(n_studies):
                study_imp_subject_imgs = imp_subject_imgs[k_study]
                perm_study_imp_subject_imgs = permute_assignments(
                    study_imp_subject_imgs,
                    seed=seed_counter_2,
                    design_type=design_type,
                )
                perm_imp_subject_imgs.append(perm_study_imp_subject_imgs)
                seed_counter_2 += 1

            perm_imp_hedges_imgs, perm_imp_hedges_var_imgs = calculate_hedges_maps(
                perm_imp_subject_imgs,
            )

            # Step 7: Meta-analyze imputed effect size maps.
            (
                perm_meta_effect_size_img,
                perm_meta_tau2_img,
                perm_cochrans_q_img,
                perm_i2_img,
            ) = run_variance_meta(
                perm_imp_hedges_imgs,
                perm_imp_hedges_var_imgs,
                design=design,
            )
            perm_meta_effect_size_imgs.append(perm_meta_effect_size_img)
            perm_meta_tau2_imgs.append(perm_meta_tau2_img)
            perm_cochrans_q_imgs.append(perm_cochrans_q_img)
            perm_i2_imgs.append(perm_i2_img)

        # Step 8: Heterogeneity statistics and combine with Rubin's rules.
        perm_meta_d_img, perm_meta_var_img, perm_meta_z_img = combine_imputation_results(
            perm_meta_effect_size_imgs,
            perm_meta_tau2_imgs,
            perm_cochrans_q_imgs,
            perm_i2_imgs,
        )

        # Log maximum statistics for multiple comparisons correction later
        # NOTE: Probably should use two-sided test.
        perm_max_size, perm_max_mass = _calculate_cluster_measures(
            perm_meta_z_img,
            threshold,
            conn,
            tail="upper",
        )
        max_stats["z"].append(np.maximum(perm_meta_z_img))
        max_stats["size"].append(perm_max_size)
        max_stats["mass"].append(perm_max_mass)

        seed_counter += n_studies

    return meta_d_img, meta_var_img, meta_z_img, max_stats
