"""Seed-based d-mapping-related methods."""
import nibabel as nib
import numpy as np
from scipy import stats

# from tqdm import tqdm, trange
from tqdm.autonotebook import tqdm, trange

from nimare.meta.utils import compute_sdm_ma
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
    meta_effect_size_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic effect-size map.
    meta_tau_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic tau (variance) map.
    """
    meta_effect_size_img, meta_tau_img = lower_bound_imgs, upper_bound_imgs
    return meta_effect_size_img, meta_tau_img


def impute_studywise_imgs(
    meta_effect_size_img,
    meta_tau_img,
    lower_bound_imgs,
    upper_bound_imgs,
    seed=0,
):
    """Impute study-wise images.

    .. todo::

        Implement.

    Parameters
    ----------
    meta_effect_size_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic effect size map.
    meta_tau_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Meta-analytic variance map.
    lower_bound_imgs : S-length :obj:`list` of :obj:`~nibabel.nifti1.Nifti1Image`
        Study-wise lower-bound effect size maps.
    upper_bound_imgs : S-length :obj:`list` of :obj:`~nibabel.nifti1.Nifti1Image`
        Study-wise upper-bound effect size maps.
    seed : :obj:`int`, optional
        Random seed. Default is 0.

    Returns
    -------
    study_effect_size_imgs : S-length :obj:`list` of :obj:`~nibabel.nifti1.Nifti1Image`
        Study-wise effect size maps.
    study_var_imgs : S-length :obj:`list` of :obj:`~nibabel.nifti1.Nifti1Image`
        Study-wise effect variance maps.
    """
    # Nonsense for now
    study_effect_size_imgs = lower_bound_imgs[:]
    study_var_imgs = lower_bound_imgs[:]
    return study_effect_size_imgs, study_var_imgs


def run_permutations(all_subject_effect_size_imgs, all_subject_var_imgs, seed=0):
    """Run permutations.

    .. todo::

        Implement.

    Parameters
    ----------
    all_subject_effect_size_imgs : \
            I-length :obj:`list` of S-length lists of numpy.ndarray of shape (N, V)
        I = imputations
        S = studies
        N = study sample sizes
        V = voxels
    all_subject_var_imgs : I-length :obj:`list` of S-length lists of numpy.ndarray of shape (N, V)
        I = imputations
        S = studies
        N = study sample sizes
        V = voxels

    Returns
    -------
    permuted_subject_effect_size_imgs : \
            I-length :obj:`list` of S-length lists of numpy.ndarray of shape (N, V)
    permuted_subject_var_imgs : \
            I-length :obj:`list` of S-length lists of numpy.ndarray of shape (N, V)

    Notes
    -----
    "PSI methods must randomly assign "1" or "-1" to each subject of a one-sample study,
    or randomly reassign each of the subjects of a two-sample study to one of the two groups...
    PSI methods must also swap subjects in a correlation meta-analysis."
    The assignments must be the same across imputations.
    """
    permuted_subject_effect_size_imgs = all_subject_effect_size_imgs[:]
    permuted_subject_var_imgs = all_subject_var_imgs[:]
    return permuted_subject_effect_size_imgs, permuted_subject_var_imgs


def extract_max_statistics(effect_size_img):
    """Extract maximum statistics from permuted data for Monte Carlo FWE correction."""
    ...


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


def scale_subject_maps(
    studylevel_effect_size_maps,
    studylevel_variance_maps,
    prelim_subjectlevel_maps,
):
    """Scale the "preliminary" set of subject-level maps for each dataset for a given imputation.

    Parameters
    ----------
    studylevel_effect_size_maps : :obj:`~numpy.ndarray` of shape (S, V)
        S is study, V is voxel.
    studylevel_variance_maps : :obj:`~numpy.ndarray` of shape (S, V)
    prelim_subjectlevel_maps : S-length :obj:`list` of :obj:`~numpy.ndarray` of shape (N, V)
        List with one entry per study (S), where each entry is an array that is subjects (N) by
        voxels (V).

    Returns
    -------
    scaled_subjectlevel_maps : S-length :obj:`list` of :obj:`~numpy.ndarray` of shape (N, V)
        List with one entry per study (S), where each entry is an array that is subjects (N) by
        voxels (V).

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
    assert len(prelim_subjectlevel_maps) == studylevel_effect_size_maps.shape[0]

    scaled_subjectlevel_maps = []
    for i_study in range(studylevel_effect_size_maps.shape[0]):
        study_mean = studylevel_effect_size_maps[i_study, :]
        study_var = studylevel_variance_maps[i_study, :]

        study_subject_level_maps = prelim_subjectlevel_maps[i_study]
        study_subject_level_maps_scaled = (study_subject_level_maps * study_var) + study_mean
        scaled_subjectlevel_maps.append(study_subject_level_maps_scaled)

    return scaled_subjectlevel_maps


def calculate_hedges_maps(subject_effect_size_imgs, subject_var_imgs):
    """Calculate study-level Hedges' g maps.

    .. todo::

        Support multiple sample sizes.

    Parameters
    ----------
    subject_effect_size_imgs : S-length :obj:`list` of :obj:`~numpy.ndarray` of shape (N, V)
        Subject-level effect size data.

        -   S = studies
        -   N = study sample sizes
        -   V = voxels
    subject_var_imgs : S-length :obj:`list` of :obj:`~numpy.ndarray` of shape (N, V)
        Subject-level effect variance data.

    Returns
    -------
    out_g_arr : :obj:`~numpy.ndarray` of shape (S, V)
    out_g_var_arr : :obj:`~numpy.ndarray` of shape (S, V)
    """
    n_studies = len(subject_effect_size_imgs)
    n_voxels = subject_effect_size_imgs[0].shape[1]
    sample_sizes = [arr.shape[0] for arr in subject_effect_size_imgs]
    max_sample_size = np.max(sample_sizes)

    out_g_arr = np.empty((n_studies, n_voxels))
    out_g_var_arr = np.empty((n_studies, n_voxels))

    for i_voxel in range(n_voxels):
        effect_size_arr = np.full((n_studies, max_sample_size), np.nan)
        var_arr = np.full((n_studies, max_sample_size), np.nan)

        # Reorganize data into numpy.ndarray of shape (S, max(N))
        for j_study in range(n_studies):
            study_effect_size_arr = subject_effect_size_imgs[j_study][:, i_voxel]
            study_var_arr = subject_var_imgs[j_study][:, i_voxel]

            effect_size_arr[j_study, : sample_sizes[j_study]] = study_effect_size_arr
            var_arr[j_study, : sample_sizes[j_study]] = study_var_arr

        g_arr = hedges_g(y=effect_size_arr, n_subjects1=sample_sizes)
        g_var_arr = hedges_g_var(g=g_arr, n_subjects1=sample_sizes)
        out_g_arr[:, i_voxel] = g_arr
        out_g_var_arr[:, i_voxel] = g_var_arr

    return out_g_arr, out_g_var_arr


def run_variance_meta(study_hedges_imgs, study_hedges_var_imgs):
    """Meta-analyze imputed effect size maps.

    .. todo::

        Implement.

    Parameters
    ----------
    study_hedges_imgs : :obj:`~numpy.ndarray` of shape (S, V)
        Study-wise Hedges g maps.

        -   I = imputations
        -   S = studies
        -   V = voxels
    study_hedges_var_imgs : :obj:`~numpy.ndarray` of shape (S, V)
        Study-wise Hedges g variance maps.

    Returns
    -------
    meta_effect_size_img : :obj:`~numpy.ndarray` of shape (V,)
        Meta-analytic effect size map.
    """
    meta_effect_size_img = study_hedges_imgs * study_hedges_var_imgs
    return meta_effect_size_img


def combine_imputation_results(coefficient_maps, covariance_maps, i_stats, q_stats):
    """Use Rubin's rules to combine meta-analysis results across imputations.

    .. todo::

        Implement.

    Parameters
    ----------
    coefficient_maps
    covariance_maps
    i_stats
    q_stats

    Returns
    -------
    perm_meta_effect_size_img

    Notes
    -----
    "Finally, SDM-PSI uses Rubin's rules to combine the coefficients of the model,
    their covariance and the heterogeneity statistics I and Q of the different imputed datasets.
    Note that Q follows a χ2 distribution, but its combined statistic follows an F distribution.
    For convenience, SDM-PSI converts FQ back into a Q (i.e. converts an F statistic to a χ2
    statistic with the same p-value). It also derives H-combined from I-combined."

    Clues from https://stats.stackexchange.com/a/476849.
    """
    point_estimates = np.mean(coefficient_maps, axis=0)
    mean_within_imputation_var = np.mean(covariance_maps, axis=0)
    # The B term is the sum of squared differences
    b = (1 / (coefficient_maps.shape[0] - 1)) * 5

    # Delete the variables for linting purposes
    del point_estimates, mean_within_imputation_var, b


def run_sdm(coords, masker, correlation_maps, n_imputations=50, n_iters=1000):
    """Run the SDM-PSI algorithm.

    The algorithm is implemented as described in :footcite:t:`albajes2019meta,albajes2019voxel`.

    Parameters
    ----------
    coords
        Coordinates.
    masker
    correlation_maps
    n_imputations : int
        Number of imputations. Default is 50, based on the SDM software.
    n_iters : int
        Number of iterations for the Monte Carlo FWE correction procedure.
        Default is 1000, based on the SDM software.

    Returns
    -------
    meta_effect_size_img
        Meta-analytic effect size map.
    meta_tau_img
        Meta-analytic variance map.
    max_stats : :obj:`dict`
        Dictionary of maximum statistics from Monte Carlo permutations.

    Notes
    -----
    1.  Use anisotropic Gaussian kernels, plus effect size estimates and metadata,
        to produce lower-bound and upper-bound effect size maps from the coordinates.

        -   We need generic inter-voxel correlation maps for this.
            NOTE: Correlation maps are unidirectional. There are 26 directions for each voxel,
            but the flipped versions (e.g., right and left) use equivalent maps so there are only
            13 correlation maps.
        -   We also need a fast implementation of Dijkstra's algorithm to estimate the shortest
            path (i.e., "virtual distance") between two voxels based on the map of correlations
            between each voxel and its neighbors. I think ``dijkstra3d`` might be useful here.

    2.  Use maximum likelihood estimation to estimate the most likely effect size and variance
        maps across studies (i.e., a meta-analytic map).

        -   Can we use NiMARE IBMAs for this?

    3.  Use the MLE map and each study's upper- and lower-bound effect size maps to impute
        study-wise effect size and variance images that meet specific requirements.
    4.  For each imputed pair of effect size and variance images, simulate subject-level images.

        -   The mean across subject-level images, for each voxel, must equal the value from the
            study-level effect size map.
        -   Values for each voxel, across subjects, must correlate with the values for the same
            voxel at 1 in all other imputations.
        -   Values of adjacent voxels must show "realistic" correlations as well.
            SDM uses tissue-type masks for this.
        -   SDM simplifies the simulation process by creating a single "preliminary" set of
            subject-level maps for each dataset (across imputations), and scaling it across
            imputations.

    5.  Subject-based permutation test on all of the pre-generated imputations.

        -   Create one random permutation of the subjects and apply it to the subject images of
            the different imputed datasets.
        -   For one-sample tests, randomly assign "1" or "-1" to each subject.
        -   For two-sample tests, randomly reassign each of the subjects.
        -   For correlation meta-analyses, randomly swap subjects.

    6.  Separately for each imputed dataset, conduct a group analysis of the permuted subject
        images to obtain one study image per study, and then conduct a meta-analysis of the
        study images to obtain one meta-analysis image.

        -   "In SDM-PSI, the group analysis is the estimation of Hedge-corrected effect sizes.
            In practice, this estimation simply consists of calculating the mean
            (or the difference of means in two-sample studies) and multiplying by J,
            given that imputed subject values have unit variance."

    7.  Perform meta-analysis across study-level effect size maps using random effects model.
        Performed separately for each imputation.

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

    8.  Compute imputation-wise heterogeneity statistics.
    9.  Use "Rubin's rules" to combine heterogeneity statistics, coefficients, and variance for
        each imputed dataset. This should result in a combined meta-analysis image.
    10. Save a maximum statistic from the combined meta-analysis image (e.g., the largest z-value).
        This is the last step within the iteration loop.
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

    # Step 1: Estimate lower- and upper-bound effect size maps from coordinates.
    lower_bound_imgs, upper_bound_imgs = compute_sdm_ma(coords)

    # Step 2: Estimate the most likely effect size and variance maps across studies.
    # This should be the meta-analysis map we care about.
    meta_effect_size_img, meta_tau_img = mle_estimation(lower_bound_imgs, upper_bound_imgs)

    # Step 4a: Create base set of simulated subject maps.
    raw_subject_effect_size_imgs = simulate_subject_maps(n_subjects, masker, correlation_maps)

    all_subject_effect_size_imgs, all_subject_var_imgs = [], []
    for i_imp in range(n_imputations):
        # Step 3: Impute study-wise effect size and variance maps.
        study_effect_size_imgs, study_var_imgs = impute_studywise_imgs(
            meta_effect_size_img,
            meta_tau_img,
            lower_bound_imgs,
            upper_bound_imgs,
            seed=i_imp,
        )

        # Step 4: Simulate subject-wise effect size and variance maps.
        # NOTE: The function below only returns scaled subject-level maps,
        # without associated variance maps.
        imp_subject_effect_size_imgs, imp_subject_var_imgs = scale_subject_maps(
            study_effect_size_imgs,
            study_var_imgs,
            raw_subject_effect_size_imgs,
        )
        # This is just a stand-in.
        all_subject_effect_size_imgs.append(imp_subject_effect_size_imgs)
        all_subject_var_imgs.append(imp_subject_var_imgs)

    # Step 5: Permutations...
    max_stats = {}
    for i_iter in range(n_iters):
        permuted_subject_effect_size_imgs, permuted_subject_var_imgs = run_permutations(
            all_subject_effect_size_imgs,
            all_subject_var_imgs,
            seed=i_iter,
        )

        # Step 6: Calculate study-level Hedges-corrected effect size maps.
        perm_meta_effect_size_imgs = []
        for j_imp in range(n_imputations):
            imp_hedges_imgs, imp_hedges_var_imgs = calculate_hedges_maps(
                permuted_subject_effect_size_imgs[j_imp],
                permuted_subject_var_imgs[j_imp],
            )

            # Step 7: Meta-analyze imputed effect size maps.
            perm_meta_effect_size_img = run_variance_meta(imp_hedges_imgs, imp_hedges_var_imgs)
            perm_meta_effect_size_imgs.append(perm_meta_effect_size_img)

        # Step 8: Heterogeneity statistics and combine with Rubin's rules.
        perm_meta_effect_size_img = combine_imputation_results(perm_meta_effect_size_imgs)

        # Max statistic
        perm_max_stats = extract_max_statistics(perm_meta_effect_size_img)
        max_stats.update(perm_max_stats)

    return meta_effect_size_img, meta_tau_img, max_stats
