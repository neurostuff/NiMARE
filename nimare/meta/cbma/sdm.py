"""Seed-based d-mapping-related methods."""
from datetime import datetime

import dijkstra3d
import nibabel as nib
import numpy as np
from nilearn import masking
from pymare import Dataset, estimators
from scipy import spatial, stats

# from tqdm import tqdm, trange
from tqdm.autonotebook import tqdm, trange


def _mle_estimation():
    ...


def _impute_studywise_imgs():
    ...


def _run_permutations():
    """Run permutations.

    Notes
    -----
    "PSI methods must randomly assign "1" or "-1" to each subject of a one-sample study,
    or randomly reassign each of the subjects of a two-sample study to one of the two groups...
    PSI methods must also swap subjects in a correlation meta-analysis."
    The assignments must be the same across imputations.
    """
    ...


def _extract_max_statistics():
    """Extract maximum statistics from permuted data for Monte Carlo FWE correction."""
    ...


def _simulate_voxel_with_no_neighbors(n_subjects):
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


def _simulate_voxel_with_one_neighbor(A, r_ay):
    """Simulate the data for a voxel with one neighbor that has already been simulated.

    Parameters
    ----------
    A : numpy.ndarray of shape (n_subjects,)
        Subject values for the neighboring voxel.
    r_ay : float
        Desired correlation between A and Y.

    Returns
    -------
    y : numpy.ndarray of shape (n_subjects,)
        Data for the voxel across subjects.

    Notes
    -----
    For imputing subject values in a voxel that has one neighboring voxel already imputed,
    SDM-PSI conducts a weighted average of the subject values of the neighboring voxel (A) and
    new standardized random normal values.
    -   Y = (w_a * A) + (w_r * R), where w are the weights that ensure that the resulting
        subject values have unit variance and the desired correlation.
    -   w_a = r_ay - (w_r * r_ar * w_r) = np.sqrt((1 - (r_ay ** 2)) / (1 - (r_ar ** 2)))
    """
    # Random normal values with null mean and unit variance.
    R = _simulate_voxel_with_no_neighbors(A.size)
    # Correlation between A and R.
    r_ar = np.corrcoef((A, R))[1, 0]

    w_r = np.sqrt((1 - (r_ay**2)) / (1 - (r_ar**2)))
    w_a = r_ay - (w_r * r_ar)

    y = (w_a * A) + (w_r * R)
    y = stats.zscore(y)  # NOTE: The zscore isn't part of the math. It should be z-scores by w*
    return y


def _simulate_voxel_with_two_neighbors(A, B, r_ay, r_by):
    """Simulate the data for a voxel with two neighbors that have already been simulated.

    Parameters
    ----------
    A, B : numpy.ndarray of shape (n_subjects,)
        Subject values for the neighboring voxels.
    r_ay, r_by : float
        Desired correlation between A and Y, and between B and Y.

    Returns
    -------
    y : numpy.ndarray of shape (n_subjects,)
        Data for the voxel across subjects.

    Notes
    -----
    For imputing subject values in a voxel that has two neighboring voxels already imputed,
    SDM-PSI conducts again a weighted average of the subject values of the neighboring voxels
    (A and B) and new standardized random normal values.
    -   Y = (w_a * A) + (w_b * B) + (w_r * R), where again w are the weights that ensure that
        the resulting subject values have unit variance and the desired correlations.
    -   w_a = r_ay - (w_b * r_ab) - (w_r * r_ar)
    -   w_b = ((r_by - (r_ab * r_ay)) - (w_r * (r_br - (r_ab * r_ar))) / (1 - (r_ab ** 2)))
    -   w_r = np.sqrt((1 - (r_ab ** 2) - (r_ay ** 2) - (r_by ** 2) + (2 * r_ab * r_ay * r_by))
        / (1 - (r_ab ** 2) - (r_ar ** 2) - (r_br ** 2) + (2 * r_ab * r_ar * r_br))
    """
    R = _simulate_voxel_with_no_neighbors(A.size)
    r_ab = np.corrcoef((A, B))[1, 0]
    r_ar = np.corrcoef((A, R))[1, 0]
    r_br = np.corrcoef((B, R))[1, 0]

    w_r = (
        (1 - (r_ab**2) - (r_ay**2) - (r_by**2) + (2 * r_ab * r_ay * r_by))
        / (1 - (r_ab**2) - (r_ar**2) - (r_br**2) + (2 * r_ab * r_ar * r_br))
    )
    w_r = np.sqrt(np.abs(w_r))  # NOTE: The abs isn't part of the math
    w_b = ((r_by - (r_ab * r_ay)) - (w_r * (r_br - (r_ab * r_ar)))) / (1 - (r_ab**2))
    w_a = r_ay - (w_b * r_ab) - (w_r * r_ar)

    y = (w_a * A) + (w_b * B) + (w_r * R)
    y = stats.zscore(y)  # NOTE: The zscore isn't part of the math. It should be z-scores by w*
    return y


def _simulate_voxel_with_three_neighbors(A, B, C, r_ay, r_by, r_cy):
    """Simulate the data for a voxel with three neighbors that have already been simulated.

    Parameters
    ----------
    A, B, C : numpy.ndarray of shape (n_subjects,)
        Subject values for the neighboring voxels.
    r_ay, r_by, r_cy : float
        Desired correlation between A and Y, B and Y, and C and Y.

    Returns
    -------
    y : numpy.ndarray of shape (n_subjects,)
        Data for the voxel across subjects.
    """
    R = _simulate_voxel_with_no_neighbors(A.size)
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


def _simulate_subject_maps(n_subjects, masker, correlation_maps):
    """Simulate the subject maps.

    Parameters
    ----------
    n_subjects
    masker
    correlation_maps : dict of 3 91x109x91 arrays
        right, posterior, and inferior

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
    mask_vec = masker.transform(masker.mask_img)
    mask_arr = masker.mask_img.get_fdata()
    n_x, n_y, n_z = masker.mask_img.shape
    xyz = np.vstack(np.where(mask_arr)).T
    # From https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/
    xyz = xyz[xyz[:, 2].argsort()]  # sort by z
    xyz = xyz[xyz[:, 1].argsort(kind="mergesort")]  # sort by y
    xyz = xyz[xyz[:, 0].argsort(kind="mergesort")]  # sort by x
    n_voxels = mask_vec.size
    assert xyz.shape[0] == n_voxels
    subject_maps = np.empty((np.sum(n_subjects), n_voxels))
    subject_counter = 0
    bad_directions, good_directions = [], []
    for i_study, n_study_subjects in enumerate(tqdm(n_subjects, desc="Studies")):
        study_subject_maps = np.full((n_x, n_y, n_z, n_study_subjects), fill_value=np.nan)
        for coord in trange(xyz.shape[0], leave=False, desc="Voxel"):
            i_x, j_y, k_z = xyz[coord, :]

            if mask_arr[i_x - 1, j_y, k_z] == 1:
                use_right = True
            else:
                use_right = False

            if mask_arr[i_x, j_y - 1, k_z] == 1:
                use_posterior = True
            else:
                use_posterior = False

            if mask_arr[i_x, j_y, k_z - 1] == 1:
                use_inferior = True
            else:
                use_inferior = False

            n_directions = sum((use_right, use_posterior, use_inferior))
            if n_directions == 0:
                voxel_values = _simulate_voxel_with_no_neighbors(n_study_subjects)

            elif n_directions == 1:
                if use_right:
                    A_data = study_subject_maps[i_x - 1, j_y, k_z, :]
                    r_ay = correlation_maps["right"][i_x, j_y, k_z]
                elif use_posterior:
                    A_data = study_subject_maps[i_x, j_y - 1, k_z, :]
                    r_ay = correlation_maps["posterior"][i_x, j_y, k_z]
                else:
                    A_data = study_subject_maps[i_x, j_y, k_z - 1, :]
                    r_ay = correlation_maps["inferior"][i_x, j_y, k_z]

                voxel_values = _simulate_voxel_with_one_neighbor(A_data, r_ay)

            elif n_directions == 2:
                if use_right:
                    A_data = study_subject_maps[i_x - 1, j_y, k_z, :]
                    r_ay = correlation_maps["right"][i_x, j_y, k_z]

                    if use_posterior:
                        B_data = study_subject_maps[i_x, j_y - 1, k_z, :]
                        r_by = correlation_maps["posterior"][i_x, j_y, k_z]
                    else:
                        B_data = study_subject_maps[i_x, j_y, k_z - 1, :]
                        r_by = correlation_maps["inferior"][i_x, j_y, k_z]
                elif use_posterior:
                    A_data = study_subject_maps[i_x, j_y - 1, k_z, :]
                    r_ay = correlation_maps["posterior"][i_x, j_y, k_z]

                    B_data = study_subject_maps[i_x, j_y, k_z - 1, :]
                    r_by = correlation_maps["inferior"][i_x, j_y, k_z]

                voxel_values = _simulate_voxel_with_two_neighbors(A_data, B_data, r_ay, r_by)

            else:
                A_data = study_subject_maps[i_x - 1, j_y, k_z, :]
                B_data = study_subject_maps[i_x, j_y - 1, k_z, :]
                C_data = study_subject_maps[i_x, j_y, k_z - 1, :]
                r_ay = correlation_maps["right"][i_x, j_y, k_z]
                r_by = correlation_maps["posterior"][i_x, j_y, k_z]
                r_cy = correlation_maps["inferior"][i_x, j_y, k_z]
                voxel_values = _simulate_voxel_with_three_neighbors(
                    A_data,
                    B_data,
                    C_data,
                    r_ay,
                    r_by,
                    r_cy,
                )

            study_subject_maps[i_x, j_y, k_z, :] = voxel_values
            if np.any(np.isnan(voxel_values)):
                bad_directions.append(n_directions)
            else:
                good_directions.append(n_directions)

        print(np.unique(bad_directions))
        print(np.unique(good_directions))
        study_subject_img = nib.Nifti1Image(
            study_subject_maps, masker.mask_img.affine, masker.mask_img.header
        )
        study_subject_data = masker.transform(study_subject_img)
        subject_maps[subject_counter : subject_counter + n_study_subjects, :] = study_subject_data
        subject_counter += n_study_subjects

    return subject_maps


def _scale_subject_maps(
    studylevel_effect_size_maps,
    studylevel_variance_maps,
    prelim_subjectlevel_maps,
):
    """Scale the "preliminary" set of subject-level maps for each dataset for a given imputation.

    Parameters
    ----------
    studylevel_effect_size_maps : numpy.ndarray of shape (S, V)
        S is study, V is voxel.
    studylevel_variance_maps : numpy.ndarray of shape (S, V)
    prelim_subjectlevel_maps : S-length list of numpy.ndarray of shape (N, V)
        List with one entry per study (S), where each entry is an array that is subjects (N) by
        voxels (V).

    Returns
    -------
    scaled_subjectlevel_maps : S-length list of numpy.ndarray of shape (N, V)
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
    scaled_subjectlevel_maps = []
    for i_study in range(studylevel_effect_size_maps.shape[0]):
        study_mean = studylevel_effect_size_maps[i_study, :]
        study_var = studylevel_variance_maps[i_study, :]

        study_subject_level_maps = prelim_subjectlevel_maps[i_study]
        study_subject_level_maps_scaled = (study_subject_level_maps * study_var) + study_mean
        scaled_subjectlevel_maps.append(study_subject_level_maps_scaled)

    return scaled_subjectlevel_maps


def _calculate_hedges_maps():
    ...


def _run_variance_meta():
    ...


def _combine_imputation_results(coefficient_maps, covariance_maps, i_stats, q_stats):
    """Use Rubin's rules to combine meta-analysis results across imputations.

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
    """Run the SDM algorithm.

    Parameters
    ----------
    coords
        Coordinates.
    n_imputations : int
        Number of imputations. Default is 50, based on the SDM software.
    n_iters : int
        Number of iterations for the Monte Carlo FWE correction procedure.
        Default is 1000, based on the SDM software.

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
    """
    # Extract sample size information from the coordinates DataFrame.
    n_subjects = coords.groupby("id")["sample_sizes"].values

    # Step 1: Estimate lower- and upper-bound effect size maps from coordinates.
    lower_bound_imgs, upper_bound_imgs = compute_sdm_ma(coords)

    # Step 2: Estimate the most likely effect size and variance maps across studies.
    # This should be the meta-analysis map we care about.
    meta_effect_size_img, meta_tau_img = _mle_estimation(lower_bound_imgs, upper_bound_imgs)

    # Step 4a: Create base set of simulated subject maps.
    raw_subject_effect_size_imgs = _simulate_subject_maps(n_subjects, masker, correlation_maps)

    all_subject_effect_size_imgs, all_subject_var_imgs = [], []
    for i_imp in range(n_imputations):
        # Step 3: Impute study-wise effect size and variance maps.
        study_effect_size_imgs, study_var_imgs = _impute_studywise_imgs(
            meta_effect_size_img,
            meta_tau_img,
            lower_bound_imgs,
            upper_bound_imgs,
            seed=i_imp,
        )

        # Step 4: Simulate subject-wise effect size and variance maps.
        imp_subject_effect_size_imgs, imp_subject_var_imgs = _scale_subject_maps(
            study_effect_size_imgs,
            study_var_imgs,
            raw_subject_effect_size_imgs,
        )
        # This is just a stand-in.
        all_subject_effect_size_imgs.append(imp_subject_effect_size_imgs)
        all_subject_var_imgs.append(imp_subject_var_imgs)

    # Step 5: Permutations...
    max_stats = {}
    for j_iter in range(n_iters):
        permuted_subject_effect_size_imgs, permuted_subject_var_imgs = _run_permutations(
            all_subject_effect_size_imgs, all_subject_var_imgs
        )

        # Step 6: Calculate study-level Hedges-corrected effect size maps.
        perm_study_hedge_imgs = _calculate_hedges_maps(
            permuted_subject_effect_size_imgs, permuted_subject_var_imgs
        )

        # Step 7: Meta-analyze imputed effect size maps.
        perm_meta_effect_size_imgs = _run_variance_meta(perm_study_hedge_imgs)

        # Step 8: Heterogeneity statistics and combine with Rubin's rules.
        perm_meta_effect_size_img = _combine_imputation_results(perm_meta_effect_size_imgs)

        # Max statistic
        perm_max_stats = _extract_max_statistics(perm_meta_effect_size_img)
        max_stats.update(perm_max_stats)

    return meta_effect_size_img, meta_tau_img, max_stats


def hedges_g(y, n_subjects1, n_subjects2=None):
    """Calculate Hedges' G.

    Parameters
    ----------
    y : 2D array of shape (n_studies, max_study_size)
        Subject-level values for which to calculate Hedges G.
        Multiple studies may be provided.
        The array contains as many rows as there are studies, and as many columns as the maximum
        sample size in the studyset. Extra columns in each row should be filled with NaNs.
    n_subjects1 : :obj:`numpy.ndarray` of shape (n_studies)
        Number of subjects in the first group of each study.
    n_subjects2 : None or int
        Number of subjects in the second group of each study.
        If None, the dataset is assumed to be have one sample.
        Technically, this parameter is probably unnecessary, since the second group's sample size
        can be inferred from ``n_subjects1`` and ``y``.

    Notes
    -----
    Clues for Python version from https://en.wikipedia.org/wiki/Effect_size#Hedges'_g.

    I also updated the original code to support varying sample sizes across studies.

    R Code
    ------
    g <- function (y) { # Calculate Hedges' g
        J * apply(y, 2, function (y_i) {
            (mean(y_i[1:n.subj]) - mean(y_i[-(1:n.subj)])) /
            sqrt((var(y_i[1:n.subj]) + var(y_i[-(1:n.subj)])) / 2)
        })
    }
    """
    from scipy.special import gamma

    if n_subjects2 is not None:
        # Must be an array
        assert n_subjects2.shape == n_subjects1.shape

        # Constants
        df = (n_subjects1 + n_subjects2) - 2  # degrees of freedom
        J_arr = gamma(df / 2) / (gamma((df - 1) / 2) * np.sqrt(df / 2))  # Hedges' correction

        g_arr = np.zeros(y.shape[0])
        for i_study in range(y.shape[0]):
            n1, n2 = n_subjects1[i_study], n_subjects2[i_study]
            study_data = y[i_study, : n1 + n2]
            var1 = np.var(study_data[:n1], axis=0)
            var2 = np.var(study_data[n1:], axis=0)

            mean_diff = np.mean(study_data[:n1]) - np.mean(study_data[n1:])
            pooled_std = np.sqrt((((n1 - 1) * var1) + ((n2 - 1) * var2)) / ((n1 + n2) - 2))
            g = mean_diff / pooled_std
            g_arr[i_study] = g

        g_arr = g_arr * J_arr * df

    else:
        df = n_subjects1 - 1
        J_arr = gamma(df / 2) / (gamma((df - 1) / 2) * np.sqrt(df / 2))  # Hedges' correction
        g_arr = J_arr * df * np.nanmean(y, axis=1) / np.nanstd(y, axis=1)

    return g_arr


def hedges_g_var(g, n_subjects1, n_subjects2=None):
    """Calculate variance of Hedges' G.

    Parameters
    ----------
    g : :obj:`numpy.ndarray` of shape (n_studies,)
        Hedges' G values.
    n_subjects1 : :obj:`numpy.ndarray` of shape (n_studies,)
    n_subjects2 : None or :obj:`numpy.ndarray` of shape (n_studies,)
        Can be None if the G values come from one-sample tests.

    Returns
    -------
    g_var : :obj:`numpy.ndarray` of shape (n_studies,)
        Hedges' G variance values.

    Notes
    -----
    Clues for Python version from https://constantinyvesplessen.com/post/g/#hedges-g,
    except using proper J instead of approximation.

    I also updated the original code to support varying sample sizes across studies.
    I still need to support one-sample tests though.

    R Code
    ------
    df <- 2 * n.subj - 2 # Degrees of freedom
    J <- gamma(df / 2) / gamma((df - 1) / 2) * sqrt(2 / df) # Hedges' correction
    g_var <- function (g) { # Variance of Hedges' g
        1 / n.subj + (1 - (df - 2) / (df * J^2)) * g^2
    }
    """
    if n_subjects2 is not None:
        assert g.shape == n_subjects1.shape == n_subjects2.shape
        g_var = ((n_subjects1 + n_subjects2) / (n_subjects1 * n_subjects2)) + (
            (g**2) / (2 * (n_subjects1 + n_subjects2))
        )
    else:
        raise ValueError("One-sample tests are not yet supported.")

    return g_var


def permute_study_effects(g, n_studies):
    """Permute study effects.

    Randomly sign-flip ~50% of the studies.

    Notes
    -----
    This is included for the simulations, but I don't think it would be used for SDM-PSI.

    R Code
    ------
    perm1 <- function (g) { # Permute study effects
        code <- which(runif(n.stud) > 0.5)
        g[code] <- -1 * g[code]
        g
    }
    """
    code = np.random.randint(0, 2, size=n_studies).astype(bool)
    out_g = g.copy()
    out_g[code] *= -1
    return out_g


def permute_subject_values(y, n_subjects):
    """Permute subject values.

    Seems to shuffle columns in each row independently.
    When "size" isn't provided to ``sample()``, it just permutes the array.

    Parameters
    ----------
    y : 2D array of shape (n_studies, max_study_size)
        Subject-level values for which to calculate Hedges G.
        Multiple studies may be provided.
        The array contains as many rows as there are studies, and as many columns as the maximum
        sample size in the studyset. Extra columns in each row should be filled with NaNs.
    n_subjects : :obj:`numpy.ndarray` of shape (n_studies,)
        The total number of subjects in each study.

    Notes
    -----
    This works for one estimate per particiipant per study.
    We need something that works for statistical maps.

    I also think this might not be usable on one-sample studies.

    R Code
    ------
    perm2 <- function (y) { # Permute subject values
        apply(y, 2, sample)
    }
    """
    permuted = np.full(y.shape, np.nan)
    for i_study in range(y.shape[0]):
        study_data = y[i_study, : n_subjects[i_study]]
        permuted[i_study, : n_subjects[i_study]] = np.random.permutation(study_data)

    return permuted


def simulate_subject_values(n_subjects1, n_subjects2):
    """Simulate (true) subject values.

    Simply draw from a normal distribution, then add a small effect size (0.2) to the first group.

    Parameters
    ----------
    n_subjects1, n_subjects2 : :obj:`numpy.ndarray` of shape (n_studies,)
        Number of subjects in each sample of each study. This does not support one-sample analyses.

    Returns
    -------
    y : 2D array of shape (n_studies, max_study_size)
        Subject-level values for which to calculate Hedges G.
        Multiple studies may be provided.
        The array contains as many rows as there are studies, and as many columns as the maximum
        sample size in the studyset. Extra columns in each row should be filled with NaNs.

    R Code
    ------
    sim.y <- function () { # Simulate (true) subject values
        y <- matrix(rnorm(n.stud * n.subj * 2), ncol = n.stud)
        y[1:n.subj,] <- y[1:n.subj,] + 0.2 # Add a small effect size
        y
    }
    """
    n_studies = len(n_subjects1)
    max_total_sample = np.max(n_subjects1 + n_subjects2)

    y = np.random.normal(size=(n_studies, max_total_sample))
    for i_study in range(y.shape[0]):
        # Any columns outside of the individual study's sample size will have NaNs
        y[i_study, (n_subjects1[i_study] + n_subjects2[i_study]) :] = np.nan
        # Add a small effect size
        y[i_study, : n_subjects1[i_study]] += 0.2

    return y


def run_simulations2(n_perms=1000, n_sims=10, n_subjects=20, n_studies=10):
    """Run the second simulation.

    Parameters
    ----------
    n_perms : int
        Number of permutations.
    n_sims : int
        Number of simulations.
    n_subjects : int
        Number of subjects per group of a study.
        2*n_subjects is the total number of subjects in each study.
    n_studies : int
        Number of studies in the meta-analysis.

    Notes
    -----
    TODO: Support one-sample studies.
    TODO: Support voxel-wise data, instead of single points per subject.
    See nimare/meta/cbma/sdmpsi_supplement_analysis2.r.
    """
    from pprint import pprint

    # Create simulation-specific sample sizes
    n_subs1 = np.random.randint(10, 50, size=(n_studies, n_sims))
    n_subs2 = np.random.randint(10, 50, size=(n_studies, n_sims))

    # Predefine outputs dictionary
    out_dict = {
        "time_perm_stud": np.empty(n_sims),
        "time_perm_subj": np.empty(n_sims),
        "p_z": np.empty(n_sims),
        "p_perm_stud": np.empty(n_sims),
        "p_perm_subj": np.empty(n_sims),
    }

    # Next is a parallelized for loop of 1:n_sims that uses metafor.
    for i_sim in range(n_sims):

        n_subs1_sim = n_subs1[:, i_sim]
        n_subs2_sim = n_subs2[:, i_sim]
        n_subjects_total = n_subs1_sim + n_subs2_sim

        # Simulate subject data of all studies
        y_unperm = simulate_subject_values(n_subs1_sim, n_subs2_sim)

        # Calculate Hedges' g
        g_unperm = hedges_g(y_unperm, n_subs1_sim, n_subs2_sim)
        g_var_unperm = hedges_g_var(g_unperm, n_subs1_sim, n_subs2_sim)

        # Meta-analysis
        dset = Dataset(y=g_unperm, v=g_var_unperm)
        est = estimators.VarianceBasedLikelihoodEstimator(method="REML")
        est.fit_dataset(dset)
        m_unperm = est.summary()
        z_unperm = m_unperm.get_fe_stats()["z"]

        # Save null distributions of z-values
        # NOTE: Not sure why the original z-values would be included in the null distribution
        # so I replaced them with empty lists.
        # nd_z_perm_stud = z_unperm
        # nd_z_perm_subj = z_unperm
        nd_z_perm_stud = np.empty(n_perms)
        nd_z_perm_subj = np.empty(n_perms)

        # Time before study-based permutation test
        time0 = datetime.now()

        # Study-based permutation test
        for j_perm in range(n_perms):
            # Permute study data
            g_stud_perm = permute_study_effects(g_unperm, n_studies)

            # Meta-analysis of permuted study data
            dset_stud_perm = Dataset(y=g_stud_perm, v=g_var_unperm)
            est_stud_perm = estimators.VarianceBasedLikelihoodEstimator(method="REML")
            est_stud_perm.fit_dataset(dset_stud_perm)
            m_stud_perm = est_stud_perm.summary()

            # Save null distribution of z-values
            zvals = m_stud_perm.get_fe_stats()["z"]
            nd_z_perm_stud[j_perm] = zvals

        # Time between study-based and subject-based permutation tests
        time1 = datetime.now()

        # Subject-based permutation test
        for j_perm in range(n_perms):
            # Permute subject data
            y_subj_perm = permute_subject_values(y_unperm, n_subjects_total)

            # Calculate Hedges' g of permuted subject data
            g_subj_perm = hedges_g(y_subj_perm, n_subs1_sim, n_subs2_sim)
            g_var_subj_perm = hedges_g_var(g_subj_perm, n_subs1_sim, n_subs2_sim)

            # Meta-analysis of permuted subject data
            dset_subj_perm = Dataset(y=g_subj_perm, v=g_var_subj_perm)
            est_subj_perm = estimators.VarianceBasedLikelihoodEstimator(method="REML")
            est_subj_perm.fit_dataset(dset_subj_perm)
            m_subj_perm = est_subj_perm.summary()

            # Save null distribution of z-values
            zvals = m_subj_perm.get_fe_stats()["z"]
            nd_z_perm_subj[j_perm] = zvals

        # Time after subject-based permutation tests
        time2 = datetime.now()

        # Save times and two-tailed p-values
        out_dict["time_perm_stud"][i_sim] = (time1 - time0).total_seconds()
        out_dict["time_perm_subj"][i_sim] = (time2 - time1).total_seconds()
        out_dict["p_z"][i_sim] = m_unperm.get_fe_stats()["p"]
        out_dict["p_perm_stud"][i_sim] = 1 - 2 * np.abs(np.mean(z_unperm > nd_z_perm_stud) - 0.5)
        out_dict["p_perm_subj"][i_sim] = 1 - 2 * np.abs(np.mean(z_unperm > nd_z_perm_subj) - 0.5)

    # Output results
    time_perm_stud = out_dict["time_perm_stud"]
    time_perm_subj = out_dict["time_perm_subj"]
    mse_perm_stud = (out_dict["p_perm_stud"] - out_dict["p_z"]) ** 2
    mse_perm_subj = (out_dict["p_perm_subj"] - out_dict["p_z"]) ** 2
    print(
        f"Decrease in execution time: {np.mean(time_perm_subj) - np.mean(time_perm_stud)}\n",
    )
    print(
        "Increase in MSE: "
        f"{(np.mean(mse_perm_stud) - np.mean(mse_perm_subj)) / np.mean(mse_perm_subj)}"
    )

    pprint(out_dict)


def compute_sdm_ma(
    ijk,
    effect_sizes,
    sample_sizes,
    significance_level,
    mask_img,
    corr_map,
    alpha=0.5,
    kernel_sigma=5,
):
    """Apply anisotropic kernel to coordinates.

    Parameters
    ----------
    ijk
    effect_sizes
    sample_sizes
    significance_level
    mask_img
    corr_map
    alpha : float
        User-selected degree of anisotropy. Default is 0.5.
    kernel_sigma : float
        User-specified sigma of kernel, in mm. Default is 5.

    Returns
    -------
    y_lower_img, y_upper_img

    Notes
    -----
    Use anisotropic Gaussian kernels, plus effect size estimates and metadata,
    to produce lower-bound and upper-bound effect size maps from the coordinates.

    References
    ----------
    *   Radua, J., Rubia, K., Canales, E. J., Pomarol-Clotet, E., Fusar-Poli, P., &
        Mataix-Cols, D. (2014). Anisotropic kernels for coordinate-based meta-analyses of
        neuroimaging studies. Frontiers in psychiatry, 5, 13.
    """
    df = np.sum(sample_sizes) - 2
    effect_size_threshold = stats.t.isf(significance_level, df)
    min_effect_size = -effect_size_threshold  # smallest possible effect size
    max_effect_size = effect_size_threshold  # largest possible effect size
    mask_data = mask_img.get_fdata()
    mask_ijk = np.vstack(np.where(mask_data)).T  # X x 3
    masked_distances = masking.unmask(masking.apply_mask(corr_map, mask_img), mask_img).get_fdata()
    masked_distances = 1 - masked_distances

    peak_corrs = []
    kept_peaks = []
    for i_peak in range(ijk.shape[0]):
        peak_ijk = ijk[i_peak, :]
        peak_t = effect_sizes[i_peak]
        if mask_data[tuple(peak_ijk)] == 0:
            # Skip peaks outside the mask
            continue

        # peak_corr is correlation between target voxel and peak.
        #   For non-adjacent voxels, peak_corr must be estimated with Dijkstra's algorithm.
        peak_corr = dijkstra3d.distance_field(masked_distances, source=peak_ijk)
        peak_corrs.append(peak_corr)
        kept_peaks.append(i_peak)

    kept_ijk = ijk[kept_peaks, :]
    peak_corrs = np.vstack(peak_corrs)
    # kept_effect_sizes = effect_sizes[kept_peaks]

    # real_distance is physical distance between voxel and peak
    # we need some way to select the appropriate peak for each voxel
    real_distances = spatial.distance.cdist(kept_ijk, mask_ijk)
    # closest_peak = np.argmin(real_distances, axis=0)
    virtual_distances = np.sqrt(
        (1 - alpha) * (real_distances**2) + alpha * 2 * kernel_sigma * np.log(peak_corr**-1)
    )
    y_lower = min_effect_size + np.exp((-(virtual_distances**2)) / (2 * kernel_sigma)) * (
        peak_t - min_effect_size
    )
    y_upper = max_effect_size + np.exp((-(virtual_distances**2)) / (2 * kernel_sigma)) * (
        peak_t - max_effect_size
    )
    y_lower_img = masking.unmask(y_lower, mask_img)
    y_upper_img = masking.unmask(y_upper, mask_img)
    return y_lower_img, y_upper_img
