"""Seed-based d-mapping-related methods."""
from datetime import datetime

import dijkstra3d
import numpy as np
from nilearn import masking
from pymare import Dataset, estimators
from scipy import spatial, stats


def hedges_g(y, J, n_subjects=None):
    """Calculate Hedges' G.

    NOTE: We probably want to support both two-sample and one-sample versions.

    R Code
    ------
    g <- function (y) { # Calculate Hedges' g
        J * apply(y, 2, function (y_i) {
            (mean(y_i[1:n.subj]) - mean(y_i[-(1:n.subj)])) /
            sqrt((var(y_i[1:n.subj]) + var(y_i[-(1:n.subj)])) / 2)
        })
    }
    """
    if n_subjects is None:
        g_arr = J * (y.shape[0] - 1) * np.mean(y, axis=0) / np.std(y, axis=0)

    else:
        n1 = n_subjects
        n2 = y.shape[0] - n_subjects
        var1 = np.var(y[:n1], axis=0)
        var2 = np.var(y[n1:], axis=0)

        g_arr = np.zeros(y.shape[1])
        for i_col in range(y.shape[1]):
            mean_diff = np.mean(y[:n_subjects, i_col]) - np.mean(y[n_subjects:, i_col])
            pooled_std = np.sqrt(
                (((n1 - 1) * var1[i_col]) + ((n2 - 1) * var2[i_col])) / (n_subjects - 2)
            )
            g = mean_diff / pooled_std
            g_arr[i_col] = g

        g_arr = g_arr * (J * (n1 + n2 - 2))

    return g_arr


def hedges_g_var(g, n_subjects, df, J):
    """Calculate variance of Hedges' G.

    R Code
    ------
    df <- 2 * n.subj - 2 # Degrees of freedom
    J <- gamma(df / 2) / gamma((df - 1) / 2) * sqrt(2 / df) # Hedges' correction
    g_var <- function (g) { # Variance of Hedges' g
        1 / n.subj + (1 - (df - 2) / (df * J^2)) * g^2
    }
    """
    g_var = (1 / n_subjects) + (1 - (df - 2) / (df * (J ** 2))) * (g ** 2)
    return g_var


def permute_study_effects(g, n_studies):
    """Permute study effects.

    Randomly sign-flip ~50% of the studies.

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


def permute_subject_values(y):
    """Permute subject values.

    Seems to shuffle rows in each column independently.
    When "size" isn't provided to ``sample()``, it just permutes the array.

    R Code
    ------
    perm2 <- function (y) { # Permute subject values
        apply(y, 2, sample)
    }
    """
    permuted = np.apply_along_axis(np.random.permutation, 0, y)
    return permuted


def simulate_subject_values(n_studies, n_subjects):
    """Simulate subject values.

    R Code
    ------
    sim.y <- function () { # Simulate (true) subject values
        y <- matrix(rnorm(n.stud * n.subj * 2), ncol = n.stud)
        y[1:n.subj,] <- y[1:n.subj,] + 0.2 # Add a small effect size
        y
    }
    """
    ...


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
    TODO: Support array for number of subjects, so that each study has its own sample size.
    TODO: Support one-sample studies.

    R Code
    ------
    library(doParallel)
    library(metafor)
    registerDoParallel(cores = detectCores() - 1)
    par(mfrow = 1:2)

    # Simulation parameters #######################################################
    n.perm <- 1000 # Number of permutations
    n.sim <- 1000 # Number of simulations
    n.subj <- 20 # Number of subjects per group of a study
    n.stud <- 10 # Number of studies

    # Constants
    df <- 2 * n.subj - 2 # Degrees of freedom
    J <- gamma(df / 2) / gamma((df - 1) / 2) * sqrt(2 / df) # Hedges' correction

    sim <- do.call(rbind.data.frame, foreach (i.sim = 1:n.sim, .packages = "metafor")
    %dopar% {
        # Simulate subject data of all studies
        y.unperm <- sim.y()

        # Calculate Hedges' g
        g.unperm <- g(y.unperm)
        g_var.unperm <- g_var(g.unperm)

        # Meta-analysis
        m.unperm <- rma(g.unperm, g_var.unperm)
        z.unperm <- m.unperm$zval

        # Save null distributions of z-values
        nd.z.perm_stud <- z.unperm
        nd.z.perm_subj <- z.unperm

        # Time before study-based permutation test
        time0 <- Sys.time()

        # Study-based permutation test
        for (i.perm in 1:(n.perm - 1)) {
            # Permute study data
            g.stud_perm <- perm1(g.unperm)

            # Meta-analysis of permuted study data
            m.stud_perm <- rma(g.stud_perm, g_var.unperm)

            # Save null distribution of z-values
            nd.z.perm_stud <- c(nd.z.perm_stud, m.stud_perm$zval)
        }

        # Time between study-based and subject-based permutation tests
        time1 <- Sys.time()

        # Subject-based permutation test
        for (i.perm in 1:(n.perm - 1)) {
            # Permute subject data
            y.subj_perm <- perm2(y.unperm)

            # Calculate Hedges' g of permuted subject data
            g.subj_perm <- g(y.subj_perm)
            g_var.subj_perm <- g_var(g.subj_perm)

            # Meta-analysis of permuted subject data
            m.subj_perm <- rma(g.subj_perm, g_var.subj_perm)

            # Save null distribution of z-values
            nd.z.perm_subj <- c(nd.z.perm_subj, m.subj_perm$zval)
        }

        # Time after subject-based permutation tests
        time2 <- Sys.time()

        # Save times and two-tailed p-values
        data.frame(
            time.perm_stud = as.numeric(time1 - time0),
            time.perm_subj = as.numeric(time2 - time1),
            p.z = m.unperm$pval,
            p.perm_stud = 1 - 2 * abs(mean(z.unperm > nd.z.perm_stud) - 0.5),
            p.perm_subj = 1 - 2 * abs(mean(z.unperm > nd.z.perm_subj) - 0.5)
        )
    })

    # Output results
    time.perm_stud <- sim$time.perm_stud
    time.perm_subj <- sim$time.perm_subj
    mse.perm_stud <- (sim$p.perm_stud - sim$p.z)^2
    mse.perm_subj <- (sim$p.perm_subj - sim$p.z)^2
    cat("Decrease in execution time: ", round(
        (mean(time.perm_subj) - mean(time.perm_stud)) / mean(time.perm_subj),
        2) * 100, "%\n", sep = "")
    cat("Increase in mean squared error: ", round(
        (mean(mse.perm_stud) - mean(mse.perm_subj)) / mean(mse.perm_subj),
        2) * 100, "%\n", sep = "")
    """
    from math import gamma

    import numpy as np

    # Constants
    df = (2 * n_subjects) - 2  # degrees of freedom
    J = gamma(df / 2) / gamma((df - 1) / 2) * np.sqrt(2 / df)  # Hedges' correction

    # Next is a parallelized for loop of 1:n_sims that somehow uses metafor.
    for i_sim in range(n_sims):

        # Simulate subject data of all studies
        y_unperm = simulate_subject_values(n_studies, n_subjects)

        # Calculate Hedges' g
        g_unperm = hedges_g(y_unperm, n_subjects, J)
        g_var_unperm = hedges_g_var(g_unperm, n_subjects, df, J)

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
        nd_z_perm_stud = []
        nd_z_perm_subj = []

        # Time before study-based permutation test
        time0 = datetime.now()

        # Study-based permutation test
        for j_perm in range(n_perms):
            # Permute study data
            g_stud_perm = permute_study_effects(g_unperm)

            # Meta-analysis of permuted study data
            dset_stud_perm = Dataset(y=g_stud_perm, v=g_var_unperm)
            est_stud_perm = estimators.VarianceBasedLikelihoodEstimator(method="REML")
            est_stud_perm.fit_dataset(dset_stud_perm)
            m_stud_perm = est_stud_perm.summary()

            # Save null distribution of z-values
            # TODO: Fix this after testing. zvals should be an array across samples,
            # so appending to a list doesn't make sense.
            zvals = m_stud_perm.get_fe_stats()["z"]
            nd_z_perm_stud.append(zvals)

        # Time between study-based and subject-based permutation tests
        time1 = datetime.now()

        # Subject-based permutation test
        for j_perm in range(n_perms):
            # Permute subject data
            y_subj_perm = permute_subject_values(y_unperm)

            # Calculate Hedges' g of permuted subject data
            g_subj_perm = hedges_g(y_subj_perm)
            g_var_subj_perm = hedges_g_var(g_subj_perm)

            # Meta-analysis of permuted subject data
            dset_subj_perm = Dataset(y=g_subj_perm, v=g_var_subj_perm)
            est_subj_perm = estimators.VarianceBasedLikelihoodEstimator(method="REML")
            est_subj_perm.fit_dataset(dset_subj_perm)
            m_subj_perm = est_subj_perm.summary()

            # Save null distribution of z-values
            zvals = m_subj_perm.get_fe_stats()["z"]
            nd_z_perm_subj.append(zvals)

        # Time after subject-based permutation tests
        time2 = datetime.now()

        # Save times and two-tailed p-values
        # TODO: Store data across simulations
        out_dict = {
            "time_perm_stud": time1 - time0,
            "time_perm_subj": time2 - time1,
            "p_z": m_unperm.get_fe_stats()["p"],
            "p_perm_stud": 1 - 2 * np.abs(np.mean(z_unperm > nd_z_perm_stud) - 0.5),
            "p_perm_subj": 1 - 2 * np.abs(np.mean(z_unperm > nd_z_perm_subj) - 0.5),
        }

    # Output results
    time_perm_stud = out_dict["time_perm_stud"]
    time_perm_subj = out_dict["time_perm_subj"]
    mse_perm_stud = (out_dict["p_perm_stud"] - out_dict["p_z"]) ** 2
    mse_perm_subj = (out_dict["p_perm_subj"] - out_dict["p_z"]) ** 2
    print(
        "Decrease in execution time: "
        f"{np.round(np.mean(time_perm_subj) - np.mean(time_perm_stud))}\n",
    )
    print(
        "Increase in MSE: "
        f"{np.round((np.mean(mse_perm_stud) - np.mean(mse_perm_stud)) / np.mean(mse_perm_subj))}"
    )


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

    References
    ----------
    * Radua, J., Rubia, K., Canales, E. J., Pomarol-Clotet, E., Fusar-Poli, P., &
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
        (1 - alpha) * (real_distances ** 2) + alpha * 2 * kernel_sigma * np.log(peak_corr ** -1)
    )
    y_lower = min_effect_size + np.exp((-(virtual_distances ** 2)) / (2 * kernel_sigma)) * (
        peak_t - min_effect_size
    )
    y_upper = max_effect_size + np.exp((-(virtual_distances ** 2)) / (2 * kernel_sigma)) * (
        peak_t - max_effect_size
    )
    y_lower_img = masking.unmask(y_lower, mask_img)
    y_upper_img = masking.unmask(y_upper, mask_img)
    return y_lower_img, y_upper_img
