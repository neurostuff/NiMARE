"""Python implementations of simulation code from the SDM-PSI paper's appendix."""
from datetime import datetime

import numpy as np
from pymare import Dataset, estimators

from nimare.meta.cbma.sdm import hedges_g, hedges_g_var


def permute_study_effects(g, n_studies):
    """Permute study effects.

    Randomly sign-flip ~50% of the studies.

    Notes
    -----
    This is included for the simulations, but I don't think it would be used for SDM-PSI.

    Notes
    -----
    R Code:

    .. highlight:: r
    .. code-block:: r

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

    Notes
    -----
    R Code:

    .. highlight:: r
    .. code-block:: r

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

    Notes
    -----
    R Code:

    .. highlight:: r
    .. code-block:: r

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

    .. todo::

        Support one-sample studies.

    .. todo::

        Support voxel-wise data, instead of single points per subject.

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
