###############################################################################
#          PERMUTATION OF STUDY DATA AND PERMUTATION OF SUBJECT DATA          #
###############################################################################

# Simulation parameters #######################################################
n.perm <- 1000 # Number of permutations
n.sim <- 1000 # Number of simulations
n.subj <- 20 # Number of subjects per group of a study
n.stud <- 10 # Number of studies

# Small functions #############################################################
g <- function (y) { # Calculate Hedges' g
    J * apply(y, 2, function (y_i) {
        (mean(y_i[1:n.subj]) - mean(y_i[-(1:n.subj)])) /
            sqrt((var(y_i[1:n.subj]) + var(y_i[-(1:n.subj)])) / 2)
    })
}
g_var <- function (g) { # Variance of Hedges' g
    1 / n.subj + (1 - (df - 2) / (df * J^2)) * g^2
}
perm1 <- function (g) { # Permute study effects
    code <- which(runif(n.stud) > 0.5)
    g[code] <- -1 * g[code]
    g
}
perm2 <- function (y) { # Permute subject values
    apply(y, 2, sample)
}
sim.y <- function () { # Simulate (true) subject values
    y <- matrix(rnorm(n.stud * n.subj * 2), ncol = n.stud)
    y[1:n.subj,] <- y[1:n.subj,] + 0.2 # Add a small effect size
    y
}

# Main ########################################################################
library(doParallel)
library(metafor)
registerDoParallel(cores = detectCores() - 1)
par(mfrow = 1:2)

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
