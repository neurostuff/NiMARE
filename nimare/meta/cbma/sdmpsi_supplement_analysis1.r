#################################################################################
#         EFFECTS OF THE CORRELATION BETWEEN IMPUTATIONS AND THE USE OF         #
#           VARIABLE PERMUTATION CODES IN THE STATISTICAL SIGNIFICANCE          #
#        SIMULATION OF MULTIPLE IMPUTATION OF EFFECT SIZE, IMPUTATION OF        #
#             SUBJECT VALUES AND PERMUTATION TEST FOR A SINGLE STUDY            #
#################################################################################

# Simulation parameters #######################################################
n.imp <- 50 # Number of imputations
n.perm <- 1000 # Number of permutations
n.sim <- 10 # Number of simulations
n.subj <- 20 # Number of subjects per study

# Small functions #############################################################
g <- function (y) { # Calculate Hedges' g
    if (is.vector(y)) {
        return(J * mean(y) / sd(y))
    }
    J * apply(y, 2, function (y_i) {
        mean(y_i) / sd(y_i)
    })
}
g_var <- function (g) { # Variance of Hedges' g
    1 / n.subj + (1 - (df - 2) / (df * J^2)) * g^2
}
imp.subj <- function () { # Impute subject values
    y <- rnorm(n.subj)
    (y - mean(y)) / sd(y)
}
perm <- function (y, code) { # Permute subject values
    y[code] <- -1 * y[code]
    y
}
perm.code <- function () { # Create a random permutation code
    which(runif(n.subj) > 0.5)
}
combine.g <- function (g) { # Combine Hedges' g from different imputations
    mean(g)
}
combine.g_var <- function (g, g_var) { # Combine variance from different imp.
    mean(g_var) + (1 + 1 / n.imp) * var(g)
}
sim.g.imp <- function (g) { # Simulate multiple imputation of Hedges' g
    g + rnorm(n.imp, 0, 0.2)
}
sim.y.true <- function () { # Simulate (true) subject values
    rnorm(n.subj)
}

# Main ########################################################################
library(doParallel)
registerDoParallel(cores = detectCores() - 1)
par(mfrow = c(2, 2))

# Constants
df <- n.subj - 1 # Degrees of freedom
J <- gamma(df / 2) / gamma((df - 1) / 2) * sqrt(2 / df) # Hedges' correction

# Effects to test: perfect correlation between any two imputations?
for (PERFECT_CORRELATION in c(TRUE, FALSE)) {
    # Effects to test: apply the same permutation code to all imputations?
    for (SAME_PERM_CODE in c(TRUE, FALSE)) {
        # Start simulations
        sim <- do.call(rbind.data.frame, foreach (i.sim = 1:n.sim) %dopar% {

            # Simulate (true) subject data
            y.true <- sim.y.true()

            # Calculate Hedges' g
            g.true <- g(y.true)
            g_var.true <- g_var(g.true)

            # Calculate z-value
            z.true <- g.true / sqrt(g_var.true)

            # Simulate multiple imputaion of Hedges' g
            g.imp <- sim.g.imp(g.true)
            g_var.imp <- g_var(g.imp)

            # Combine imputed Hedges' g
            g.imp.combined <- combine.g(g.imp)
            g_var.imp.combined <- combine.g_var(g.imp, g_var.imp)

            # Calculate z-value of combined imputed data
            z.imp.combined <- g.imp.combined / sqrt(g_var.imp.combined)

            # Impute subject data
            replicate.d.imp <- t(replicate(n.subj, g.imp / J))
            if (PERFECT_CORRELATION) {
                common.imp.subj <- imp.subj()
                y.imp <- replicate(n.imp, common.imp.subj) + replicate.d.imp
            } else {
                y.imp <- replicate(n.imp, imp.subj()) + replicate.d.imp
            }

            # Save variance between imputations and null distributions of z-values
            var.g.imp <- var(g.imp)
            nd.z.true <- z.true
            nd.z.imp.combined <- z.imp.combined

            for (i.perm in 1:(n.perm - 1)) {

                # Permute (true) subject data
                y.true.perm <- perm(y.true, perm.code())
                # Calculate Hedges' g of permuted data
                g.true.perm <- g(y.true.perm)
                g_var.true.perm <- g_var(g.true.perm)
                # Calculate z-value of permuted data
                z.true.perm <- g.true.perm / sqrt(g_var.true.perm)
                # Permute imputed subject data
                if (SAME_PERM_CODE) {
                    common.perm.code <- perm.code()
                    y.imp.perm <- apply(y.imp, 2, function (y.imp.i) {
                        perm(y.imp.i, common.perm.code)
                    })
                } else {
                    y.imp.perm <- apply(y.imp, 2, function (y.imp.i) {
                        perm(y.imp.i, perm.code())
                    })
                }

                # Calculate Hedges' g of permuted imputed data
                g.imp.perm <- g(y.imp.perm)
                g_var.imp.perm <- g_var(g.imp.perm)

                # Calculate z-value of permuted imputed data
                g.imp.perm.combined <- combine.g(g.imp.perm)
                g_var.imp.perm.combined <- combine.g_var(g.imp.perm, g_var.imp.perm)
                z.imp.perm.combined <- g.imp.perm.combined / sqrt(g_var.imp.perm.combined)

                # Save variance between imputations and null distributions of z-values
                var.g.imp <- c(var.g.imp, var(g.imp.perm))
                nd.z.true <- c(nd.z.true, z.true.perm)
                nd.z.imp.combined <- c(nd.z.imp.combined, z.imp.perm.combined)

            }

            # Save variance between imputations and one-tailed p-values
            data.frame(
                var.g.imp = mean(var.g.imp),
                p.true = mean(z.true > nd.z.true),
                p.imp.combined = mean(z.imp.combined > nd.z.imp.combined)
            )
        })

        # Create output plot
        plot(0:1, 0:1, type = "l", col = "#999999", cex.lab = 1.2,
            xlab = "p-values from (true) subject data",
            ylab = "p-values from imputed subject data",
            main = paste(
                ifelse(PERFECT_CORRELATION, "Perfect", "No"),
                "correlation between imputations.",
                ifelse(SAME_PERM_CODE, "Same", "Different"),
                "permutation code.\n",
                "Variance between imputations:",
                signif(mean(sim$var.g.imp), 1)
            ), font.main = 1)
        points(sim$p.true, sim$p.imp.combined)
    }
}
