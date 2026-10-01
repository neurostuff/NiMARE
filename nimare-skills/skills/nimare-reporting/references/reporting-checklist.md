# Pre-submission checklist

Work through it; each unchecked line is a predictable reviewer comment.

## Study set

- [ ] Search strategy, databases, dates, and full query recorded.
- [ ] Inclusion/exclusion criteria stated, with counts at each screening stage.
- [ ] PRISMA flow diagram.
- [ ] Articles, independent samples, and experiments counted separately and all reported.
- [ ] Rule for collapsing same-sample contrasts stated and applied.
- [ ] `audit_independence.py` run clean, or its findings explained.
- [ ] >= 17-20 experiments, or the power limitation stated prominently.
- [ ] All coordinates in one space; conversions named.
- [ ] Out-of-mask foci filtered (`FocusFilter`) and the count reported.
- [ ] Sample sizes present for every experiment, or the fixed kernel justified.
- [ ] Study set shared as NiMADS or Sleuth.

## Analysis

- [ ] NiMARE version and Python version.
- [ ] Estimator named with its parameters, not just its name.
- [ ] Kernel and FWHM rule.
- [ ] Null method.
- [ ] Correction: method, cluster-forming threshold, corrected alpha, iterations.
- [ ] `random_state` set and reported.
- [ ] The specific map key reported (e.g. `z_desc-mass_level-cluster_corr-FWE_method-montecarlo`).
- [ ] For subtraction: groups disjoint; both group sizes; balanced or not.
- [ ] For conjunction: inputs thresholded and corrected.
- [ ] For IBMA: `groupby`, `weight_scheme`, `rho`, masking mode, n groups, any PyMARE warning.
- [ ] For decoding: vocabulary restriction, n terms, null model or an explicit "rankings only".

## Robustness

- [ ] Jackknife / leave-one-experiment-out run and reported.
- [ ] Focus counter run; clusters driven by one or two experiments flagged in the text.
- [ ] Any cluster that fails either diagnostic described as exploratory.
- [ ] Sensitivity to the thresholding choice checked, if the result is near the boundary.

## Results presentation

- [ ] Cluster table: coordinates, extent, peak statistic, **and the map the statistic
      came from** (corrected cluster map vs uncorrected z).
- [ ] Anatomical labels with the atlas named and its version.
- [ ] Unthresholded maps uploaded to NeuroVault and linked.
- [ ] Figures show the threshold used.

## Language

- [ ] No reverse-inference claims from forward-inference maps.
- [ ] No "region X does Y" from convergence.
- [ ] Null results framed as undetected, not absent.
- [ ] Decoding described as spatial similarity to meta-analytic term maps.

## Code and data

- [ ] Script in the supplement or a repository, with pinned versions.
- [ ] Input study set shared.
- [ ] Outputs (maps, tables, provenance) archived with a DOI.
