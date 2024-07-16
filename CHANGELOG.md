# What's New

All notable changes to NiMARE releases are documented in this page.

## [Unreleased](https://github.com/neurostuff/NiMARE/compare/0.3.0...HEAD)

## [0.3.0](https://github.com/neurostuff/NiMARE/compare/0.2.2...0.3.0) - 2024-07-16

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### 🎉 Exciting New Features

* Add correction for multiple contrasts within a study in Stouffer's IBMA by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/882
* Add support for the `concordant` mode test in Fisher's and Stouffer's estimators by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/884
* Add support for publication size weighting. General refactoring of IBMA estimators by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/887

#### Other Changes

* min joblib 1.3.0 by @jdkent in https://github.com/neurostuff/NiMARE/pull/880
* Update Nilearn API _check_same_fov by @adelavega in https://github.com/neurostuff/NiMARE/pull/873
* [FIX] update codecov by @jdkent in https://github.com/neurostuff/NiMARE/pull/883
* [FIX] bump matplotlib version and update usage of get_cmap by @jdkent in https://github.com/neurostuff/NiMARE/pull/885
* [MAINT] bump cognitiveatlas version by @jdkent in https://github.com/neurostuff/NiMARE/pull/890
* Support Python 3.12 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/853

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.2.2...0.3.0

## [0.2.2](https://github.com/neurostuff/NiMARE/compare/0.2.1...0.2.2) - 2024-02-07

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

This release changes the default behavior of the number of iterations chosen for the FWECorrector, we reduced the default number of montecarlo simulations from 10,000 to 5,000 as this allows for a more manageable runtime with limited impact on the stability of the results.

#### 🎉 Exciting New Features

* Add advanced plots to IBMA report by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/864

#### 🐛 Bug Fixes

* Fix ridgeline plot in IBMA report by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/863
* [FIX] tqdm not displaying on notebooks by @jdkent in https://github.com/neurostuff/NiMARE/pull/878
* [FIX] fwe docstrings by @jdkent in https://github.com/neurostuff/NiMARE/pull/868

#### Other Changes

* Clarify FWECorrector arguments by @adelavega in https://github.com/neurostuff/NiMARE/pull/865
* [MAINT] limit version of nilearn to 0.10.2 by @jdkent in https://github.com/neurostuff/NiMARE/pull/877
* [MAINT] use updated black by @jdkent in https://github.com/neurostuff/NiMARE/pull/876

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.2.1...0.2.2

## [0.2.1](https://github.com/neurostuff/NiMARE/compare/0.2.0...0.2.1) - 2024-01-11

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### 🎉 Exciting New Features

* Implement caching for Estimators and Transformers by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/845
* Add similarity correlation matrix figure to IBMA reports by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/852
* Support liberal mask in IBMA estimators by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/848
* Add DoF map to IBMA report by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/854

#### 👎 Deprecations

* Deprecate `ale_sleuth_workflow` in favor of `CBMAWorkflow` and `PairwiseCBMAWorkflow` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/851

#### Enhancements

* Optimize compute_kda_ma for memory and speed by @adelavega in https://github.com/neurostuff/NiMARE/pull/857 :rocket:

#### Fixes

* [FIX] do not check id columns for image extensions by @jdkent in https://github.com/neurostuff/NiMARE/pull/860
* [FIX] bump min numpy/seaborn versions by @jdkent in https://github.com/neurostuff/NiMARE/pull/861

#### Other Changes

* Disable computation of probabilities by default for MKDAChi2 by @adelavega in https://github.com/neurostuff/NiMARE/pull/856

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.2.0...0.2.1

## [0.2.0](https://github.com/neurostuff/NiMARE/compare/0.1.1...0.2.0) - 2023-11-02

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### 🎉 Exciting New Features

- Add Workflow and CBMAWorkflow classes. Support pairwise CBMA workflows by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/809
- Remove `resample` argument from IBMA estimators by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/823
- Add IBMAWorkflow by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/817
- Make `torch` optional by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/836
- Add Conjunction Analysis Workflow by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/841

#### 🐛 Bug Fixes

- Fix the aspect ratio and size of the heatmap in Reports by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/815
- Addresses new RTD configuration file requirements by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/829
- Fix compatibility of ImageTransformer with Pandas 2.1.2 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/843
- [FIX] handle index errors by @jdkent in https://github.com/neurostuff/NiMARE/pull/839
- [FIX] regular expression for no moderators by @jdkent in https://github.com/neurostuff/NiMARE/pull/821
- Fix the NeuroLibre badge by @tsalo in https://github.com/neurostuff/NiMARE/pull/824
- [FIX] handle null values in metadata by @jdkent in https://github.com/neurostuff/NiMARE/pull/831

#### Other Changes

- Add badges and citations for Aperture Neuro article by @tsalo in https://github.com/neurostuff/NiMARE/pull/834
- Remove pytorch warning message by @yifan0330 in https://github.com/neurostuff/NiMARE/pull/828

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.1.1...0.2.0

## [0.1.1](https://github.com/neurostuff/NiMARE/compare/0.1.0...0.1.1) - 2023-06-12

<!-- Release notes generated using configuration in .github/release.yml at main -->
Main change is to include `default.yml` and `default.tpl` in the python package distribution

### What's Changed

#### 🎉 Exciting New Features

- Combine analyses in Studyset by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/810

#### Other Changes

- [FIX] include default.yml as part of nimare package by @jdkent in https://github.com/neurostuff/NiMARE/pull/812

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.1.0...0.1.1

## [0.1.0](https://github.com/neurostuff/NiMARE/compare/0.0.14...0.1.0) - 2023-06-02

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

This is a big release, there are several large items we added:

- Coordinate Based Meta Regression! This is a great alternative to the kernel-based methods to detect convergence with more sensitivity and more flexibly compare between groups.
- Reports Module: now you can generate an html report for the simple kernel based methods, more estimators will be supported in upcoming releases

#### 🎉 Exciting New Features

- [ENH] Support pre-generated maps in `CorrelationDecoder` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/782
- [ENH] add Coordinate Based Meta Regression by @yifan0330 in https://github.com/neurostuff/NiMARE/pull/721
- [ENH] Add Corrector and Diagnostics attributes to MetaResult object by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/804
- [ENH] Add NiMAREBase features to the Corrector base class by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/807
- [ENH] Add `reports` module by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/802

#### 🐛 Bug Fixes

- [FIX] Set `n_iters` defaults only for estimators with `null_method="montecarlo"` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/803
- [FIX] handle case of no sample size being reported by @jdkent in https://github.com/neurostuff/NiMARE/pull/792
- [FIX] math display by @yifan0330 in https://github.com/neurostuff/NiMARE/pull/805
- [FIX] allow analysis to have null points when converting from nimads to dataset by @jdkent in https://github.com/neurostuff/NiMARE/pull/808

#### Other Changes

- [MAINT] Drop support for Python 3.6 and 3.7 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/780
- [MAINT] remove codecov by @jdkent in https://github.com/neurostuff/NiMARE/pull/788
- [DOC] change readthedocs badge from latest to stable by @jdkent in https://github.com/neurostuff/NiMARE/pull/786
- [DOC] Add neurolibre link by @jdkent in https://github.com/neurostuff/NiMARE/pull/789
- [MAINT] make indexed_gzip install optional by @jdkent in https://github.com/neurostuff/NiMARE/pull/791
- [MAINT] Remove RC versions from Changelog by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/790
- [MAINT] Unpin numpy version by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/794
- [REF] Replace `_get_clusters_table` with nilearn's `get_clusters_table` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/793
- [MAINT] Support Python 3.11 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/796
- [MAINT] fix readthedocs by @jdkent in https://github.com/neurostuff/NiMARE/pull/797

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.14...0.1.0

## [0.0.14](https://github.com/neurostuff/NiMARE/compare/0.0.13...0.0.14) - 2023-03-31

### What's Changed

#### 🛠 Breaking Changes

- Support clusters table in Diagnostics by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/765

#### 🎉 Exciting New Features

- Add `save()` and `load()` methods to `MetaResult` objects by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/771
- Incorporate Estimator and Corrector descriptions into MetaResult objects by @tsalo in https://github.com/neurostuff/NiMARE/pull/724
- Add `cluster_threshold` option to Diagnostics by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/777
- Add CBMA workflow by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/761

#### 🐛 Bug Fixes

- Do not zero out one-tailed z-statistics for p-values > 0.5 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/693

#### Other Changes

- Support nibabel 5.0.0 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/762
- Link to NeuroStars software support category instead of neuro questions by @tsalo in https://github.com/neurostuff/NiMARE/pull/768
- Revert "Do not zero out one-tailed z-statistics for p-values > 0.5" by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/769
- [DOC] add note about SDM by @jdkent in https://github.com/neurostuff/NiMARE/pull/764
- Replace `pandas.DataFrame.append` with `pandas.concat` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/774
- [ENH] rudimentary support for nimads by @jdkent in https://github.com/neurostuff/NiMARE/pull/763
- Major refactoring of Diagnostics module by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/776
- [DOC] add proper documentation to nimads module by @jdkent in https://github.com/neurostuff/NiMARE/pull/778

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.13...0.0.14

## [0.0.13](https://github.com/neurostuff/NiMARE/compare/0.0.12...0.0.13) - 2023-01-17

### What's Changed

This release was motivated because there were updates to underlying packages that broke the installation of NiMARE.
In addition, there are multiple breaking changes as well as new features outlined below.

#### 🛠 Breaking Changes

- Remove Peaks2Maps from NiMARE by @tsalo in https://github.com/neurostuff/NiMARE/pull/644
- Remove duecredit in favor of BibTeX references by @tsalo in https://github.com/neurostuff/NiMARE/pull/736
- Switch from face+edge connectivity to face-only by @tsalo in https://github.com/neurostuff/NiMARE/pull/733
- Remove conperm and scale CLI workflows by @tsalo in https://github.com/neurostuff/NiMARE/pull/740

#### 🎉 Exciting New Features

- Add `tables` attribute to MetaResult class by @tsalo in https://github.com/neurostuff/NiMARE/pull/734
- Add FocusFilter class for removing coordinates outside of a mask by @tsalo in https://github.com/neurostuff/NiMARE/pull/732
- Add parallelization option to `CorrelationDecoder` and `CorrelationDistributionDecoder` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/738
- Append the top 3 words to LDA topic names by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/741
- Enhance LDA annotator by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/742

#### 🐛 Bug Fixes

- Shift centers of mass into clusters in Jackknife/FocusCounter by @tsalo in https://github.com/neurostuff/NiMARE/pull/735
- fix a bug in conversion from z statistics to p values by @yifan0330 in https://github.com/neurostuff/NiMARE/pull/749
- Remove "dataset" `return_type` option from kernel transformers by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/752

#### Other Changes

- Fix import in download_neurosynth example by @PTDZ in https://github.com/neurostuff/NiMARE/pull/743
- Optimize compute_kda_ma by @liuzhenqi77 in https://github.com/neurostuff/NiMARE/pull/745
- Optimize dataset.get by @liuzhenqi77 in https://github.com/neurostuff/NiMARE/pull/746
- Fix MACM analysis example by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/750
- Remove upper bound for matplotlib version by @ghisvail in https://github.com/neurostuff/NiMARE/pull/751
- Fix neurosyth download_abstracts example; inc biopython by @WillForan in https://github.com/neurostuff/NiMARE/pull/753
- Raise deprecation warnings with Python 3.6 and 3.7 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/754
- [MAINT] Fix various errors due to major version changes in dependencies by @jdkent in https://github.com/neurostuff/NiMARE/pull/757

### New Contributors

- @PTDZ made their first contribution in https://github.com/neurostuff/NiMARE/pull/743
- @liuzhenqi77 made their first contribution in https://github.com/neurostuff/NiMARE/pull/745
- @yifan0330 made their first contribution in https://github.com/neurostuff/NiMARE/pull/749
- @ghisvail made their first contribution in https://github.com/neurostuff/NiMARE/pull/751
- @WillForan made their first contribution in https://github.com/neurostuff/NiMARE/pull/753

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.12...0.0.13

## [0.0.12](https://github.com/neurostuff/NiMARE/compare/0.0.11...0.0.12) - 2022-07-22

This release continues ongoing work on improving memory usage. We have eliminated the `memory_limit` option in our Estimators in favor of using sparse arrays. We expect to see a corresponding increase in fit times, especially for Monte Carlo FWE correction- however, we plan to address this in future releases.

### What's Changed

#### 🛠 Breaking Changes

- Replace multiprocessing with joblib for parallelization and change n_cores default to 1 by @tsalo in https://github.com/neurostuff/NiMARE/pull/597
- Incorporate joblib into ALESubtraction and fix SCALE docstring by @tsalo in https://github.com/neurostuff/NiMARE/pull/641
- Stop storing MetaResults as attributes of fitted Estimators by @tsalo in https://github.com/neurostuff/NiMARE/pull/657
- Refactor Correctors and remove statsmodels requirement by @tsalo in https://github.com/neurostuff/NiMARE/pull/679

#### 🎉 Exciting New Features

- Add FocusCounter diagnostic tool by @tsalo in https://github.com/neurostuff/NiMARE/pull/649
- Support cluster-level Monte Carlo FWE correction in the MKDAChi2 Estimator by @tsalo in https://github.com/neurostuff/NiMARE/pull/650
- Support `vfwe_only` in CBMAEstimator even when `null_method` isn't `montecarlo` by @tsalo in https://github.com/neurostuff/NiMARE/pull/678
- Add warning when coordinates dataset contains both positive and negative z_stats by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/699
- Add parameter estimate standard error to IBMA results by @tsalo in https://github.com/neurostuff/NiMARE/pull/691
- Use sparse array in ALE, ALESubtraction, SCALE, KDA, and MKDADensity by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/725

#### 🐛 Bug Fixes

- Retain updated Estimator in Corrector-generated MetaResults by @tsalo in https://github.com/neurostuff/NiMARE/pull/633
- Do not inherit IBMAEstimator's aggressive_mask from previous Datasets by @tsalo in https://github.com/neurostuff/NiMARE/pull/652
- Use beta maps in PermutedOLS instead of z maps by @tsalo in https://github.com/neurostuff/NiMARE/pull/715

#### Other Changes

- Reduce SCALE memory usage by @tsalo in https://github.com/neurostuff/NiMARE/pull/632
- Improve memory management in MKDAChi2 Estimator by @tsalo in https://github.com/neurostuff/NiMARE/pull/638
- Remove Peaks2Maps-related tests by @tsalo in https://github.com/neurostuff/NiMARE/pull/643
- Disable MA map pre-generation in CorrelationDecoder by @tsalo in https://github.com/neurostuff/NiMARE/pull/637
- Switch testing from CircleCI to GitHub Actions by @tsalo in https://github.com/neurostuff/NiMARE/pull/642
- Override unusable methods and improve documentation by @tsalo in https://github.com/neurostuff/NiMARE/pull/645
- Document other meta-analysis tools outside our ecosystem by @tsalo in https://github.com/neurostuff/NiMARE/pull/654
- Reorganize and streamline examples by @tsalo in https://github.com/neurostuff/NiMARE/pull/656
- Convert CBMAEstimator method to function by @tsalo in https://github.com/neurostuff/NiMARE/pull/658
- Add explicit support for Python 3.10 by @tsalo in https://github.com/neurostuff/NiMARE/pull/648
- Use BibTeX citations in documentation by @tsalo in https://github.com/neurostuff/NiMARE/pull/670
- Replace relative imports with absolute ones by @tsalo in https://github.com/neurostuff/NiMARE/pull/674
- Simplify organization of base classes by @tsalo in https://github.com/neurostuff/NiMARE/pull/675
- Note why we don't implement TFCE in NiMARE (currently) by @tsalo in https://github.com/neurostuff/NiMARE/pull/680
- Dropping the memory-mapping option for Estimators and kernel transformers by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/681
- Optimize locating coordinates in convert_neurosynth_to_dataset by @ryanhammonds in https://github.com/neurostuff/NiMARE/pull/682
- Reduce memory usage of `KernelTransformer.transform` and `meta.utils.compute_kda_ma` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/676
- Generate automatic CHANGELOG from release note and add it to docs by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/684
- Add manual changelog to documentation by @tsalo in https://github.com/neurostuff/NiMARE/pull/635
- Automatically update `CHANGELOG.md` for prereleases as well by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/688
- Fix tag-name issue in update-changelog workflow by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/689
- Optimize numpy operations in MKDADensity Estimator and (M)KDAKernel by @adelavega in https://github.com/neurostuff/NiMARE/pull/685
- Add PAT to automatically commit release notes to `CHANGELOG.md` by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/695
- Fix CHANGELOG formatting issues by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/701
- Add citation information to documentation by @tsalo in https://github.com/neurostuff/NiMARE/pull/712
- Add a glossary page to the documentation by @tsalo in https://github.com/neurostuff/NiMARE/pull/706
- Remove extraneous `copy()` statements by @jdkent in https://github.com/neurostuff/NiMARE/pull/662
- Add information about maintaining NiMARE to developer's guide by @tsalo in https://github.com/neurostuff/NiMARE/pull/703
- Pin minimum version of pandas by @jdkent in https://github.com/neurostuff/NiMARE/pull/722

### New Contributors

- @ryanhammonds made their first contribution in https://github.com/neurostuff/NiMARE/pull/682
- @adelavega made their first contribution in https://github.com/neurostuff/NiMARE/pull/685

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.11...0.0.12

## [0.0.11](https://github.com/neurostuff/NiMARE/compare/0.0.10...0.0.11) - 2022-01-06

This release continues fixes and improvements we've made for the NiMARE manuscript.
Additionally, we are starting to dramatically refactor how NiMARE stores meta-analytic data,
with the end goals of (1) synchronizing NiMARE data storage with the NIMADS standard,
(2) internally representing data and results in a manner that is compatible with NeuroStore,
and (3) implementing a meta-analysis model specification that can be employed with both NiMARE and Neurosynth 2.0.

### What's Changed

#### 🛠 Breaking Changes

- Replace Nilearn templates with ones stored in resources folder and unset maximum Nilearn version by @tsalo in https://github.com/neurostuff/NiMARE/pull/621
- Implement cluster mass-based cluster-level Monte Carlo correction for CBMA algorithms by @tsalo in https://github.com/neurostuff/NiMARE/pull/609
- Use scikit-learn for LDAModel by @tsalo in https://github.com/neurostuff/NiMARE/pull/607

#### 🎉 Exciting New Features

- New diagnostics module, with post-meta-analysis Jackknife method by @tsalo in https://github.com/neurostuff/NiMARE/pull/592

#### 👎 Deprecations

- Flag peaks2maps for removal in 0.0.13/0.1.0 by @tsalo in https://github.com/neurostuff/NiMARE/pull/616

#### 🐛 Bug Fixes

- Only download group maps when creating dataset and raise error if no images are found for a contrast by @jdkent in https://github.com/neurostuff/NiMARE/pull/580
- Force maskers to be array images instead of proxy images by @tsalo in https://github.com/neurostuff/NiMARE/pull/588

#### Other Changes

- Add test steps and explicit support for Python 3.9 by @JulioAPeraza in https://github.com/neurostuff/NiMARE/pull/578
- Add major classes/functions to parent namespaces by @tsalo in https://github.com/neurostuff/NiMARE/pull/600
- Make non-user-facing utility functions semi-private and improve docs by @tsalo in https://github.com/neurostuff/NiMARE/pull/604
- Move files used by examples from tests to resources by @tsalo in https://github.com/neurostuff/NiMARE/pull/605
- Improve documentation of CBMA and add methods pages by @tsalo in https://github.com/neurostuff/NiMARE/pull/610
- Use tmpdir for memmap files instead of the NiMARE data directory by @tsalo in https://github.com/neurostuff/NiMARE/pull/599

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.10...0.0.11

## [0.0.10](https://github.com/neurostuff/NiMARE/compare/0.0.9...0.0.10) - 2021-10-06

The 0.0.10 release includes a number of bug fixes and improvements.
The two biggest changes are (1) support for fetching and using NeuroQuery's database, and
(2) a reorganization of how Neurosynth's database is fetched and converted.
Both databases now use a shared format, which separates coordinates, metadata, and
features into different files with a semi-standardized naming structure.
The fetching and conversion functions also now support accessing multiple vocabularies provided by these databases.

### What's Changed

#### 🔧 Breaking changes

- The fetching functions for the Neurosynth and NeuroQuery databases now have a `data_dir` parameter, instead of a `path` parameter.
- The Neurosynth and NeuroQuery databases are now in a very different format,
- so the fetching and conversion functions operate quite differently.
- The `low_memory` parameter employed in many NiMARE classes and functions has been replaced with `memory_limit`.

#### ✨ Enhancements

- Add new methods to fetch and convert the NeuroQuery database.
- Support the new format for the Neurosynth database.
- A new decoding method for regions of interest taken from Neurosynth: `ROIAssociationDecoder`.
- Custom `__repr__` methods in many NiMARE classes, so now printing the object will show relevant information.
- Reduce memory usage in CBMA Estimators.

#### 🐛 Bug fixes

- Extract relevant metadata in kernel transformers for Dataset-based transform calls.
- Reference selected features instead of initial features in CorrelationDecoder.
- Separate IJK calculation from coordinate space conversion.

#### Changes since last stable release

- [TST] Test minimum versions of dependencies (#567) @tsalo
- [FIX] Add pytest to docs dependencies (#572) @tsalo
- [ENH, DOC] Document and use data directory approach in fetching functions (#570) @tsalo
- [TST] Reorganize tests for speed (#571) @tsalo
- [ENH] Add NeuroQuery 6308 vocab to resources (#568) @tsalo
- [DOC] Improve documentation of decoders (#506) @JulioAPeraza
- [DOC] Fill out CBMA docstrings (#564) @tsalo
- [REF] Reduce memory in CBMA Estimators (#562) @tsalo
- [FIX] Drop NQ tfidf features from manifest (#561) @tsalo
- [FIX] Separate IJK calculation from coordinate space conversion (#556) @tsalo
- [FIX] Reference selected features instead of initial features in CorrelationDecoder (#560) @tsalo
- [DOC] Replace map and threshold for MACM example (#558) @tsalo
- [FIX, DOC] Use appropriate structure in Neurosynth download example (#554) @tsalo
- [DOC] Add admonition to about.rst linking to new site (#552) @tsalo
- [DOC] Update docstrings (#551) @tsalo
- [ENH] Support new format for Neurosynth and NeuroQuery data (#535) @tsalo
- [DOC] Update citation for Enge et al. (2021) (#549) @alexenge
- [FIX] Use resample=True in IBMA examples (#546) @tsalo
- [FIX] Extract relevant metadata in kernel transformers for Dataset-based transform calls (#548) @tsalo
- [DOC] Update ecosystem figure and documentation (#545) @tsalo
- [ENH] Do not apply IBMA methods to voxels with zeros or NaNs (#544) @tsalo
- [REF] Remove unused dependencies and unimplemented workflow (#541) @tsalo
- [DOC] Change napoleon settings (#540) @tsalo
- [ENH] Add ROI association decoder (#536) @tsalo
- [ENH] Add custom `__repr__` methods (#538) @tsalo
- [FIX] Update CircleCI config to fix recent bug (#537) @tsalo
- [ENH] Replace low_memory with memory_limit and reduce memory bottlenecks (#520) @tsalo

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.9...0.0.10

## [0.0.9](https://github.com/neurostuff/NiMARE/compare/0.0.8...0.0.9) - 2021-07-04

This release primarily improves testing and documentation, but there are a few new features as well.
The new features include (1) a new "merge" method for Datasets, to combine two Datasets into a new one,
(2) a new ImageTransformer class for generating new images from existing ones of different types,
(3) a new "inspect" method for Correctors, to make it easier for users to see what kinds of multiple comparisons correction
methods are available for a Corrector type and a specific MetaResult.

Special thanks to @alexenge for the new ALE subtraction and conjunction analysis example and to @JulioAPeraza for
enhancing NiMARE's docstrings with "versionadded" and "versionchanged" directives!

### What's Changed

- [TST, REF] Address pandas SettingWithCopyWarnings (#534) @tsalo
- [ENH] Accept multiple annotations files in Neurosynth converter (#531) @tsalo
- [ENH] Add Corrector.inspect class method (#530) @tsalo
- [TST] Cover cases where some studies are missing coordinates (#527) @tsalo
- [ENH] Add `drop_invalid` option to `Dataset.get()` and `Estimator.fit()` (#526) @tsalo
- [ENH] Support lists of targets in ImageTransformer (#518) @tsalo
- [DOC] Add `versionadded` and `versionchanged` directives to docstrings (#501) @JulioAPeraza
- [DOC] Add example for ALE subtraction and conjunction analysis (#519) @alexenge
- [ENH] Add merge method to Dataset class (#517) @tsalo
- [ENH] Add ImageTransformer class (#513) @tsalo
- [ENH] Add overwrite option to transform_images (#509) @tsalo

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.8...0.0.9

## [0.0.8](https://github.com/neurostuff/NiMARE/compare/0.0.7...0.0.8) - 2021-05-17

This release includes a number of bug-fixes, along with enhancements to how many tools within NiMARE implement low-memory options.
In addition, we have renamed the CBMA estimators' null methods.
The "analytic" method is now "approximate" and the "empirical" method is now "montecarlo".

.. warning:: Known Bugs This version contains some bugs that were identified after it was released.
\   - The ALESubtraction class from this release should not be used, as it uses a symmetric null distribution, which does not work properly for comparisons between Datasets with different sizes.

### What's Changed

- [REF] Rename CBMA null distribution generation methods (#494) @tsalo
- [FIX] Add informative error when NeuroVault collection is not found (#500) @tsalo
- [ENH] Support symmetric GCLDA topics with more than two subregions (#499) @tsalo
- [DOC] Add sphinx-copybutton to docs requirements (#502) @tsalo
- [ENH] Incorporate information about valid masking approaches into IBMA Estimators (#495) @tsalo
- [FIX] Deal with extreme t-values in t_to_z by truncating associated p-values (#498) @tsalo
- [TST] Add flake8-isort to test dependencies (#493) @tsalo
- [REF] Miscellaneous GCLDA cleanup (#486) @tsalo
- [DOC] Add new functions and classes to API documentation (#490) @tsalo
- [ENH] add images_to_coordinates (#446) @jdkent
- [ENH] Add check_type function (#480) @tsalo
- [REF] Add low_memory option to Estimators and add function for moving metadata from Dataset to DataFrame (#476) @tsalo
- [FIX] Set Dataset.basepath using absolute path (#474) @tsalo
- [FIX] Find common stem in find_stem instead of largest common substring (#472) @tsalo
- [FIX] Replace misspelled "log_p" with "logp" (#468) @tsalo
- [FIX] Assume non-symmetric null distribution in ALESubtraction (#464) @tsalo
- [TST] Add memmap test. (#463) @jdkent
- [REF] Write temporary files to the NiMARE data directory (#460) @tsalo
- [REF] Use saved MA maps, when available, in CBMA estimators (#462) @tsalo
- [FIX] Neurovault name collisions (#457) @jdkent
- [FIX] Update niftimasker in dataset blob (#459) @jdkent
- [FIX] Add work-around for maskers that do not accept 1D input (#455) @jdkent
- [ENH] Add low-memory option for kernel transformers (#453) @tsalo
- [ENH] add function to convert neurovault collections to a NiMARE dataset (#432) @jdkent
- [FIX] Ensure IBMA results have the expected number of dimensions (#450) @jdkent
- [STY, TST] Add flake8-docstrings to requirements (#435) @tsalo

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.7...0.0.8

## [0.0.7](https://github.com/neurostuff/NiMARE/compare/0.0.6...0.0.7) - 2021-02-25

This release involves two changes worth mentioning.
First, we have fixed a bug in how permutation-based p-values are calculated (thanks to @alexenge for identifying and reporting).
Second, we have changed how the "empirical" null method is performed.
The "empirical" method is now much slower, but more accurate, than the "analytic" approach.

This release successfully deployed to PyPi, unlike 0.0.6.

### What's Changed

- [FIX] Permutation p-values (#447) @tyarkoni
- [FIX,REF] start changing how to handle resampling (#439) @jdkent
- [FIX] transform_images extra dimension (#445) @jdkent
- [DOC] Add decoding description page (#443) @tsalo
- [MAINT] Switch to GitHub Actions for PyPi deployment (#441) @tsalo
- [ENH] Implement full coordinate-set empirical null method  (#424) @tsalo
- [DOC] Fix NeuroStars link (#434) @tsalo
- [DOC] Add specialized issue templates (#433) @tsalo
- [MAINT] Add indexed_gzip as a dependency (#431) @tsalo

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.6...0.0.7

## [0.0.6](https://github.com/neurostuff/NiMARE/compare/0.0.5...0.0.6) - 2021-02-25

.. important:: \ This release was not deployed to PyPi. However, 0.0.7 is the same as 0.0.6, so just use that one.

This release involves two changes worth mentioning.
First, we have fixed a bug in how permutation-based p-values are calculated (thanks to @alexenge for identifying and reporting).
Second, we have changed how the "empirical" null method is performed.
The "empirical" method is now much slower, but more accurate, than the "analytic" approach.

### What's Changed

- [FIX] Permutation p-values (#447) @tyarkoni
- [FIX,REF] start changing how to handle resampling (#439) @jdkent
- [FIX] transform_images extra dimension (#445) @jdkent
- [DOC] Add decoding description page (#443) @tsalo
- [MAINT] Switch to GitHub Actions for PyPi deployment (#441) @tsalo
- [ENH] Implement full coordinate-set empirical null method  (#424) @tsalo
- [DOC] Fix NeuroStars link (#434) @tsalo
- [DOC] Add specialized issue templates (#433) @tsalo
- [MAINT] Add indexed_gzip as a dependency (#431) @tsalo

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.5...0.0.6

## [0.0.5](https://github.com/neurostuff/NiMARE/compare/0.0.4...0.0.5) - 2020-12-31

This release is focused on fixing two bugs in v0.0.4.
One bug affected which files were packaged with the library, such that some templates were missing.
The other bug was introduced in v0.0.4 and invalidates cluster-level Monte Carlo-based FWE-correction in coordinate-based meta-analyses.

### What's Changed

- [FIX] Convert histogram weights to null distribution in p-to-stat conversion (#430) @tsalo
- [FIX] Fix packaging of resources (#428) @tsalo
- [FIX] Include resources in library data files (#427) @tsalo

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.4...0.0.5

## [0.0.4](https://github.com/neurostuff/NiMARE/compare/0.0.3...0.0.4) - 2020-12-28

This release includes a number of substantial changes to `NiMARE`.

### What's Changed

#### Major changes

1. We've added PyMARE as a dependency! PyMARE is a general-purpose meta-analysis library in Python that we now use to perform our image-based meta-analyses.
2. For image-based meta-analyses, we also now have a transforms module to calculate new image types from available data.
3. Datasets now have a number of attributes retained as properties, which will break compatibility with Datasets from older versions of NiMARE.
4. We now have multiple methods for converting summary statistics (e.g., ALE, OF) to p-values in all of our major CBMA algorithms, thanks to @tyarkoni! The two current methods for each algorithm are a fast, but slightly less accurate, "analytic" method and a slower, but more accurate, "empirical" method. For ALE, We generally recommend the "analytic" method for maximum compatibility with GingerALE. The implementations of these algorithms have also been streamlined and sped up somewhat.
5. We have a new generate module for simulating coordinate-based datasets, thanks to @jdkent!
6. A number of modules, classes, and functions that were not yet implemented have been pruned from the API to make it easier to work with. Don't worry, we're still planning to get around to them at some point.

### All changes

- [FIX] Fix the warnings about mismatched kernels and estimators (#425) @tsalo
- [FIX] Add nullhist_to_p and crop invalid p-values (#409) @tsalo
- [TST] Do not download test peaks2maps to tmpdir (#419) @tsalo
- [FIX] Restructure Peaks2MapsKernel to operate like other kernels (#410) @tsalo
- [ENH] Improve convergence between ALE null methods (#411) @tsalo
- [DOC] Add warnings for CBMA kernel/estimator mismatch (#416) @tsalo
- [FIX] Remove rows with empty abstract before running LDAModel (#414) @JulioAPeraza
- [FIX] Sort all arrays and DataFrames in Dataset by ID (#402) @tsalo
- [FIX] Allow no coordinates in a dataset (#407) @jdkent
- [ENH] Add analytic null method to KDA estimator (#397) @tsalo
- [FIX] Use unzipped mask as temporary fix (#401) @tsalo
- [DOC] Update API and examples (#395) @tsalo
- [REF] CBMA re-organization and improvement (#393) @tyarkoni
- [MAINT] Pin to PyMARE 0.0.2 (#391) @tsalo
- [TST] Test both analytic and empirical methods in ALE and MKDA (#380) @jdkent
- [FIX] Change default seed to None (#392) @jdkent
- [PERF] Various performance improvements (#386) @tyarkoni
- Add performance tweaks to ALE analytical null generation (#390) @tyarkoni
- fix tests (#387) @tyarkoni
- [FIX] respect n_noise_foci value (#382) @jdkent
- [ENH] Add analytic null method to MKDADensity (#375) @tsalo
- [ENH] Add empirical null method to density-based CBMA Estimators (#372) @tsalo
- [REF] Refactor KernelTransformer hierarchy (#369) @tyarkoni
- [ENH] Add generate module (#343) @jdkent
- [FIX] enforce correct lowest p-value (#365) @jdkent
- [FIX] Treat vfwe as an array of floats for KDA (#362) @jdkent
- [DOC] Update roadmap.rst (#359) @tsalo
- [DOC] Add example of combining kernels and CBMA estimators (#346) @koudyk
- [MAINT] Add Dorota Jarecka to Zenodo file (#358) @djarecka
- [MAINT] Add Enrico Glerean's affiliation and ORCID (#357) @eglerean
- [ENH] Clip p-values based on number of permutations (#353) @tsalo
- [REF] Remove unused alpha argument in statsmodels call (#354) @tsalo
- [ENH] Replace TTest with PermutedOLS (#304) @tsalo
- [REF] Reduce dependencies (#345) @tsalo
- [ENH] Add Neurosynth data fetcher (#342) @tsalo
- [INFRA] Add json describing filename convention (#338) @tsalo
- [DOC] Enable CBMA example (#337) @tsalo
- [FIX] Add private setter method for Dataset.ids (#336) @tsalo
- [REF] More low-memory work (#334) @tsalo
- [FIX, DOC] Change natural log to base-ten and document output naming convention (#333) @tsalo
- [FIX] Pin setuptools again (#331) @tsalo
- [FIX] Update setuptools version (#330) @tsalo
- [FIX] Add setuptools to requirements (#329) @tsalo
- [TST] Add test for peaks2maps (#328) @tsalo
- [FIX, TST] Fix and test CorrelationDistributionDecoder (#327) @tsalo
- [TST] Use temporary directories with automatic teardown (#326) @tsalo
- [REF] Speed up CorrelationDecoder (#324) @tsalo
- [ENH] Support Dataset transformations in kernel transformers (#320) @tsalo
- [ENH] Add PairwiseCBMAEstimator class and add low_memory option to ALESubtraction (#319) @tsalo
- [TST] Improve meta-analysis tests (#318) @tsalo
- [DOC] Fix Lancaster xform and Sleuth conversion docstrings (#317) @tsalo
- [TST] Improve nimare.io test coverage (#314) @tsalo
- [REF] Reduce duplication by calling _check_ncores (#313) @tsalo
- [REF] Remove generate_cooccurrence (#312) @tsalo
- [REF] Operate on arrays in ALESubtraction (#311) @tsalo
- [TST] Add flake8-black to test requirements (#300) @akimbler
- [FIX] Support multiple header lines in Sleuth text files (#310) @tsalo
- [FIX] Operate on copy of df in extract_cogat() (#306) @tsalo
- [MAINT] Update setup configuration (#303) @tsalo
- [REF] Sort imports alphabetically (#299) @tsalo
- [REF] Run automated code formatting with black (#296) @tsalo
- [DOC] Remove whitespace from README (#295) @tsalo
- [MAINT, TST] Drop 3.5 support. Add tests for Python 3.7 and 3.8. (#293) @tsalo
- [MAINT] Delete unused files (#291) @tsalo
- [MAINT] Increase minimum tensorflow to 2.0.0 (#290) @tsalo
- [FIX] Update peaks2maps w.r.t. recent changes in the API (#287) @tsalo
- [FIX] Raise an error in Decoders if no features remain (#284) @tsalo
- [REF] Move CBMA methods up a level (#283) @tsalo
- [REF] Rename RandomEffectsGLM to TTest (#282) @tsalo
- [ENH] Split DerSimonianLaird and Hedges IBMA estimators (#281) @tsalo
- [DOC] Expand IBMA example (#280) @tsalo
- [ENH] Use PyMARE for image-based meta-analyses (#273) @tsalo
- [FIX] Replace NaNs in Datasets with Nones (#276) @tsalo
- [ENH] Support initialized and uninitialized kernels for CBMA (#275) @tsalo
- [ENH] Add functions to convert image types (#272) @tsalo
- [REF] Convert Dataset attributes to properties (#270) @tsalo
- [REF] Drop unimplemented annotators (#269) @tsalo
- [REF] Drop unimplemented parcellate module and meta-ICA workflow (#264) @tsalo
- [ENH] Use nearest-neighbor interpolation for masks (#258) @tsalo

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.3...0.0.4

## [0.0.3](https://github.com/neurostuff/NiMARE/compare/0.0.2...0.0.3) - 2020-07-12

This release consolidates changes prior to PyMARE integration.
In addition to a number of bug fixes, this release also includes substantial changes to
ALESubtraction, annotation storage, and Dataset size.

### What's Changed

- [FIX] Preallocate ALE cFWE p-value array with ones instead of zeros (#254) @tsalo
- [ENH] Convert decoders to classes (#252) @tsalo
- [ENH] Add prefix to annotations to delineate sources (#250) @tsalo
- [REF] Drop Dataset.data attribute (#249) @tsalo
- [ENH] Remove voxel selection in ALESubtraction (#245) @tsalo
- [REF] Eliminate duplication in ALESubtraction (#244) @tsalo
- [FIX] Add minimum nibabel version (#237) @tsalo
- [TST] Fix CodeCov config file (#240) @tsalo
- [ENH] Add transforms module (#239) @tsalo
- [TST] Add workflow tests (#235) @tsalo
- [REF, ENH] Add CBMAEstimator base class (#232) @tsalo
- [MAINT] Add @nicholst's info to Zenodo file (#228) @tsalo
- [MAINT] Consolidate requirements files (#230) @tsalo
- [ENH] Dataset.get_X methods return available types when type is not provided (#205) @tsalo
- [FIX] Update workflows given recent changes (#226) @tsalo
- [REF] Reorganize submodules (#225) @tsalo
- [REF, DOC] Update meta-analysis output map names (#224) @tsalo
- [REF] Rename "permutation" to "montecarlo" (#223) @tsalo
- [DOC] Fix annotate API docs (#222) @tsalo
- [DOC] Improve API rendering and run some examples (#219) @tsalo
- [REF, DOC] Remove unused base classes and improve docs (#216) @tsalo
- [REF] Rename kernel_estimator attribute to kernel_transformer (#197) @tsalo
- [ENH] Make convert_sleuth_to_dataset more flexible (#166) @62442katieb

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.2...0.0.3

## [0.0.2](https://github.com/neurostuff/NiMARE/compare/0.0.1...0.0.2) - 2020-05-11

### What's Changed

- [FIX] Mimic PyMARE's config more completely (#214)
- [FIX] Retain ALESubtraction results (#213)
- [TST] Add CI step for building docs (#212)
- [MAINT] Add workflow to autodeploy to PyPi (#210)
- [FIX] Fix bugs in LDAModel and paths to MALLET (#202)
- [DOC] Fix examples and add gallery back in (#201)
- [REF] Consolidate Dataset loading methods and refactor nimare.extract (#200)
- [ENH] Add extract submodule (again) (#199)
- [ENH] Improve Corrector transparency (#192)
- [TST] Expand test coverage and refactor LDA/CogAt (#193)
- [DOC] Update API documentation (#191)
- [FIX] Fix IO bug (#190)
- [FIX] Fix Neurosynth conversion. (#188)
- [MAINT] Add badges to README and RTD (#187)
- [MAINT] Add PyPi badges
- [MAINT] Add Zenodo DOI badge

**Full Changelog**: https://github.com/neurostuff/NiMARE/compare/0.0.1...0.0.2

## 0.0.1 - 2019-11-20

This is NiMARE's initial release.
NiMARE is still alpha software, so the package may change dramatically from release to release and there may be bugs in the code.
In this first release, NiMARE supports a range of coordinate- and image-based meta-analytic algorithms,
ingestion of coordinate-based datasets from Sleuth and Neurosynth, dataset searching, a set of command-line workflows,
and some miscellaneous tools like functional decoding.

### What's Changed

- [MAINT] Add release drafter (#186) @tsalo
- [FIX] Update examples to prepare for release (#185) @tsalo
- [TST] Add test for FDRCorrector (#184) @tsalo
- [FIX] Fix missing parameter q of FDRCorrector (#182) @alexprz
- [REF] Distribute base classes into their associated modules (#177) @tsalo
- [FIX] Remove click from conf.py (#178) @tsalo
- [REF] Switch CLI from click to argparse (#169) @tsalo
- [FIX, STY] Fix flake8 issues (#176) @tsalo
- [REF] changes to IBMAEstimator hierarchy and masking functionality (#173) @tyarkoni
- [REF] Save pain test dataset to local directory (#174) @tyarkoni
- [FIX, DOC] Add m2r to docs-specific requirements file (#168) @tsalo
- [FIX, DOC] Add m2r to doc requirements (#167) @tsalo
- [DOC, REF] Reorganize docs and setup configuration (#165) @tsalo
- [DOC, REF] Add references submodule (#164) @tsalo
- [FIX, DOC] Improve docstrings and update Peaks2Maps (#162) @tsalo
- [REF] Separate multiple comparisons correction from estimators (#157) @tsalo
- [ENH] Initialize Estimators with hyperparameters, fit to Datasets (#155) @tsalo
- [FIX] Fix MetaResult (#154) @tsalo
- [REF, DOC] Update docs and remove dataset.extract (#153) @tsalo
- [REF] Update MetaResult and move general IBMA functions into new module (#152) @tsalo
- [REF] Remove unused base classes (#151) @tsalo
- [TST] Improve CircleCI configuration (#150) @tsalo
- [FIX] Don't run gallery examples (#147) @tsalo
- [DOC] Update Manifest to work with examples. (#149) @tsalo
- [FIX, TST] Fix CircleCI (#148) @tsalo
- [FIX] Support multiline experiment names in Sleuth converter (#146) @tsalo
- [DOC] Add examples gallery (#144) @tsalo
- [DOC] Further update docstrings (#143) @tsalo
- [DOC] Update docstrings (#142) @tsalo
- [ENH] Use log-P values for FWE maps in CBMA estimators (#136) @tsalo
- [MAINT] Add Julio to Zenodo file (#141) @tsalo
- [ENH] Support additional space descriptors (#140) @tsalo
- [ENH] Add FWHM argument to ALE CLI workflow. (#138) @tsalo
- [MAINT] Remove download_test_data.py (#139) @tsalo
- [REF] Refactor GCLDA model (#137) @tsalo
- [FIX] Clean up examples after changes to API (#135) @tsalo
- [DOC] Change outdated Slack link to Mattermost (#127) @tsalo
- [FIX] Fix up GCLDA and text extraction methods (#126) @tsalo
- [DOC] Add ecosystem info to documentation (#125) @tsalo
- [ENH] Add texts field to Dataset (#124) @tsalo
- [FIX] Remove absolute paths in dataset (#122) @tsalo
- [ENH] Add slice method for Datasets (#121) @tsalo
- [FIX, ENH] Fix discrete decoding and speed up Dataset initialization (#120) @tsalo
- [FIX] Loop through requested IDs in kernel transformers (#118) @tsalo
- [FIX] Remove deprecated sklearn dependency (#119) @tsalo
- [DOC] Add favicon for website (#116) @tsalo
- [DOC] Add badges (#114) @tsalo
- [DOC] Update documentation and website (#112) @tsalo
- [ENH] Add MACM and Neurosynth/Sleuth conversion workflows (#111) @tsalo
- [STY, FIX] Fix style problems (#110) @tsalo
- [REF] Reorganize package to incorporate Transformers and Estimators (#107) @tsalo
- [TST] Add CodeCov and linting to CI (#100) @tsalo
- [ENH] Add ALE subtraction analysis to CLI (#106) @tsalo
- [DOC] Minor fixes to documentation. (#105) @tsalo
- [FIX] Assign transform to ALE MNI template (#104) @tsalo
- [FIX] Fix ALE subtraction analysis (#103) @tsalo
- [DOC] Draft boilerplates for ALE and SCALE workflows (#98) @tsalo
- [REF] Replace printing with logging (#99) @tsalo
- [MAINT] Add Dylan to zenodo (#102) @Shotgunosine
- [DOC] Add Puck Reeders to zenodo.json (#101) @puckr
- Update (#1) @puckr
- [WIP, DOC] populate contributors.rst with maintainer information (#54) @jdkent
- [ENH] Use maximum number of cores in ALE/MKDA by default (#93) @tsalo
- [DOC] Add ImportWarnings to untested modules (#90) @tsalo
- [ENH] Add CI (circleci) (#62) @jdkent
- [FIX] Fix docs API and conf.py (#94) @tsalo
- Fix a typo (#91) @chrisgorgo
- Update, test `scale` cli. (#89) @62442katieb
- Adding workflows for `metacluster` and `scale` (#84) @62442katieb
- skeleton of MC correction hierarchy (#87) @tyarkoni
- ale CLI cleanup (#86) @chrisgorgo
- Updating the documentation of the website and logo? (#88) @JesseyWright
- Small speedups in kernel transforms (#85) @Shotgunosine
- adding the `peaks2maps` CLI command (#83) @chrisgorgo
- adding the `conperm` CLI command (#80) @chrisgorgo
- [ENH] Speed up ALE MA generation (#81) @tsalo
- ADD when ALE is called with 1 core, don't use a pool (#79) @Shotgunosine
- fix: suppress all warnings for now (#75) @satra
- [HOTFIX] Feed IBMA MetaResults function names instead of estimators (#76) @tsalo
- updating Dockerfile and create_dockerfiles  (#69) @djarecka
- store originating estimator in MetaResult (#67) @tyarkoni
- Website Template Updates (#68) @JesseyWright
- Update .zenodo.json (#71) @bilgelm
- fix dep (#65) @chrisgorgo
- Adding command line interface (#63) @chrisgorgo
- changes in setup and requirements (#61) @djarecka
- Adding progress bars for permutations (#60) @chrisgorgo
- lazy load tensorflow, improve error reporting (#55) @chrisgorgo
- convert_sleuth_to_database (#56) @chrisgorgo
- [FIX] save_nidm_to_dset.ipynb (#51) @jdkent
- [FIX] Delete extra base class for parcellators (#52) @tsalo
- Added Peaks2MapsKernel (#42) @chrisgorgo
- [FIX] generate_ma_maps.ipynb (#50) @jdkent
- Add Angie to Zenodo file (#48) @tsalo
- Fix license ID (#47) @chrisgorgo
- [ENH] populate notebook with good images (#46) @jdkent
- [FIX] attempt to get nidm_pain_meta-analyses.ipynb working (#35) @jdkent
- quick fix for sphinx docs generation (#39) @bilgelm
- [ENH] Add Zenodo file and instructions to contributing guidelines (#38) @tsalo
- Small typo under example notebooks. (#29) @eglerean
- [FIX] remove abc as dependency (#32) @jdkent
- [FIX] Fix tests (#33) @tsalo
- [ENH] Add ability to convert Sleuth text files to NiMARE-compatible json files (#28) @tsalo
- [ENH] Add automated annotation tools (#18) @tsalo
- [ENH] Add decoders (#17) @tsalo
- [DOC, TST] Add tests and docs (#15) @tsalo
- [FIX] Correct subpeaks in pain dataset and use nilearn for templates/masks (#26) @tsalo
- Decoding (#4) @tsalo
- Annotation (#3) @tsalo
- Add ni18 poster (#2) @tsalo
- Add contributing guidelines and code of conduct (#14) @tsalo
- Add tail argument to IBMAs and use FSL for FFX GLM (#12) @tsalo
- Add image- and coordinate-based meta-analyses. (#11) @tsalo
