.. include:: links.rst

Changelog
=========

0.0.1: 2019-11-20
-----------------

This is NiMARE's initial release.
NiMARE is still alpha software, so the package may change dramatically from release to release and there may be bugs in the code.
In this first release, NiMARE supports a range of coordinate- and image-based meta-analytic algorithms,
ingestion of coordinate-based datasets from Sleuth and Neurosynth, dataset searching, a set of command-line workflows,
and some miscellaneous tools like functional decoding.

Changes
```````

* [MAINT] Add release drafter (#186) @tsalo
* [FIX] Update examples to prepare for release (#185) @tsalo
* [TST] Add test for FDRCorrector (#184) @tsalo
* [FIX] Fix missing parameter q of FDRCorrector (#182) @alexprz
* [REF] Distribute base classes into their associated modules (#177) @tsalo
* [FIX] Remove click from conf.py (#178) @tsalo
* [REF] Switch CLI from click to argparse (#169) @tsalo
* [FIX, STY] Fix flake8 issues (#176) @tsalo
* [REF] changes to IBMAEstimator hierarchy and masking functionality (#173) @tyarkoni
* [REF] Save pain test dataset to local directory (#174) @tyarkoni
* [FIX, DOC] Add m2r to docs-specific requirements file (#168) @tsalo
* [FIX, DOC] Add m2r to doc requirements (#167) @tsalo
* [DOC, REF] Reorganize docs and setup configuration (#165) @tsalo
* [DOC, REF] Add references submodule (#164) @tsalo
* [FIX, DOC] Improve docstrings and update Peaks2Maps (#162) @tsalo
* [REF] Separate multiple comparisons correction from estimators (#157) @tsalo
* [ENH] Initialize Estimators with hyperparameters, fit to Datasets (#155) @tsalo
* [FIX] Fix MetaResult (#154) @tsalo
* [REF, DOC] Update docs and remove dataset.extract (#153) @tsalo
* [REF] Update MetaResult and move general IBMA functions into new module (#152) @tsalo
* [REF] Remove unused base classes (#151) @tsalo
* [TST] Improve CircleCI configuration (#150) @tsalo
* [FIX] Don't run gallery examples (#147) @tsalo
* [DOC] Update Manifest to work with examples. (#149) @tsalo
* [FIX, TST] Fix CircleCI (#148) @tsalo
* [FIX] Support multiline experiment names in Sleuth converter (#146) @tsalo
* [DOC] Add examples gallery (#144) @tsalo
* [DOC] Further update docstrings (#143) @tsalo
* [DOC] Update docstrings (#142) @tsalo
* [ENH] Use log-P values for FWE maps in CBMA estimators (#136) @tsalo
* [MAINT] Add Julio to Zenodo file (#141) @tsalo
* [ENH] Support additional space descriptors (#140) @tsalo
* [ENH] Add FWHM argument to ALE CLI workflow. (#138) @tsalo
* [MAINT] Remove download_test_data.py (#139) @tsalo
* [REF] Refactor GCLDA model (#137) @tsalo
* [FIX] Clean up examples after changes to API (#135) @tsalo
* [DOC] Change outdated Slack link to Mattermost (#127) @tsalo
* [FIX] Fix up GCLDA and text extraction methods (#126) @tsalo
* [DOC] Add ecosystem info to documentation (#125) @tsalo
* [ENH] Add texts field to Dataset (#124) @tsalo
* [FIX] Remove absolute paths in dataset (#122) @tsalo
* [ENH] Add slice method for Datasets (#121) @tsalo
* [FIX, ENH] Fix discrete decoding and speed up Dataset initialization (#120) @tsalo
* [FIX] Loop through requested IDs in kernel transformers (#118) @tsalo
* [FIX] Remove deprecated sklearn dependency (#119) @tsalo
* [DOC] Add favicon for website (#116) @tsalo
* [DOC] Add badges (#114) @tsalo
* [DOC] Update documentation and website (#112) @tsalo
* [ENH] Add MACM and Neurosynth/Sleuth conversion workflows (#111) @tsalo
* [STY, FIX] Fix style problems (#110) @tsalo
* [REF] Reorganize package to incorporate Transformers and Estimators (#107) @tsalo
* [TST] Add CodeCov and linting to CI (#100) @tsalo
* [ENH] Add ALE subtraction analysis to CLI (#106) @tsalo
* [DOC] Minor fixes to documentation. (#105) @tsalo
* [FIX] Assign transform to ALE MNI template (#104) @tsalo
* [FIX] Fix ALE subtraction analysis (#103) @tsalo
* [DOC] Draft boilerplates for ALE and SCALE workflows (#98) @tsalo
* [REF] Replace printing with logging (#99) @tsalo
* [MAINT] Add Dylan to zenodo (#102) @Shotgunosine
* [DOC] Add Puck Reeders to zenodo.json (#101) @puckr
* Update (#1) @puckr
* [WIP, DOC] populate contributors.rst with maintainer information (#54) @jdkent
* [ENH] Use maximum number of cores in ALE/MKDA by default (#93) @tsalo
* [DOC] Add ImportWarnings to untested modules (#90) @tsalo
* [ENH] Add CI (circleci) (#62) @jdkent
* [FIX] Fix docs API and conf.py (#94) @tsalo
* Fix a typo (#91) @chrisgorgo
* Update, test `scale` cli. (#89) @62442katieb
* Adding workflows for `metacluster` and `scale` (#84) @62442katieb
* skeleton of MC correction hierarchy (#87) @tyarkoni
* ale CLI cleanup (#86) @chrisgorgo
* Updating the documentation of the website and logo? (#88) @JesseyWright
* Small speedups in kernel transforms (#85) @Shotgunosine
* adding the `peaks2maps` CLI command (#83) @chrisgorgo
* adding the `conperm` CLI command (#80) @chrisgorgo
* [ENH] Speed up ALE MA generation (#81) @tsalo
* ADD when ALE is called with 1 core, don't use a pool (#79) @Shotgunosine
* fix: suppress all warnings for now (#75) @satra
* [HOTFIX] Feed IBMA MetaResults function names instead of estimators (#76) @tsalo
* updating Dockerfile and create_dockerfiles  (#69) @djarecka
* store originating estimator in MetaResult (#67) @tyarkoni
* Website Template Updates (#68) @JesseyWright
* Update .zenodo.json (#71) @bilgelm
* fix dep (#65) @chrisgorgo
* Adding command line interface (#63) @chrisgorgo
* changes in setup and requirements (#61) @djarecka
* Adding progress bars for permutations (#60) @chrisgorgo
* lazy load tensorflow, improve error reporting (#55) @chrisgorgo
* convert_sleuth_to_database (#56) @chrisgorgo
* [FIX] save_nidm_to_dset.ipynb (#51) @jdkent
* [FIX] Delete extra base class for parcellators (#52) @tsalo
* Added Peaks2MapsKernel (#42) @chrisgorgo
* [FIX] generate_ma_maps.ipynb (#50) @jdkent
* Add Angie to Zenodo file (#48) @tsalo
* Fix license ID (#47) @chrisgorgo
* [ENH] populate notebook with good images (#46) @jdkent
* [FIX] attempt to get nidm_pain_meta-analyses.ipynb working (#35) @jdkent
* quick fix for sphinx docs generation (#39) @bilgelm
* [ENH] Add Zenodo file and instructions to contributing guidelines (#38) @tsalo
* Small typo under example notebooks. (#29) @eglerean
* [FIX] remove abc as dependency (#32) @jdkent
* [FIX] Fix tests (#33) @tsalo
* [ENH] Add ability to convert Sleuth text files to NiMARE-compatible json files (#28) @tsalo
* [ENH] Add automated annotation tools (#18) @tsalo
* [ENH] Add decoders (#17) @tsalo
* [DOC, TST] Add tests and docs (#15) @tsalo
* [FIX] Correct subpeaks in pain dataset and use nilearn for templates/masks (#26) @tsalo
* Decoding (#4) @tsalo
* Annotation (#3) @tsalo
* Add ni18 poster (#2) @tsalo
* Add contributing guidelines and code of conduct (#14) @tsalo
* Add tail argument to IBMAs and use FSL for FFX GLM (#12) @tsalo
* Add image- and coordinate-based meta-analyses. (#11) @tsalo
