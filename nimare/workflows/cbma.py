"""Workflow for running an coordinates-based meta-analysis from a NiMARE database."""
import itertools
import logging
import os.path as op

from nimare.correct import Corrector, FDRCorrector, FWECorrector
from nimare.dataset import Dataset
from nimare.diagnostics import Diagnostics, FocusCounter, Jackknife
from nimare.meta import ALE, KDA, SCALE, MKDADensity
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.transforms import p_to_z
from nimare.utils import _check_ncores, _check_type

LGR = logging.getLogger(__name__)


def _str_to_class(str_name):
    """Match a string to a class name without initializing the class."""
    classes = {
        "ale": ALE,
        "scale": SCALE,
        "mkdadensity": MKDADensity,
        "kda": KDA,
        "montecarlo": FWECorrector,
        "fdr": FDRCorrector,
        "bonferroni": FWECorrector,
        "jackknife": Jackknife,
        "focuscounter": FocusCounter,
    }
    return classes[str_name]


def _check_input(obj, clss, options, **kwargs):
    """Check input for workflow functions."""
    if isinstance(obj, str):
        if obj not in options:
            raise ValueError(f'"estimator" of kind string must be {", ".join(options)}')

        # Get the class from the string
        obj_str = obj
        obj = _str_to_class(obj_str)

        # Add the method to the kwargs if it's a FWECorrector
        if obj == FWECorrector:
            kwargs["method"] = obj_str

    return _check_type(obj, clss, **kwargs)


def cbma_workflow(
    dataset,
    estimator=None,
    corrector=None,
    diagnostics=None,
    output_dir=None,
    n_cores=1,
):
    """Compose a coordinate-based meta-analysis workflow.

    .. versionadded:: 0.0.14

    This workflow performs a coordinate-based meta-analysis, multiple comparison correction,
    and diagnostics analyses on corrected meta-analytic maps.

    Parameters
    ----------
    dataset : :obj:`~nimare.dataset.Dataset`
        Dataset for which to run meta-analyses to generate maps.
    estimator : :class:`~nimare.base.CBMAEstimator`, :obj:`str` {'ale', 'scale', 'mkdadensity',
    'kda'}, or optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.cbma.ale.ALE`.
    corrector : :class:`~nimare.correct.Corrector`, :obj:`str` {'montecarlo', 'fdr', 'bonferroni'}
    or optional
        Meta-analysis corrector. Default is :class:`~nimare.correct.FWECorrector`.
    diagnostics : :obj:`list` of :class:`~nimare.diagnostics.Diagnostics`,
    :class:`~nimare.diagnostics.Diagnostics`, :obj:`str` {'jackknife', 'focusCounter'}, or optional
        List of meta-analysis diagnostic classes. A single diagnostic class can also be passed.
        Default is :class:`~nimare.diagnostics.FocusCounter`.
    output_dir : :obj:`str`, optional
        Output directory in which to save results. If the directory doesn't
        exist, it will be created. Default is None (the results are not saved).
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        If estimator, corrector, or diagnostics are passed as initialized objects, this parameter
        will be ignored.
        Default is 1.

    Returns
    -------
    :obj:`~nimare.results.MetaResult`
        Results of Estimator and Corrector fitting with cluster and diagnostic tables.
    """
    n_cores = _check_ncores(n_cores)

    if not isinstance(diagnostics, list) and diagnostics is not None:
        diagnostics = [diagnostics]

    # Check dataset type
    dataset = _check_type(dataset, Dataset)

    # Options allows for string input
    estm_options = ("ale", "scale", "mkdadensity", "kda")
    corr_options = ("montecarlo", "fdr", "bonferroni")
    diag_options = ("jackknife", "focuscounter")

    # Check inputs and set defaults if input is None
    estimator = (
        ALE(n_cores=n_cores)
        if estimator is None
        else _check_input(estimator, CBMAEstimator, estm_options, n_cores=n_cores)
    )
    corrector = (
        FWECorrector(method="montecarlo", n_cores=n_cores)
        if corrector is None
        else _check_input(corrector, Corrector, corr_options, n_cores=n_cores)
    )

    pval = 0.05 if corrector.method == "montecarlo" else 0.001
    vthr = p_to_z(pval, tail="one")  # Set voxel_thresh for diagnostics
    if diagnostics is None:
        diagnostics = [Jackknife(voxel_thresh=vthr, n_cores=n_cores)]
    else:
        diagnostics = [
            _check_input(diagnostic, Diagnostics, diag_options, voxel_thresh=vthr, n_cores=n_cores)
            for diagnostic in diagnostics
        ]

    if isinstance(estimator, PairwiseCBMAEstimator):
        raise AttributeError(
            'The "cbma_workflow" function does not currently work with pairwise Estimators.'
        )

    LGR.info("Performing meta-analysis...")
    results = estimator.fit(dataset)

    LGR.info("Performing correction on meta-analysis...")
    corr_results = corrector.transform(results)

    LGR.info("Generating clusters tables and performing diagnostics on corrected meta-analyses...")
    img_keys = [
        img_key
        for img_key in corr_results.maps.keys()
        if img_key.startswith("z_") and ("_corr-" in img_key)
    ]
    for img_key, diagnostic in itertools.product(img_keys, diagnostics):
        diagnostic.target_image = img_key
        contribution_table, clusters_table, _ = diagnostic.transform(corr_results)

        diag_name = diagnostic.__class__.__name__
        clust_tbl_name = f"{img_key}_clust"
        count_tbl_name = f"{img_key}_{diag_name}"
        if not contribution_table.empty:
            corr_results.tables[clust_tbl_name] = clusters_table
            corr_results.tables[count_tbl_name] = contribution_table
        else:
            LGR.warning(
                f"Key {count_tbl_name} and {clust_tbl_name} will not be stored in "
                "MetaResult.tables dictionary."
            )

    if output_dir is not None:
        LGR.info(f"Saving meta-analytic maps, tables and boilerplate to {output_dir}...")
        corr_results.save_maps(output_dir=output_dir)
        corr_results.save_tables(output_dir=output_dir)

        boilerplate = corr_results.description_
        with open(op.join(output_dir, "boilerplate.txt"), "w") as fo:
            fo.write(boilerplate)

        bibtex = corr_results.bibtex_
        with open(op.join(output_dir, "references.bib"), "w") as fo:
            fo.write(bibtex)

    LGR.info("Workflow completed.")
    return corr_results
