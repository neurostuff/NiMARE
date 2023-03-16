"""Workflow for running an coordinates-based meta-analysis from a NiMARE database."""
import itertools
import logging
import os.path as op

from nimare.correct import FWECorrector
from nimare.diagnostics import Jackknife
from nimare.meta import ALE

LGR = logging.getLogger(__name__)


def cbma_workflow(
    dataset,
    meta_estimator=ALE(null_method="approximate"),
    corrector=FWECorrector(),
    diagnostics=(Jackknife(),),
    output_dir=None,
):
    """Compose a coordinate-based meta-analysis workflow.

    .. versionadded:: 0.0.14

    This workflow performs a coordinate-based meta-analysis, multiple comparison correction,
    and diagnostics analyses on corrected meta-analytic maps.

    Parameters
    ----------
    dataset : :obj:`~nimare.dataset.Dataset`
        Dataset for which to run meta-analyses to generate maps.
    meta_estimator : :class:`~nimare.base.CBMAEstimator`, optional
        Meta-analysis estimator. Default is :class:`~nimare.meta.cbma.ale.ALE`.
    meta_corrector : :class:`~nimare.correct.Corrector`, optional
        Meta-analysis corrector. Default is :class:`~nimare.correct.FWECorrector`.
    diagnostics : :obj:`tuple`, optional
        Tuple of meta-analysis diagnostic classes.
        Default is :class:`~nimare.diagnostics.FocusCounter`.
    output_dir : :obj:`str`, optional
        Output directory in which to save results. If the directory doesn't
        exist, it will be created. Default is None (the results are not saved).

    Returns
    -------
    :obj:`~nimare.results.MetaResult`
        Results of Estimator and Corrector fitting with cluster and diagnostic tables.
    """
    LGR.info("Performing meta-analysis...")
    results = meta_estimator.fit(dataset)

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
            LGR.warn(
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
