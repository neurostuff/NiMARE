"""Workflow for running an coordinates-based meta-analysis from a NiMARE database."""
import logging

from nilearn import reporting

from nimare.correct import FWECorrector
from nimare.diagnostics import FocusCounter
from nimare.meta import ALE

LGR = logging.getLogger(__name__)


def cbma_workflow(
    dataset,
    meta_estimator=ALE(),
    corrector=FWECorrector(),
    diagnostics=(FocusCounter(),),
    output_dir=None,
):
    """Compose a coordinate-based meta-analysis workflow.

    .. versionadded:: 0.0.14

    This workflow performs a coordinate-based meta-analysis, multiple comparison correction,
    and diagnostics analyses on corrected meta-analytic images.

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

    for img_key in corr_results.maps.keys():
        # Save cluster table only for z maps
        if img_key.startswith("z_"):
            cbma_img = corr_results.get_map(img_key)

            LGR.info("Generating cluster tables...")
            cluster_df = reporting.get_clusters_table(cbma_img, stat_threshold=0)
            # Remove prefix from name
            img_name = "_".join(img_key.split("_")[1:])

            if not cluster_df.empty:
                clust_tbl_name = f"cluster_{img_name}"
                corr_results.tables[clust_tbl_name] = cluster_df

            # Run diagnostic on corrected z maps
            if "_corr-" in img_key:
                for diagnostic in diagnostics:
                    diagnostic.target_image = img_key

                    LGR.info("Performing diagnostic on corrected meta-analysis...")
                    count_df, _ = diagnostic.transform(corr_results)

                    if not count_df.empty:
                        diag_tbl_name = f"{diagnostic.__class__.__name__}_{img_name}"
                        corr_results.tables[diag_tbl_name] = count_df

    if output_dir is not None:
        LGR.info(f"Saving meta-result object to {output_dir}...")
        corr_results.save_maps(output_dir=output_dir)
        corr_results.save_tables(output_dir=output_dir)

    return corr_results
