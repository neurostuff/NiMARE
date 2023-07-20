"""Workflow for running an ALE meta-analysis from a Sleuth text file."""
import logging
import os
import pathlib
from shutil import copyfile

import numpy as np

from nimare.correct import FWECorrector
from nimare.diagnostics import FocusCounter
from nimare.io import convert_sleuth_to_dataset
from nimare.meta import ALE, ALESubtraction

LGR = logging.getLogger(__name__)


def ale_sleuth_workflow(
    sleuth_file,
    sleuth_file2=None,
    output_dir=None,
    prefix=None,
    n_iters=10000,
    v_thr=0.001,
    fwhm=None,
    n_cores=1,
):
    """Perform ALE meta-analysis from Sleuth text file."""
    LGR.warning(
        "The ale_sleuth_workflow function is deprecated and will be removed in release 0.1.3. "
        "Use CBMAWorkflow or PairwiseCBMAWorkflow instead."
    )

    LGR.info("Loading coordinates...")

    if not sleuth_file2:
        # One ALE
        dset = convert_sleuth_to_dataset(sleuth_file, target="ale_2mm")
        n_subs = dset.get_metadata(field="sample_sizes")
        n_subs = np.sum(n_subs)

        ale = ALE(kernel__fwhm=fwhm)

        LGR.info("Performing meta-analysis...")
        results = ale.fit(dset)
        corr = FWECorrector(
            method="montecarlo",
            n_iters=n_iters,
            voxel_thresh=v_thr,
            n_cores=n_cores,
        )
        cres = corr.transform(results)
        fcounter = FocusCounter(
            target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
            voxel_thresh=None,
        )
        cres = fcounter.transform(cres)
        count_df = cres.tables[
            "z_desc-size_level-cluster_corr-FWE_method-montecarlo_diag-FocusCounter"
            "_tab-counts_tail-positive"
        ]
        boilerplate = cres.description_
        bibtex = cres.bibtex_

    else:
        # Two ALEs and an ALESubtraction
        dset1 = convert_sleuth_to_dataset(sleuth_file, target="ale_2mm")
        dset2 = convert_sleuth_to_dataset(sleuth_file2, target="ale_2mm")
        n_subs1 = dset1.get_metadata(field="sample_sizes")
        n_subs1 = np.sum(n_subs1)
        n_subs2 = dset2.get_metadata(field="sample_sizes")
        n_subs2 = np.sum(n_subs2)

        ale1 = ALE(kernel__fwhm=fwhm)
        ale2 = ALE(kernel__fwhm=fwhm)

        LGR.info("Performing meta-analyses...")
        res1 = ale1.fit(dset1)
        res2 = ale2.fit(dset2)
        corr = FWECorrector(
            method="montecarlo",
            n_iters=n_iters,
            voxel_thresh=v_thr,
            n_cores=n_cores,
        )
        cres1 = corr.transform(res1)
        boilerplate = cres1.description_

        fcounter = FocusCounter(
            target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
            voxel_thresh=None,
        )
        cres1 = fcounter.transform(cres1)
        count_df1 = cres1.tables[
            "z_desc-size_level-cluster_corr-FWE_method-montecarlo_diag-FocusCounter"
            "_tab-counts_tail-positive"
        ]

        cres2 = corr.transform(res2)
        boilerplate += "\n" + cres2.description_

        cres2 = fcounter.transform(cres2)
        count_df2 = cres2.tables[
            "z_desc-size_level-cluster_corr-FWE_method-montecarlo_diag-FocusCounter"
            "_tab-counts_tail-positive"
        ]

        sub = ALESubtraction(n_iters=n_iters, kernel__fwhm=fwhm)
        sres = sub.fit(dset1, dset2)
        boilerplate += "\n" + sres.description_

        # Inject the composite description into the ALESubtraction MetaResult to trigger
        # a re-compilation of references
        sres.description_ = boilerplate
        bibtex = sres.bibtex_  # This will now include references from all three analyses

    if output_dir is None:
        output_dir = os.path.abspath(os.path.dirname(sleuth_file))
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(sleuth_file)
        prefix, _ = os.path.splitext(base)
        prefix += "_"
    elif not prefix.endswith("_"):
        prefix = prefix + "_"

    LGR.info("Saving output maps...")
    if not sleuth_file2:
        cres.save_maps(output_dir=output_dir, prefix=prefix)
        count_df.to_csv(os.path.join(output_dir, prefix + "_clust.tsv"), index=False, sep="\t")
        copyfile(sleuth_file, os.path.join(output_dir, prefix + "input_coordinates.txt"))

    else:
        prefix1 = os.path.splitext(os.path.basename(sleuth_file))[0] + "_"
        prefix2 = os.path.splitext(os.path.basename(sleuth_file2))[0] + "_"
        prefix3 = prefix + "subtraction_"
        cres1.save_maps(output_dir=output_dir, prefix=prefix1)
        count_df1.to_csv(os.path.join(output_dir, prefix1 + "_clust.tsv"), index=False, sep="\t")
        cres2.save_maps(output_dir=output_dir, prefix=prefix2)
        count_df2.to_csv(os.path.join(output_dir, prefix2 + "_clust.tsv"), index=False, sep="\t")
        sres.save_maps(output_dir=output_dir, prefix=prefix3)
        copyfile(sleuth_file, os.path.join(output_dir, prefix + "group1_input_coordinates.txt"))
        copyfile(sleuth_file2, os.path.join(output_dir, prefix + "group2_input_coordinates.txt"))

    with open(os.path.join(output_dir, prefix + "boilerplate.txt"), "w") as fo:
        fo.write(boilerplate)

    with open(os.path.join(output_dir, prefix + "references.bib"), "w") as fo:
        fo.write(bibtex)

    LGR.info("Workflow completed.")
