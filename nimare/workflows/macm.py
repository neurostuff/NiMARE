"""Perform MACM with ALE algorithm."""

import logging
import os
import pathlib
from shutil import copyfile

from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.meta import ALE

LGR = logging.getLogger(__name__)


def macm_workflow(
    dataset_file,
    mask_file,
    output_dir=None,
    prefix=None,
    n_iters=5000,
    v_thr=0.001,
    n_cores=1,
):
    """Perform MACM with ALE algorithm."""
    LGR.info("Loading coordinates...")
    dset = Dataset(dataset_file)
    sel_ids = dset.get_studies_by_mask(mask_file)
    sel_dset = dset.slice(sel_ids)
    n_foci_db = dset.coordinates.shape[0]
    n_foci_sel = sel_dset.coordinates.shape[0]
    n_exps_db = len(dset.ids)
    n_exps_sel = len(sel_dset.ids)

    LGR.info("Performing meta-analysis...")
    ale = ALE()
    results = ale.fit(sel_dset)
    corr = FWECorrector(method="montecarlo", n_iters=n_iters, voxel_thresh=v_thr, n_cores=n_cores)
    cres = corr.transform(results)

    boilerplate = cres.description_

    boilerplate = (
        "A meta-analytic connectivity modeling (MACM; "
        "\\citealt{laird2009investigating,robinson2010metaanalytic,eickhoff2010anatomical}) "
        "analysis was performed. "
        f"The input dataset included {n_foci_db} foci across {n_exps_db} experiments, "
        "from which experiments were selected for analysis if they had at least one focus inside "
        "the target mask. "
        f"The resulting sample included {n_foci_sel} foci across {n_exps_sel} experiments. "
    ) + boilerplate

    # Inject the composite description into the ALESubtraction MetaResult to trigger
    # a re-compilation of references
    cres.description_ = boilerplate
    bibtex = cres.bibtex_  # This will now include references from all three analyses

    if output_dir is None:
        output_dir = os.path.abspath(os.path.dirname(dataset_file))
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(dataset_file)
        prefix, _ = os.path.splitext(base)
        prefix += "_"

    LGR.info("Saving output maps...")
    cres.save_maps(output_dir=output_dir, prefix=prefix)
    copyfile(dataset_file, os.path.join(output_dir, prefix + "input_dataset.json"))

    with open(os.path.join(output_dir, prefix + "boilerplate.txt"), "w") as fo:
        fo.write(boilerplate)

    with open(os.path.join(output_dir, prefix + "references.bib"), "w") as fo:
        fo.write(bibtex)

    LGR.info("Workflow completed.")
