"""
Workflow for running a SCALE meta-analysis from a Sleuth text file.
"""
import os
import logging
import pathlib
from shutil import copyfile

import numpy as np

from ..dataset import Dataset
from ..io import convert_sleuth_to_dataset
from ..meta.cbma.ale import SCALE

LGR = logging.getLogger(__name__)


def scale_workflow(dataset_file, baseline, output_dir=None, prefix=None,
                   n_iters=2500, v_thr=0.001):
    """
    Perform SCALE meta-analysis from Sleuth text file or NiMARE json file.

    Warnings
    --------
    This method is not yet implemented.
    """
    if dataset_file.endswith('.json'):
        dset = Dataset(dataset_file, target='mni152_2mm')
    if dataset_file.endswith('.txt'):
        dset = convert_sleuth_to_dataset(dataset_file, target='mni152_2mm')

    boilerplate = """
A specific coactivation likelihood estimation (SCALE; Langner et al., 2014)
meta-analysis was performed using NiMARE. The input dataset included {n}
studies/experiments.

Voxel-specific null distributions were generated using base rates from {bl}
with {n_iters} iterations. Results were thresholded at p < {thr}.

References
----------
- Langner, R., Rottschy, C., Laird, A. R., Fox, P. T., & Eickhoff, S. B. (2014).
Meta-analytic connectivity modeling revisited: controlling for activation base
rates. NeuroImage, 99, 559-570.
    """
    boilerplate = boilerplate.format(
        n=len(dset.ids),
        thr=v_thr,
        bl=baseline if baseline else 'a gray matter template',
        n_iters=n_iters)

    # At the moment, the baseline file should be an n_coords X 3 list of matrix
    # indices matching the dataset template, where the base rate for a given
    # voxel is reflected by the number of times that voxel appears in the array
    if not baseline:
        ijk = np.vstack(np.where(dset.mask.get_data())).T
    else:
        ijk = np.loadtxt(baseline)

    estimator = SCALE(dset, ijk=ijk, n_iters=n_iters)
    estimator.fit(dset.ids, voxel_thresh=v_thr, n_iters=n_iters, n_cores=2)

    if output_dir is None:
        output_dir = os.path.dirname(dataset_file)
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(dataset_file)
        prefix, _ = os.path.splitext(base)
        prefix += '_'

    estimator.results.save_maps(output_dir=output_dir, prefix=prefix)
    copyfile(dataset_file, os.path.join(output_dir, prefix + 'input_coordinates.txt'))

    LGR.info('Workflow completed.')
    LGR.info(boilerplate)
