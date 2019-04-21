"""
Workflow for running a SCALE meta-analysis from a Sleuth text file.
"""
import os
import pathlib
from shutil import copyfile

import click
import numpy as np

from ..dataset import Dataset
from ..io import convert_sleuth_to_dataset
from ..meta.cbma.ale import SCALE


N_ITERS_DEFAULT = 2500
CLUSTER_FORMING_THRESHOLD_P_DEFAULT = 0.001


@click.command(name='scale', short_help='permutation-based, modified MACM approach '
                                        'that takes activation frequency bias into '
                                        'account',
               help='Method for performing Specific CoActivation Likelihood Estimation (SCALE),'
                    'a modified meta-analytic coactivation modeling (MACM) that takes activation'
                    'frequency bias into account, for delineating distinct core networks of '
                    'coactivation, using a permutation based approach.')
@click.argument('database', required=True, type=click.Path(exists=True, readable=True))
@click.option('--baseline', type=click.Path(exists=True, readable=True),
              help='Voxelwise baseline activation rates.')
@click.option('--output_dir', required=True, type=click.Path(),
              help='Directory into which clustering results will be written.')
@click.option('--prefix', type=str, help='Common prefix for output SCALE results.')
@click.option('--n_iters', default=N_ITERS_DEFAULT, show_default=True, type=int,
              help='Number of iterations for permutation testing.')
@click.option('--v_thr', default=CLUSTER_FORMING_THRESHOLD_P_DEFAULT,
              show_default=True, help="Voxel p-value threshold used to create clusters.")
def scale_workflow(database, baseline, output_dir=None, prefix=None,
                   n_iters=N_ITERS_DEFAULT,
                   v_thr=CLUSTER_FORMING_THRESHOLD_P_DEFAULT):
    """
    Perform SCALE meta-analysis from Sleuth text file or NiMARE json file.

    Warnings
    --------
    This method is not yet implemented.
    """
    if database.endswith('.json'):
        dset = Dataset(database, target='mni152_2mm')
    if database.endswith('.txt'):
        dset = convert_sleuth_to_dataset(database, target='mni152_2mm')

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
        output_dir = os.path.dirname(database)
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(database)
        prefix, _ = os.path.splitext(base)
        prefix += '_'

    estimator.results.save_results(output_dir=output_dir, prefix=prefix)
    copyfile(database, os.path.join(output_dir, prefix + 'input_coordinates.txt'))

    click.echo("Workflow completed.")
    click.echo(boilerplate)
