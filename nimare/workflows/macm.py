"""

"""
import os
import pathlib
from shutil import copyfile

import click
import numpy as np

from ..dataset import Dataset
from ..meta.cbma import ALE

N_ITERS_DEFAULT = 10000
CLUSTER_FORMING_THRESHOLD_P_DEFAULT = 0.001
CLUSTER_SIZE_THRESHOLD_Q_DEFAULT = 0.05


@click.command(name='macm',
               short_help='Run a meta-analytic coactivation modeling (MACM) '
                          'analysis using activation likelihood estimation '
                          '(ALE) on a NiMARE dataset file and a target mask.',
               help='Method for performing coordinate based meta analysis that uses a convolution'
                    'with a 3D Gaussian to model activation. Statistical inference is performed '
                    'using a permutation based approach with Family Wise Error multiple '
                    'comparison correction.')
@click.argument('dataset_file', type=click.Path(exists=True))
@click.option('--mask', '--mask_file', type=click.Path(exists=True))
@click.option('--output_dir', help="Where to put the output maps.")
@click.option('--prefix', help="Common prefix for output maps.")
@click.option('--n_iters', default=N_ITERS_DEFAULT, show_default=True,
              help="Number of iterations for permutation testing.")
@click.option('--v_thr', default=CLUSTER_FORMING_THRESHOLD_P_DEFAULT,
              show_default=True,
              help="Voxel p-value threshold used to create clusters.")
@click.option('--c_thr', default=CLUSTER_SIZE_THRESHOLD_Q_DEFAULT,
              show_default=True,
              help="Cluster size corrected p-value threshold.")
@click.option('--n_cores', default=-1,
              show_default=True,
              help="Number of processes to use for meta-analysis. If -1, use "
                   "all available cores.")
def macm_workflow(dataset_file, mask_file, output_dir=None,
                  prefix=None,
                  n_iters=N_ITERS_DEFAULT,
                  v_thr=CLUSTER_FORMING_THRESHOLD_P_DEFAULT,
                  c_thr=CLUSTER_SIZE_THRESHOLD_Q_DEFAULT,
                  n_cores=-1):
    dset = Dataset(dataset_file)
    sel_ids = dset.get_studies_by_mask(mask_file)
    unsel_ids = sorted(list(set(dset.ids) - set(sel_ids)))
    click.echo("{0} studies selected out of {1}.".format(len(sel_ids),
                                                         len(dset.ids)))

    ale = ALE(dset)
    ale.fit(n_iters=n_iters, ids=sel_ids, ids2=unsel_ids,
            voxel_thresh=v_thr, q=c_thr, corr='FWE',
            n_cores=n_cores)

    if output_dir is None:
        output_dir = os.path.abspath(os.path.dirname(dataset_file))
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(dataset_file)
        prefix, _ = os.path.splitext(base)
        prefix += '_'

    click.echo("Saving output maps...")
    ale.results.save_results(output_dir=output_dir, prefix=prefix)
    copyfile(dataset_file, os.path.join(output_dir, prefix + 'input_dataset.json'))
    click.echo("Workflow completed.")
