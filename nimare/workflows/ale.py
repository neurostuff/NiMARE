"""
Workflow for running an ALE meta-analysis from a Sleuth text file.
"""
import os
import pathlib
from shutil import copyfile

import click
from ..dataset.extract import convert_sleuth_to_database
from ..meta.cbma import ALE

N_ITERS_DEFAULT = 10000
CLUSTER_FORMING_THRESHOLD_P_DEFAULT = 0.001
CLUSTER_SIZE_THRESHOLD_Q_DEFAULT = 0.05


@click.command(name='ale',
               short_help='Run activation likelihood estimation (ALE) on Sleuth text file. '
                          'A permutation-based meta-analysis of coordinates that uses '
                          '3D Gaussians to model activation.',
               help='Method for performing coordinate based meta analysis that uses a convolution'
                    'with a 3D Gaussian to model activation. Statistical inference is performed '
                    'using a permutation based approach with Family Wise Error multiple '
                    'comparison correction.')
@click.argument('sleuth_file', type=click.Path(exists=True))
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
def ale_sleuth_inference(sleuth_file, output_dir=None, prefix=None,
                         n_iters=N_ITERS_DEFAULT,
                         cluster_forming_threshold_p=CLUSTER_FORMING_THRESHOLD_P_DEFAULT,
                         cluster_size_threshold_q=CLUSTER_SIZE_THRESHOLD_Q_DEFAULT):
    """
    Perform ALE meta-analysis from Sleuth text file.
    """
    click.echo("Loading coordinates...")
    dset = convert_sleuth_to_database(sleuth_file).get_dataset()
    ale = ALE(dset, ids=dset.ids)

    click.echo("Estimating the null distribution...")
    ale.fit(n_iters=n_iters, ids=dset.ids,
            voxel_thresh=cluster_forming_threshold_p,
            q=cluster_size_threshold_q, corr='FWE')

    if output_dir is None:
        output_dir = os.path.dirname(sleuth_file)
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(sleuth_file)
        prefix, _ = os.path.splitext(base)
        prefix += '_'

    click.echo("Saving output maps...")
    ale.results.save_results(output_dir=output_dir, prefix=prefix)
    copyfile(sleuth_file, os.path.join(output_dir, prefix + 'input_coordinates.txt'))
