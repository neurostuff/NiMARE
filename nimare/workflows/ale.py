import os
import pathlib
from shutil import copyfile

import click
from ..dataset.extract import convert_sleuth_to_database
from ..meta.cbma import ALE

n_iters_default = 10000
cluster_forming_threshold_p_default = 0.001
cluster_size_threshold_q_default = 0.05


@click.command(name='ale', short_help='permutation based metaanalysis of coordinates that uses '
                                      '3D Gaussians to model activation',
               help='Method for performing coordinate based meta analysis that uses a convolution'
                    'with a 3D Gaussian to model activation. Statistical inference is performed '
                    'using a permutation based approach with Family Wise Error multiple '
                    'comparison correction.')
@click.argument('sleuth_file')
@click.option('--output_dir', help="Where to put the output maps.")
@click.option('--output_prefix', help="Common prefix for output maps.")
@click.option('--n_iters', default=n_iters_default, show_default=True,
              help="Number of iterations for permutation testing.")
@click.option('--cluster_forming_threshold_p', default=cluster_forming_threshold_p_default,
              show_default=True, help="Voxel p-value threshold used to create clusters.")
@click.option('--cluster_size_threshold_q', default=cluster_size_threshold_q_default,
              show_default=True, help="Cluster size corrected p-value threshold.")
def ale_sleuth_inference(sleuth_file, output_dir=None, output_prefix=None,
                         n_iters=n_iters_default,
                         cluster_forming_threshold_p=cluster_forming_threshold_p_default,
                         cluster_size_threshold_q=cluster_size_threshold_q_default):
    click.echo("Loading coordinates...")
    dset = convert_sleuth_to_database(sleuth_file).get_dataset()
    ale = ALE(dset, ids=dset.ids)

    click.echo("Estimating the null distribution...")
    ale.fit(n_iters=n_iters, ids=dset.ids, voxel_thresh=cluster_forming_threshold_p,
            q=cluster_size_threshold_q, corr='FWE')

    if output_dir is None:
        output_dir = os.path.dirname(sleuth_file)
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        base = os.path.basename(sleuth_file)
        output_prefix, _ = os.path.splitext(base)
        output_prefix += '_'

    click.echo("Saving output maps...")
    ale.results.save_results(output_dir=output_dir, prefix=output_prefix)
    copyfile(sleuth_file, os.path.join(output_dir, output_prefix + 'input_coordinates.txt'))
