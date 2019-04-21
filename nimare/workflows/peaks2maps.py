import os
import pathlib
import click
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from ..meta.ibma import rfx_glm
from ..meta.cbma import Peaks2MapsKernel
from ..io import convert_sleuth_to_dataset

n_iters_default = 10000


@click.command(name='peaks2maps',
               short_help='Permutation-based meta-analysis of coordinates '
                          'that uses deep learning to reconstruct the original '
                          'maps.',
               help='Method for performing coordinate-based meta-analysis that '
                    'uses a pretrained deep neural network to reconstruct '
                    'unthresholded maps from peak coordinates. The reconstructed '
                    'maps are evaluated for statistical significance using a '
                    'permutation-based approach with Family Wise Error multiple '
                    'comparison correction.')
@click.argument('sleuth_file', type=click.Path(exists=True))
@click.option('--output_dir', help="Where to put the output maps.")
@click.option('--output_prefix', help="Common prefix for output maps.")
@click.option('--n_iters', default=n_iters_default, show_default=True,
              help="Number of iterations for permutation testing.")
def peaks2maps_workflow(sleuth_file, output_dir=None, output_prefix=None,
                        n_iters=n_iters_default):
    click.echo("Loading coordinates...")
    dset = convert_sleuth_to_dataset(sleuth_file)

    click.echo("Reconstructing unthresholded maps...")
    k = Peaks2MapsKernel(dset.coordinates, mask=dset.mask)
    imgs = k.transform(ids=dset.ids, masked=False, resample_to_mask=False)

    mask_img = resample_to_img(dset.mask, imgs[0], interpolation='nearest')
    z_data = apply_mask(imgs, mask_img)

    click.echo("Estimating the null distribution...")
    res = rfx_glm(z_data, mask_img, null='empirical', n_iters=n_iters)

    if output_dir is None:
        output_dir = os.path.dirname(sleuth_file)
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        base = os.path.basename(sleuth_file)
        output_prefix, _ = os.path.splitext(base)
        output_prefix += '_'

    click.echo("Saving output maps...")
    res.save_results(output_dir=output_dir, prefix=output_prefix)
