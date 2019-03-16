import os
import pathlib
import click
from nilearn.masking import apply_mask

from ..utils.utils import get_template
from ..meta.ibma import rfx_glm

N_ITERS_DEFAULT = 10000
PREFIX_DEFAULT = ''


@click.command(name='conperm', short_help='Permutation-based meta-analysis of contrast maps',
               help='Meta-analysis of contrast maps using random effects and '
                    'two-sided inference with empirical (permutation-based) null distribution '
                    'and Family Wise Error multiple comparisons correction. '
                    'Input may be a list of 3D files or a single 4D file.')
@click.argument('contrast_images', nargs=-1, required=True,
                type=click.Path(exists=True))
@click.option('--output_dir', help="Where to put the output maps.")
@click.option('--prefix', help="Common prefix for output maps.",
              default=PREFIX_DEFAULT, show_default=True)
@click.option('--n_iters', default=N_ITERS_DEFAULT, show_default=True,
              help="Number of iterations for permutation testing.")
def con_perm(contrast_images, output_dir=None, prefix=PREFIX_DEFAULT,
             n_iters=N_ITERS_DEFAULT):
    target = 'mni152_2mm'
    mask_img = get_template(target, mask='brain')
    click.echo("Loading contrast maps...")
    z_data = apply_mask(contrast_images, mask_img)

    click.echo("Estimating the null distribution...")
    res = rfx_glm(z_data, mask_img, null='empirical', n_iters=n_iters)

    if output_dir is None:
        output_dir = os.getcwd()
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    click.echo("Saving output maps...")
    res.save_results(output_dir=output_dir, prefix=prefix)
