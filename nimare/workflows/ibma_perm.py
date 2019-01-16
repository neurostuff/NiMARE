import os
import pathlib
import click
from nilearn.masking import apply_mask

from ..utils import get_template
from ..meta.ibma import rfx_glm

n_iters_default = 10000
output_prefix_default = ''


@click.command(name='conperm', short_help='permutation based metaanalysis of contrast maps',
               help='Metaanalysis of contrast maps using random effects and '
                    'two-sided inference with empirical (permutation based) null distribution '
                    'and Family Wise Error multiple comparison correction.')
@click.argument('contrast_images', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--output_dir', help="Where to put the output maps.")
@click.option('--output_prefix', help="Common prefix for output maps.",
              default=output_prefix_default, show_default=True)
@click.option('--n_iters', default=n_iters_default, show_default=True,
              help="Number of iterations for permutation testing.")
def con_perm(contrast_images, output_dir=None, output_prefix=output_prefix_default,
             n_iters=n_iters_default):
    target = 'mni152_2mm'
    mask_img = get_template(target, mask='brain')
    z_data = apply_mask(contrast_images, mask_img)

    res = rfx_glm(z_data, mask_img, null='empirical', n_iters=n_iters)

    if output_dir is None:
        output_dir = os.getcwd()
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, img in res.images.items():
        print(output_dir, output_prefix, name)
        img.to_filename(os.path.join(output_dir, output_prefix + name + ".nii.gz"))
