import os
import pathlib
import click
from nilearn.masking import apply_mask

from ..utils import get_template
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
def con_perm_workflow(contrast_images, output_dir=None, prefix=PREFIX_DEFAULT,
                      n_iters=N_ITERS_DEFAULT):
    target = 'mni152_2mm'
    mask_img = get_template(target, mask='brain')
    n_studies = len(contrast_images)
    click.echo("Loading contrast maps...")
    z_data = apply_mask(contrast_images, mask_img)

    boilerplate = """
A contrast permutation analysis was performed on a sample of {n_studies}
images. A brain mask derived from the MNI 152 template (Fonov et al., 2009;
Fonov et al., 2011) was applied at 2x2x2mm resolution. The sign flipping
method used was implemented as described in Maumet & Nichols (2016), with
{n_iters} iterations used to estimate the null distribution.

References
----------
- Fonov, V., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C.,
Collins, D. L., & Brain Development Cooperative Group. (2011).
Unbiased average age-appropriate atlases for pediatric studies.
Neuroimage, 54(1), 313-327.
- Fonov, V. S., Evans, A. C., McKinstry, R. C., Almli, C. R., & Collins, D. L.
(2009). Unbiased nonlinear average age-appropriate brain templates from birth
to adulthood. NeuroImage, (47), S102.
- Maumet, C., & Nichols, T. E. (2016). Minimal Data Needed for Valid & Accurate
Image-Based fMRI Meta-Analysis. https://doi.org/10.1101/048249
    """

    click.echo("Performing meta-analysis.")
    res = rfx_glm(z_data, mask_img, null='empirical', n_iters=n_iters)

    boilerplate = boilerplate.format(
        n_studies=n_studies,
        n_iters=n_iters)

    if output_dir is None:
        output_dir = os.getcwd()
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    click.echo("Saving output maps...")
    res.save_results(output_dir=output_dir, prefix=prefix)
    click.echo("Workflow completed.")
    click.echo(boilerplate)
