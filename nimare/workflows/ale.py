"""
Workflow for running an ALE meta-analysis from a Sleuth text file.
"""
import os
import pathlib
from shutil import copyfile

import click
import numpy as np

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
@click.option('--n_cores', default=-1,
              show_default=True,
              help="Number of processes to use for meta-analysis. If -1, use "
                   "all available cores.")
def ale_sleuth_inference(sleuth_file, output_dir=None, prefix=None,
                         n_iters=N_ITERS_DEFAULT,
                         v_thr=CLUSTER_FORMING_THRESHOLD_P_DEFAULT,
                         c_thr=CLUSTER_SIZE_THRESHOLD_Q_DEFAULT,
                         n_cores=-1):
    """
    Perform ALE meta-analysis from Sleuth text file.
    """
    click.echo("Loading coordinates...")
    dset = convert_sleuth_to_database(sleuth_file).get_dataset(target='colin_2mm')

    n_subs = dset.coordinates.drop_duplicates('id')['n'].astype(float).astype(int).sum()

    boilerplate = """
An activation likelihood estimation (ALE; Turkeltaub, Eden, Jones, & Zeffiro,
2002; Eickhoff, Bzdok, Laird, Kurth, & Fox, 2012; Turkeltaub et al., 2012)
meta-analysis was performed using NiMARE. The input dataset included {n_foci}
foci from {n_subs} participants across {n_exps} studies/experiments.

Foci were convolved with Gaussian kernels determined by sample size,
implemented on the Colin27 MNI template (Holmes et al., 1998; Aubert-Broche,
Evans, & Collins, 2006) at 2x2x2mm resolution.

-> If the cluster-level FWE-corrected results were used, include the following:
A cluster-forming threshold of p < {unc} was used, along with a cluster-extent
threshold of {fwe}. {n_iters} iterations were performed to estimate a null
distribution of cluster sizes, in which the locations of coordinates were
randomly drawn from a gray matter template and the maximum cluster size was
recorded after applying an uncorrected cluster-forming threshold of p < {unc},
resulting in a minimum cluster size of {min_clust:.02f} mm3.

-> If voxel-level FWE-corrected results were used, include the following:
Voxel-level FWE-correction was performed and results were thresholded at
p < {fwe}. {n_iters} iterations were performed to estimate a null
distribution of ALE values, in which the locations of coordinates were randomly
drawn from a gray matter template and the maximum ALE value was recorded.

References
----------
- Turkeltaub, P. E., Eden, G. F., Jones, K. M., & Zeffiro, T. A. (2002).
Meta-analysis of the functional neuroanatomy of single-word reading: method
and validation. NeuroImage, 16(3 Pt 1), 765–780.
- Eickhoff, S. B., Bzdok, D., Laird, A. R., Kurth, F., & Fox, P. T. (2012).
Activation likelihood estimation meta-analysis revisited. NeuroImage,
59(3), 2349–2361.
- Turkeltaub, P. E., Eickhoff, S. B., Laird, A. R., Fox, M., Wiener, M.,
& Fox, P. (2012). Minimizing within-experiment and within-group effects in
Activation Likelihood Estimation meta-analyses. Human Brain Mapping,
33(1), 1–13.
- Holmes, C. J., Hoge, R., Collins, D. L., Woods, R., Toga, A. W., & Evans, A.
C. (1998). Enhancement of MR images using registration for signal averaging.
J Comput Assist Tomogr, 22(2), 324–33.
http://dx.doi.org/10.1097/00004728-199803000-00032
- Aubert-Broche, B., Evans, A. C., & Collins, D. L. (2006). A new improved
version of the realistic digital brain phantom. NeuroImage, 32(1), 138–45.
http://www.ncbi.nlm.nih.gov/pubmed/16750398
    """

    ale = ALE(dset)

    click.echo("Estimating the null distribution...")
    ale.fit(n_iters=n_iters, ids=dset.ids,
            voxel_thresh=v_thr, q=c_thr, corr='FWE',
            n_cores=n_cores)

    min_clust = np.percentile(ale.null['cfwe'], 100 * (1 - c_thr))
    min_clust *= np.prod(dset.mask.header.get_zooms())

    boilerplate = boilerplate.format(
        n_exps=len(dset.ids),
        n_subs=n_subs,
        n_foci=dset.coordinates.shape[0],
        unc=v_thr,
        fwe=c_thr,
        n_iters=n_iters,
        min_clust=min_clust)

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

    click.echo("Workflow completed.")
    click.echo(boilerplate)
