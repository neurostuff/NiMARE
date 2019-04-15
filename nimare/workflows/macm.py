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
                          '(ALE) on a NiMARE dataset file and a target mask.')
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
    """
    Perform MACM with ALE algorithm.
    """
    click.echo("Loading coordinates...")
    dset = Dataset(dataset_file)
    sel_ids = dset.get_studies_by_mask(mask_file)

    # override sample size
    n_subs_db = dset.coordinates.drop_duplicates('id')['n'].astype(float).astype(int).sum()
    sel_coords = dset.coordinates.loc[dset.coordinates['id'].isin(sel_ids)]
    n_subs_sel = sel_coords.drop_duplicates('id')['n'].astype(float).astype(int).sum()
    click.echo("{0} studies selected out of {1}.".format(len(sel_ids),
                                                         len(dset.ids)))

    boilerplate = """
Meta-analytic connectivity modeling (MACM; Laird et al., 2009; Robinson et al.,
2009; Eickhoff et al., 2010) analysis was performed with the activation
likelihood estimation (ALE; Turkeltaub, Eden, Jones, & Zeffiro, 2002; Eickhoff,
Bzdok, Laird, Kurth, & Fox, 2012; Turkeltaub et al., 2012) meta-analysis
algorithm using NiMARE. The input dataset included {n_foci_db}
foci from {n_subs_db} participants across {n_exps_db} studies/experiments, from
which studies/experiments were selected for analysis if they had at least one
focus inside the target mask. The resulting sample included {n_foci_sel}
foci from {n_subs_sel} participants across {n_exps_sel} studies/experiments.

Modeled activation maps were generated for each study/experiment by convolving
each focus with a Gaussian kernel determined by the study/experiment's sample
size. For voxels with overlapping kernels, the maximum value was retained.
The modeled activation maps were rendered in MNI 152 space (Fonov et al., 2009;
Fonov et al., 2011) at 2x2x2mm resolution. A map of ALE values was then
computed for the sample as the union of modeled activation values across
studies/experiments. Voxelwise statistical significance was determined based on
an analytically derived null distribution using the method described in
Eickhoff, Bzdok, Laird, Kurth, & Fox (2012), prior to multiple comparisons
correction.

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
- Eickhoff, S. B., Bzdok, D., Laird, A. R., Kurth, F., & Fox, P. T. (2012).
Activation likelihood estimation meta-analysis revisited. NeuroImage,
59(3), 2349–2361.
- Eickhoff, S. B., Jbabdi, S., Caspers, S., Laird, A. R., Fox, P. T., Zilles,
K., & Behrens, T. E. (2010). Anatomical and functional connectivity of
cytoarchitectonic areas within the human parietal operculum. Journal of
Neuroscience, 30(18), 6409-6421.
- Fonov, V., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C.,
Collins, D. L., & Brain Development Cooperative Group. (2011).
Unbiased average age-appropriate atlases for pediatric studies.
Neuroimage, 54(1), 313-327.
- Fonov, V. S., Evans, A. C., McKinstry, R. C., Almli, C. R., & Collins, D. L.
(2009). Unbiased nonlinear average age-appropriate brain templates from birth
to adulthood. NeuroImage, (47), S102.
- Laird, A. R., Eickhoff, S. B., Li, K., Robin, D. A., Glahn, D. C., &
Fox, P. T. (2009). Investigating the functional heterogeneity of the default
mode network using coordinate-based meta-analytic modeling. The Journal of
Neuroscience: The Official Journal of the Society for Neuroscience, 29(46),
14496–14505.
- Robinson, J. L., Laird, A. R., Glahn, D. C., Lovallo, W. R., & Fox, P. T.
(2009). Metaanalytic connectivity modeling: Delineating the functional
connectivity of the human amygdala. Human Brain Mapping, 31(2), 173-184.
- Turkeltaub, P. E., Eden, G. F., Jones, K. M., & Zeffiro, T. A. (2002).
Meta-analysis of the functional neuroanatomy of single-word reading: method
and validation. NeuroImage, 16(3 Pt 1), 765–780.
- Turkeltaub, P. E., Eickhoff, S. B., Laird, A. R., Fox, M., Wiener, M.,
& Fox, P. (2012). Minimizing within-experiment and within-group effects in
Activation Likelihood Estimation meta-analyses. Human Brain Mapping,
33(1), 1–13.
    """

    ale = ALE(dset)

    click.echo("Performing meta-analysis...")
    ale.fit(n_iters=n_iters, ids=sel_ids,
            voxel_thresh=v_thr, q=c_thr, corr='FWE',
            n_cores=n_cores)

    min_clust = np.percentile(ale.null['cfwe'], 100 * (1 - c_thr))
    min_clust *= np.prod(dset.mask.header.get_zooms())

    boilerplate = boilerplate.format(
        n_exps_db=len(dset.ids),
        n_subs_db=n_subs_db,
        n_foci_db=dset.coordinates.shape[0],
        n_exps_sel=len(sel_ids),
        n_subs_sel=n_subs_sel,
        n_foci_sel=sel_coords.shape[0],
        unc=v_thr,
        fwe=c_thr,
        n_iters=n_iters,
        min_clust=min_clust)

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
    click.echo(boilerplate)
