"""Workflow for running an ALE meta-analysis from a Sleuth text file."""
import logging
import os
import pathlib
from shutil import copyfile

import numpy as np

from ..correct import FWECorrector
from ..io import convert_sleuth_to_dataset
from ..meta import ALE, ALESubtraction

LGR = logging.getLogger(__name__)


def ale_sleuth_workflow(
    sleuth_file,
    sleuth_file2=None,
    output_dir=None,
    prefix=None,
    n_iters=10000,
    v_thr=0.001,
    fwhm=None,
    n_cores=-1,
):
    """Perform ALE meta-analysis from Sleuth text file."""
    LGR.info("Loading coordinates...")

    if fwhm:
        fwhm_str = "of {0} mm".format(fwhm)
    else:
        fwhm_str = "determined by sample size"

    if not sleuth_file2:
        dset = convert_sleuth_to_dataset(sleuth_file, target="ale_2mm")
        n_subs = dset.get_metadata(field="sample_sizes")
        n_subs = np.sum(n_subs)

        boilerplate = """
An activation likelihood estimation (ALE; Turkeltaub, Eden, Jones, & Zeffiro,
2002; Eickhoff, Bzdok, Laird, Kurth, & Fox, 2012; Turkeltaub et al., 2012)
meta-analysis was performed using NiMARE. The input dataset included {n_foci}
foci from {n_subs} participants across {n_exps} studies/experiments.

Modeled activation maps were generated for each study/experiment by convolving
each focus with a Gaussian kernel {fwhm_str}.
For voxels with overlapping kernels, the maximum value was retained.
The modeled activation maps were rendered in MNI 152 space (Fonov et al., 2009;
Fonov et al., 2011) at 2x2x2mm resolution. A map of ALE values was then
computed for the sample as the union of modeled activation values across
studies/experiments. Voxelwise statistical significance was determined based on
an analytically derived null distribution using the method described in
Eickhoff, Bzdok, Laird, Kurth, & Fox (2012), prior to multiple comparisons
correction.

-> If the cluster-level FWE-corrected results were used, include the following:
A cluster-forming threshold of p < {unc} was used to perform cluster-level FWE
correction. {n_iters} iterations were performed to estimate a null distribution
of cluster sizes, in which the locations of coordinates were randomly drawn
from a gray matter template and the maximum cluster size was recorded after
applying an uncorrected cluster-forming threshold of p < {unc}. The negative
log-transformed p-value for each cluster in the thresholded map was determined
based on the cluster sizes.

-> If voxel-level FWE-corrected results were used, include the following:
Voxel-level FWE-correction was performed. {n_iters} iterations were performed
to estimate a null distribution of ALE values, in which the locations of
coordinates were randomly drawn from a gray matter template and the maximum
ALE value was recorded.

References
----------
- Eickhoff, S. B., Bzdok, D., Laird, A. R., Kurth, F., & Fox, P. T. (2012).
Activation likelihood estimation meta-analysis revisited. NeuroImage,
59(3), 2349–2361.
- Fonov, V., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C.,
Collins, D. L., & Brain Development Cooperative Group. (2011).
Unbiased average age-appropriate atlases for pediatric studies.
Neuroimage, 54(1), 313-327.
- Fonov, V. S., Evans, A. C., McKinstry, R. C., Almli, C. R., & Collins, D. L.
(2009). Unbiased nonlinear average age-appropriate brain templates from birth
to adulthood. NeuroImage, (47), S102.
- Turkeltaub, P. E., Eden, G. F., Jones, K. M., & Zeffiro, T. A. (2002).
Meta-analysis of the functional neuroanatomy of single-word reading: method
and validation. NeuroImage, 16(3 Pt 1), 765–780.
- Turkeltaub, P. E., Eickhoff, S. B., Laird, A. R., Fox, M., Wiener, M.,
& Fox, P. (2012). Minimizing within-experiment and within-group effects in
Activation Likelihood Estimation meta-analyses. Human Brain Mapping,
33(1), 1–13.
        """

        ale = ALE(kernel__fwhm=fwhm)

        LGR.info("Performing meta-analysis...")
        results = ale.fit(dset)
        corr = FWECorrector(
            method="montecarlo", n_iters=n_iters, voxel_thresh=v_thr, n_cores=n_cores
        )
        cres = corr.transform(results)

        boilerplate = boilerplate.format(
            n_exps=len(dset.ids),
            n_subs=n_subs,
            n_foci=dset.coordinates.shape[0],
            unc=v_thr,
            n_iters=n_iters,
            fwhm_str=fwhm_str,
        )
    else:
        dset1 = convert_sleuth_to_dataset(sleuth_file, target="ale_2mm")
        dset2 = convert_sleuth_to_dataset(sleuth_file2, target="ale_2mm")
        n_subs1 = dset1.get_metadata(field="sample_sizes")
        n_subs1 = np.sum(n_subs1)
        n_subs2 = dset2.get_metadata(field="sample_sizes")
        n_subs2 = np.sum(n_subs2)

        boilerplate = """
Activation likelihood estimation (ALE; Turkeltaub, Eden, Jones, & Zeffiro,
2002; Eickhoff, Bzdok, Laird, Kurth, & Fox, 2012; Turkeltaub et al., 2012)
meta-analyses were performed using NiMARE for each of two datasets.
The first input dataset included {n_foci1} foci from {n_subs1} participants
across {n_exps1} studies/experiments. The second input dataset included
{n_foci2} foci from {n_subs2} participants across {n_exps2} studies/experiments.

Foci were convolved with Gaussian kernels {fwhm_str},
implemented on the MNI 152 template (Fonov et al., 2009; Fonov et al., 2011)
at 2x2x2mm resolution.

-> If the cluster-level FWE-corrected results were used, include the following:
A cluster-forming threshold of p < {unc} was used to perform cluster-level FWE
correction. {n_iters} iterations were performed to estimate a null distribution
of cluster sizes, in which the locations of coordinates were randomly drawn
from a gray matter template and the maximum cluster size was recorded after
applying an uncorrected cluster-forming threshold of p < {unc}. The negative
log-transformed p-value for each cluster in the thresholded map was determined
based on the cluster sizes.

-> If voxel-level FWE-corrected results were used, include the following:
Voxel-level FWE-correction was performed. {n_iters} iterations were performed
to estimate a null distribution of ALE values, in which the locations of
coordinates were randomly drawn from a gray matter template and the maximum
ALE value was recorded.

Following dataset-specific ALE meta-analyses, a subtraction analysis was
performed to compare the two datasets according to the procedure from Laird
et al. (2005). {n_iters} iterations were performed.

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
- Fonov, V., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C.,
Collins, D. L., & Brain Development Cooperative Group. (2011).
Unbiased average age-appropriate atlases for pediatric studies.
Neuroimage, 54(1), 313-327.
- Fonov, V. S., Evans, A. C., McKinstry, R. C., Almli, C. R., & Collins, D. L.
(2009). Unbiased nonlinear average age-appropriate brain templates from birth
to adulthood. NeuroImage, (47), S102.
- Laird, A. R., Fox, P. M., Price, C. J., Glahn, D. C., Uecker, A. M.,
Lancaster, J. L., ... & Fox, P. T. (2005). ALE meta‐analysis: Controlling the
false discovery rate and performing statistical contrasts. Human brain mapping,
25(1), 155-164.
        """

        ale1 = ALE(kernel__fwhm=fwhm)
        ale2 = ALE(kernel__fwhm=fwhm)

        LGR.info("Performing meta-analyses...")
        res1 = ale1.fit(dset1)
        res2 = ale2.fit(dset2)
        corr = FWECorrector(
            method="montecarlo", n_iters=n_iters, voxel_thresh=v_thr, n_cores=n_cores
        )
        cres1 = corr.transform(res1)
        cres2 = corr.transform(res2)
        sub = ALESubtraction(n_iters=n_iters, kernel__fwhm=fwhm)
        sres = sub.fit(dset1, dset2)

        boilerplate = boilerplate.format(
            n_exps1=len(dset1.ids),
            n_subs1=n_subs1,
            n_foci1=dset1.coordinates.shape[0],
            n_exps2=len(dset2.ids),
            n_subs2=n_subs2,
            n_foci2=dset2.coordinates.shape[0],
            unc=v_thr,
            n_iters=n_iters,
            fwhm_str=fwhm_str,
        )

    if output_dir is None:
        output_dir = os.path.abspath(os.path.dirname(sleuth_file))
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(sleuth_file)
        prefix, _ = os.path.splitext(base)
        prefix += "_"
    elif not prefix.endswith("_"):
        prefix = prefix + "_"

    LGR.info("Saving output maps...")
    if not sleuth_file2:
        cres.save_maps(output_dir=output_dir, prefix=prefix)
        copyfile(sleuth_file, os.path.join(output_dir, prefix + "input_coordinates.txt"))
    else:
        prefix1 = os.path.splitext(os.path.basename(sleuth_file))[0] + "_"
        prefix2 = os.path.splitext(os.path.basename(sleuth_file2))[0] + "_"
        prefix3 = prefix + "subtraction_"
        cres1.save_maps(output_dir=output_dir, prefix=prefix1)
        cres2.save_maps(output_dir=output_dir, prefix=prefix2)
        sres.save_maps(output_dir=output_dir, prefix=prefix3)
        copyfile(sleuth_file, os.path.join(output_dir, prefix + "group1_input_coordinates.txt"))
        copyfile(sleuth_file2, os.path.join(output_dir, prefix + "group2_input_coordinates.txt"))

    LGR.info("Workflow completed.")
    LGR.info(boilerplate)
