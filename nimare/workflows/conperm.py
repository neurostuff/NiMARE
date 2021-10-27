"""Run a contrast permutation meta-analysis on a set of images."""
import logging
import os
import pathlib

import numpy as np
from nilearn.masking import apply_mask
from nilearn.mass_univariate import permuted_ols

from ..results import MetaResult
from ..utils import get_template

LGR = logging.getLogger(__name__)


def conperm_workflow(contrast_images, mask_image=None, output_dir=None, prefix="", n_iters=10000):
    """Run a contrast permutation workflow."""
    if mask_image is None:
        target = "mni152_2mm"
        mask_image = get_template(target, mask="brain")

    n_studies = len(contrast_images)
    LGR.info("Loading contrast maps...")
    z_data = apply_mask(contrast_images, mask_image)

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

    LGR.info("Performing meta-analysis.")
    log_p_map, t_map, _ = permuted_ols(
        np.ones((z_data.shape[0], 1)),
        z_data,
        confounding_vars=None,
        model_intercept=False,  # modeled by tested_vars
        n_perm=n_iters,
        two_sided_test=True,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    res = {"logp": log_p_map, "t": t_map}
    # The t_test function will stand in for the Estimator in the results object
    res = MetaResult(permuted_ols, mask_image, maps=res)

    boilerplate = boilerplate.format(n_studies=n_studies, n_iters=n_iters)

    if output_dir is None:
        output_dir = os.getcwd()
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    LGR.info("Saving output maps...")
    res.save_maps(output_dir=output_dir, prefix=prefix)
    LGR.info("Workflow completed.")
    LGR.info(boilerplate)
