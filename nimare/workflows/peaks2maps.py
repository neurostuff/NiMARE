"""Perform meta-analysis on images constructed from coordinates using the Peaks2Maps kernel."""
import logging
import os
import pathlib

import numpy as np
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask
from nilearn.mass_univariate import permuted_ols

from ..base import MetaResult
from ..io import convert_sleuth_to_dataset
from ..meta.kernel import Peaks2MapsKernel

LGR = logging.getLogger(__name__)


def peaks2maps_workflow(sleuth_file, output_dir=None, prefix=None, n_iters=10000):
    """Run the peaks2maps workflow."""
    LGR.info("Loading coordinates...")
    dset = convert_sleuth_to_dataset(sleuth_file)

    LGR.info("Reconstructing unthresholded maps...")
    k = Peaks2MapsKernel(resample_to_mask=False)
    imgs = k.transform(dset, return_type="image")

    mask_img = resample_to_img(dset.mask, imgs[0], interpolation="nearest")
    z_data = apply_mask(imgs, mask_img)

    LGR.info("Estimating the null distribution...")
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

    res = MetaResult(permuted_ols, maps=res, mask=mask_img)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(sleuth_file))
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(sleuth_file)
        prefix, _ = os.path.splitext(base)
        prefix += "_"

    LGR.info("Saving output maps...")
    res.save_maps(output_dir=output_dir, prefix=prefix)
