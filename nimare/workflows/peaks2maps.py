"""
Workflow for contrast permutation meta-analysis on images constructed from
coordinates using the Peaks2Maps kernel.
"""
import os
import logging
import pathlib
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from ..base import MetaResult
from ..meta.ibma import rfx_glm
from ..meta.cbma import Peaks2MapsKernel
from ..io import convert_sleuth_to_dataset

LGR = logging.getLogger(__name__)


def peaks2maps_workflow(sleuth_file, output_dir=None, prefix=None, n_iters=10000):
    """
    """
    LGR.info('Loading coordinates...')
    dset = convert_sleuth_to_dataset(sleuth_file)

    LGR.info('Reconstructing unthresholded maps...')
    k = Peaks2MapsKernel(resample_to_mask=False)
    imgs = k.transform(dset, masked=False)

    mask_img = resample_to_img(dset.mask, imgs[0], interpolation='nearest')
    z_data = apply_mask(imgs, mask_img)

    LGR.info('Estimating the null distribution...')
    res = rfx_glm(z_data, null='empirical', n_iters=n_iters)
    res = MetaResult('rfx_glm', maps=res, mask=mask_img)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(sleuth_file))
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if prefix is None:
        base = os.path.basename(sleuth_file)
        prefix, _ = os.path.splitext(base)
        prefix += '_'

    LGR.info('Saving output maps...')
    res.save_maps(output_dir=output_dir, prefix=prefix)
