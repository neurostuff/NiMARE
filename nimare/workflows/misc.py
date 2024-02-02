"""Miscellaneous Workflows."""

import logging

import nibabel as nib
from nilearn._utils import check_niimg_3d
from nilearn.image import math_img

LGR = logging.getLogger(__name__)


def conjunction_analysis(imgs):
    """Perform a conjunction analysis.

    .. versionadded:: 0.2.0

    This method is described in :footcite:t:`nichols2005valid`.

    Parameters
    ----------
    imgs : :obj:`list` of 3D :obj:`~nibabel.nifti1.Nifti1Image`, or :obj:`list` of :obj:`str`
        List of images upon which to perform the conjuction analysis.
        If a list of strings is provided, it is assumed to be paths to NIfTI images.

    Returns
    -------
    :obj:`~nibabel.nifti1.Nifti1Image`
        Conjunction image.

    References
    ----------
    .. footbibliography::
    """
    if len(imgs) < 2:
        raise ValueError("Conjunction analysis requires more than one image.")

    imgs_dict = {}
    mult_formula, min_formula = "", ""
    for img_i, img_obj in enumerate(imgs):
        if isinstance(img_obj, str):
            img = nib.load(img_obj)
        elif isinstance(img_obj, nib.Nifti1Image):
            img = img_obj
        else:
            raise ValueError(
                f"Invalid image type provided: {type(img_obj)}. Must be a path to a NIfTI image "
                "or a NIfTI image object."
            )

        img = check_niimg_3d(img)

        img_label = f"img{img_i}"
        imgs_dict[img_label] = img
        mult_formula += img_label
        min_formula += img_label

        if img_i != len(imgs) - 1:
            mult_formula += " * "
            min_formula += ", "

    formula = f"np.where({mult_formula} > 0, np.minimum.reduce([{min_formula}]), 0)"
    LGR.info("Performing conjunction analysis...")
    LGR.info(f"Formula: {formula}")

    return math_img(formula, **imgs_dict)
