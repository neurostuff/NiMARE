"""Kernel transformers for CBMA algorithms.

Methods for estimating thresholded cluster maps from neuroimaging contrasts
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).
"""
from __future__ import division

import logging
import os
import warnings
from hashlib import md5

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image

from .. import references
from ..base import Transformer
from ..due import due
from ..utils import (
    _add_metadata_to_dataframe,
    _safe_transform,
    mm2vox,
    use_memmap,
    vox2mm,
)
from .utils import compute_ale_ma, compute_kda_ma, compute_p2m_ma, get_ale_kernel

LGR = logging.getLogger(__name__)


class KernelTransformer(Transformer):
    """Base class for modeled activation-generating methods in :mod:`~nimare.meta.kernel`.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.

    Notes
    -----
    All extra (non-ijk) parameters for a given kernel should be overrideable as
    parameters to __init__, so we can access them with get_params() and also
    apply them to datasets with missing data.
    """

    def _infer_names(self, **kwargs):
        """Determine filename pattern and image type for files created with this transformer.

        The parameters used to construct the filenames come from the transformer's
        parameters (attributes saved in ``__init__()``).

        Parameters
        ----------
        **kwargs
            Additional key/value pairs to incorporate into the image name.
            A common example is the hash for the target template's affine.

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for images that will be saved by the transformer.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
        """
        params = self.get_params()
        params = dict(**params, **kwargs)

        # Determine names for kernel-specific files
        keys = sorted(params.keys())
        param_str = "_".join(f"{k}-{str(params[k])}" for k in keys)
        self.filename_pattern = (
            f"study-[[id]]_{param_str}_{self.__class__.__name__}.nii.gz".replace(
                "[[", "{"
            ).replace("]]", "}")
        )
        self.image_type = f"{param_str}_{self.__class__.__name__}"

    @use_memmap(LGR)
    def transform(self, dataset, masker=None, return_type="image"):
        """Generate modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        masker : img_like or None, optional
            Mask to apply to MA maps. Required if ``dataset`` is a DataFrame.
            If None (and ``dataset`` is a Dataset), the Dataset's masker attribute will be used.
            Default is None.
        return_type : {'array', 'image', 'dataset'}, optional
            Whether to return a numpy array ('array'), a list of niimgs ('image'),
            or a Dataset with MA images saved as files ('dataset').
            Default is 'image'.

        Returns
        -------
        imgs : (C x V) :class:`numpy.ndarray` or :obj:`list` of :class:`nibabel.Nifti1Image` \
               or :class:`~nimare.dataset.Dataset`
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).
            If return_type is 'dataset', a new Dataset object with modeled
            activation images saved to files and referenced in the
            Dataset.images attribute.

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for MA maps that will be saved by the transformer.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
        """
        if return_type not in ("array", "image", "dataset"):
            raise ValueError('Argument "return_type" must be "image", "array", or "dataset".')

        if isinstance(dataset, pd.DataFrame):
            assert (
                masker is not None
            ), "Argument 'masker' must be provided if dataset is a DataFrame."
            mask = masker.mask_img
            coordinates = dataset.copy()
            assert (
                return_type != "dataset"
            ), "Input dataset must be a Dataset if return_type='dataset'."

            # Calculate IJK. Must assume that the masker is in same space,
            # but has different affine, from original IJK.
            coordinates[["i", "j", "k"]] = mm2vox(coordinates[["x", "y", "z"]], mask.affine)
        else:
            masker = dataset.masker if not masker else masker
            mask = masker.mask_img
            coordinates = dataset.coordinates.copy()

            # Determine MA map filenames. Must happen after parameters are set.
            self._infer_names(affine=md5(mask.affine).hexdigest())

            # Check for existing MA maps
            # Use coordinates to get IDs instead of Dataset.ids bc of possible
            # mismatch between full Dataset and contrasts with coordinates.
            if self.image_type in dataset.images.columns:
                files = dataset.get_images(ids=coordinates["id"].unique(), imtype=self.image_type)
                if all(f is not None for f in files):
                    LGR.debug("Files already exist. Using them.")
                    if return_type == "array":
                        masked_data = _safe_transform(files, masker)
                        return masked_data
                    elif return_type == "image":
                        return [nib.load(f) for f in files]
                    elif return_type == "dataset":
                        return dataset.copy()

            # Calculate IJK
            if not np.array_equal(mask.affine, dataset.masker.mask_img.affine):
                LGR.warning("Mask affine does not match Dataset affine. Assuming same space.")

            coordinates[["i", "j", "k"]] = mm2vox(coordinates[["x", "y", "z"]], mask.affine)

            # Add any metadata the Transformer might need to the coordinates DataFrame
            # This approach is probably inferior to one which uses a _required_inputs attribute
            # (like the MetaEstimators), but it should work just fine as long as individual
            # requirements are written in here.
            if (
                hasattr(self, "sample_size")
                and (self.sample_size is None)
                and ("sample_size" not in coordinates.columns)
            ):
                coordinates = _add_metadata_to_dataframe(
                    dataset,
                    coordinates,
                    metadata_field="sample_sizes",
                    target_column="sample_size",
                    filter_func=np.mean,
                )

        # Generate the MA maps if they weren't already available as images
        if return_type == "array":
            mask_data = mask.get_fdata().astype(bool)
        elif return_type == "image":
            dtype = type(self.value) if hasattr(self, "value") else float
            mask_data = mask.get_fdata().astype(dtype)
        elif return_type == "dataset":
            if dataset.basepath is None:
                raise ValueError(
                    "Dataset output path is not set. Set the path with Dataset.update_path()."
                )
            elif not os.path.isdir(dataset.basepath):
                raise ValueError(
                    "Output directory does not exist. Set the path to an existing folder with "
                    "Dataset.update_path()."
                )
            dataset = dataset.copy()

        transformed_maps = self._transform(mask, coordinates)

        # This will be a numpy.ndarray or numpy.memmap if the kernel is an (M)KDAKernel or
        # memory_limit is set, respectively.
        if not isinstance(transformed_maps[0], (list, tuple)):
            if return_type == "array":
                ma_arr = transformed_maps[0][:, mask_data]
                # If this array is a memmap, then the file needs to be closed
                if isinstance(transformed_maps[0], np.memmap):
                    LGR.debug(f"Closing memmap at {transformed_maps[0].filename}")
                    transformed_maps[0]._mmap.close()

                return ma_arr
            else:
                # Transform into an length-N list of length-2 tuples,
                # composed of a 3D array/memmap and a string with the ID.
                transformed_maps = list(zip(*transformed_maps))

        imgs = []
        for (kernel_data, id_) in transformed_maps:
            if isinstance(kernel_data, np.memmap):
                # Convert data to a numpy array if it's a memmap
                kernel_data = np.array(kernel_data)

            if return_type == "array":
                # NOTE: This will never be a memmap because memory_limit[!None]+return_type[array]
                # is dealt with above.
                img = kernel_data[mask_data]
                imgs.append(img)
            elif return_type == "image":
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
                imgs.append(img)
            elif return_type == "dataset":
                img = nib.Nifti1Image(kernel_data, mask.affine)
                out_file = os.path.join(dataset.basepath, self.filename_pattern.format(id=id_))
                img.to_filename(out_file)
                dataset.images.loc[dataset.images["id"] == id_, self.image_type] = out_file

        # If this array is a memmap, then the file needs to be closed
        if isinstance(transformed_maps[0][0], np.memmap):
            LGR.debug(f"Closing memmap at {transformed_maps[0][0].filename}")
            transformed_maps[0][0]._mmap.close()

        del kernel_data, transformed_maps

        if return_type == "array":
            return np.vstack(imgs)
        elif return_type == "image":
            return imgs
        elif return_type == "dataset":
            # Replace NaNs with Nones
            dataset.images[self.image_type] = dataset.images[self.image_type].where(
                dataset.images[self.image_type].notnull(), None
            )
            # Infer relative path
            dataset.images = dataset.images
            return dataset

    def _transform(self, mask, coordinates):
        """Apply the kernel's unique transformer.

        Parameters
        ----------
        mask : niimg-like
            Mask image. Should contain binary-like integer data.
        coordinates : pandas.DataFrame
            DataFrame containing IDs and coordinates.
            The DataFrame must have the following columns: "id", "i", "j", "k".
            Additionally, individual kernels may require other columns
            (e.g., "sample_size" for ALE).

        Returns
        -------
        transformed_maps : N-length list of (3D array, str) tuples or (4D array, 1D array) tuple \
                or (4D memmap, 1D array) tuple
            Transformed data, containing one element for each study.

            -   Case 1: A kernel that is not an (M)KDAKernel, with no memory limit.
                Each list entry is composed of a 3D array (the MA map) and the study's ID.

            -   Case 2: (M)KDAKernel, with no memory limit.
                There is a length-2 tuple with a 4D numpy array of the shape (N, X, Y, Z),
                containing all of the MA maps, and a numpy array of shape (N,) with the study IDs.

            -   Case 3: Any kernel, with memory_limit set, so memmaps will be used.
                There is a length-2 tuple with a 4D memmap array of the shape (N, X, Y, Z),
                containing all of the MA maps, and a numpy array of shape (N,) with the study IDs.
        """
        pass


@due.dcite(
    references.ALE2,
    description=(
        "Modifies ALE algorithm to eliminate within-experiment "
        "effects and generate MA maps based on subject group "
        "instead of experiment."
    ),
)
class ALEKernel(KernelTransformer):
    """Generate ALE modeled activation images from coordinates and sample size.

    By default (if neither ``fwhm`` nor ``sample_size`` is provided), the FWHM of the kernel
    will be determined on a study-wise basis based on the sample sizes available in the input,
    via the method described in [1]_.

    .. versionchanged:: 0.0.8

        * [ENH] Add low-memory option for kernel transformers.

    Parameters
    ----------
    fwhm : :obj:`float`, optional
        Full-width half-max for Gaussian kernel, if you want to have a
        constant kernel across Contrasts. Mutually exclusive with
        ``sample_size``.
    sample_size : :obj:`int`, optional
        Sample size, used to derive FWHM for Gaussian kernel based on
        formulae from Eickhoff et al. (2012). This sample size overwrites
        the Contrast-specific sample sizes in the dataset, in order to hold
        kernel constant across Contrasts. Mutually exclusive with ``fwhm``.
    memory_limit : :obj:`str` or None, optional
        Memory limit to apply to data. If None, no memory management will be applied.
        Otherwise, the memory limit will be used to (1) assign memory-mapped files and
        (2) restrict memory during array creation to the limit.
        Default is None.

    References
    ----------
    .. [1] Eickhoff, Simon B., et al. "Activation likelihood estimation
           meta-analysis revisited." Neuroimage 59.3 (2012): 2349-2361.
    """

    def __init__(self, fwhm=None, sample_size=None, memory_limit=None):
        if fwhm is not None and sample_size is not None:
            raise ValueError('Only one of "fwhm" and "sample_size" may be provided.')
        self.fwhm = fwhm
        self.sample_size = sample_size
        self.memory_limit = memory_limit

    def _transform(self, mask, coordinates):
        kernels = {}  # retain kernels in dictionary to speed things up
        exp_ids = coordinates["id"].unique()

        if self.memory_limit:
            # Use a memmapped 4D array
            transformed_shape = (len(exp_ids),) + mask.shape
            transformed = np.memmap(
                self.memmap_filenames[0],
                dtype=float,
                mode="w+",
                shape=transformed_shape,
            )
        else:
            # Use a list of tuples
            transformed = []

        for i_exp, id_ in enumerate(exp_ids):
            data = coordinates.loc[coordinates["id"] == id_]

            ijk = np.vstack((data.i.values, data.j.values, data.k.values)).T.astype(int)
            if self.sample_size is not None:
                sample_size = self.sample_size
            elif self.fwhm is None:
                # Extract from input
                sample_size = data.sample_size.astype(float).values[0]

            if self.fwhm is not None:
                assert np.isfinite(self.fwhm), "FWHM must be finite number"
                if self.fwhm not in kernels.keys():
                    _, kern = get_ale_kernel(mask, fwhm=self.fwhm)
                    kernels[self.fwhm] = kern
                else:
                    kern = kernels[self.fwhm]

            else:
                assert np.isfinite(sample_size), "Sample size must be finite number"
                if sample_size not in kernels.keys():
                    _, kern = get_ale_kernel(mask, sample_size=sample_size)
                    kernels[sample_size] = kern
                else:
                    kern = kernels[sample_size]

            kernel_data = compute_ale_ma(mask.shape, ijk, kern)

            if self.memory_limit:
                transformed[i_exp, :, :, :] = kernel_data

                # Write changes to disk
                transformed.flush()
            else:
                transformed.append((kernel_data, id_))

        if self.memory_limit:
            return transformed, exp_ids
        else:
            return transformed


class KDAKernel(KernelTransformer):
    """Generate KDA modeled activation images from coordinates.

    .. versionchanged:: 0.0.8

        * [ENH] Add low-memory option for kernel transformers.

    Parameters
    ----------
    r : :obj:`int`, optional
        Sphere radius, in mm.
    value : :obj:`int`, optional
        Value for sphere.
    memory_limit : :obj:`str` or None, optional
        Memory limit to apply to data. If None, no memory management will be applied.
        Otherwise, the memory limit will be used to (1) assign memory-mapped files and
        (2) restrict memory during array creation to the limit.
        Default is None.
    """

    _sum_overlap = True

    def __init__(self, r=10, value=1, memory_limit=None):
        self.r = float(r)
        self.value = value
        self.memory_limit = memory_limit

    def _transform(self, mask, coordinates):
        dims = mask.shape
        vox_dims = mask.header.get_zooms()

        ijks = coordinates[["i", "j", "k"]].values
        exp_idx = coordinates["id"].values
        transformed = compute_kda_ma(
            dims,
            vox_dims,
            ijks,
            self.r,
            self.value,
            exp_idx,
            sum_overlap=self._sum_overlap,
            memory_limit=self.memory_limit,
            memmap_filename=self.memmap_filenames[0],
        )
        exp_ids = np.unique(exp_idx)
        return transformed, exp_ids


class MKDAKernel(KDAKernel):
    """Generate MKDA modeled activation images from coordinates.

    .. versionchanged:: 0.0.8

        * [ENH] Add low-memory option for kernel transformers.

    Parameters
    ----------
    r : :obj:`int`, optional
        Sphere radius, in mm.
    value : :obj:`int`, optional
        Value for sphere.
    memory_limit : :obj:`str` or None, optional
        Memory limit to apply to data. If None, no memory management will be applied.
        Otherwise, the memory limit will be used to (1) assign memory-mapped files and
        (2) restrict memory during array creation to the limit.
        Default is None.
    """

    _sum_overlap = False


class Peaks2MapsKernel(KernelTransformer):
    """Generate peaks2maps modeled activation images from coordinates.

    .. deprecated:: 0.0.11
        `Peaks2MapsKernel` will be removed in NiMARE 0.0.13.

    Parameters
    ----------
    model_dir : :obj:`str`, optional
        Path to model directory. Default is "auto".

    Warning
    -------
    Peaks2MapsKernel is not intended for serious research.
    We strongly recommend against using it for any meaningful analyses.
    """

    def __init__(self, model_dir="auto"):
        warnings.warn(
            "Peaks2MapsKernel is deprecated, and will be removed in NiMARE version 0.0.13.",
            DeprecationWarning,
        )

        # Use private attribute to hide value from get_params.
        # get_params will find model_dir=None, which is *very important* when a path is provided.
        self._model_dir = model_dir

    def _transform(self, mask, coordinates):
        transformed = []
        coordinates_list = []
        ids = []
        for id_, data in coordinates.groupby("id"):
            ijk = np.vstack((data.i.values, data.j.values, data.k.values)).T.astype(int)
            xyz = vox2mm(ijk, mask.affine)
            coordinates_list.append(xyz)
            ids.append(id_)

        imgs = compute_p2m_ma(coordinates_list, skip_out_of_bounds=True, model_dir=self._model_dir)
        resampled_imgs = []
        for img in imgs:
            resampled_imgs.append(image.resample_to_img(img, mask).get_fdata())
        transformed = list(zip(resampled_imgs, ids))
        return transformed
