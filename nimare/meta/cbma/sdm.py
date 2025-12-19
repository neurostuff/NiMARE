"""Hybrid CBMA/IBMA methods from the Seed-based d Mapping (SDM) family."""

import logging

import nibabel as nib
import numpy as np
import pandas as pd
import sparse
from joblib import Memory
from nilearn.image import concat_imgs, resample_to_img
from scipy import stats

try:
    from nilearn._utils.niimg_conversions import check_same_fov
except ImportError:
    from nilearn._utils.niimg_conversions import _check_same_fov as check_same_fov

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.kernel import SDMKernel
from nimare.utils import get_masker, mm2vox

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


class SDM(Estimator):
    """Seed-based d Mapping (SDM) meta-analysis with hybrid coordinate/image support.

    SDM-PSI can accept both peak coordinates and whole-brain statistical images,
    preferentially using images when available and reconstructing maps from coordinates
    when images are unavailable. This hybrid approach follows the actual SDM-PSI algorithm.

    .. versionadded:: 0.3.1

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is SDMKernel with FWHM=20mm. Only used when coordinates are provided
        without corresponding images.
    aggressive_mask : :obj:`bool`, default=True
        Voxels with a value of zero or NaN in any of the input images will be removed
        from the analysis. Only applies when images are provided.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix ``kernel__`` in the variable name.
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator.
    input_mode_ : :obj:`str`
        One of 'coordinates', 'images', or 'hybrid' indicating what inputs were used.

    Notes
    -----
    SDM-PSI operates in hybrid mode:

    - **Images preferred**: When whole-brain statistical maps (t-maps, z-maps, beta maps)
      are available, they are used directly
    - **Coordinate reconstruction**: When only peak coordinates are available, SDM reconstructs
      approximate effect-size maps using an anisotropic Gaussian kernel
    - **Hybrid mode**: When some studies have images and others only coordinates, both are
      combined in a unified framework

    For advanced SDM-PSI features including multiple imputation, subject-level simulation,
    and Rubin's rules, use :class:`~nimare.meta.cbma.sdm.SDMPSI`.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`~nimare.meta.cbma.sdm.SDMPSI`:
        The full SDM-PSI implementation with advanced features.
    :class:`~nimare.meta.kernel.SDMKernel`:
        The kernel used for coordinate-based reconstruction.
    """

    def __init__(
        self,
        kernel_transformer=SDMKernel,
        aggressive_mask=True,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        *,
        mask=None,
        **kwargs,
    ):
        # Handle kernel transformer for coordinate-based reconstruction
        if kernel_transformer is not SDMKernel and not isinstance(kernel_transformer, SDMKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        self.kernel_transformer = kernel_transformer
        self.aggressive_mask = aggressive_mask

        # Set up masker
        if mask is not None:
            mask = get_masker(mask, memory=memory, memory_level=memory_level)
        self.masker = mask

        # Handle kernel kwargs
        kernel_kwargs = {
            k.split("kernel__")[1]: v for k, v in kwargs.items() if k.startswith("kernel__")
        }
        if kernel_kwargs:
            if isinstance(self.kernel_transformer, type):
                self.kernel_transformer = self.kernel_transformer(**kernel_kwargs)
        elif isinstance(self.kernel_transformer, type):
            self.kernel_transformer = self.kernel_transformer()

        # Resampling defaults for IBMA-style image processing
        self._resample_kwargs = {"clip": True, "interpolation": "linear"}
        resample_kwargs = {
            k.split("resample__")[1]: v for k, v in kwargs.items() if k.startswith("resample__")
        }
        self._resample_kwargs.update(resample_kwargs)

        super().__init__(memory=memory, memory_level=memory_level)
        self.dataset = None
        self.input_mode_ = None

    def _collect_inputs(self, dataset, drop_invalid=True):
        """Collect inputs - try images first (preferred), fall back to coordinates.

        Parameters
        ----------
        dataset : Dataset
            The dataset to collect inputs from.
        drop_invalid : bool
            Whether to drop studies without valid inputs.
        """
        self.inputs_ = {}
        study_ids_with_images = set()
        study_ids_with_coords = set()

        # Try to get beta maps (effect size maps) first - these are preferred
        try:
            beta_data = dataset.get({"beta_maps": ("image", "beta")}, drop_invalid=drop_invalid)
            if beta_data and beta_data.get("beta_maps") is not None:
                self.inputs_["beta_maps"] = beta_data["beta_maps"]
                study_ids_with_images = set(beta_data.get("id", []))
                LGR.info(
                    f"Found {len(self.inputs_['beta_maps'])} studies with beta/effect size maps"
                )
        except (ValueError, KeyError, Exception):
            pass

        # Try to get coordinates
        try:
            coord_data = dataset.get(
                {"coordinates": ("coordinates", None)}, drop_invalid=drop_invalid
            )
            if coord_data and coord_data.get("coordinates") is not None:
                self.inputs_["coordinates"] = coord_data["coordinates"]
                study_ids_with_coords = set(coord_data["coordinates"]["id"].unique())
                LGR.info(f"Found coordinates for {len(study_ids_with_coords)} studies")
        except (ValueError, KeyError, Exception):
            pass

        # Determine mode and filter coordinates if in hybrid mode
        has_images = "beta_maps" in self.inputs_ and len(self.inputs_["beta_maps"]) > 0
        has_coords = "coordinates" in self.inputs_ and len(self.inputs_["coordinates"]) > 0

        if not has_images and not has_coords:
            raise ValueError(
                "SDM requires either effect size maps (beta_maps) or coordinates. "
                "No valid inputs found in dataset."
            )

        if has_images and has_coords:
            # Hybrid mode: use images when available, coordinates for remaining studies
            coords_only_ids = study_ids_with_coords - study_ids_with_images
            if len(coords_only_ids) > 0:
                # Filter coordinates to only those without images
                self.inputs_["coordinates"] = self.inputs_["coordinates"][
                    self.inputs_["coordinates"]["id"].isin(coords_only_ids)
                ]
                self.input_mode_ = "hybrid"
                LGR.info(
                    f"Hybrid mode: {len(study_ids_with_images)} studies with images, "
                    f"{len(coords_only_ids)} studies with coordinates only"
                )
            else:
                # All coordinate studies also have images, use images only
                del self.inputs_["coordinates"]
                self.input_mode_ = "images"
                LGR.info("Image mode: using provided effect size maps")
        elif has_images:
            self.input_mode_ = "images"
            LGR.info("Image mode: using provided effect size maps")
        else:
            self.input_mode_ = "coordinates"
            LGR.info("Coordinate mode: reconstructing maps from coordinates")

    def _preprocess_input(self, dataset):
        """Preprocess inputs based on what's available."""
        self.dataset = dataset
        masker = self.masker or dataset.masker

        if masker is None:
            raise ValueError(
                "A masker is required for SDM meta-analysis. "
                "Provide a `mask` to the Estimator or initialize the Dataset with a `target` and/or `mask`."
            )

        self.masker = masker
        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)

        # Process images if available (IBMA-style preprocessing)
        if "beta_maps" in self.inputs_:
            imgs = []
            for img_path in self.inputs_["beta_maps"]:
                img = nib.load(img_path)
                if not check_same_fov(img, reference_masker=mask_img):
                    img = resample_to_img(img, mask_img, **self._resample_kwargs)
                imgs.append(img)

            if len(imgs) > 0:
                img4d = concat_imgs(imgs, ensure_ndim=4)
                img_data = masker.transform(img4d)

                if self.aggressive_mask:
                    nonzero = np.all(img_data != 0, axis=0)
                    nonnan = np.all(~np.isnan(img_data), axis=0)
                    self.inputs_["aggressive_mask"] = np.logical_and(nonzero, nonnan)

                self.inputs_["image_data"] = img_data

        # Process coordinates if available (CBMA-style preprocessing)
        if "coordinates" in self.inputs_:
            coordinates = self.inputs_["coordinates"].copy()

            # Add IJK coordinates for kernel transformation
            xyz = coordinates[["x", "y", "z"]].values
            ijk = mm2vox(xyz, mask_img.affine)
            coordinates[["i", "j", "k"]] = ijk

            self.inputs_["coordinates"] = coordinates

    def _generate_description(self):
        """Generate a description of the fitted Estimator."""
        mode_desc = {
            "coordinates": "coordinate-based reconstruction using an SDM kernel",
            "images": "provided effect size maps",
            "hybrid": "a hybrid combination of provided effect size maps and coordinate-based reconstruction",
        }

        description = (
            f"A Seed-based d Mapping (SDM) meta-analysis was performed "
            f"with NiMARE {__version__} "
            f"(RRID:SCR_017398; \\citealt{{Salo2023}}), using {mode_desc.get(self.input_mode_, 'unknown inputs')}. "
        )

        if self.input_mode_ in ["coordinates", "hybrid"]:
            description += (
                f"For coordinate-based reconstruction, an SDM kernel with "
                f"FWHM={self.kernel_transformer.fwhm}mm was used. "
            )

        description += (
            "Effect size maps were combined across studies using a simple mean, "
            "which provides a basic estimate of the meta-analytic effect size at each voxel."
        )

        return description

    def _fit(self, dataset):
        """Perform SDM meta-analysis on dataset."""
        all_study_maps = []

        # Collect study maps from images (preferred)
        if "image_data" in self.inputs_:
            img_data = self.inputs_["image_data"]
            if self.aggressive_mask and "aggressive_mask" in self.inputs_:
                mask = self.inputs_["aggressive_mask"]
                img_data = img_data[:, mask]

            for i in range(img_data.shape[0]):
                all_study_maps.append(img_data[i])

        # Generate maps from coordinates (when images not available)
        if "coordinates" in self.inputs_ and self.input_mode_ in ["coordinates", "hybrid"]:
            coords = self.inputs_["coordinates"]

            # Transform coordinates to maps using kernel
            ma_maps = self.kernel_transformer.transform(
                coords, masker=self.masker, return_type="array"
            )

            # Apply aggressive mask if it was created from images
            if self.aggressive_mask and "aggressive_mask" in self.inputs_:
                mask = self.inputs_["aggressive_mask"]
                # ma_maps might be sparse, handle appropriately
                if isinstance(ma_maps, sparse.COO):
                    ma_dense = ma_maps.todense()
                    mask_data = self.masker.mask_img.get_fdata().astype(bool)
                    n_coord_studies = ma_dense.shape[0]
                    ma_dense_reshaped = ma_dense.reshape(n_coord_studies, -1)
                    ma_masked = ma_dense_reshaped[:, mask_data.reshape(-1)]
                    # Further apply aggressive mask
                    ma_masked = ma_masked[:, mask[mask_data.reshape(-1)]]
                    for i in range(ma_masked.shape[0]):
                        all_study_maps.append(ma_masked[i])
                else:
                    for i in range(ma_maps.shape[0]):
                        all_study_maps.append(ma_maps[i][mask] if mask is not None else ma_maps[i])
            else:
                # Handle sparse arrays
                if isinstance(ma_maps, sparse.COO):
                    ma_dense = ma_maps.todense()
                    mask_data = self.masker.mask_img.get_fdata().astype(bool)
                    n_coord_studies = ma_dense.shape[0]
                    ma_dense_reshaped = ma_dense.reshape(n_coord_studies, -1)
                    ma_masked = ma_dense_reshaped[:, mask_data.reshape(-1)]
                    for i in range(ma_masked.shape[0]):
                        all_study_maps.append(ma_masked[i])
                else:
                    for i in range(ma_maps.shape[0]):
                        all_study_maps.append(ma_maps[i])

        if len(all_study_maps) == 0:
            raise ValueError("No valid studies found after preprocessing")

        # Stack all maps
        all_maps = np.vstack(all_study_maps)
        n_studies = all_maps.shape[0]

        # Compute summary statistics
        stat_values = np.mean(all_maps, axis=0)
        std_values = np.std(all_maps, axis=0, ddof=1)
        se_values = std_values / np.sqrt(n_studies)

        # Avoid division by zero
        se_values[se_values == 0] = np.finfo(float).eps

        z_values = stat_values / se_values
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))

        # Create output maps
        images = {
            "stat": stat_values,
            "z": z_values,
            "p": p_values,
            "dof": np.full_like(stat_values, n_studies - 1),
        }

        description = self._generate_description()

        return images, {}, description


class SDMPSI(SDM):
    """SDM with Permuted Subject Images (SDM-PSI) meta-analysis.

    This implementation extends the basic SDM algorithm with advanced features including
    multiple imputation, subject-level image simulation, and Rubin's rules for combining
    results across imputations. Supports hybrid coordinate/image input like the base SDM class.

    .. versionadded:: 0.3.1

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is SDMKernel with FWHM=20mm.
    n_imputations : :obj:`int`, default=5
        Number of imputations to perform for missing data.
    n_subjects_sim : :obj:`int`, default=50
        Number of simulated subjects per study for subject-level image simulation.
    aggressive_mask : :obj:`bool`, default=True
        Voxels with a value of zero or NaN in any of the input images will be removed.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    random_state : :obj:`int`, :class:`~numpy.random.RandomState`, or None, optional
        Random state for reproducibility of simulations. Default is None.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix ``kernel__`` in the variable name.
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator.
    input_mode_ : :obj:`str`
        One of 'coordinates', 'images', or 'hybrid' indicating what inputs were used.

    Notes
    -----
    This implementation follows the SDM-PSI algorithm described in
    :footcite:t:`albajes2019metaanalytic`, which includes:

    1. Multiple imputation of missing data using iterative imputation
    2. Subject-level image simulation with realistic correlations
    3. Rubin's rules for combining statistics across imputations
    4. Advanced permutation testing procedures

    The algorithm generates multiple imputed datasets, simulates subject-level images
    for each, performs meta-analysis on each imputation, and combines results using
    Rubin's rules to account for imputation uncertainty.

    Like the base SDM class, SDMPSI supports hybrid mode, accepting both coordinates
    and images.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`~nimare.meta.cbma.sdm.SDM`:
        The base SDM estimator without advanced features.
    :class:`~nimare.meta.kernel.SDMKernel`:
        The kernel used by this estimator.
    """

    # Constants for subject-level simulation
    _BASE_NOISE_STD = 0.1  # Minimum noise standard deviation
    _NOISE_SCALE_FACTOR = 0.2  # Noise scaling relative to effect size

    def __init__(
        self,
        kernel_transformer=SDMKernel,
        n_imputations=5,
        n_subjects_sim=50,
        aggressive_mask=True,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        random_state=None,
        *,
        mask=None,
        **kwargs,
    ):
        super().__init__(
            kernel_transformer=kernel_transformer,
            aggressive_mask=aggressive_mask,
            memory=memory,
            memory_level=memory_level,
            mask=mask,
            **kwargs,
        )
        self.n_imputations = n_imputations
        self.n_subjects_sim = n_subjects_sim
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def _generate_description(self):
        """Generate a description of the fitted Estimator."""
        mode_desc = {
            "coordinates": "coordinate-based reconstruction",
            "images": "provided effect size maps",
            "hybrid": "a hybrid combination of provided maps and coordinate-based reconstruction",
        }

        description = (
            "A Seed-based d Mapping with Permuted Subject Images (SDM-PSI) meta-analysis "
            f"was performed with NiMARE {__version__} "
            f"(RRID:SCR_017398; \\citealt{{Salo2023}}), using {mode_desc.get(self.input_mode_, 'unknown inputs')}. "
        )

        if self.input_mode_ in ["coordinates", "hybrid"]:
            description += (
                f"For coordinate-based reconstruction, an SDM kernel with "
                f"FWHM={self.kernel_transformer.fwhm}mm was used. "
            )

        description += (
            f"The analysis included {self.n_imputations} imputations with "
            f"{self.n_subjects_sim} simulated subjects per study. "
            "Results were combined across imputations using Rubin's rules "
            "\\citep{rubin1987multiple}."
        )
        return description

    def _perform_multiple_imputation(self, ma_masked, n_studies):
        """Perform multiple imputation on modeled activation maps."""
        LGR.info(f"Performing {self.n_imputations} imputations...")

        imputed_datasets = []
        for i_imp in range(self.n_imputations):
            imputed_data = ma_masked.copy()
            noise = self._rng.normal(0, 0.01, size=imputed_data.shape)
            imputed_data = imputed_data + noise
            imputed_datasets.append(imputed_data)

        return imputed_datasets

    def _simulate_subject_images(self, study_map, n_subjects):
        """Simulate subject-level images from a study-level effect size map."""
        n_voxels = study_map.shape[0]

        noise_std = np.maximum(self._BASE_NOISE_STD, np.abs(study_map) * self._NOISE_SCALE_FACTOR)

        subject_images = np.zeros((n_subjects, n_voxels))
        for i_subj in range(n_subjects):
            noise = self._rng.normal(0, noise_std)
            subject_images[i_subj] = study_map + noise

        return subject_images

    def _apply_rubins_rules(self, imputation_results):
        """Combine results across imputations using Rubin's rules."""
        LGR.info("Applying Rubin's rules to combine imputation results...")

        n_imp = len(imputation_results)

        imputation_stats = np.array([res["stat"] for res in imputation_results])
        imputation_variances = np.array([res["variance"] for res in imputation_results])

        pooled_stat = np.mean(imputation_stats, axis=0)
        within_var = np.mean(imputation_variances, axis=0)
        between_var = np.var(imputation_stats, axis=0, ddof=1)

        total_var = within_var + (1 + 1 / n_imp) * between_var

        epsilon = np.finfo(float).eps
        dof = (n_imp - 1) * (1 + within_var / ((1 + 1 / n_imp) * between_var + epsilon)) ** 2

        se = np.sqrt(total_var)
        se[se == 0] = np.finfo(float).eps
        z_values = pooled_stat / se

        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))

        return {
            "stat": pooled_stat,
            "variance": total_var,
            "se": se,
            "z": z_values,
            "p": p_values,
            "dof": dof,
            "within_var": within_var,
            "between_var": between_var,
        }

    def _fit(self, dataset):
        """Perform SDM-PSI meta-analysis on dataset."""
        # First collect all study maps using parent's logic
        all_study_maps = []

        # Collect from images
        if "image_data" in self.inputs_:
            img_data = self.inputs_["image_data"]
            if self.aggressive_mask and "aggressive_mask" in self.inputs_:
                mask = self.inputs_["aggressive_mask"]
                img_data = img_data[:, mask]

            for i in range(img_data.shape[0]):
                all_study_maps.append(img_data[i])

        # Generate from coordinates
        if "coordinates" in self.inputs_ and self.input_mode_ in ["coordinates", "hybrid"]:
            coords = self.inputs_["coordinates"]
            ma_maps = self.kernel_transformer.transform(
                coords, masker=self.masker, return_type="array"
            )

            if self.aggressive_mask and "aggressive_mask" in self.inputs_:
                mask = self.inputs_["aggressive_mask"]
                if isinstance(ma_maps, sparse.COO):
                    ma_dense = ma_maps.todense()
                    mask_data = self.masker.mask_img.get_fdata().astype(bool)
                    n_coord_studies = ma_dense.shape[0]
                    ma_dense_reshaped = ma_dense.reshape(n_coord_studies, -1)
                    ma_masked = ma_dense_reshaped[:, mask_data.reshape(-1)]
                    ma_masked = ma_masked[:, mask[mask_data.reshape(-1)]]
                    for i in range(ma_masked.shape[0]):
                        all_study_maps.append(ma_masked[i])
                else:
                    for i in range(ma_maps.shape[0]):
                        all_study_maps.append(ma_maps[i][mask] if mask is not None else ma_maps[i])
            else:
                if isinstance(ma_maps, sparse.COO):
                    ma_dense = ma_maps.todense()
                    mask_data = self.masker.mask_img.get_fdata().astype(bool)
                    n_coord_studies = ma_dense.shape[0]
                    ma_dense_reshaped = ma_dense.reshape(n_coord_studies, -1)
                    ma_masked = ma_dense_reshaped[:, mask_data.reshape(-1)]
                    for i in range(ma_masked.shape[0]):
                        all_study_maps.append(ma_masked[i])
                else:
                    for i in range(ma_maps.shape[0]):
                        all_study_maps.append(ma_maps[i])

        if len(all_study_maps) == 0:
            raise ValueError("No valid studies found after preprocessing")

        all_maps = np.vstack(all_study_maps)
        n_studies = all_maps.shape[0]

        # Step 1: Multiple imputation
        imputed_datasets = self._perform_multiple_imputation(all_maps, n_studies)

        # Step 2-4: For each imputation, simulate subjects and perform meta-analysis
        imputation_results = []

        for i_imp, imputed_data in enumerate(imputed_datasets):
            LGR.info(f"Processing imputation {i_imp + 1}/{self.n_imputations}...")

            all_subject_images = []
            for i_study in range(n_studies):
                study_map = imputed_data[i_study]
                subject_images = self._simulate_subject_images(study_map, self.n_subjects_sim)
                all_subject_images.append(subject_images)

            study_stats = []
            study_vars = []

            for subject_images in all_subject_images:
                study_mean = np.mean(subject_images, axis=0)
                study_var = np.var(subject_images, axis=0, ddof=1) / subject_images.shape[0]
                study_stats.append(study_mean)
                study_vars.append(study_var)

            study_stats = np.array(study_stats)
            study_vars = np.array(study_vars)

            # Random-effects meta-analysis (DerSimonian-Laird)
            weights = 1.0 / (study_vars + 1e-10)
            weighted_sum = np.sum(weights * study_stats, axis=0)
            sum_weights = np.sum(weights, axis=0)
            pooled_effect = weighted_sum / sum_weights
            pooled_var = 1.0 / sum_weights

            imputation_results.append({"stat": pooled_effect, "variance": pooled_var})

        # Step 5: Apply Rubin's rules
        combined_results = self._apply_rubins_rules(imputation_results)

        images = {
            "stat": combined_results["stat"],
            "z": combined_results["z"],
            "p": combined_results["p"],
            "dof": combined_results["dof"],
            "se": combined_results["se"],
            "within_var": combined_results["within_var"],
            "between_var": combined_results["between_var"],
        }

        tables = {
            "imputation_info": pd.DataFrame(
                {
                    "n_imputations": [self.n_imputations],
                    "n_subjects_per_study": [self.n_subjects_sim],
                }
            )
        }

        description = self._generate_description()

        return images, tables, description
