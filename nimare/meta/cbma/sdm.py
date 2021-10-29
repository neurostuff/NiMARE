"""Seed-based d-mapping-related methods."""
import dijkstra3d
import numpy as np
from nilearn import masking
from scipy import spatial, stats


def compute_sdm_ma(
    ijk,
    effect_sizes,
    sample_sizes,
    significance_level,
    mask_img,
    corr_map,
    alpha=0.5,
    kernel_sigma=5,
):
    """Apply anisotropic kernel to coordinates.

    Parameters
    ----------
    ijk
    effect_sizes
    sample_sizes
    significance_level
    mask_img
    corr_map
    alpha : float
        User-selected degree of anisotropy. Default is 0.5.
    kernel_sigma : float
        User-specified sigma of kernel. Default is 5.
    """
    df = np.sum(sample_sizes) - 2
    effect_size_threshold = stats.t.isf(significance_level, df)
    min_effect_size = -effect_size_threshold  # smallest possible effect size
    max_effect_size = effect_size_threshold  # largest possible effect size
    mask_data = mask_img.get_fdata()
    mask_ijk = np.vstack(np.where(mask_data)).T  # X x 3
    masked_distances = masking.unmask(masking.apply_mask(corr_map, mask_img), mask_img).get_fdata()
    masked_distances = 1 - masked_distances

    peak_corrs = []
    kept_peaks = []
    for i_peak in range(ijk.shape[0]):
        peak_ijk = ijk[i_peak, :]
        peak_t = effect_sizes[i_peak]
        if mask_data[tuple(peak_ijk)] == 0:
            # Skip peaks outside the mask
            continue

        # peak_corr is correlation between target voxel and peak.
        #   For non-adjacent voxels, peak_corr must be estimated with Dijkstra's algorithm.
        peak_corr = dijkstra3d.distance_field(masked_distances, source=peak_ijk)
        peak_corrs.append(peak_corr)
        kept_peaks.append(i_peak)

    kept_ijk = ijk[kept_peaks, :]
    peak_corrs = np.vstack(peak_corrs)
    kept_effect_sizes = effect_sizes[kept_peaks]

    # real_distance is physical distance between voxel and peak
    # we need some way to select the appropriate peak for each voxel
    real_distances = spatial.distance.cdist(kept_ijk, mask_ijk)
    closest_peak = np.argmin(real_distances, axis=0)
    virtual_distances = np.sqrt(
        (1 - alpha) * (real_distances ** 2) + alpha * 2 * kernel_sigma * np.log(peak_corr ** -1)
    )
    y_lower = min_effect_size + np.exp((-(virtual_distances ** 2)) / (2 * kernel_sigma)) * (
        peak_t - min_effect_size
    )
    y_upper = max_effect_size + np.exp((-(virtual_distances ** 2)) / (2 * kernel_sigma)) * (
        peak_t - max_effect_size
    )
    y_lower_img = masking.unmask(y_lower, mask_img)
    y_upper_img = masking.unmask(y_upper, mask_img)
    return y_lower_img, y_upper_img
