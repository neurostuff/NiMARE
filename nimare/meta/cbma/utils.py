"""
Utilities for coordinate-based meta-analysis estimators
"""
import numpy as np
from scipy import ndimage
from scipy.stats import norm

from ...due import due, Doi, BibTeX


def p_to_z(p, sign):
    """From Neurosynth.
    """
    p = p/2  # convert to two-tailed
    # prevent underflow
    p[p < 1e-240] = 1e-240
    # Convert to z and assign tail
    z = np.abs(norm.ppf(p)) * sign
    # Set very large z's to max precision
    z[np.isinf(z)] = norm.ppf(1e-240)*-1
    return z


def compute_ma(shape, ijk, kernel):
    """
    Generate modeled activation (MA) maps.
    Replaces the values around each focus in ijk with the contrast-specific kernel.
    Takes the element-wise maximum when looping through foci, which accounts for foci
    which are near to one another and may have overlapping kernels.
    Parameters
    ----------
    shape : tuple
        Shape of brain image + buffer. Typically (91, 109, 91) + (30, 30, 30).
    ijk : array-like
        Indices of foci. Each row is a coordinate, with the three columns
        corresponding to index in each of three dimensions.
    kernel : array-like
        3D array of smoothing kernel. Typically of shape (30, 30, 30).

    Returns
    -------
    ma_values : array-like
        1d array of modeled activation values.
    """
    ma_values = np.zeros(shape)
    mid = int(np.floor(kernel.shape[0] / 2.))
    for j_peak in range(ijk.shape[0]):
        i, j, k = ijk[j_peak, :]
        xl = max(i-mid, 0)
        xh = min(i+mid+1, ma_values.shape[0])
        yl = max(j-mid, 0)
        yh = min(j+mid+1, ma_values.shape[1])
        zl = max(k-mid, 0)
        zh = min(k+mid+1, ma_values.shape[2])
        xlk = mid - (i - xl)
        xhk = mid - (i - xh)
        ylk = mid - (j - yl)
        yhk = mid - (j - yh)
        zlk = mid - (k - zl)
        zhk = mid - (k - zh)
        ma_values[xl:xh, yl:yh, zl:zh] = np.maximum(ma_values[xl:xh, yl:yh, zl:zh],
                                                    kernel[xlk:xhk, ylk:yhk, zlk:zhk])
    return ma_values


@due.dcite(Doi('10.1002/hbm.20718'),
           description='Introduces sample size-dependent kernels to ALE.')
def get_ale_kernel(img, n=None, fwhm=None):
    """
    Estimate 3D Gaussian and sigma (in voxels) for ALE kernel given
    sample size (n) or fwhm (in mm).
    """
    if n is not None and fwhm is not None:
        raise ValueError('Only one of n and fwhm may be specified')
    elif n is None and fwhm is None:
        raise ValueError('Either n or fwhm must be provided')
    elif n is not None:
        uncertain_templates = (5.7/(2.*np.sqrt(2./np.pi)) * \
                               np.sqrt(8.*np.log(2.)))  # pylint: disable=no-member
        # Assuming 11.6 mm ED between matching points
        uncertain_subjects = (11.6/(2*np.sqrt(2/np.pi)) * \
                              np.sqrt(8*np.log(2))) / np.sqrt(n)  # pylint: disable=no-member
        fwhm = np.sqrt(uncertain_subjects**2 + uncertain_templates**2)

    fwhm_vox = fwhm / np.sqrt(np.prod(img.header.get_zooms()))
    sigma_vox = fwhm_vox * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)  # pylint: disable=no-member

    data = np.zeros((31, 31, 31))
    mid = int(np.floor(data.shape[0] / 2.))
    data[mid, mid, mid] = 1.
    kernel = ndimage.filters.gaussian_filter(data, sigma_vox, mode='constant')

    # Crop kernel to drop surrounding zeros
    mn = np.min(np.where(kernel > np.spacing(1))[0])
    mx = np.max(np.where(kernel > np.spacing(1))[0])
    kernel = kernel[mn:mx+1, mn:mx+1, mn:mx+1]
    mid = int(np.floor(data.shape[0] / 2.))
    return sigma_vox, kernel
