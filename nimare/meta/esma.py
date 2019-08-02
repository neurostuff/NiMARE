"""
Effect-size meta-analysis functions
"""
from __future__ import division

import logging

import numpy as np
from scipy import stats

from ..stats import null_to_p, p_to_z
from ..due import due
from .. import references

LGR = logging.getLogger(__name__)


@due.dcite(references.FISHERS, description='Fishers citation.')
def fishers(z_maps, two_sided=True):
    """
    Run a Fisher's image-based meta-analysis on z-statistics.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistics.
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    Returns
    -------
    result : :obj:`dict`
        Dictionary containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    # Get test-value signs for p-to-z conversion
    sign = np.sign(np.mean(z_maps, axis=0))
    sign[sign == 0] = 1

    k = z_maps.shape[0]

    if two_sided:
        # two-tailed method
        ffx_stat_map = -2 * np.sum(np.log(stats.norm.sf(np.abs(z_maps), loc=0,
                                                        scale=1) * 2), axis=0)
    else:
        # one-tailed method
        ffx_stat_map = -2 * np.sum(np.log(stats.norm.cdf(-z_maps, loc=0,
                                                         scale=1)), axis=0)
    p_map = stats.chi2.sf(ffx_stat_map, 2 * k)
    z_map = p_to_z(p_map, tail='two') * sign
    log_p_map = -np.log10(p_map)

    return dict(ffx_stat=ffx_stat_map, p=p_map, z=z_map, log_p=log_p_map)


@due.dcite(references.STOUFFERS, description='Stouffers citation.')
def stouffers(z_maps, inference='ffx', null='theoretical', n_iters=None,
              two_sided=True):
    """
    Run a Stouffer's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    inference : {'ffx', 'rfx'}, optional
        Whether to use fixed-effects inference (default) or random-effects
        inference.
    null : {'theoretical', 'empirical'}, optional
        Whether to use a theoretical null T distribution or an empirically-
        derived null distribution determined via sign flipping. Empirical null
        is only possible if ``inference = 'rfx'``.
    n_iters : :obj:`int` or :obj:`None`, optional
        The number of iterations to run in estimating the null distribution.
        Only used if ``inference = 'rfx'`` and ``null = 'empirical'``.
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    Returns
    -------
    result : :obj:`dict`
        Dictionary containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    sign = np.sign(np.mean(z_maps, axis=0))
    sign[sign == 0] = 1

    if inference == 'rfx':
        t_map, p_map = stats.ttest_1samp(z_maps, popmean=0, axis=0)
        t_map[np.isnan(t_map)] = 0
        p_map[np.isnan(p_map)] = 1

        if not two_sided:
            # MATLAB one-tailed method
            p_map = stats.t.cdf(-t_map, df=z_maps.shape[0] - 1)

        if null == 'empirical':
            k = z_maps.shape[0]
            p_map = np.ones(t_map.shape)
            iter_t_maps = np.zeros((n_iters, t_map.shape[0]))

            data_signs = np.sign(z_maps[z_maps != 0])
            data_signs[data_signs < 0] = 0
            posprop = np.mean(data_signs)
            for i in range(n_iters):
                # Randomly flip signs of z-maps based on proportion of z-value
                # signs across all maps.
                iter_z_maps = np.copy(z_maps)
                signs = np.random.choice(a=2, size=k, p=[1 - posprop, posprop])
                signs[signs == 0] = -1
                iter_z_maps *= signs[:, None]
                iter_t_maps[i, :], _ = stats.ttest_1samp(iter_z_maps,
                                                         popmean=0, axis=0)
            iter_t_maps[np.isnan(iter_t_maps)] = 0

            for voxel in range(iter_t_maps.shape[1]):
                p_map[voxel] = null_to_p(t_map[voxel], iter_t_maps[:, voxel])

            # Crop p-values of 0 or 1 to nearest values that won't evaluate to
            # 0 or 1. Prevents inf z-values.
            p_map[p_map < 1e-16] = 1e-16
            p_map[p_map > (1. - 1e-16)] = 1. - 1e-16
        elif null != 'theoretical':
            raise ValueError('Input null must be "theoretical" or "empirical".')

        # Convert p to z, preserving signs
        z_map = p_to_z(p_map, tail='two') * sign
        log_p_map = -np.log10(p_map)
        images = {'t': t_map,
                  'p': p_map,
                  'z': z_map,
                  'log_p': log_p_map}
    elif inference == 'ffx':
        if null == 'theoretical':
            k = z_maps.shape[0]
            z_map = np.sum(z_maps, axis=0) / np.sqrt(k)
            sign = np.sign(z_map)
            sign[sign == 0] = 1
            if two_sided:
                # two-tailed method
                p_map = stats.norm.sf(np.abs(z_map)) * 2
            else:
                # one-tailed method
                p_map = stats.norm.cdf(-z_map, loc=0, scale=1)

            # Convert p to z, preserving signs
            z_map = p_to_z(p_map, tail='two') * sign
            log_p_map = -np.log10(p_map)
            images = {'z': z_map,
                      'p': p_map,
                      'log_p': log_p_map}
        else:
            raise ValueError('Only theoretical null distribution may be used '
                             'for FFX Stouffers.')
    else:
        raise ValueError('Input inference must be "rfx" or "ffx".')
    return images


@due.dcite(references.WEIGHTED_STOUFFERS, description='Weighted Stouffers citation.')
def weighted_stouffers(z_maps, sample_sizes, two_sided=True):
    """
    Run a Stouffer's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    sample_sizes : (n_contrasts,) :obj:`numpy.ndarray`
        A 1D array of sample sizes associated with contrasts in ``z_maps``.
        Must be in same order as rows in ``z_maps``.
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    Returns
    -------
    result : :obj:`dict`
        Dictionary containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    assert z_maps.shape[0] == sample_sizes.shape[0]

    weighted_z_maps = z_maps * np.sqrt(sample_sizes)[:, None]
    ffx_stat_map = np.sum(weighted_z_maps, axis=0) / np.sqrt(np.sum(sample_sizes))

    if two_sided:
        # two-tailed method
        p_map = stats.norm.sf(np.abs(ffx_stat_map)) * 2
    else:
        # one-tailed method
        p_map = stats.norm.cdf(-ffx_stat_map, loc=0, scale=1)

    # Convert p to z, preserving signs
    sign = np.sign(ffx_stat_map)
    sign[sign == 0] = 1
    z_map = p_to_z(p_map, tail='two') * sign
    log_p_map = -np.log10(p_map)
    images = {'ffx_stat': ffx_stat_map,
              'p': p_map,
              'z': z_map,
              'log_p': log_p_map}
    return images


def rfx_glm(con_maps, null='theoretical', n_iters=None, two_sided=True):
    """
    Run a random-effects (RFX) GLM on contrast maps.

    Parameters
    ----------
    con_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast maps in the same space, after masking.
    null : {'theoretical', 'empirical'}, optional
        Whether to use a theoretical null T distribution or an empirically-
        derived null distribution determined via sign flipping.
        Default is 'theoretical'.
    n_iters : :obj:`int` or :obj:`None`, optional
        The number of iterations to run in estimating the null distribution.
        Only used if ``null = 'empirical'``.
    two_sided : :obj:`bool`, optional
        Whether to do a two- or one-sided test. Default is True.

    Returns
    -------
    result : :obj:`dict`
        Dictionary object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    # Normalize contrast maps to have unit variance
    con_maps = con_maps / np.std(con_maps, axis=1)[:, None]
    t_map, p_map = stats.ttest_1samp(con_maps, popmean=0, axis=0)
    t_map[np.isnan(t_map)] = 0
    p_map[np.isnan(p_map)] = 1

    if not two_sided:
        # MATLAB one-tailed method
        p_map = stats.t.cdf(-t_map, df=con_maps.shape[0] - 1)

    if null == 'empirical':
        k = con_maps.shape[0]
        p_map = np.ones(t_map.shape)
        iter_t_maps = np.zeros((n_iters, t_map.shape[0]))

        data_signs = np.sign(con_maps[con_maps != 0])
        data_signs[data_signs < 0] = 0
        posprop = np.mean(data_signs)
        for i in range(n_iters):
            iter_con_maps = np.copy(con_maps)
            signs = np.random.choice(a=2, size=k, p=[1 - posprop, posprop])
            signs[signs == 0] = -1
            iter_con_maps *= signs[:, None]
            iter_t_maps[i, :], _ = stats.ttest_1samp(iter_con_maps, popmean=0,
                                                     axis=0)
        iter_t_maps[np.isnan(iter_t_maps)] = 0

        for voxel in range(iter_t_maps.shape[1]):
            p_map[voxel] = null_to_p(t_map[voxel], iter_t_maps[:, voxel])

        # Crop p-values of 0 or 1 to nearest values that won't evaluate to
        # 0 or 1. Prevents inf z-values.
        p_map[p_map < 1e-16] = 1e-16
        p_map[p_map > (1. - 1e-16)] = 1. - 1e-16
    elif null != 'theoretical':
        raise ValueError('Input null must be "theoretical" or "empirical".')

    # Convert p to z, preserving signs
    sign = np.sign(t_map)
    sign[sign == 0] = 1
    z_map = p_to_z(p_map, tail='two') * sign
    log_p_map = -np.log10(p_map)
    images = {'t': t_map,
              'z': z_map,
              'p': p_map,
              'log_p': log_p_map}
    return images
