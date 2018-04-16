"""
Image-based meta-analysis estimators
"""
from __future__ import division

from os import mkdir
import os.path as op
from shutil import rmtree

import numpy as np
import nibabel as nib
from scipy import stats
from nipype.interfaces import fsl
from nilearn.masking import unmask, apply_mask
from statsmodels.sandbox.stats.multicomp import multipletests

from .base import IBMAEstimator
from ..base import MetaResult
from ...utils import null_to_p
from ...due import due, BibTeX


@due.dcite(BibTeX("""
           @article{fisher1932statistical,
              title={Statistical methods for research workers, Edinburgh:
                     Oliver and Boyd, 1925},
              author={Fisher, RA},
              journal={Google Scholar},
              year={1932}
              }
           """),
           description='Fishers citation.')
def fishers(z_maps, mask, q=0.05, corr='FWE'):
    """
    Run a Fisher's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
    q : :obj:`float`, optional
        Alpha for multiple comparisons correction.
    corr : :obj:`str` or :obj:`None`, optional
        Multiple comparisons correction method to employ. May be None.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    if corr == 'FDR':
        method = 'fdr_bh'
    elif corr == 'FWE':
        method = 'bonferroni'
    elif corr is None:
        method = None
    else:
        raise ValueError('{0} correction not supported.'.format(corr))

    # Get test-value signs for p-to-z conversion
    sign = np.sign(np.mean(z_maps, axis=0))
    sign[sign == 0] = 1

    k = z_maps.shape[0]
    ffx_stat_map = -2 * np.sum(np.log10(stats.norm.cdf(-z_maps, loc=0,
                                                       scale=1)), axis=0)
    p_map = stats.chi2.sf(ffx_stat_map, 2*k)

    # Multiple comparisons correction
    if corr is not None:
        _, p_corr_map, _, _ = multipletests(p_map, alpha=q, method=method,
                                            is_sorted=False, returnsorted=False)
    else:
        p_corr_map = p_map.copy()
    z_corr_map = -1 * stats.norm.ppf(p_corr_map) * sign
    log_p_map = -np.log10(p_corr_map)

    result = MetaResult(mask=mask, ffx_stat=ffx_stat_map, p=p_corr_map,
                        z=z_corr_map, log_p=log_p_map)
    return result


class Fishers(IBMAEstimator):
    """
    An image-based meta-analytic test using t- or z-statistic images.
    Sum of -log P-values (from T/Zs converted to Ps)

    Requirements:
        - t OR z
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.results = None

    def fit(self, q=0.05, corr='FWE'):
        z_maps = self.dataset.get(self.ids, 'z')
        result = fishers(z_maps, self.mask, q=q, corr=corr)
        self.results = result


@due.dcite(BibTeX("""
           @article{stouffer1949american,
             title={The American soldier: Adjustment during army life.(Studies
                    in social psychology in World War II), Vol. 1},
             author={Stouffer, Samuel A and Suchman, Edward A and DeVinney,
                     Leland C and Star, Shirley A and Williams Jr, Robin M},
             year={1949},
             publisher={Princeton Univ. Press}
             }
           """),
           description='Stouffers citation.')
def stouffers(z_maps, mask, inference='ffx', null='theoretical', n_iters=None,
              q=0.05, corr='FWE'):
    """
    Run a Stouffer's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
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
    q : :obj:`float`, optional
        Alpha for multiple comparisons correction.
    corr : :obj:`str` or :obj:`None`, optional
        Multiple comparisons correction method to employ. May be None.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    if corr == 'FDR':
        method = 'fdr_bh'
    elif corr == 'FWE':
        method = 'bonferroni'
    elif corr is None:
        method = None
    else:
        raise ValueError('{0} correction not supported.'.format(corr))

    sign = np.sign(np.mean(z_maps, axis=0))
    sign[sign == 0] = 1

    if inference == 'rfx':
        t_map, p_map = stats.ttest_1samp(z_maps, popmean=0, axis=0)
        t_map[np.isnan(t_map)] = 0
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
                signs = np.random.choice(a=2, size=k, p=[1-posprop, posprop])
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

        # Multiple comparisons correction
        if corr is not None:
            _, p_corr_map, _, _ = multipletests(p_map, alpha=q, method=method,
                                                is_sorted=False, returnsorted=False)
        else:
            p_corr_map = p_map.copy()

        # Convert p to z, preserving signs
        z_corr_map = -1 * stats.norm.ppf(p_corr_map) * sign
        log_p_map = -np.log10(p_corr_map)
        result = MetaResult(mask=mask, t=t_map, p=p_corr_map, z=z_corr_map,
                            log_p=log_p_map)
    elif inference == 'ffx':
        if null == 'theoretical':
            k = z_maps.shape[0]
            z_map = np.sum(z_maps, axis=0) / np.sqrt(k)
            p_map = stats.norm.cdf(-z_map, loc=0, scale=1)

            # Multiple comparisons correction
            if corr is not None:
                _, p_corr_map, _, _ = multipletests(p_map, alpha=q, method=method,
                                                    is_sorted=False, returnsorted=False)
            else:
                p_corr_map = p_map.copy()

            # Convert p to z, preserving signs
            z_corr_map = -1 * stats.norm.ppf(p_corr_map) * sign
            log_p_map = -np.log10(p_corr_map)
            result = MetaResult(mask=mask, z=z_corr_map, p=p_corr_map,
                                log_p=log_p_map)
        else:
            raise ValueError('Only theoretical null distribution may be used '
                             'for FFX Stouffers.')
    else:
        raise ValueError('Input inference must be "rfx" or "ffx".')
    return result


class Stouffers(IBMAEstimator):
    """
    A t-test on z-statistic images.

    Parameters
    ----------
    dataset : :obj:`nimare.dataset.Dataset`
        Dataset to analyze.
    inference : {'rfx', 'ffx'}
        Whether to run a random- or fixed-effects model.
    null : {'theoretical', 'empirical'}
        Whether to compare test statistics to theoretical or empirical null
        distribution. Empirical null distribution is only possible when
        inference is set to 'rfx'.

    Requirements:
        - z
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.inference = None
        self.null = None
        self.n_iters = None
        self.results = None

    def fit(self, inference='ffx', null='theoretical', n_iters=None, q=0.05,
            corr='FWE'):
        self.inference = inference
        self.null = null
        self.n_iters = n_iters
        z_maps = self.dataset.get(self.ids, 'z')
        result = stouffers(z_maps, self.mask, inference=inference, null=null,
                           n_iters=n_iters, q=q, corr=corr)
        self.results = result


@due.dcite(BibTeX("""
           @article{zaykin2011optimally,
             title={Optimally weighted Z-test is a powerful method for
                    combining probabilities in meta-analysis},
             author={Zaykin, Dmitri V},
             journal={Journal of evolutionary biology},
             volume={24},
             number={8},
             pages={1836--1841},
             year={2011},
             publisher={Wiley Online Library}
             }
           """),
           description='Weighted Stouffers citation.')
def weighted_stouffers(z_maps, sample_sizes, mask, q=0.05, corr='FWE'):
    """
    Run a Stouffer's image-based meta-analysis on z-statistic maps.

    Parameters
    ----------
    z_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of z-statistic maps in the same space, after masking.
    sample_sizes : (n_contrasts,) :obj:`numpy.ndarray`
        A 1D array of sample sizes associated with contrasts in ``z_maps``.
        Must be in same order as rows in ``z_maps``.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
    q : :obj:`float`, optional
        Alpha for multiple comparisons correction.
    corr : :obj:`str` or :obj:`None`, optional
        Multiple comparisons correction method to employ. May be None.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    assert z_maps.shape[0] == sample_sizes.shape[0]
    if corr == 'FDR':
        method = 'fdr_bh'
    elif corr == 'FWE':
        method = 'bonferroni'
    elif corr is None:
        method = None
    else:
        raise ValueError('{0} correction not supported.'.format(corr))

    weighted_z_maps = z_maps * np.sqrt(sample_sizes)[:, None]
    ffx_stat_map = np.sum(weighted_z_maps, axis=0) / np.sqrt(np.sum(sample_sizes))

    p_map = stats.norm.cdf(-ffx_stat_map, loc=0, scale=1)

    # Multiple comparisons correction
    if corr is not None:
        _, p_corr_map, _, _ = multipletests(p_map, alpha=q, method=method,
                                            is_sorted=False, returnsorted=False)
    else:
        p_corr_map = p_map.copy()

    # Convert p to z, preserving signs
    sign = np.sign(ffx_stat_map)
    sign[sign == 0] = 1
    z_corr_map = -1 * stats.norm.ppf(p_corr_map) * sign
    log_p_map = -np.log10(p_corr_map)
    result = MetaResult(mask=mask, ffx_stat=ffx_stat_map, p=p_corr_map,
                        z=z_corr_map, log_p=log_p_map)
    return result


class WeightedStouffers(IBMAEstimator):
    """
    An image-based meta-analytic test using z-statistic images and
    sample sizes.
    Zs from bigger studies get bigger weight

    Requirements:
        - z
        - n
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.results = None

    def fit(self):
        z_maps = self.dataset.get(self.ids, 'z')
        sample_sizes = self.dataset.get(self.ids, 'n')
        result = weighted_stouffers(z_maps, sample_sizes, self.mask)
        self.results = result


def rfx_glm(con_maps, mask, null='theoretical', n_iters=None, q=0.05,
            corr='FWE'):
    """
    Run a random-effects (RFX) GLM on contrast maps.

    Parameters
    ----------
    con_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast maps in the same space, after masking.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
    null : {'theoretical', 'empirical'}, optional
        Whether to use a theoretical null T distribution or an empirically-
        derived null distribution determined via sign flipping.
    n_iters : :obj:`int` or :obj:`None`, optional
        The number of iterations to run in estimating the null distribution.
        Only used if ``null = 'empirical'``.
    q : :obj:`float`, optional
        Alpha for multiple comparisons correction.
    corr : :obj:`str` or :obj:`None`, optional
        Multiple comparisons correction method to employ. May be None.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    if corr == 'FDR':
        method = 'fdr_bh'
    elif corr == 'FWE':
        method = 'bonferroni'
    elif corr is None:
        method = None
    else:
        raise ValueError('{0} correction not supported.'.format(corr))

    # Normalize contrast maps to have unit variance
    con_maps = con_maps / np.std(con_maps, axis=1)[:, None]
    t_map, p_map = stats.ttest_1samp(con_maps, popmean=0, axis=0)
    t_map[np.isnan(t_map)] = 0

    if null == 'empirical':
        k = con_maps.shape[0]
        p_map = np.ones(t_map.shape)
        iter_t_maps = np.zeros((n_iters, t_map.shape[0]))

        data_signs = np.sign(con_maps[con_maps != 0])
        data_signs[data_signs < 0] = 0
        posprop = np.mean(data_signs)
        for i in range(n_iters):
            iter_con_maps = np.copy(con_maps)
            signs = np.random.choice(a=2, size=k, p=[1-posprop, posprop])
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

    # Multiple comparisons correction
    if corr is not None:
        _, p_corr_map, _, _ = multipletests(p_map, alpha=q, method=method,
                                            is_sorted=False, returnsorted=False)
    else:
        p_corr_map = p_map.copy()

    # Convert p to z, preserving signs
    sign = np.sign(t_map)
    sign[sign == 0] = 1
    z_corr_map = -1 * stats.norm.ppf(p_corr_map) * sign
    log_p_map = -np.log10(p_corr_map)
    result = MetaResult(mask=mask, t=t_map, z=z_corr_map, p=p_corr_map,
                        log_p=log_p_map)
    return result


class RFX_GLM(IBMAEstimator):
    """
    A t-test on contrast images.

    Requirements:
        - con
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.null = None
        self.n_iters = None
        self.results = None

    def fit(self, null='theoretical', n_iters=None, q=0.05, corr='FWE'):
        self.null = null
        self.n_iters = n_iters
        con_maps = self.dataset.get(self.ids, 'con')
        result = rfx_glm(con_maps, self.mask, null=self.null,
                         n_iters=self.n_iters, q=q, corr=corr)
        self.results = result


def ffx_glm(con_maps, var_maps, sample_sizes, mask, equal_var=True, q=0.05,
            corr='FWE'):
    """
    Run a fixed-effects GLM on contrast and standard error images.

    Parameters
    ----------
    con_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast maps in the same space, after masking.
    var_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast standard error maps in the same space, after
        masking. Must match shape and order of ``con_maps``.
    sample_sizes : (n_contrasts,) :obj:`numpy.ndarray`
        A 1D array of sample sizes associated with contrasts in ``con_maps``
        and ``var_maps``. Must be in same order as rows in ``con_maps`` and
        ``var_maps``.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
    equal_var : :obj:`bool`, optional
        Whether equal variance is assumed across contrasts. Default is True.
        False is not yet implemented.
    q : :obj:`float`, optional
        Alpha for multiple comparisons correction.
    corr : :obj:`str` or :obj:`None`, optional
        Multiple comparisons correction method to employ. May be None.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.
    """
    assert con_maps.shape == var_maps.shape
    assert con_maps.shape[0] == sample_sizes.shape[0]

    if corr == 'FDR':
        method = 'fdr_bh'
    elif corr == 'FWE':
        method = 'bonferroni'
    elif corr is None:
        method = None
    else:
        raise ValueError('{0} correction not supported.'.format(corr))

    if equal_var:
        weighted_con_maps = con_maps * sample_sizes[:, None]
        sum_weighted_con_map = np.sum(weighted_con_maps, axis=0)
        adj_con_map = (1. / np.sqrt(np.sum(sample_sizes))) * sum_weighted_con_map
        weighted_ss_maps = var_maps * (sample_sizes[:, None] - 1)
        sum_weighted_ss_map = np.sum(weighted_ss_maps, axis=0)
        est_ss_map = (1. / np.sqrt(np.sum(sample_sizes - 1))) * sum_weighted_ss_map
        ffx_stat_map = adj_con_map / np.sqrt(est_ss_map)
        dof = np.sum(sample_sizes - 1)
    else:
        raise Exception('Unequal variances not available yet.')

    p_map = stats.t.cdf(-ffx_stat_map, df=dof, loc=0, scale=1)

    # Multiple comparisons correction
    if corr is not None:
        _, p_corr_map, _, _ = multipletests(p_map, alpha=q, method=method,
                                            is_sorted=False, returnsorted=False)
    else:
        p_corr_map = p_map.copy()

    # Convert p to z, preserving signs
    sign = np.sign(ffx_stat_map)
    sign[sign == 0] = 1
    z_corr_map = -1 * stats.norm.ppf(p_corr_map) * sign
    log_p_map = -np.log10(p_corr_map)
    result = MetaResult(mask=mask, ffx_stat=ffx_stat_map, z=z_corr_map,
                        p=p_corr_map, log_p=log_p_map)
    return result


class FFX_GLM(IBMAEstimator):
    """
    An image-based meta-analytic test using contrast and standard error images.
    Don't estimate variance, just take from first level

    Requirements:
        - con
        - se
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids
        self.sample_sizes = None
        self.equal_var = None

    def fit(self, sample_sizes=None, equal_var=True, q=0.05, corr='FWE'):
        """
        Perform meta-analysis given parameters.
        """
        self.sample_sizes = sample_sizes
        self.equal_var = equal_var
        con_maps = self.dataset.get(self.ids, 'con')
        var_maps = self.dataset.get(self.ids, 'con_se')
        k = con_maps.shape[0]
        if self.sample_sizes is not None:
            sample_sizes = np.repeat(self.sample_sizes, k)
        else:
            sample_sizes = self.dataset.get(self.ids, 'n')
        result = ffx_glm(con_maps, var_maps, sample_sizes, self.mask,
                         equal_var=self.equal_var, q=q, corr=corr)
        self.results = result


def mfx_glm(con_maps, se_maps, sample_sizes, mask, q=0.05, cdt=0.01,
            work_dir='mfx_glm'):
    """
    Run a mixed-effects GLM on contrast and standard error images.

    Parameters
    ----------
    con_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast maps in the same space, after masking.
    var_maps : (n_contrasts, n_voxels) :obj:`numpy.ndarray`
        A 2D array of contrast standard error maps in the same space, after
        masking. Must match shape and order of ``con_maps``.
    sample_sizes : (n_contrasts,) :obj:`numpy.ndarray`
        A 1D array of sample sizes associated with contrasts in ``con_maps``
        and ``var_maps``. Must be in same order as rows in ``con_maps`` and
        ``var_maps``.
    mask : :obj:`nibabel.Nifti1Image`
        Mask image, used to unmask results maps in compiling output.
    equal_var : :obj:`bool`, optional
        Whether equal variance is assumed across contrasts. Default is True.
        False is not yet implemented.
    q : :obj:`float`, optional
        Alpha for multiple comparisons correction.
    corr : :obj:`str` or :obj:`None`, optional
        Multiple comparisons correction method to employ. May be None.

    Returns
    -------
    result : :obj:`nimare.meta.MetaResult`
        MetaResult object containing maps for test statistics, p-values, and
        negative log(p) values.

    TODO
    ----
    Step 1: Concatenate con_maps into 4D image and save to file in working
            directory
    Step 2: Repeat with var_maps
    Step 3: Write out mask image to file in working directory
    Step 4: Create design file
    Step 5: Create t contrast file
    Step 6: Create covariance split file
    Step 7: Create DOF file for varcopes
    Step 8: Run flameo with --runmode=flame1, setting --logdir to working
            directory
    """
    assert con_maps.shape == se_maps.shape
    assert con_maps.shape[0] == sample_sizes.shape[0]

    if 0 < cdt < 1:
        cdt_z = stats.norm.ppf(1-(cdt/2))
    else:
        cdt_z = cdt

    work_dir = op.abspath(work_dir)
    if op.isdir(work_dir):
        raise ValueError('Working directory already '
                         'exists: "{0}"'.format(work_dir))

    mkdir(work_dir)
    cope_file = op.join(work_dir, 'cope.nii.gz')
    varcope_file = op.join(work_dir, 'varcope.nii.gz')
    mask_file = op.join(work_dir, 'mask.nii.gz')
    design_file = op.join(work_dir, 'design.mat')
    tcon_file = op.join(work_dir, 'design.con')
    cov_split_file = op.join(work_dir, 'cov_split.mat')
    dof_file = op.join(work_dir, 'dof.nii.gz')

    dofs = (np.array(sample_sizes) - 1).astype(str)

    con_maps[np.isnan(con_maps)] = 0
    cope_4d_img = unmask(con_maps, mask)
    se_maps[np.isnan(se_maps)] = 0
    varcope_4d_img = unmask(se_maps, mask)
    dof_maps = np.ones(con_maps.shape)
    for i in range(len(dofs)):
        dof_maps[i, :] = dofs[i]
    dof_4d_img = unmask(dof_maps, mask)

    # Covariance splitting file
    cov_data = ['/NumWaves\t1',
                '/NumPoints\t{0}'.format(con_maps.shape[0]),
                '',
                '/Matrix']
    cov_data += ['1'] * con_maps.shape[0]
    with open(cov_split_file, 'w') as fo:
        fo.write('\n'.join(cov_data))

    # T contrast file
    tcon_data = ['/ContrastName1 MFX-GLM',
                 '/NumWaves\t1',
                 '/NumPoints\t1',
                 '',
                 '/Matrix',
                 '1']
    with open(tcon_file, 'w') as fo:
        fo.write('\n'.join(tcon_data))

    cope_4d_img.to_filename(cope_file)
    varcope_4d_img.to_filename(varcope_file)
    dof_4d_img.to_filename(dof_file)
    mask.to_filename(mask_file)

    design_matrix = ['/NumWaves\t1',
                     '/NumPoints\t{0}'.format(con_maps.shape[0]),
                     '/PPheights\t1',
                     '',
                     '/Matrix']
    design_matrix += ['1'] * con_maps.shape[0]
    with open(design_file, 'w') as fo:
        fo.write('\n'.join(design_matrix))

    flameo = fsl.FLAMEO()
    flameo.inputs.cope_file = cope_file
    flameo.inputs.var_cope_file = varcope_file
    flameo.inputs.cov_split_file = cov_split_file
    flameo.inputs.design_file = design_file
    flameo.inputs.t_con_file = tcon_file
    flameo.inputs.mask_file = mask_file
    flameo.inputs.run_mode = 'flame1'
    flameo.inputs.dof_var_cope_file = dof_file
    res = flameo.run()

    # FWE correction
    # Estimate smoothness
    est = fsl.model.SmoothEstimate()
    est.inputs.dof = con_maps.shape[0] - 1
    est.inputs.mask_file = mask_file
    est.inputs.residual_fit_file = res.outputs.res4d
    est_res = est.run()

    # Cluster
    cl = fsl.model.Cluster()
    cl.inputs.threshold = cdt_z
    cl.inputs.pthreshold = q
    cl.inputs.in_file = res.outputs.zstats
    cl.inputs.cope_file = res.outputs.copes
    cl.inputs.use_mm = True
    cl.inputs.dlh = est_res.outputs.dlh
    cl.inputs.volume = est_res.outputs.volume
    cl.inputs.out_threshold_file = op.join(work_dir, 'thresh_zstat1.nii.gz')
    cl.inputs.connectivity = 26
    cl.inputs.out_localmax_txt_file = op.join(work_dir, 'lmax_zstat1_tal.txt')
    cl_res = cl.run()

    out_cope_img = nib.load(res.outputs.copes)
    out_t_img = nib.load(res.outputs.tstats)
    out_z_img = nib.load(res.outputs.zstats)
    out_cope_map = apply_mask(out_cope_img, mask)
    out_t_map = apply_mask(out_t_img, mask)
    out_z_map = apply_mask(out_z_img, mask)
    thresh_z_map = apply_mask(nib.load(cl_res.outputs.threshold_file), mask)

    print('Cleaning up...')
    rmtree(work_dir)
    rmtree(res.outputs.stats_dir)

    # Compile outputs
    out_p_map = stats.norm.sf(abs(out_z_map)) * 2
    log_p_map = -np.log10(out_p_map)
    result = MetaResult(mask=mask, cope=out_cope_map, z=out_z_map,
                        thresh_z=thresh_z_map,
                        t=out_t_map, p=out_p_map, log_p=log_p_map)
    return result


class MFX_GLM(IBMAEstimator):
    """
    The gold standard image-based meta-analytic test. Uses contrast and standard
    error images.

    Requirements:
        - con
        - se
    """
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.mask = self.dataset.mask
        self.ids = ids

    def fit(self):
        con_maps = self.dataset.get(self.ids, 'con')
        var_maps = self.dataset.get(self.ids, 'con_se')
