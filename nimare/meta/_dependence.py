"""Which images share participants, and the quantities that follow from it.

NiMARE's role here is narrow: work out which images come from the same participants, and
hand PyMARE the labels and weights that follow. The statistics -- cluster-robust covariance,
Satterthwaite degrees of freedom, Brown's scaled chi-squared, Stouffer's variance inflation
term -- all belong to PyMARE.
"""

import numpy as np
from pymare.stats import encode_groups, group_mean


def hashable_label(value):
    """Return a hashable stand-in for one group label.

    Metadata fields arrive as lists (``sample_sizes`` is ``[25]``, not ``25``), which cannot
    go into the set used to assign group codes. Tuples hash by value, so images carrying
    equal metadata still land in the same group.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(np.asarray(value).ravel().tolist())
    return value


class DependenceModel:
    """One resolved grouping of images, and the quantities derived from it.

    Built per fitted model rather than stored on the estimator: the liberal-mask path fits
    bags holding only some of a group's images, so the grouping has to describe the bag.

    Parameters
    ----------
    codes : :obj:`numpy.ndarray` of shape (K,)
        One integer group code per image. Images sharing a code are assumed to come from the
        same participants.
    image_indices : None or :obj:`numpy.ndarray` of shape (K,), optional
        Which images of the full dataset these rows are. Default is all of them, in order.
    enabled : :obj:`bool`, optional
        False (``groupby=False``) makes every image independent regardless of the codes.
        Default is True.
    """

    def __init__(self, codes, image_indices=None, enabled=True):
        self.codes = np.asarray(codes)
        self.image_indices = (
            np.arange(self.codes.size) if image_indices is None else np.asarray(image_indices)
        )
        self.enabled = bool(enabled)

    def for_images(self, image_mask):
        """Restrict the grouping to a subset of images, such as one liberal-mask bag."""
        image_mask = np.asarray(image_mask)
        return DependenceModel(
            self.codes[image_mask],
            image_indices=self.image_indices[image_mask],
            enabled=self.enabled,
        )

    @property
    def has_dependence(self):
        """Whether any group holds more than one image."""
        return self.enabled and np.unique(self.codes).size < self.codes.size

    @property
    def labels(self):
        """Group labels for PyMARE, or None when no group holds more than one image.

        None is meaningful rather than missing: PyMARE then uses model-based standard errors
        and a normal reference, which is the right inference for independent images.
        """
        return self.codes if self.has_dependence else None

    @property
    def blocks(self):
        """Exchangeability blocks, one label per image, never None.

        Ungrouped images each become their own block, so collapsing a block to its mean is
        the identity. Those blocks are dataset-wide image indices, which keeps a null drawn
        once for the whole brain indexable from every liberal-mask bag.
        """
        return self.codes if self.has_dependence else self.image_indices

    @property
    def group_order(self):
        """Unique block labels, ordered by first occurrence.

        The order :func:`~pymare.stats.encode_groups` assigns, and so the column order that
        per-group weights and sign flips must follow.
        """
        return encode_groups(self.blocks, n_observations=self.blocks.size)[1]

    @property
    def n_groups(self):
        """Number of independent units backing the inference."""
        return int(self.group_order.size)

    @property
    def dof(self):
        """Degrees of freedom from the group count, floored at one.

        For the combination tests only. The meta-regression estimators report PyMARE's
        Satterthwaite degrees of freedom instead.
        """
        return max(self.n_groups - 1, 1)

    def per_group(self, values):
        """Reduce one value per image to one per group, in :attr:`group_order`.

        The mean, not the first value: a group whose images disagree about their sample size
        has no right answer, and averaging beats privileging row order.
        """
        codes = encode_groups(self.blocks, n_observations=self.blocks.size)[0]
        return group_mean(np.asarray(values, dtype=float)[:, None], codes).ravel()
