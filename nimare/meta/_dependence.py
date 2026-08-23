"""Which images share participants, and the quantities that follow from it.

NiMARE's role here is narrow: work out which images come from the same participants, and
hand PyMARE the labels and weights that follow. The statistics -- cluster-robust covariance,
Satterthwaite degrees of freedom, Brown's scaled chi-squared, Stouffer's variance inflation
term -- all belong to PyMARE.
"""

import numpy as np
from pymare.stats import encode_groups, group_mean

#: Independent units needed to estimate a variance. One unit says nothing about how much a
#: second sample would have differed, so every floor in the library derives from this one:
#: :attr:`DependenceModel.supports_inference` applies it to groups, and
#: :attr:`~nimare.meta.ibma.IBMAEstimator._min_analyses` to the analyses that form them.
MIN_INDEPENDENT_UNITS = 2


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
    blocks : None or :obj:`numpy.ndarray` of shape (K,), optional
        Already-resolved block labels. Only :meth:`for_images` passes this; see
        :attr:`blocks`.

    Attributes
    ----------
    blocks : :obj:`numpy.ndarray` of shape (K,)
        Exchangeability blocks, one label per image, never None. Ungrouped images each become
        their own block, so collapsing a block to its mean is the identity. Resolved once for
        the full set of images and carried through every restriction, so that a label means
        the same thing in every bag.
    """

    def __init__(self, codes, image_indices=None, blocks=None):
        self.codes = np.asarray(codes)
        self.image_indices = (
            np.arange(self.codes.size) if image_indices is None else np.asarray(image_indices)
        )
        if blocks is None:
            blocks = self.codes if self.has_dependence else self.image_indices
        self.blocks = np.asarray(blocks)
        self._encoded = None

    def for_images(self, image_mask):
        """Restrict the grouping to a subset of images, such as one liberal-mask bag."""
        image_mask = np.asarray(image_mask)
        return DependenceModel(
            self.codes[image_mask],
            image_indices=self.image_indices[image_mask],
            blocks=self.blocks[image_mask],
        )

    @property
    def _encoded_blocks(self):
        """Cache :func:`~pymare.stats.encode_groups`, which several properties below want."""
        if self._encoded is None:
            self._encoded = encode_groups(self.blocks, n_observations=self.blocks.size)
        return self._encoded

    @property
    def has_dependence(self):
        """Whether any group holds more than one image.

        ``groupby=False`` is expressed upstream by giving every image its own code, so this
        is the only place dependence is decided.
        """
        return np.unique(self.codes).size < self.codes.size

    @property
    def labels(self):
        """Group labels for PyMARE, or None when no group holds more than one image.

        None is meaningful rather than missing: PyMARE then uses model-based standard errors
        and a normal reference, which is the right inference for independent images.
        """
        return self.codes if self.has_dependence else None

    @property
    def group_order(self):
        """Unique block labels, ordered by first occurrence.

        The order :func:`~pymare.stats.encode_groups` assigns, and so the column order that
        per-group weights and sign flips must follow.
        """
        return self._encoded_blocks[1]

    @property
    def n_groups(self):
        """Number of independent units backing the inference."""
        return int(self.group_order.size)

    def group_rows(self):
        """Yield ``(label, rows)`` for every group, in :attr:`group_order`.

        The enumeration PyMARE's combination tests run over the labels they are handed, so a
        per-group quantity computed here describes the block PyMARE will aggregate.

        Yields
        ------
        label : :obj:`object`
            The block label.
        rows : :obj:`numpy.ndarray`
            Row indices of that block's images, into this grouping rather than the full
            dataset; :attr:`image_indices` translates them back.
        """
        codes, labels = self._encoded_blocks
        for index, label in enumerate(labels):
            yield label, np.flatnonzero(codes == index)

    @property
    def supports_inference(self):
        """Whether there are at least two independent units to compare.

        With one block every image comes from the same participants, so nothing in these
        images says how much a second sample would have differed. Every estimator here needs
        that spread: PyMARE's cluster-robust sandwich, Stouffer's between-group variance and
        the sign-flip null all reject a single block outright.
        """
        return self.n_groups >= MIN_INDEPENDENT_UNITS

    @property
    def dof(self):
        """Degrees of freedom from the group count.

        For the combination tests only. The meta-regression estimators report PyMARE's
        Satterthwaite degrees of freedom instead.
        """
        return self.n_groups - 1

    def per_group(self, values):
        """Reduce one value per image to one per group, in :attr:`group_order`.

        The mean, not the first value: a group whose images disagree about their sample size
        has no right answer, and averaging beats privileging row order.
        """
        return group_mean(
            np.asarray(values, dtype=float)[:, None],
            self._encoded_blocks[0],
        ).ravel()

    def per_image(self, values):
        """Replace each image's value with its group's mean, keeping one value per image.

        The group-constant form of :meth:`per_group`, for the PyMARE estimators that take a
        weight per image but require every image in a group to agree on it.
        """
        return self.per_group(values)[self._encoded_blocks[0]]
