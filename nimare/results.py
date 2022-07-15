"""Tools for managing meta-analytic results."""
import copy
import logging
import os

from nibabel.funcs import squeeze_image

from nimare.utils import get_description_references, get_masker

LGR = logging.getLogger(__name__)


class MetaResult(object):
    """Base class for meta-analytic results.

    .. versionchanged:: 0.0.12
        - Added the description attribute.

    Parameters
    ----------
    estimator : :class:`~nimare.base.Estimator`
        The Estimator used to generate the maps in the MetaResult.
    mask : Niimg-like or `nilearn.input_data.base_masker.BaseMasker`
        Mask for converting maps between arrays and images.
    maps : :obj:`dict` or None, optional
        Maps to store in the object. Default is None.
    description : :obj:`str`, optional
        Description of the method that generated the result. Default is "".

    Attributes
    ----------
    estimator : :class:`~nimare.base.Estimator`
        The Estimator used to generate the maps in the MetaResult.
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    maps : :obj:`dict`
        Keys are map names and values are arrays.
    description : :obj:`str`
        A textual description of the method that generated the result.

        Citations in this description are formatted according to ``natbib``'s LaTeX format.
    bibtex : :obj:`str`
        The BibTeX entries for any citations in ``description``.
        These entries are extracted from NiMARE's references.bib file and filtered based on the
        description automatically.

        Users should be able to copy the contents of the ``bibtex`` attribute into their own
        BibTeX file without issue.
    """

    def __init__(self, estimator, mask, maps=None, description=""):
        self.estimator = copy.deepcopy(estimator)
        self.masker = get_masker(mask)
        self.maps = maps or {}
        self.description = description
        self.bibtex = get_description_references(description)

    def get_map(self, name, return_type="image"):
        """Get stored map as image or array.

        Parameters
        ----------
        name : :obj:`str`
            Name of the map. Used to index self.maps.
        return_type : {'image', 'array'}, optional
            Whether to return a niimg ('image') or a numpy array.
            Default is 'image'.
        """
        m = self.maps.get(name)
        if m is None:
            raise ValueError(f"No map with name '{name}' found.")
        if return_type == "image":
            # pending resolution of https://github.com/nilearn/nilearn/issues/2724
            try:
                return self.masker.inverse_transform(m)
            except IndexError:
                return squeeze_image(self.masker.inverse_transform([m]))
        return m

    def save_maps(self, output_dir=".", prefix="", prefix_sep="_", names=None):
        """Save results to files.

        Parameters
        ----------
        output_dir : :obj:`str`, optional
            Output directory in which to save results. If the directory doesn't
            exist, it will be created. Default is current directory.
        prefix : :obj:`str`, optional
            Prefix to prepend to output file names.
            Default is None.
        prefix_sep : :obj:`str`, optional
            Separator to add between prefix and default file names.
            Default is _.
        names : None or :obj:`list` of :obj:`str`, optional
            Names of specific maps to write out. If None, save all maps.
            Default is None.
        """
        if prefix == "":
            prefix_sep = ""

        if not prefix.endswith(prefix_sep):
            prefix = prefix + prefix_sep

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        names = names or list(self.maps.keys())
        maps = {k: self.get_map(k) for k in names}

        for imgtype, img in maps.items():
            filename = prefix + imgtype + ".nii.gz"
            outpath = os.path.join(output_dir, filename)
            img.to_filename(outpath)

    def copy(self):
        """Return copy of result object."""
        new = MetaResult(
            self.estimator,
            mask=self.masker,
            maps=copy.deepcopy(self.maps),
            description=self.description,
        )
        return new
