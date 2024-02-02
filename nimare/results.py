"""Tools for managing meta-analytic results."""

import copy
import logging
import os

import numpy as np
import pandas as pd
from nibabel.funcs import squeeze_image

from nimare.base import NiMAREBase
from nimare.utils import get_description_references, get_masker

LGR = logging.getLogger(__name__)


class MetaResult(NiMAREBase):
    """Base class for meta-analytic results.

    .. versionchanged:: 0.1.0

        - Added corrector and diagnostics attributes.

    .. versionchanged:: 0.0.12

        - Added the description attribute.

    Parameters
    ----------
    estimator : :class:`~nimare.base.Estimator`
        The Estimator used to generate the maps in the MetaResult.
    corrector : :class:`~nimare.correct.Corrector`
        The Corrector used to correct the maps in the MetaResult.
    diagnostics : :obj:`list` of :class:`~nimare.diagnostics.Diagnostics`
        List of diagnostic classes.
    mask : Niimg-like or `nilearn.input_data.base_masker.BaseMasker`
        Mask for converting maps between arrays and images.
    maps : None or :obj:`dict` of :obj:`numpy.ndarray`, optional
        Maps to store in the object. The maps must be provided as 1D numpy arrays. Default is None.
    tables : None or :obj:`dict` of :obj:`pandas.DataFrame`, optional
        Pandas DataFrames to store in the object. Default is None.
    description_ : :obj:`str`, optional
        Description of the method that generated the result. Default is "".

    Attributes
    ----------
    estimator : :class:`~nimare.base.Estimator`
        The Estimator used to generate the maps in the MetaResult.
    corrector : :class:`~nimare.correct.Corrector`
        The Corrector used to correct the maps in the MetaResult.
    diagnostics : :obj:`list` of :class:`~nimare.diagnostics.Diagnostics`
        List of diagnostic classes.
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    maps : :obj:`dict`
        Keys are map names and values are 1D arrays.
    tables : :obj:`dict`
        Keys are table levels and values are pandas DataFrames.
    description_ : :obj:`str`
        A textual description of the method that generated the result.

        Citations in this description are formatted according to ``natbib``'s LaTeX format.
    bibtex_ : :obj:`str`
        The BibTeX entries for any citations in ``description``.
        These entries are extracted from NiMARE's references.bib file and filtered based on the
        description automatically.

        Users should be able to copy the contents of the ``bibtex`` attribute into their own
        BibTeX file without issue.
    """

    def __init__(
        self,
        estimator,
        corrector=None,
        diagnostics=None,
        mask=None,
        maps=None,
        tables=None,
        description="",
    ):
        self.estimator = copy.deepcopy(estimator)
        self.corrector = copy.deepcopy(corrector)
        diagnostics = diagnostics or []
        self.diagnostics = [copy.deepcopy(diagnostic) for diagnostic in diagnostics]
        self.masker = get_masker(mask)

        maps = maps or {}
        tables = tables or {}

        for map_name, map_ in maps.items():
            if not isinstance(map_, np.ndarray):
                raise ValueError(f"Maps must be numpy arrays. '{map_name}' is a {type(map_)}")

            if map_.ndim != 1:
                LGR.warning(f"Map '{map_name}' should be 1D, not {map_.ndim}D. Squeezing.")
                map_ = np.squeeze(map_)

        for table_name, table in tables.items():
            if not isinstance(table, pd.DataFrame):
                raise ValueError(f"Tables must be DataFrames. '{table_name}' is a {type(table)}")

        self.maps = maps
        self.tables = tables
        self.metadata = {}
        self.description_ = description

    @property
    def description_(self):
        """:obj:`str`: A textual description of the method that generated the result."""
        return self.__description

    @description_.setter
    def description_(self, desc):
        """Automatically extract references when the description is set."""
        self.__description = desc
        self.bibtex_ = get_description_references(desc)

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
        maps = {k: self.get_map(k) for k in names if self.maps[k] is not None}

        for imgtype, img in maps.items():
            filename = prefix + imgtype + ".nii.gz"
            outpath = os.path.join(output_dir, filename)
            img.to_filename(outpath)

    def save_tables(self, output_dir=".", prefix="", prefix_sep="_", names=None):
        """Save result tables to TSV files.

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
            Names of specific tables to write out. If None, save all tables.
            Default is None.
        """
        if prefix == "":
            prefix_sep = ""

        if not prefix.endswith(prefix_sep):
            prefix = prefix + prefix_sep

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        names = names or list(self.tables.keys())
        tables = {k: self.tables[k] for k in names}

        for tabletype, table in tables.items():
            filename = prefix + tabletype + ".tsv"
            outpath = os.path.join(output_dir, filename)
            if table is not None:
                table.to_csv(outpath, sep="\t", index=False)
            else:
                LGR.warning(f"Table {tabletype} is None. Not saving.")

    def copy(self):
        """Return copy of result object."""
        new = MetaResult(
            estimator=self.estimator,
            corrector=self.corrector,
            diagnostics=self.diagnostics,
            mask=self.masker,
            maps=copy.deepcopy(self.maps),
            tables=copy.deepcopy(self.tables),
            description=self.description_,
        )
        return new
