# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
NiMARE: Neuroimaging Meta-Analysis Research Environment
"""
from .dataset import Dataset
from .meta import cbma
from .version import __version__

del cbma, Dataset
