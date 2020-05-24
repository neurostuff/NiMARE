"""
Automated annotation tools
"""
import warnings

from . import (boltzmann, cogat, cogpo, gclda, lda, text, text2brain, utils,
               word2brain)

__all__ = ['boltzmann', 'cogat', 'cogpo', 'gclda', 'lda', 'text', 'text2brain',
           'utils', 'word2brain']

warnings.simplefilter('default')

warnings.warn(
    "{} is an experimental module under active development; use it at your "
    "own risk.".format(__name__),
    ImportWarning
)
