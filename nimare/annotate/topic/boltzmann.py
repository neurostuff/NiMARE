"""
Topic modeling with deep Boltzmann machines.
"""
from ...base import AnnotationModel
from ...due import due, BibTeX


@due.dcite(BibTeX("""
    @article{DBLP:journals/corr/MontiLLAM16,
      author    = {Ricardo Pio Monti and Romy Lorenz and Robert Leech and
                   Christoforos Anagnostopoulos and Giovanni Montana},
      title     = {Text-mining the NeuroSynth corpus using Deep Boltzmann
                   Machines},
      journal   = {CoRR},
      volume    = {abs/1605.00223},
      year      = {2016},
      url       = {http://arxiv.org/abs/1605.00223},
      archivePrefix = {arXiv},
      eprint    = {1605.00223},
      timestamp = {Wed, 07 Jun 2017 14:42:40 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/MontiLLAM16},
      bibsource = {dblp computer science bibliography, https://dblp.org}}
    """), description='Uses Deep Boltzmann Machines for annotation.')
class BoltzmannModel(AnnotationModel):
    """
    Generate a deep Boltzmann machine topic model [1]_.

    Warnings
    --------
    This method is not yet implemented.

    References
    ----------
    .. [1] Monti, Ricardo, et al. "Text-mining the NeuroSynth corpus using deep
        Boltzmann machines." 2016 International Workshop on Pattern Recognition
        in NeuroImaging (PRNI). IEEE, 2016.
        https://doi.org/10.1109/PRNI.2016.7552329
    """
    def __init__(self, text_df, coordinates_df):
        pass

    def fit(self):
        pass
