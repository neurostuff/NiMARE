"""References to be imported and injected at relevant places throughout the library."""
from .due import BibTeX, Doi

TEXT2BRAIN = Doi("https://doi.org/10.1007/978-3-030-00931-1_67")

WORD2BRAIN = Doi("10.1101/299024")

BOLTZMANNMODEL = BibTeX(
    """
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
    """
)

GCLDAMODEL = Doi("10.1371/journal.pcbi.1005649")

LDA = BibTeX(
    """
    @article{blei2003latent,
      title={Latent dirichlet allocation},
      author={Blei, David M and Ng, Andrew Y and Jordan, Michael I},
      journal={Journal of machine Learning research},
      volume={3},
      number={Jan},
      pages={993--1022},
      year={2003}}
    """
)

MALLET = BibTeX(
    """
    @article{mallettoolbox,
      title={MALLET: A Machine Learning for Language Toolkit.},
      author={McCallum, Andrew K},
      year={2002}}
    """
)

LDAMODEL = Doi("10.1371/journal.pcbi.1002707")

COGNITIVE_ATLAS = Doi("10.3389/fninf.2011.00017")

COGNITIVE_PARADIGM_ONTOLOGY = Doi("10.1007/s12021-011-9126-x")

ATHENA = Doi("10.3389/fnins.2019.00494")

LOG_LIKELIHOOD = Doi("10.1145/1577069.1755845")

GCLDA_DECODING = Doi("10.1371/journal.pcbi.1005649")

NEUROSYNTH = Doi("10.1038/nmeth.1635")

BRAINMAP_DECODING = Doi("10.1007/s00429-013-0698-0")

ALE1 = BibTeX(
    """
    @article{turkeltaub2002meta,
      title={Meta-analysis of the functional neuroanatomy of single-word
             reading: method and validation},
      author={Turkeltaub, Peter E and Eden, Guinevere F and Jones,
              Karen M and Zeffiro, Thomas A},
      journal={Neuroimage},
      volume={16},
      number={3},
      pages={765--780},
      year={2002},
      publisher={Elsevier}
    }
    """
)

ALE2 = Doi("10.1002/hbm.21186")

ALE3 = Doi("10.1016/j.neuroimage.2011.09.017")

ALE_KERNEL = Doi("10.1002/hbm.20718")

SCALE = Doi("10.1016/j.neuroimage.2014.06.007")

MKDA = Doi("10.1093/scan/nsm015")

KDA1 = Doi("10.1016/S1053-8119(03)00078-8")

KDA2 = Doi("10.1016/j.neuroimage.2004.03.052")

BHICP = Doi("10.1198/jasa.2011.ap09735")

HPGRF = BibTeX(
    """
    @article{kang2014bayesian,
      title={A Bayesian hierarchical spatial point process model for
             multi-type neuroimaging meta-analysis},
      author={Kang, Jian and Nichols, Thomas E and Wager, Tor D and
              Johnson, Timothy D},
      journal={The annals of applied statistics},
      volume={8},
      number={3},
      pages={1800},
      year={2014},
      publisher={NIH Public Access}
      }
    """
)

SBLFR = Doi("10.1111/biom.12713")

SBR = Doi("10.1214/11-AOAS523")

PEAKS2MAPS = Doi("10.7490/f1000research.1116395.1")

FISHERS = BibTeX(
    """
    @article{fisher1932statistical,
       title={Statistical methods for research workers, Edinburgh:
              Oliver and Boyd, 1925},
       author={Fisher, RA},
       journal={Google Scholar},
       year={1932}
       }
    """
)

STOUFFERS = BibTeX(
    """
    @article{stouffer1949american,
      title={The American soldier: Adjustment during army life. (Studies
             in social psychology in World War II), Vol. 1},
      author={Stouffer, Samuel A and Suchman, Edward A and DeVinney,
              Leland C and Star, Shirley A and Williams Jr, Robin M},
      year={1949},
      publisher={Princeton Univ. Press}
      }
    """
)

WEIGHTED_STOUFFERS = BibTeX(
    """
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
    """
)

CBP = Doi("10.1002/hbm.22138")

MAMP = Doi("10.1016/j.neuroimage.2015.08.027")

MAPBOT = Doi("10.1016/j.neuroimage.2017.06.032")

T2Z_TRANSFORM = BibTeX(
    """
    @article{hughett2007accurate,
      title={Accurate Computation of the F-to-z and t-to-z Transforms
             for Large Arguments},
      author={Hughett, Paul and others},
      journal={Journal of Statistical Software},
      volume={23},
      number={1},
      pages={1--5},
      year={2007},
      publisher={Foundation for Open Access Statistics}
    }
    """
)

T2Z_IMPLEMENTATION = Doi("10.5281/zenodo.32508")

LANCASTER_TRANSFORM = Doi("10.1002/hbm.20345")

LANCASTER_TRANSFORM_VALIDATION = Doi("10.1016/j.neuroimage.2010.02.048")

META_CLUSTER = Doi("10.1016/j.neuroimage.2015.06.044")

META_CLUSTER2 = Doi("10.1162/netn_a_00050")

META_ICA = Doi("10.1162/jocn_a_00077")

META_ICA2 = Doi("10.1162/jocn_a_00077")
