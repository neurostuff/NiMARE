"""
Topic modeling with latent Dirichlet allocation via MALLET.
"""
import os
import os.path as op
import shutil
import logging
import subprocess

import pandas as pd

from ..base import AnnotationModel
from ...utils import get_resource_path
from ...due import due
from ... import references

LGR = logging.getLogger(__name__)


@due.dcite(references.LDA, description='Introduces LDA.')
@due.dcite(references.MALLET, description='Citation for MALLET toolbox')
@due.dcite(references.LDAMODEL,
           description='First use of LDA for automated annotation of '
           'neuroimaging literature.')
class LDAModel(AnnotationModel):
    """
    Perform topic modeling using Latent Dirichlet Allocation [1]_ with the
    Java toolbox MALLET [2]_, as performed in [3]_.

    Parameters
    ----------
    text_df : :obj:`pandas.DataFrame`
        A pandas DataFrame with two columns ('id' and 'text') containing
        article text_df.
    n_topics : :obj:`int`, optional
        Number of topics to generate. Default=50.
    n_words : :obj:`int`, optional
        Number of top words to return for each topic. Default=31, based on
        Poldrack et al. (2012). Not used.
    n_iters : :obj:`int`, optional
        Number of iterations to run in training topic model. Default=1000.
    alpha : :obj:`float`, optional
        The Dirichlet prior on the per-document topic distributions.
        Default: 50 / n_topics, based on Poldrack et al. (2012).
    beta : :obj:`float`, optional
        The Dirichlet prior on the per-topic word distribution. Default: 0.001,
        based on Poldrack et al. (2012).

    References
    ----------
    .. [1] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent
        dirichlet allocation." Journal of machine Learning research 3.Jan
        (2003): 993-1022.
    .. [2] McCallum, Andrew Kachites. "Mallet: A machine learning for language
        toolkit." (2002).
    .. [3] Poldrack, Russell A., et al. "Discovering relations between mind,
        brain, and mental disorders using topic mapping." PLoS computational
        biology 8.10 (2012): e1002707.
        https://doi.org/10.1371/journal.pcbi.1002707
    """
    def __init__(self, text_df, n_topics=50, n_iters=1000, alpha='auto',
                 beta=0.001):
        resdir = op.abspath(get_resource_path())
        tempdir = op.join(resdir, 'topic_models')
        text_dir = op.join(tempdir, 'texts')
        if not op.isdir(tempdir):
            os.mkdir(tempdir)

        if alpha == 'auto':
            alpha = 50. / n_topics
        elif not isinstance(alpha, float):
            raise ValueError('Argument alpha must be float or "auto"')

        self.params = {
            'n_topics': n_topics,
            'n_iters': n_iters,
            'alpha': alpha,
            'beta': beta,
        }

        # Check for presence of text files and convert if necessary
        if not op.isdir(text_dir):
            LGR.info('Texts folder not found. Creating text files...')
            os.mkdir(text_dir)
            for id_ in text_df.index.values:
                text = text_df.loc[id_]['text']
                with open(op.join(text_dir, str(id_) + '.txt'), 'w') as fo:
                    fo.write(text)

        # Run MALLET topic modeling
        LGR.info('Generating topics...')
        mallet_bin = op.join(op.dirname(op.dirname(__file__)),
                             'resources/mallet/bin/mallet')
        import_str = ('{mallet} import-dir '
                      '--input {text_dir} '
                      '--output {outdir}/topic-input.mallet '
                      '--keep-sequence '
                      '--remove-stopwords').format(mallet=mallet_bin,
                                                   text_dir=text_dir,
                                                   outdir=tempdir)

        train_str = ('{mallet} train-topics '
                     '--input {out}/topic-input.mallet '
                     '--num-topics {n_topics} '
                     '--output-doc-topics {out}/doc_topics.txt '
                     '--topic-word-weights-file {out}/topic_word_weights.txt '
                     '--num-iterations {n_iters} '
                     '--output-model {out}/saved_model.mallet '
                     '--random-seed 1 '
                     '--alpha {alpha} '
                     '--beta {beta}').format(mallet=mallet_bin, out=tempdir,
                                             n_topics=self.params['n_topics'],
                                             n_iters=self.params['n_iters'],
                                             alpha=self.params['alpha'],
                                             beta=self.params['beta'])

        subprocess.call(import_str, shell=True)
        subprocess.call(train_str, shell=True)

        # Read in and convert doc_topics and topic_keys.
        topic_names = ['topic_{0:03d}'.format(i) for i in range(self.params['n_topics'])]

        # doc_topics: Topic weights for each paper.
        # The conversion here is pretty ugly at the moment.
        # First row should be dropped. First column is row number and can be used
        # as the index.
        # Second column is 'file: /full/path/to/id.txt' <-- Parse to get id.
        # After that, odd columns are topic numbers and even columns are the
        # weights for the topics in the preceding column. These columns are sorted
        # on an individual id basis by the weights.
        n_cols = (2 * self.params['n_topics']) + 1
        dt_df = pd.read_csv(op.join(tempdir, 'doc_topics.txt'),
                            delimiter='\t', skiprows=1, header=None,
                            index_col=0)
        dt_df = dt_df[dt_df.columns[:n_cols]]

        # Get ids from filenames
        dt_df[1] = dt_df[1].apply(self._clean_str)

        # Put weights (even cols) and topics (odd cols) into separate dfs.
        weights_df = dt_df[dt_df.columns[2::2]]
        weights_df.index = dt_df[1]
        weights_df.columns = range(self.params['n_topics'])

        topics_df = dt_df[dt_df.columns[1::2]]
        topics_df.index = dt_df[1]
        topics_df.columns = range(self.params['n_topics'])

        # Sort columns in weights_df separately for each row using topics_df.
        sorters_df = topics_df.apply(self._get_sort, axis=1)
        weights = weights_df.as_matrix()
        sorters = sorters_df.as_matrix()
        # there has to be a better way to do this.
        for i in range(sorters.shape[0]):
            weights[i, :] = weights[i, sorters[i, :]]

        # Define topic names (e.g., topic_000)
        p_topic_g_doc_df = pd.DataFrame(columns=topic_names, data=weights,
                                        index=dt_df[1])
        p_topic_g_doc_df.index.name = 'id'
        self.p_topic_g_doc = p_topic_g_doc_df.values

        # Topic word weights
        p_word_g_topic_df = pd.read_csv(op.join(tempdir, 'topic_word_weights.txt'),
                                        dtype=str, keep_default_na=False,
                                        na_values=[], sep='\t', header=None,
                                        names=['topic', 'word', 'weight'])
        p_word_g_topic_df['weight'] = p_word_g_topic_df['weight'].astype(float)
        p_word_g_topic_df['topic'] = p_word_g_topic_df['topic'].astype(int)
        p_word_g_topic_df = p_word_g_topic_df.pivot(index='topic',
                                                    columns='word',
                                                    values='weight')
        p_word_g_topic_df = p_word_g_topic_df.div(p_word_g_topic_df.sum(axis=1),
                                                  axis=0)
        self.p_word_g_topic = p_word_g_topic_df.values

        # Remove all temporary files (text files, model, and outputs).
        shutil.rmtree(tempdir)

    def _clean_str(self, string):
        return op.basename(op.splitext(string)[0])

    def _get_sort(self, lst):
        return [i[0] for i in sorted(enumerate(lst), key=lambda x: x[1])]
