"""Topic modeling with latent Dirichlet allocation via MALLET."""
import logging
import os
import os.path as op
import shutil
import subprocess

import numpy as np
import pandas as pd

from .. import references
from ..base import NiMAREBase
from ..due import due
from ..extract import download_mallet, utils

LGR = logging.getLogger(__name__)


@due.dcite(references.LDA, description="Introduces LDA.")
@due.dcite(references.MALLET, description="Citation for MALLET toolbox")
@due.dcite(
    references.LDAMODEL,
    description="First use of LDA for automated annotation of neuroimaging literature.",
)
class LDAModel(NiMAREBase):
    """
    Perform topic modeling using Latent Dirichlet Allocation (LDA).

    Build an LDA [1]_ topic model with the Java toolbox MALLET [2]_, as
    performed in [3]_.

    Parameters
    ----------
    text_df : :obj:`pandas.DataFrame`
        A pandas DataFrame with two columns ('id' and text_column) containing
        article text.
    text_column : :obj:`str`, optional
        Name of column in text_df that contains text. Default is 'abstract'.
    n_topics : :obj:`int`, optional
        Number of topics to generate. Default=50.
    n_iters : :obj:`int`, optional
        Number of iterations to run in training topic model. Default=1000.
    alpha : :obj:`float` or 'auto', optional
        The Dirichlet prior on the per-document topic distributions.
        Default: auto, which calculates 50 / n_topics, based on Poldrack et al.
        (2012).
    beta : :obj:`float`, optional
        The Dirichlet prior on the per-topic word distribution. Default: 0.001,
        based on Poldrack et al. (2012).

    Attributes
    ----------
    commands_ : :obj:`list` of :obj:`str`
        List of MALLET commands called to fit model.

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

    def __init__(
        self, text_df, text_column="abstract", n_topics=50, n_iters=1000, alpha="auto", beta=0.001
    ):
        mallet_dir = download_mallet()
        mallet_bin = op.join(mallet_dir, "bin/mallet")

        model_dir = utils._get_dataset_dir("mallet_model")
        text_dir = op.join(model_dir, "texts")

        if not op.isdir(model_dir):
            os.mkdir(model_dir)

        if alpha == "auto":
            alpha = 50.0 / n_topics
        elif not isinstance(alpha, float):
            raise ValueError('Argument alpha must be float or "auto"')

        self.params = {"n_topics": n_topics, "n_iters": n_iters, "alpha": alpha, "beta": beta}
        self.model_dir = model_dir

        # Check for presence of text files and convert if necessary
        if not op.isdir(text_dir):
            LGR.info("Texts folder not found. Creating text files...")
            os.mkdir(text_dir)

            # Remove rows with empty text cells
            orig_ids = text_df["id"].tolist()
            text_df = text_df.dropna(subset=[text_column])
            keep_ids = text_df["id"].tolist()

            if len(keep_ids) != len(orig_ids):
                LGR.info("Retaining {0}/{1} studies".format(len(keep_ids), len(orig_ids)))

            for id_ in text_df["id"].values:
                text = text_df.loc[text_df["id"] == id_, text_column].values[0]
                with open(op.join(text_dir, str(id_) + ".txt"), "w") as fo:
                    fo.write(text)

        # Run MALLET topic modeling
        LGR.info("Generating topics...")
        import_str = (
            "{mallet} import-dir "
            "--input {text_dir} "
            "--output {outdir}/topic-input.mallet "
            "--keep-sequence "
            "--remove-stopwords"
        ).format(mallet=mallet_bin, text_dir=text_dir, outdir=model_dir)

        train_str = (
            "{mallet} train-topics "
            "--input {out}/topic-input.mallet "
            "--num-topics {n_topics} "
            "--output-doc-topics {out}/doc_topics.txt "
            "--topic-word-weights-file {out}/topic_word_weights.txt "
            "--num-iterations {n_iters} "
            "--output-model {out}/saved_model.mallet "
            "--random-seed 1 "
            "--alpha {alpha} "
            "--beta {beta}"
        ).format(
            mallet=mallet_bin,
            out=model_dir,
            n_topics=self.params["n_topics"],
            n_iters=self.params["n_iters"],
            alpha=self.params["alpha"],
            beta=self.params["beta"],
        )
        self.commands_ = [import_str, train_str]

    def fit(self):
        """
        Fit LDA model to corpus.

        Attributes
        ----------
        p_topic_g_doc_ : :obj:`numpy.ndarray`
            Probability of each topic given a document
        p_word_g_topic_ : :obj:`numpy.ndarray`
            Probability of each word given a topic
        """
        subprocess.call(self.commands_[0], shell=True)
        subprocess.call(self.commands_[1], shell=True)

        # Read in and convert doc_topics and topic_keys.
        topic_names = ["topic_{0:03d}".format(i) for i in range(self.params["n_topics"])]

        # doc_topics: Topic weights for each paper.
        # The conversion here is pretty ugly at the moment.
        # First row should be dropped. First column is row number and can be used
        # as the index.
        # Second column is 'file: /full/path/to/id.txt' <-- Parse to get id.
        # After that, odd columns are topic numbers and even columns are the
        # weights for the topics in the preceding column. These columns are sorted
        # on an individual id basis by the weights.
        n_cols = (2 * self.params["n_topics"]) + 1
        dt_df = pd.read_csv(
            op.join(self.model_dir, "doc_topics.txt"),
            delimiter="\t",
            skiprows=1,
            header=None,
            index_col=0,
        )
        dt_df = dt_df[dt_df.columns[:n_cols]]

        # Get ids from filenames
        dt_df[1] = dt_df[1].apply(self._clean_str)

        # Put weights (even cols) and topics (odd cols) into separate dfs.
        weights_df = dt_df[dt_df.columns[2::2]]
        weights_df.index = dt_df[1]
        weights_df.columns = range(self.params["n_topics"])

        topics_df = dt_df[dt_df.columns[1:-1:2]]
        topics_df.index = dt_df[1]
        topics_df.columns = range(self.params["n_topics"])

        # Sort columns in weights_df separately for each row using topics_df.
        sorters_df = topics_df.apply(self._get_sort, axis=1)
        weights = weights_df.values
        sorters = np.vstack(sorters_df.values)
        # there has to be a better way to do this.
        for i in range(sorters.shape[0]):
            weights[i, :] = weights[i, sorters[i, :]]

        # Define topic names (e.g., topic_000)
        p_topic_g_doc_df = pd.DataFrame(columns=topic_names, data=weights, index=dt_df[1])
        p_topic_g_doc_df.index.name = "id"
        self.p_topic_g_doc_ = p_topic_g_doc_df.values
        self.p_topic_g_doc_df_ = p_topic_g_doc_df

        # Topic word weights
        p_word_g_topic_df = pd.read_csv(
            op.join(self.model_dir, "topic_word_weights.txt"),
            dtype=str,
            keep_default_na=False,
            na_values=[],
            sep="\t",
            header=None,
            names=["topic", "word", "weight"],
        )
        p_word_g_topic_df["weight"] = p_word_g_topic_df["weight"].astype(float)
        p_word_g_topic_df["topic"] = p_word_g_topic_df["topic"].astype(int)
        p_word_g_topic_df = p_word_g_topic_df.pivot(index="topic", columns="word", values="weight")
        p_word_g_topic_df = p_word_g_topic_df.div(p_word_g_topic_df.sum(axis=1), axis=0)
        self.p_word_g_topic_ = p_word_g_topic_df.values
        self.p_word_g_topic_df_ = p_word_g_topic_df

        # Remove all temporary files (text files, model, and outputs).
        shutil.rmtree(self.model_dir)

    def _clean_str(self, string):
        return op.basename(op.splitext(string)[0])

    def _get_sort(self, lst):
        return [i[0] for i in sorted(enumerate(lst), key=lambda x: x[1])]
