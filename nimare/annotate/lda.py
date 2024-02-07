"""Topic modeling with latent Dirichlet allocation."""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

from nimare.annotate.text import generate_counts
from nimare.base import NiMAREBase
from nimare.utils import _check_ncores


class LDAModel(NiMAREBase):
    """Generate a latent Dirichlet allocation (LDA) topic model.

    This class is a light wrapper around scikit-learn tools for tokenization and LDA.

    Parameters
    ----------
    n_topics : :obj:`int`
        Number of topics for topic model. This corresponds to the model's ``n_components``
        parameter. Must be an integer >= 1.
    max_iter : :obj:`int`, optional
        Maximum number of iterations to use during model fitting. Default = 1000.
    alpha : :obj:`float` or None, optional
        The ``alpha`` value for the model. This corresponds to the model's ``doc_topic_prior``
        parameter. Default is None, which evaluates to ``1 / n_topics``,
        as was used in :footcite:t:`poldrack2012discovering`.
    beta : :obj:`float` or None, optional
        The ``beta`` value for the model. This corresponds to the model's ``topic_word_prior``
        parameter. If None, it evaluates to ``1 / n_topics``.
        Default is 0.001, which was used in :footcite:t:`poldrack2012discovering`.
    text_column : :obj:`str`, optional
        The source of text to use for the model. This should correspond to an existing column
        in the :py:attr:`~nimare.dataset.Dataset.texts` attribute. Default is "abstract".
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Attributes
    ----------
    model : :obj:`~sklearn.decomposition.LatentDirichletAllocation`

    Notes
    -----
    Latent Dirichlet allocation was first developed in :footcite:t:`blei2003latent`,
    and was first applied to neuroimaging articles in :footcite:t:`poldrack2012discovering`.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`~sklearn.feature_extraction.text.CountVectorizer`: Used to build a vocabulary of terms
        and their associated counts from texts in the ``self.text_column`` of the Dataset's
        ``texts`` attribute.
    :class:`~sklearn.decomposition.LatentDirichletAllocation`: Used to train the LDA model.
    """

    def __init__(
        self, n_topics, max_iter=1000, alpha=None, beta=0.001, text_column="abstract", n_cores=1
    ):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.text_column = text_column
        self.n_cores = _check_ncores(n_cores)

        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="batch",
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            n_jobs=n_cores,
        )

    def fit(self, dset):
        """Fit the LDA topic model to text from a Dataset.

        Parameters
        ----------
        dset : :obj:`~nimare.dataset.Dataset`
            A Dataset with, at minimum, text available in the ``self.text_column`` column of its
            :py:attr:`~nimare.dataset.Dataset.texts` attribute.

        Returns
        -------
        dset : :obj:`~nimare.dataset.Dataset`
            A new Dataset with an updated :py:attr:`~nimare.dataset.Dataset.annotations` attribute.

        Attributes
        ----------
        distributions_ : :obj:`dict`
            A dictionary containing additional distributions produced by the model, including:

                -   ``p_topic_g_word``: :obj:`numpy.ndarray` of shape (n_topics, n_tokens)
                    containing the topic-term weights for the model.
                -   ``p_topic_g_word_df``: :obj:`pandas.DataFrame` of shape (n_topics, n_tokens)
                    containing the topic-term weights for the model.
        """
        counts_df = generate_counts(
            dset.texts,
            text_column=self.text_column,
            tfidf=False,
            max_df=len(dset.ids) - 2,
            min_df=2,
        )
        vocabulary = counts_df.columns.to_numpy()
        count_values = counts_df.values
        study_ids = counts_df.index.tolist()

        doc_topic_weights = self.model.fit_transform(count_values)
        topic_word_weights = self.model.components_

        # Get top 3 words for each topic for annotation
        sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
        top_tokens = [
            "_".join(vocabulary[sorted_weights_idxs[topic_i, :]][:3])
            for topic_i in range(self.n_topics)
        ]
        topic_names = [
            f"LDA{self.n_topics}__{i + 1}_{top_tokens[i]}" for i in range(self.n_topics)
        ]

        doc_topic_weights_df = pd.DataFrame(
            index=study_ids,
            columns=topic_names,
            data=doc_topic_weights,
        )
        topic_word_weights_df = pd.DataFrame(
            index=topic_names,
            columns=vocabulary,
            data=topic_word_weights,
        )
        self.distributions_ = {
            "p_topic_g_word": topic_word_weights,
            "p_topic_g_word_df": topic_word_weights_df,
        }

        annotations = dset.annotations.copy()
        annotations = pd.merge(annotations, doc_topic_weights_df, left_on="id", right_index=True)
        new_dset = dset.copy()
        new_dset.annotations = annotations
        return new_dset
