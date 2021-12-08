"""Topic modeling with latent Dirichlet allocation."""
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

from nimare import references
from nimare.annotate.text import generate_counts
from nimare.base import Annotator
from nimare.due import due


@due.dcite(references.LDA, description="Introduces LDA.")
@due.dcite(
    references.LDAMODEL,
    description="First use of LDA for automated annotation of neuroimaging literature.",
)
class LDAModel(Annotator):
    """Generate a latent Dirichlet allocation (LDA) topic model.

    This class is a light wrapper around scikit-learn tools for tokenization and LDA.

    Parameters
    ----------
    n_topics
    max_iter
    text_column

    Attributes
    ----------
    model : :obj:`sklearn.decomposition.LatentDirichletAllocation`

    See Also
    --------
    :class:`sklearn.feature_extraction.text.CountVectorizer`
    :class:`sklearn.decomposition.LatentDirichletAllocation`
    """

    def __init__(self, n_topics, max_iter, text_column="abstract"):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.text_column = text_column
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="online",
        )

    def _transform(self, dset):
        """Fit the LDA topic model to text from a Dataset.

        Parameters
        ----------
        dset

        Returns
        -------
        dset

        Attributes
        ----------
        distributions_ : :obj:`dict`
            A dictionary containing additional distributions produced by the model, including:

                -   p_topic_g_word: numpy ndarray of shape (n_topics, n_tokens) containing the
                    topic-term weights for the model.
                -   p_topic_g_word_df: pandas DataFrame of shape (n_topics, n_tokens) containing
                    the topic-term weights for the model.
        """
        counts_df = generate_counts(
            dset.texts,
            text_column=self.text_column,
            tfidf=False,
            max_df=len(dset.ids) - 2,
            min_df=2,
        )
        vocabulary = counts_df.columns.tolist()
        count_values = counts_df.values
        study_ids = counts_df.index.tolist()
        # LDA50__1_word1_word2_word3
        topic_names = [f"LDA{self.n_topics}__{i + 1}" for i in range(self.n_topics)]

        doc_topic_weights = self.model.fit_transform(count_values)
        doc_topic_weights_df = pd.DataFrame(
            index=study_ids,
            columns=topic_names,
            data=doc_topic_weights,
        )
        topic_word_weights = self.model.components_
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
        new_annotations = pd.merge(
            annotations, doc_topic_weights_df, left_on="id", right_index=True
        )
        new_dset = dset.copy()
        new_dset.annotations = new_annotations
        return new_dset
