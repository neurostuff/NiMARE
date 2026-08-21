"""Benchmark GC-LDA on a reduced Neurosynth-derived corpus."""

import os

from nimare import annotate
from nimare.nimads import Studyset
from nimare.tests.utils import get_test_data_path


class TimeGCLDA:
    """Time GC-LDA fitting on a small abstract subset."""

    def setup(self):
        """Load a small Neurosynth-derived subset and build integer term counts."""
        studyset = Studyset(
            os.path.join(get_test_data_path(), "neurosynth_laird_studyset.json"),
            target="mni152_2mm",
        )
        texts = studyset.texts.iloc[:12].copy()
        ids = set(texts["id"])

        self.mask = studyset.masker.mask_img
        self.coordinates = studyset.coordinates.loc[studyset.coordinates["id"].isin(ids)].copy()
        self.counts_df = annotate.text.generate_counts(
            texts,
            text_column="abstract",
            tfidf=False,
            max_df=0.99,
            min_df=0.01,
        )

    def time_fit_small(self):
        """Fit a reduced GC-LDA model for a few iterations."""
        model = annotate.gclda.GCLDAModel(
            self.counts_df,
            self.coordinates,
            mask=self.mask,
            n_topics=8,
            n_regions=4,
            symmetric=True,
            seed_init=1,
        )
        model.fit(n_iters=8, loglikely_freq=8)
