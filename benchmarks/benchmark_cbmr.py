"""Benchmark the CBMR estimators."""

import numpy as np

from nimare.generate import create_coordinate_dataset
from nimare.meta import models
from nimare.meta.cbmr import CBMREstimator
from nimare.transforms import StandardizeField


class TimeCBMR:
    """Time CBMR estimators."""

    def setup(self):
        """
        Setup the data.

        Simulates and standardizes the dataset required for the benchmarks.
        """
        _, dataset = create_coordinate_dataset(
            foci=10,
            sample_size=(20, 40),
            n_studies=100,
            seed=100,
        )
        n_rows = dataset.annotations.shape[0]
        dataset.annotations["diagnosis"] = [
            "schizophrenia" if i % 2 == 0 else "depression" for i in range(n_rows)
        ]
        dataset.annotations["drug_status"] = [
            "Yes" if i % 2 == 0 else "No" for i in range(n_rows)
        ]
        dataset.annotations["drug_status"] = (
            dataset.annotations["drug_status"]
            .sample(frac=1, random_state=100)
            .reset_index(drop=True)
        )
        dataset.annotations["sample_sizes"] = [
            dataset.metadata.sample_sizes[i][0] for i in range(n_rows)
        ]
        dataset.annotations["avg_age"] = np.arange(n_rows)

        self.dataset = StandardizeField(fields=["sample_sizes", "avg_age"]).transform(dataset)
        self.group_categories = ["diagnosis", "drug_status"]
        self.moderators = ["standardized_sample_sizes", "standardized_avg_age"]

    def _fit_cbmr(self, model):
        """Fit a CBMR estimator with common benchmark options."""
        meta = CBMREstimator(
            group_categories=self.group_categories,
            moderators=self.moderators,
            spline_spacing=100,
            model=model,
            penalty=False,
            n_iter=200,
            lr=1,
            tol=1e4,
            device="cpu",
        )
        meta.fit(self.dataset)

    def time_poisson(self):
        """
        Time the Poisson CBMR estimator.

        Fits the Poisson CBMR estimator to the dataset and measures the time taken.
        """
        self._fit_cbmr(models.PoissonEstimator)

    def time_negative_binomial(self):
        """
        Time the Negative Binomial CBMR estimator.

        Fits the Negative Binomial CBMR estimator to the dataset and measures the time taken.
        """
        self._fit_cbmr(models.NegativeBinomialEstimator)

    def time_clustered_negative_binomial(self):
        """
        Time the Clustered Negative Binomial CBMR estimator.

        Fits the Clustered Negative Binomial CBMR estimator to the dataset and measures the time
        taken.
        """
        self._fit_cbmr(models.ClusteredNegativeBinomialEstimator)
