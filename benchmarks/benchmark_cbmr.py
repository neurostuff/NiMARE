"""Benchmark the CBMR estimators."""

import numpy as np

from nimare.generate import create_coordinate_dataset
from nimare.meta import models
from nimare.meta.cbmr import CBMREstimator, CBMRInference
from nimare.transforms import StandardizeField


def _make_cbmr_dataset():
    """Simulate and standardize a CBMR benchmark dataset."""
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
    dataset.annotations["drug_status"] = ["Yes" if i % 2 == 0 else "No" for i in range(n_rows)]
    dataset.annotations["drug_status"] = (
        dataset.annotations["drug_status"]
        .sample(frac=1, random_state=100)
        .reset_index(drop=True)
    )
    dataset.annotations["sample_sizes"] = [
        dataset.metadata.sample_sizes[i][0] for i in range(n_rows)
    ]
    dataset.annotations["avg_age"] = np.arange(n_rows)
    return StandardizeField(fields=["sample_sizes", "avg_age"]).transform(dataset)


def _fit_cbmr(dataset, group_categories, moderators, model=models.PoissonEstimator):
    """Fit a CBMR estimator with common benchmark options."""
    meta = CBMREstimator(
        group_categories=group_categories,
        moderators=moderators,
        spline_spacing=100,
        model=model,
        penalty=False,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cpu",
    )
    return meta.fit(dataset)


class _CBMRBenchmarkMixin:
    """Shared setup for CBMR benchmarks."""

    def setup(self):
        """
        Setup the data.

        Simulates and standardizes the dataset required for the benchmarks.
        """
        self.dataset = _make_cbmr_dataset()
        self.group_categories = ["diagnosis", "drug_status"]
        self.moderators = ["standardized_sample_sizes", "standardized_avg_age"]


class TimeCBMR(_CBMRBenchmarkMixin):
    """Time CBMR estimators."""

    def _fit_cbmr(self, model):
        """Fit a CBMR estimator with common benchmark options."""
        _fit_cbmr(self.dataset, self.group_categories, self.moderators, model=model)

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


class TimeCBMRInference(_CBMRBenchmarkMixin):
    """Time CBMR inference routines."""

    def setup(self):
        """Setup a fitted CBMR result and reusable contrasts for inference benchmarks."""
        super().setup()
        self.result = _fit_cbmr(self.dataset, self.group_categories, self.moderators)
        self.group_contrast_name = "DepressionYes-DepressionNo"
        self.moderator_contrast_name = "standardized_sample_sizes-standardized_avg_age"
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        self.group_contrast = inference.create_contrast(
            [self.group_contrast_name],
            source="groups",
        )
        self.moderator_contrast = inference.create_contrast(
            [self.moderator_contrast_name],
            source="moderators",
        )
        self.multi_group_contrast = [
            [[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 1, -1]],
        ]

    def time_fit(self):
        """
        Time fitting the CBMRInference object.

        Copies the fitted CBMR result and constructs group/moderator lookup structures.
        """
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)

    def time_create_group_contrast(self):
        """
        Time named group contrast construction.

        Parses a pairwise group contrast into the contrast matrix used downstream.
        """
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        inference.create_contrast([self.group_contrast_name], source="groups")

    def time_create_moderator_contrast(self):
        """
        Time named moderator contrast construction.

        Parses a pairwise moderator contrast into the contrast matrix used downstream.
        """
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        inference.create_contrast([self.moderator_contrast_name], source="moderators")

    def time_group_inference(self):
        """
        Time group-level CBMR inference.

        Runs spatial intensity inference for one pairwise group contrast.
        """
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        inference.transform(t_con_groups=self.group_contrast, t_con_moderators=False)

    def time_moderator_inference(self):
        """
        Time moderator-level CBMR inference.

        Runs scalar moderator inference for one pairwise moderator contrast.
        """
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        inference.transform(t_con_groups=False, t_con_moderators=self.moderator_contrast)

    def time_combined_inference(self):
        """
        Time combined group and moderator CBMR inference.

        Runs both group-level and moderator-level inference in one transform call.
        """
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        inference.transform(
            t_con_groups=self.group_contrast,
            t_con_moderators=self.moderator_contrast,
        )

    def time_group_glh_inference(self):
        """
        Time multi-row GLH group inference.

        Runs a generalized linear hypothesis test over all fitted group intensities.
        """
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        inference.transform(t_con_groups=self.multi_group_contrast, t_con_moderators=False)
