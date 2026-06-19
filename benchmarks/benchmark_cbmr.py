"""Benchmark the CBMR estimators."""

import inspect

import numpy as np

from nimare.generate import create_coordinate_dataset
from nimare.meta import models
from nimare.meta.cbmr import CBMREstimator, CBMRInference
from nimare.transforms import StandardizeField

GROUP_CATEGORIES = ["diagnosis", "drug_status"]
MODERATORS = ["standardized_sample_sizes", "standardized_avg_age"]
N_STUDIES = 100
RANDOM_STATE = 100


def _make_cbmr_dataset():
    """Simulate and standardize a CBMR benchmark dataset."""
    _, dataset = create_coordinate_dataset(
        foci=10,
        sample_size=(20, 40),
        n_studies=N_STUDIES,
        seed=RANDOM_STATE,
    )
    n_rows = dataset.annotations.shape[0]
    dataset.annotations["diagnosis"] = [
        "schizophrenia" if i % 2 == 0 else "depression" for i in range(n_rows)
    ]
    dataset.annotations["drug_status"] = ["Yes" if i % 2 == 0 else "No" for i in range(n_rows)]
    dataset.annotations["drug_status"] = (
        dataset.annotations["drug_status"]
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    dataset.annotations["sample_sizes"] = [
        dataset.metadata.sample_sizes[i][0] for i in range(n_rows)
    ]
    dataset.annotations["avg_age"] = np.arange(n_rows)
    return StandardizeField(fields=["sample_sizes", "avg_age"]).transform(dataset)


def _supports_moderator_effect():
    """Return whether the installed CBMR estimator exposes the moderator_effect selector."""
    return "moderator_effect" in inspect.signature(CBMREstimator).parameters


def _fit_cbmr(
    dataset,
    group_categories,
    moderators,
    model=models.PoissonEstimator,
    moderator_effect=None,
):
    """Fit a CBMR estimator with common benchmark options."""
    kwargs = {}
    if moderator_effect is not None and _supports_moderator_effect():
        kwargs["moderator_effect"] = moderator_effect

    meta = CBMREstimator(
        group_categories=list(group_categories),
        moderators=list(moderators),
        spline_spacing=100,
        model=model,
        penalty=False,
        n_iter=200,
        lr=1,
        tol=1e4,
        device="cpu",
        **kwargs,
    )
    return meta.fit(dataset)


def _fit_cbmr_inference(result):
    """Fit a CBMRInference object with common benchmark options."""
    inference = CBMRInference(device="cpu")
    inference.fit(result)
    return inference


class _CBMRBenchmarkMixin:
    """Shared setup for CBMR benchmarks."""

    def setup(self):
        """
        Set up the data.

        Simulates and standardizes the dataset required for the benchmarks.
        """
        self.dataset = _make_cbmr_dataset()
        self.group_categories = list(GROUP_CATEGORIES)
        self.moderators = list(MODERATORS)


class TimeCBMR(_CBMRBenchmarkMixin):
    """Time CBMR estimators."""

    def _fit_cbmr(self, model, moderator_effect=None):
        """Fit a CBMR estimator with common benchmark options."""
        _fit_cbmr(
            self.dataset,
            self.group_categories,
            self.moderators,
            model=model,
            moderator_effect=moderator_effect,
        )

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
        self._fit_cbmr(models.NegativeBinomialEstimator, moderator_effect="global")

    def time_clustered_negative_binomial(self):
        """
        Time the Clustered Negative Binomial CBMR estimator.

        Fits the Clustered Negative Binomial CBMR estimator to the dataset and measures the time
        taken.
        """
        self._fit_cbmr(
            models.ClusteredNegativeBinomialEstimator,
            moderator_effect="global",
        )


class TimeCBMRInference(_CBMRBenchmarkMixin):
    """Time CBMR inference routines."""

    def setup(self):
        """Set up a fitted CBMR result and reusable contrasts for inference benchmarks."""
        super().setup()
        self.result = _fit_cbmr(self.dataset, self.group_categories, self.moderators)
        self.group_contrast_name = "DepressionYes-DepressionNo"
        self.moderator_contrast_name = "standardized_sample_sizes-standardized_avg_age"
        self.contrast_inference = _fit_cbmr_inference(self.result)
        self.group_inference = _fit_cbmr_inference(self.result)
        self.moderator_inference = _fit_cbmr_inference(self.result)
        self.combined_inference = _fit_cbmr_inference(self.result)
        self.group_glh_inference = _fit_cbmr_inference(self.result)
        self.group_contrast = self.contrast_inference.create_contrast(
            [self.group_contrast_name],
            source="groups",
        )
        self.moderator_contrast = self.contrast_inference.create_contrast(
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
        _fit_cbmr_inference(self.result)

    def time_create_group_contrast(self):
        """
        Time named group contrast construction.

        Parses a pairwise group contrast into the contrast matrix used downstream.
        """
        self.contrast_inference.create_contrast([self.group_contrast_name], source="groups")

    def time_create_moderator_contrast(self):
        """
        Time named moderator contrast construction.

        Parses a pairwise moderator contrast into the contrast matrix used downstream.
        """
        self.contrast_inference.create_contrast(
            [self.moderator_contrast_name],
            source="moderators",
        )

    def time_group_inference(self):
        """
        Time group-level CBMR inference.

        Runs spatial intensity inference for one pairwise group contrast.
        """
        self.group_inference.transform(
            t_con_groups=self.group_contrast,
            t_con_moderators=False,
        )

    def time_moderator_inference(self):
        """
        Time moderator-level CBMR inference.

        Runs scalar moderator inference for one pairwise moderator contrast.
        """
        self.moderator_inference.transform(
            t_con_groups=False,
            t_con_moderators=self.moderator_contrast,
        )

    def time_combined_inference(self):
        """
        Time combined group and moderator CBMR inference.

        Runs both group-level and moderator-level inference in one transform call.
        """
        self.combined_inference.transform(
            t_con_groups=self.group_contrast,
            t_con_moderators=self.moderator_contrast,
        )

    def time_group_glh_inference(self):
        """
        Time multi-row GLH group inference.

        Runs a generalized linear hypothesis test over all fitted group intensities.
        """
        self.group_glh_inference.transform(
            t_con_groups=self.multi_group_contrast,
            t_con_moderators=False,
        )
