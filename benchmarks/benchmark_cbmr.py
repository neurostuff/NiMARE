"""Benchmark coordinate-based meta-regression.

Timed around where the cost actually is. Fitting cost is driven by the number of *distinct
spatial patterns* a design produces, not by how many terms it has: a grouped design collapses
onto one log-intensity map per group, while a continuously varying spatial term gives every
experiment its own, which is the expensive end. Inference cost is dominated by the covariance --
the Fisher information is a dense (parameters x parameters) Hessian, and a sandwich estimator
adds a meat matrix of the same size on top.

ASV runs this benchmark file against both the PR commit and the base commit. The base commit has
the legacy ``CBMREstimator`` API, while the PR has the formula-based ``CBMR`` API, so this module
keeps a small compatibility layer for the base side of ``asv continuous``.
"""

import numpy as np

from nimare.transforms import StandardizeField

try:
    from nimare.generate import create_coordinate_studyset
    from nimare.meta.cbmr import CBMR

    HAS_FORMULA_CBMR = True
except ImportError:
    from nimare.generate import create_coordinate_dataset
    from nimare.meta import models
    from nimare.meta.cbmr import CBMREstimator, CBMRInference

    HAS_FORMULA_CBMR = False

GROUP_CATEGORIES = ["diagnosis", "drug_status"]
N_STUDIES = 100
RANDOM_STATE = 100

# Coarse spacing and a loose tolerance, so a timing run measures the code path rather than how
# long the optimizer wanders. See the golden-fixture module for why tightening the tolerance on
# coordinate data does not buy a better fit.
FIT_KWARGS = dict(
    spline_spacing=100,
    n_iter=200,
    lr=1,
    tol=1e4,
    device="cpu",
    random_state=RANDOM_STATE,
    generate_description=False,
)


def _make_studyset():
    """Simulate and standardize a Studyset with two factors and two moderators."""
    if HAS_FORMULA_CBMR:
        _, studyset = create_coordinate_studyset(
            foci=10,
            sample_size=(20, 40),
            n_studies=N_STUDIES,
            seed=RANDOM_STATE,
        )
        annotations = studyset.annotations_df.copy()
        n_rows = annotations.shape[0]
        annotations["diagnosis"] = [
            "schizophrenia" if i % 2 == 0 else "depression" for i in range(n_rows)
        ]
        annotations["drug_status"] = ["Yes" if i % 2 == 0 else "No" for i in range(n_rows)]
        annotations["drug_status"] = (
            annotations["drug_status"]
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        annotations["sample_sizes"] = [studyset.metadata.sample_sizes[i][0] for i in range(n_rows)]
        annotations["avg_age"] = np.arange(n_rows)
        studyset = studyset.with_annotations_df(annotations, name="moderators", replace=True)
        return StandardizeField(fields=["sample_sizes", "avg_age"]).transform(studyset)

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


def _legacy_model(distribution):
    """Map a formula-API distribution name to the legacy CBMR model class."""
    return {
        "poisson": models.PoissonEstimator,
        "negativebinomial": models.NegativeBinomialEstimator,
        "clusterednegativebinomial": models.ClusteredNegativeBinomialEstimator,
    }[distribution]


def _legacy_terms(formula):
    """Map representative formula benchmark cases onto the legacy CBMR API."""
    if formula == "~ 1":
        return None, []
    if "standardized_sample_sizes" in formula and "standardized_avg_age" in formula:
        return GROUP_CATEGORIES, ["standardized_sample_sizes", "standardized_avg_age"]
    if "standardized_sample_sizes" in formula:
        return GROUP_CATEGORIES, ["standardized_sample_sizes"]
    if "standardized_avg_age" in formula:
        return GROUP_CATEGORIES, ["standardized_avg_age"]
    return GROUP_CATEGORIES, []


class _CBMRBenchmarkMixin:
    """Shared setup for CBMR benchmarks."""

    def setup(self):
        """Simulate the data the benchmarks fit."""
        self.studyset = _make_studyset()

    def _fit(self, formula, distribution="poisson", **overrides):
        """Fit one formula with the shared benchmark options."""
        fit_kwargs = {**FIT_KWARGS, **overrides}
        if HAS_FORMULA_CBMR:
            return CBMR(formula, distribution=distribution, **fit_kwargs).fit(
                dataset=self.studyset,
            )

        group_categories, moderators = _legacy_terms(formula)
        fit_kwargs.pop("generate_description")
        estimator = CBMREstimator(
            group_categories=None if group_categories is None else list(group_categories),
            moderators=list(moderators),
            model=_legacy_model(distribution),
            penalty=False,
            **fit_kwargs,
        )
        return estimator.fit(self.studyset)


class TimeCBMRDistributions(_CBMRBenchmarkMixin):
    """Time each observation distribution on the same grouped design."""

    def time_poisson(self):
        """Time the Poisson fit, the default and by far the cheapest."""
        self._fit("~ s(diagnosis:drug_status)")

    def time_negative_binomial(self):
        """Time the negative binomial fit, which adds one overdispersion parameter per group."""
        self._fit("~ s(diagnosis:drug_status)", distribution="negativebinomial", lr=1e-2)

    def time_clustered_negative_binomial(self):
        """Time the clustered negative binomial fit."""
        self._fit(
            "~ s(diagnosis:drug_status)",
            distribution="clusterednegativebinomial",
            lr=1e-2,
        )


class TimeCBMRDesigns(_CBMRBenchmarkMixin):
    """Time designs whose spatial-pattern counts differ, which is what drives fitting cost."""

    def time_pooled(self):
        """One shared map: a single spatial pattern, the cheapest possible design."""
        self._fit("~ 1")

    def time_grouped(self):
        """One map per cell: four spatial patterns."""
        self._fit("~ s(diagnosis:drug_status)")

    def time_scalar_moderator(self):
        """Time a scalar moderator, which adds one coefficient and no new patterns."""
        self._fit("~ s(diagnosis:drug_status) + standardized_sample_sizes")

    def time_spatial_moderator(self):
        """Time a continuous spatial term, which gives every experiment its own pattern.

        The expensive end of the range, and the case the old implementation needed a separate
        model class for. Worth watching: this is where a regression would show up first.
        """
        self._fit("~ s(diagnosis:drug_status) + s(standardized_avg_age)")

    def time_sum_to_zero(self):
        """Additive spatial factors, via the sum-to-zero reparameterization."""
        self._fit("~ sz(diagnosis) + sz(drug_status)")


class TimeCBMRInference(_CBMRBenchmarkMixin):
    """Time hypothesis testing, whose cost is dominated by the covariance estimate."""

    def setup(self):
        """Fit once, then time inference against the fitted result."""
        super().setup()
        self.result = self._fit("~ s(diagnosis:drug_status) + standardized_sample_sizes")
        if not HAS_FORMULA_CBMR:
            self.contrast_name = "DepressionYes-DepressionNo"
            self.group_contrast = self._legacy_inference().create_contrast(
                [self.contrast_name],
                source="groups",
            )
            self.moderator_contrast = [[1]]

    def _legacy_inference(self):
        """Return a fitted legacy inference object."""
        inference = CBMRInference(device="cpu")
        inference.fit(self.result)
        return inference

    def time_spatial_contrast_fisher(self):
        """Time a per-voxel contrast using the Fisher-information covariance."""
        if HAS_FORMULA_CBMR:
            self.result.test("schizophrenia-Yes = schizophrenia-No", name="bench")
        else:
            self._legacy_inference().transform(
                t_con_groups=self.group_contrast,
                t_con_moderators=False,
            )

    def time_spatial_contrast_sandwich(self):
        """Time the same contrast with a clustered sandwich covariance.

        The sandwich adds a meat matrix the same size as the bread, so this is the more
        expensive of the two covariance paths.
        """
        if HAS_FORMULA_CBMR:
            self.result.test(
                "schizophrenia-Yes = schizophrenia-No",
                name="bench",
                cov_type="sandwich",
                meat="cluster",
            )
        else:
            self._legacy_inference().transform(
                t_con_groups=self.group_contrast,
                t_con_moderators=False,
            )

    def time_joint_contrast(self):
        """Time a generalized linear hypothesis, which solves a small system per voxel."""
        if HAS_FORMULA_CBMR:
            self.result.test(
                ["schizophrenia-Yes = schizophrenia-No", "depression-Yes = depression-No"],
                name="bench",
            )
        else:
            self._legacy_inference().transform(
                t_con_groups=[[[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 1, -1]]],
                t_con_moderators=False,
            )

    def time_scalar_contrast(self):
        """Time a scalar term's test, which needs no per-voxel work at all."""
        if HAS_FORMULA_CBMR:
            self.result.test("standardized_sample_sizes = 0", name="bench")
        else:
            self._legacy_inference().transform(
                t_con_groups=False,
                t_con_moderators=self.moderator_contrast,
            )
