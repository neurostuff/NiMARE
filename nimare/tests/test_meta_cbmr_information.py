"""Tests for the closed-form observed information and the covariance strategies.

Checked against ``torch.func.hessian``, the path these replace, so a shared mistake cannot pass.
The derivations are proved at https://github.com/jdkent/cbmr-proofs; what is checked here is the
code that implements them.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta.cbmr.covariance import (
        blockwise_inverse,
        bordered_inverse,
        cholesky_inverse,
        is_block_diagonal,
        spatial_components,
        symmetric_condition_number,
    )
    from nimare.meta.cbmr.information import closed_form_information
    from nimare.meta.cbmr.model import CBMRModel
    from nimare.meta.cbmr.predictor import CBMRPredictor
    from nimare.meta.cbmr.terms import Design, bind

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")

N_PER_GROUP = 12
N_VOXELS = 30
N_BASES = 4
# A cell-means factor, the same with one and two moderators, and a sum-to-zero term whose
# loadings are neither one-hot nor positive.
FORMULAS = [
    "~ s(diagnosis)",
    "~ s(diagnosis) + n",
    "~ s(diagnosis) + n + age",
    "~ sz(diagnosis) + sz(drug)",
]
DISTRIBUTIONS = ["poisson", "negativebinomial", "clusterednegativebinomial"]


@pytest.fixture(scope="module")
def data():
    """Simulate counts with an interior maximum, as the model tests do."""
    rng = np.random.default_rng(23)
    n_experiments = 2 * N_PER_GROUP
    annotations = pd.DataFrame(
        {
            "diagnosis": ["schiz"] * N_PER_GROUP + ["dep"] * N_PER_GROUP,
            "drug": ["yes", "no"] * N_PER_GROUP,
            "n": rng.normal(size=n_experiments),
            "age": rng.normal(size=n_experiments),
        }
    )
    raw = rng.uniform(0.05, 1.0, (N_VOXELS, N_BASES))
    return annotations, raw / raw.sum(axis=1, keepdims=True)


def _fit(formula, distribution, data, n_iter=60):
    """Fit ``distribution`` to counts drawn from a matching generative model.

    The draw matters. Fitting a negative binomial to Poisson counts pushes its overdispersion to
    zero, where the closed form loses precision and the comparison measures that instead of the
    derivation. Overdispersed counts keep the fit in the range the distribution is meant for.
    """
    annotations, bases = data
    predictor = CBMRPredictor(bind(Design.from_formula(formula), annotations), bases)
    rng = np.random.default_rng(29)
    spatial = torch.tensor(
        rng.normal(1.3, 0.25, (predictor.n_spatial_columns, predictor.n_bases)),
        dtype=torch.float64,
    )
    intensity = torch.exp(predictor.log_intensity_by_pattern(spatial)).detach().numpy()
    mean = intensity[predictor.patterns.assignment]
    size = 6.0
    if distribution == "poisson":
        foci = rng.poisson(mean).astype(float)
    elif distribution == "clusterednegativebinomial":
        # One factor per experiment, shared across the brain. That is what separates this model
        # from the plain negative binomial, and a voxelwise mixture leaves it nothing to fit.
        factor = rng.gamma(size, 1.0 / size, size=mean.shape[0])[:, None]
        foci = rng.poisson(mean * factor).astype(float)
    else:
        # Independent gamma variation at each voxel.
        foci = rng.poisson(rng.gamma(size, mean / size)).astype(float)
    model = CBMRModel(predictor, distribution=distribution)
    model.fit(foci, n_iter=n_iter, tol=1e-8)
    return model, foci


def _autodiff_information(model, foci):
    """The path the closed forms replace, called directly."""
    flat = model.coefficients.detach().clone()
    nuisance = None if model.nuisance is None else model.nuisance.detach().clone()

    def negative_log_likelihood(vector):
        return -model.log_likelihood(foci, flat=vector, nuisance=nuisance)

    hessian = torch.func.hessian(negative_log_likelihood)(flat)
    return hessian.reshape(model.n_parameters, model.n_parameters).detach().cpu().numpy()


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("formula", FORMULAS)
def test_closed_form_matches_autodiff(formula, distribution, data):
    """The closed form is the same matrix automatic differentiation produces."""
    model, foci = _fit(formula, distribution, data)
    reference = _autodiff_information(model, foci)
    analytic = closed_form_information(model.distribution)(model, foci)
    assert np.abs(analytic - reference).max() < 1e-8 * np.abs(reference).max()


@pytest.mark.parametrize("distribution", ["negativebinomial", "clusterednegativebinomial"])
def test_overdispersion_stays_off_the_poisson_boundary(distribution, data):
    """Guard the premise of the test above.

    If the counts stop being overdispersed, the fits collapse toward Poisson and the equality
    tests start measuring rounding error. This fails first and says so.
    """
    model, _ = _fit("~ s(diagnosis) + n", distribution, data)
    fitted = model.overdispersion()
    assert fitted is not None
    assert fitted.min() > 1e-3, f"overdispersion collapsed to {fitted}; counts are not dispersed"


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("formula", FORMULAS)
def test_information_is_symmetric(formula, distribution, data):
    """A Hessian is symmetric, and each closed form mirrors its cross block by hand.

    That last step is repeated once per distribution, so a fourth one could omit it and produce
    a matrix that inverts fine but is wrong above the diagonal.
    """
    model, foci = _fit(formula, distribution, data)
    information = model.information_matrix(foci)
    np.testing.assert_allclose(information, information.T, rtol=1e-12, atol=0)


def test_poisson_information_ignores_the_foci(data):
    """Under a log link the counts drop out of the Poisson information."""
    model, foci = _fit("~ s(diagnosis)", "poisson", data)
    one = closed_form_information(model.distribution)(model, foci)
    two = closed_form_information(model.distribution)(model, foci * 3)
    np.testing.assert_allclose(one, two, rtol=1e-12)


@pytest.mark.parametrize("distribution", ["negativebinomial", "clusterednegativebinomial"])
def test_overdispersed_information_uses_the_foci(distribution, data):
    """The other two must respond, or an implementation ignoring the counts would pass."""
    model, foci = _fit("~ s(diagnosis)", distribution, data)
    one = closed_form_information(model.distribution)(model, foci)
    two = closed_form_information(model.distribution)(model, foci * 3)
    assert np.abs(one - two).max() > 1e-6 * np.abs(one).max()


def test_unknown_distribution_falls_back_to_autodiff(data):
    """A distribution with no derivation must not get another one's Hessian."""
    from nimare.meta.cbmr.distributions import Distribution
    from nimare.meta.cbmr.predictor import poisson_log_likelihood

    class Unlisted(Distribution):
        name = "Unlisted"

        def log_likelihood(self, predictor, spatial_coef, global_coef, nuisance, foci):
            return poisson_log_likelihood(predictor, spatial_coef, global_coef, foci)

    assert closed_form_information(Unlisted()) is None
    model, foci = _fit("~ s(diagnosis)", "poisson", data)
    model.distribution = Unlisted()
    np.testing.assert_allclose(
        model.information_matrix(foci), _autodiff_information(model, foci), rtol=1e-9
    )


def test_information_is_block_diagonal_for_a_cell_means_factor(data):
    """Off-block entries are exactly zero, not merely small."""
    model, foci = _fit("~ s(diagnosis)", "poisson", data)
    information = model.information_matrix(foci)
    blocks = spatial_components(model.predictor.patterns.loadings, model.predictor.n_bases)
    assert len(blocks) == 2
    assert is_block_diagonal(information, blocks)
    mask = np.ones_like(information, dtype=bool)
    for index in blocks:
        mask[np.ix_(index, index)] = False
    assert np.abs(information[mask]).max() == 0.0


def test_a_moderator_couples_every_column_but_not_the_spatial_block(data):
    """Moderators end block diagonality but leave the spatial blocks separable."""
    model, foci = _fit("~ s(diagnosis) + n", "poisson", data)
    information = model.information_matrix(foci)
    spatial = spatial_components(model.predictor.patterns.loadings, model.predictor.n_bases)
    assert len(spatial) == 2, "the spatial columns must still separate"
    assert is_block_diagonal(information[: model.n_spatial, : model.n_spatial], spatial)
    assert not is_block_diagonal(information, spatial), "gamma must couple the whole matrix"


def test_every_inverse_route_agrees():
    """Blockwise, bordered and Cholesky must give one matrix, not three."""
    n_blocks, width, border = 3, 6, 2
    rng = np.random.default_rng(11)
    blocks, index = [], []
    for i in range(n_blocks):
        root = rng.normal(size=(width, width)) / np.sqrt(width)
        blocks.append(root @ root.T + np.eye(width))
        index.append(np.arange(i * width, (i + 1) * width))
    n_spatial = n_blocks * width
    diagonal = np.zeros((n_spatial, n_spatial))
    for slot, block in zip(index, blocks):
        diagonal[np.ix_(slot, slot)] = block

    reference = np.linalg.inv(diagonal)
    np.testing.assert_allclose(blockwise_inverse(diagonal, index), reference, atol=1e-10)
    np.testing.assert_allclose(cholesky_inverse(diagonal), reference, atol=1e-10)
    np.testing.assert_allclose(
        symmetric_condition_number(diagonal, index), np.linalg.cond(diagonal), rtol=1e-10
    )
    np.testing.assert_allclose(
        symmetric_condition_number(diagonal), np.linalg.cond(diagonal), rtol=1e-10
    )

    edge = rng.normal(size=(n_spatial, border)) * 0.1
    root = rng.normal(size=(border, border))
    full = np.zeros((n_spatial + border, n_spatial + border))
    full[:n_spatial, :n_spatial] = diagonal
    full[:n_spatial, n_spatial:] = edge
    full[n_spatial:, :n_spatial] = edge.T
    full[n_spatial:, n_spatial:] = (
        edge.T @ np.linalg.solve(diagonal, edge) + root @ root.T + np.eye(border)
    )
    np.testing.assert_allclose(
        bordered_inverse(full, index, n_spatial), np.linalg.inv(full), atol=1e-8
    )
    np.testing.assert_allclose(cholesky_inverse(full), np.linalg.inv(full), atol=1e-8)

    # With no border the Schur complement must match the blockwise inverse.
    shrunk = full.copy()
    shrunk[:n_spatial, n_spatial:] = 0.0
    shrunk[n_spatial:, :n_spatial] = 0.0
    np.testing.assert_allclose(
        bordered_inverse(shrunk, index, n_spatial)[:n_spatial, :n_spatial],
        blockwise_inverse(diagonal, index),
        atol=1e-10,
    )


def test_a_singular_design_explains_itself_on_the_blockwise_route(data):
    """The message must not depend on which route hit the singularity.

    The blockwise route inverts each block directly, so without a shared handler it would raise
    numpy's bare "Singular matrix". This design takes that route.
    """
    annotations, bases = data
    # Duplicated basis columns make each block exactly singular.
    predictor = CBMRPredictor(
        bind(Design.from_formula("~ s(diagnosis)"), annotations), np.hstack([bases, bases])
    )
    rng = np.random.default_rng(29)
    spatial = torch.tensor(
        rng.normal(1.3, 0.25, (predictor.n_spatial_columns, predictor.n_bases)),
        dtype=torch.float64,
    )
    intensity = torch.exp(predictor.log_intensity_by_pattern(spatial)).detach().numpy()
    foci = rng.poisson(intensity[predictor.patterns.assignment]).astype(float)
    model = CBMRModel(predictor, distribution="poisson")
    model.fit(foci, n_iter=60, tol=1e-8)

    with pytest.raises(np.linalg.LinAlgError, match="spline_spacing"):
        model.covariance(foci)


def test_cholesky_inverse_declines_a_non_positive_definite_matrix():
    """It returns None rather than raising, so the caller can fall back."""
    assert cholesky_inverse(np.array([[1.0, 2.0], [2.0, 1.0]])) is None


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("formula", FORMULAS)
def test_covariance_matches_a_dense_inverse(formula, distribution, data):
    """Whichever rung the design reaches, the answer is the dense inverse."""
    model, foci = _fit(formula, distribution, data)
    information = model.information_matrix(foci)
    condition = np.linalg.cond(information)
    if condition > 1.0 / np.finfo(float).eps:
        pytest.skip("information matrix is numerically singular; no inverse to compare against")
    reference = np.linalg.inv(information)
    produced = model.covariance(foci)
    tolerance = max(1e-10, 100 * condition * np.finfo(float).eps)
    assert np.abs(produced - reference).max() / np.abs(reference).max() < tolerance


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_sparse_and_dense_foci_agree(distribution, data):
    """One code path serves both, so the two representations must not diverge.

    The estimator always builds a sparse matrix, but ``fit`` and ``information_matrix`` accept
    an array too, and NiMARE's other CBMR tests pass one.
    """
    import scipy.sparse

    annotations, bases = data
    predictor = CBMRPredictor(bind(Design.from_formula("~ s(diagnosis)"), annotations), bases)
    rng = np.random.default_rng(5)
    dense = rng.poisson(0.4, (predictor.patterns.n_experiments, N_VOXELS)).astype(float)
    sparse = scipy.sparse.csr_matrix(dense)

    np.testing.assert_array_equal(
        predictor.patterns.marginal_by_pattern(dense),
        predictor.patterns.marginal_by_pattern(sparse),
    )

    model = CBMRModel(predictor, distribution=distribution)
    model.fit(dense, n_iter=20, tol=1e-8)
    np.testing.assert_allclose(
        float(model.log_likelihood(dense)), float(model.log_likelihood(sparse)), rtol=1e-12
    )
    np.testing.assert_allclose(
        model.information_matrix(dense), model.information_matrix(sparse), rtol=1e-12
    )
