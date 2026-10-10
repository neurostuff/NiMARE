r"""Restricted spatial structure, and borrowing it from coordinates without borrowing amplitude.

Sections 2.2 and 4 of the revised plan. The model is

.. math::
    m(v) = B(v)^\top\beta + r_0(v), \qquad
    \theta_i(v) = m(v) + x_i^\top\gamma(v) + B(v)^\top b_i + r_i(v),
    \\
    b_i \sim N(0, D), \qquad Y_i \mid \theta_i \sim N(\theta_i, S_i),

with :math:`S_i` the **sampling** covariance, including spatial correlation, and :math:`D` the
**between-study** structure. Keeping them separate is not tidiness: the exchangeable block in
:mod:`nimare.meta.cbma.blocks` ties them together, and so cannot represent a smooth field with
no heterogeneity. :mod:`nimare.meta.cbma.orthant` takes an arbitrary :math:`S_i` for that reason.

**The one thing this module refuses.** A dense unrestricted :math:`D` learned from a handful of
images. :func:`between_study_covariance` offers ``"scaled"`` (:math:`D = \tau^2 D_0` with
:math:`D_0` supplied and externally justified), ``"diagonal"`` and ``"grouped"``, and rejects
``"dense"`` unless the study count exceeds the free-parameter count by the margin
:data:`DENSE_MARGIN` -- which eight images never do for a basis of any useful rank.

**Borrowing, and the comparator that makes it interpretable.** Section 4's warning is a theorem.
The generalised-least-squares map onto a basis,
:math:`P_B = B(B^\top V^{-1}B)^{-1}B^\top V^{-1}`, has

.. math::
    \operatorname{tr}(P_B V P_B^\top) = \sigma^2 q, \qquad q = \operatorname{rank}(B),

for :math:`V = \sigma^2 I` -- a function of the **rank alone**. Two bases of equal rank, one from
coordinates and one from random directions, deliver *identical* variance reduction. So a gain
measured against an unregularised image-only baseline is the rank's, not the coordinates'. The
only thing a coordinate basis can win is the bias term :math:`\|(P_B-I)m\|^2`, and
:func:`compare_against_equal_regularisation` runs that comparison because leaving it optional
makes every reported gain ambiguous. Derived in ``proofs/spatial_borrowing_projection.py``.

Notes
-----
**A basis fitted against the target images makes its own bias term meaningless.** The plan asks
for a basis "fitted independently of target image outcomes"; :func:`borrowed_mean` cannot check
that and says so. Coordinate-derived basis uncertainty and cohort overlap are likewise outside
this algebra.
"""

from __future__ import annotations

import numpy as np

#: How many independent studies per free covariance parameter before a dense ``D`` is allowed.
#: Not a statistical threshold so much as a refusal to fit what cannot be fitted: the per-study
#: information bound in ``proofs/information_per_study_is_bounded.py`` caps what any number of
#: voxels can say about between-study structure, so the study count is the binding quantity.
DENSE_MARGIN = 10.0

STRUCTURES = ("scaled", "diagonal", "grouped", "dense")


def between_study_covariance(
    structure, rank, *, tau2=None, reference=None, groups=None, variances=None, n_studies=None
):
    r"""Build a **restricted** between-study covariance :math:`D`, or refuse.

    Parameters
    ----------
    structure : {"scaled", "diagonal", "grouped", "dense"}
        ``"scaled"`` is :math:`\tau^2 D_0` with ``reference`` supplying :math:`D_0`: one free
        parameter, and the plan's recommended starting point. ``"diagonal"`` has ``rank`` free
        parameters, ``"grouped"`` one per group. ``"dense"`` has :math:`q(q+1)/2` and is refused
        unless ``n_studies`` exceeds that by :data:`DENSE_MARGIN`.
    reference : :obj:`numpy.ndarray`, optional
        :math:`D_0`, which must be supplied rather than estimated. Its justification is
        external, and sensitivity to it belongs in the analysis rather than here.
    """
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}; got {structure!r}.")
    rank = int(rank)
    if rank < 1:
        raise ValueError("The basis must have at least one column.")

    if structure == "scaled":
        if tau2 is None or reference is None:
            raise ValueError(
                "A scaled structure needs tau2 and an externally justified reference D0. "
                "Estimating D0 from the same images is what this structure exists to avoid."
            )
        reference = np.atleast_2d(np.asarray(reference, dtype=float))
        if reference.shape != (rank, rank):
            raise ValueError(f"reference must be {rank} by {rank}; got {reference.shape}.")
        if float(tau2) < 0:
            raise ValueError("tau2 cannot be negative.")
        return float(tau2) * reference, 1

    if structure == "diagonal":
        if variances is None:
            raise ValueError("A diagonal structure needs one variance per basis column.")
        diagonal = np.asarray(variances, dtype=float).reshape(-1)
        if diagonal.size != rank or np.any(diagonal < 0):
            raise ValueError("variances must be non-negative with one entry per column.")
        return np.diag(diagonal), rank

    if structure == "grouped":
        if groups is None or variances is None:
            raise ValueError("A grouped structure needs group labels and one variance per group.")
        labels = np.asarray(groups).reshape(-1)
        if labels.size != rank:
            raise ValueError("groups must label every basis column.")
        codes, positions = np.unique(labels, return_inverse=True)
        values = np.asarray(variances, dtype=float).reshape(-1)
        if values.size != codes.size or np.any(values < 0):
            raise ValueError("variances must be non-negative with one entry per group.")
        return np.diag(values[positions]), int(codes.size)

    free = rank * (rank + 1) // 2
    if n_studies is None:
        raise ValueError(
            "A dense structure needs n_studies, because whether it can be fitted is a question "
            "about the number of independent studies and not about the voxel count."
        )
    if float(n_studies) < DENSE_MARGIN * free:
        raise ValueError(
            f"A dense D of rank {rank} has {free} free parameters and would need at least "
            f"{DENSE_MARGIN * free:.0f} independent studies at the margin this module applies; "
            f"{float(n_studies):g} were supplied. Per-study information about between-study "
            "structure is capped regardless of how many voxels each study has, so no quantity "
            "of voxels substitutes. Use 'scaled' with a justified D0 and report sensitivity to "
            "it."
        )
    raise ValueError(
        "A dense D is permitted by the study count but is not constructed here: it must be "
        "supplied explicitly so that its provenance is visible."
    )


def gls_projection(basis, sampling_covariance):
    r"""Project onto the basis in the sampling metric.

    :math:`P_B = B(B^\top V^{-1}B)^{-1}B^\top V^{-1}`.
    """
    basis = np.atleast_2d(np.asarray(basis, dtype=float))
    covariance = np.atleast_2d(np.asarray(sampling_covariance, dtype=float))
    voxels = basis.shape[0]
    if covariance.shape != (voxels, voxels):
        raise ValueError(f"sampling_covariance must be {voxels} by {voxels}.")
    inverse = np.linalg.inv(covariance)
    gram = basis.T @ inverse @ basis
    if np.linalg.matrix_rank(gram) < gram.shape[0]:
        raise ValueError(
            "The basis is rank deficient in the sampling metric; drop the redundant columns "
            "rather than regularising them silently, since the rank is what sets the variance."
        )
    return basis @ np.linalg.solve(gram, basis.T @ inverse)


def projection_bias(basis, sampling_covariance, truth):
    r""":math:`(P_B - I)m`: what the basis moves, and to where.

    Exposed because it is the whole cost of borrowing. A constant basis applied to a truth of
    :math:`(1,0)` returns :math:`(-\tfrac12, +\tfrac12)` -- half the effect relocated onto a
    voxel that had none.
    """
    projection = gls_projection(basis, sampling_covariance)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    return (projection - np.eye(projection.shape[0])) @ truth


def variance_trace(basis, sampling_covariance):
    r""":math:`\operatorname{tr}(P_B V P_B^\top)`, which depends on the basis only via its rank.

    Reported so that a claimed gain can be checked against the rank rather than credited to the
    basis's provenance.
    """
    projection = gls_projection(basis, sampling_covariance)
    covariance = np.atleast_2d(np.asarray(sampling_covariance, dtype=float))
    return float(np.trace(projection @ covariance @ projection.T))


def borrowed_mean(basis, values, sampling_covariance, *, residual_variance=0.0):
    r"""GLS estimate of the mean map on the basis, with a residual mean field retained.

    ``residual_variance`` is the variance of :math:`r_0(v)`, the population-mean structure the
    basis does not represent. Leaving it at zero asserts the basis is complete, which is the
    assumption the bias term measures -- so a non-zero value is the honest default for real data
    and the caller must choose it.

    .. warning::
        This conditions on a **supplied** sampling covariance. It does not estimate one from a
        single image, and the plan is explicit that distinguishing sampling covariance from
        spatially structured signal needs subject-level maps or a sampling model that an
        arbitrary group statistic map does not provide.
    """
    projection = gls_projection(basis, sampling_covariance)
    values = np.asarray(values, dtype=float)
    fitted = projection @ values.mean(axis=0) if values.ndim > 1 else projection @ values
    if residual_variance < 0:
        raise ValueError("residual_variance cannot be negative.")
    covariance = np.atleast_2d(np.asarray(sampling_covariance, dtype=float))
    count = values.shape[0] if values.ndim > 1 else 1
    variance = np.diag(projection @ covariance @ projection.T) / count + float(residual_variance)
    return {
        "estimate": fitted,
        "se": np.sqrt(variance),
        "rank": int(np.linalg.matrix_rank(np.atleast_2d(np.asarray(basis, dtype=float)))),
        "variance_trace": variance_trace(basis, sampling_covariance),
        "residual_variance": float(residual_variance),
    }


#: Comparator families for :func:`compare_against_equal_regularisation`. ``"smooth"`` is the one
#: that matters: generic spatial smoothing is the realistic alternative explanation for a gain,
#: and random directions are a weak comparator that almost any basis beats.
COMPARATORS = ("smooth", "random")


def compare_against_equal_regularisation(
    basis,
    values,
    sampling_covariance,
    *,
    truth=None,
    rng=None,
    draws=32,
    comparators=COMPARATORS,
    positions=None,
):
    r"""Compare a candidate basis against **equally regularised** random bases of the same rank.

    Section 4 requires this and it is not optional here, because the variance reduction is the
    rank's: without it, a gain over an unregularised image-only estimator is uninterpretable.

    Returns a dict with the candidate's and the comparators' variance traces -- which must
    agree -- and, when ``truth`` is supplied, their squared biases, which are the only quantity
    that can separate them.

    ``comparators`` defaults to both families, and ``"smooth"`` is the one that decides
    anything: generic spatial smoothing at the same rank is the alternative explanation for a
    gain, so a basis that beats only random directions has shown nothing. ``verdict`` is
    ``"basis earns its gain"`` only when the candidate's squared bias is below the **hardest**
    comparator's; otherwise it names which families it did and did not beat.
    """
    basis = np.atleast_2d(np.asarray(basis, dtype=float))
    voxels, columns = basis.shape
    rank = int(np.linalg.matrix_rank(basis))
    generator = np.random.default_rng() if rng is None else rng

    for name in comparators:
        if name not in COMPARATORS:
            raise ValueError(f"comparators must come from {COMPARATORS}; got {name!r}.")
    grid = (
        np.arange(voxels, dtype=float)
        if positions is None
        else np.asarray(positions, dtype=float).reshape(-1)
    )
    if grid.size != voxels:
        raise ValueError("positions must give one coordinate per voxel.")
    span = max(float(np.ptp(grid)), 1.0)

    candidate_trace = variance_trace(basis, sampling_covariance)
    comparator_traces = []
    comparator_biases = {name: [] for name in comparators}
    for _ in range(int(draws)):
        built = {}
        if "random" in comparators:
            built["random"] = generator.normal(size=(voxels, rank))
        if "smooth" in comparators:
            # Generic smoothing at the same rank: evenly spaced Gaussian bumps with a jittered
            # offset and width. This is the comparator the plan actually needs, because a gain
            # from smoothing looks exactly like a gain from coordinates.
            width = float(generator.uniform(0.05, 0.25)) * span
            offset = float(generator.uniform(0.0, 1.0)) * span / max(rank, 1)
            centres = np.linspace(grid.min(), grid.max(), rank) + offset
            built["smooth"] = np.stack(
                [np.exp(-0.5 * ((grid - centre) / width) ** 2) for centre in centres], axis=1
            )
        for name, matrix in built.items():
            if np.linalg.matrix_rank(matrix) < rank:
                continue
            comparator_traces.append(variance_trace(matrix, sampling_covariance))
            if truth is not None:
                comparator_biases[name].append(
                    float(np.sum(projection_bias(matrix, sampling_covariance, truth) ** 2))
                )

    result = {
        "rank": rank,
        "candidate_variance_trace": candidate_trace,
        "comparator_variance_trace_mean": float(np.mean(comparator_traces)),
        "variance_traces_agree": bool(np.allclose(comparator_traces, candidate_trace, rtol=1e-8)),
        "candidate_squared_bias": None,
        "comparator_squared_bias_min": None,
        "verdict": "gain is the rank's",
    }
    if truth is not None:
        candidate_bias = float(np.sum(projection_bias(basis, sampling_covariance, truth) ** 2))
        result["candidate_squared_bias"] = candidate_bias
        per_family = {
            name: float(np.min(found)) for name, found in comparator_biases.items() if found
        }
        result["comparator_squared_bias_min"] = per_family
        # The verdict is the *hardest* comparator's. A basis that beats only random directions
        # has shown nothing: generic smoothing is the explanation a reader will reach for.
        if per_family and candidate_bias < min(per_family.values()):
            result["verdict"] = "basis earns its gain"
        elif per_family:
            beaten = [name for name, value in per_family.items() if candidate_bias < value]
            missed = [name for name in per_family if name not in beaten]
            result["verdict"] = f"gain is the rank's (beats {beaten or ['nothing']}, not {missed})"
    del values  # the comparison is of bases, not of a particular realisation
    return result
