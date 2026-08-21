"""Named hypothesis tests on a fitted term-based CBMR model.

The older interface took contrasts positionally::

    infer(group_contrasts=[[[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 1, -1]]])

which is unreadable and silently depends on group ordering: reorder the levels and the same
matrix tests a different hypothesis. With named terms the same tests are written out::

    "diagnosis[schizophrenia] = diagnosis[depression]"
    "s(avg_age) = 0"
    ["diagnosis[a] = diagnosis[b]", "diagnosis[b] = diagnosis[c]"]   # joint

Statistics come from the joint coefficient covariance, so a contrast spanning a spatial term and
a scalar one is handled without special-casing.

For a spatial term a contrast is evaluated per voxel. Writing ``c`` for the contrast over the
term's columns and ``B(v)`` for the basis row at voxel ``v``, the contrast vector in the flat
parameter space is ``c kron B(v)``, so::

    estimate(v) = c' beta B(v)
    variance(v) = B(v)' M B(v),   M[p, q] = sum_ab c_a c_b V[a, p, b, q]

with ``V`` the term's covariance block reshaped to ``(columns, bases, columns, bases)``. That
collapses the per-voxel quadratic form into one basis-sized matrix, so cost is
O(n_voxels x n_bases^2) rather than a per-voxel inversion.
"""

import re

import numpy as np
import pandas as pd

from nimare.transforms import chi2_to_nlogp, nlogp_to_z, z_to_nlogp
from nimare.utils import DEFAULT_FLOAT_DTYPE, _clip_p_values, _nlogp_to_logp_values


class ContrastError(ValueError):
    """Raised when a hypothesis cannot be interpreted against a design."""


def _column_index(bound_design, reference):
    """Locate a coefficient column named ``reference``, returning (term name, column index).

    Accepts either the patsy column name (``diagnosis[schiz]``) or a bare level (``schiz``),
    since the map labels users see are the bare form.
    """
    matches = []
    for block in bound_design.blocks:
        for index, column in enumerate(block.column_names):
            if reference == column:
                matches.append((str(block.term), index, True))
                continue
            levels = re.findall(r"\[(?:T\.)?([^\]]+)\]", column)
            if reference == column.split("[")[0] and block.n_columns == 1:
                matches.append((str(block.term), index, False))
            elif reference in levels or reference == "-".join(levels):
                matches.append((str(block.term), index, False))

    exact = [m for m in matches if m[2]]
    candidates = exact or matches
    if not candidates:
        available = sorted(
            column for block in bound_design.blocks for column in block.column_names
        )
        raise ContrastError(
            f"No coefficient named {reference!r}. Available columns: {', '.join(available)}."
        )
    if len({(name, index) for name, index, _ in candidates}) > 1:
        where = ", ".join(sorted({f"{name}[{index}]" for name, index, _ in candidates}))
        raise ContrastError(
            f"{reference!r} is ambiguous; it matches {where}. Use the full column name."
        )
    name, index, _ = candidates[0]
    return name, index


def _parse_statement(statement, bound_design):
    """Parse one ``left = right`` hypothesis into (term name, contrast over that term's columns).

    Both sides must belong to the same term. Cross-term contrasts are a coherent idea but their
    estimate has no single natural scale -- one side is a map and the other a scalar -- so they
    are refused rather than guessed at.
    """
    if statement.count("=") != 1:
        raise ContrastError(
            f"Hypothesis {statement!r} must contain exactly one '=', as in "
            "'diagnosis[a] = diagnosis[b]' or 's(age) = 0'."
        )
    left, right = (side.strip() for side in statement.split("="))
    if not left:
        raise ContrastError(f"Hypothesis {statement!r} has an empty left-hand side.")

    term_name, index = _column_index(bound_design, left)
    block = next(b for b in bound_design.blocks if str(b.term) == term_name)
    contrast = np.zeros(block.n_columns, dtype=float)
    contrast[index] = 1.0

    if right in ("0", "0.0", ""):
        return term_name, contrast

    other_term, other_index = _column_index(bound_design, right)
    if other_term != term_name:
        raise ContrastError(
            f"Hypothesis {statement!r} compares {left!r} in term {term_name} with {right!r} in "
            f"term {other_term}. Contrasts must stay within one term, since a spatial term's "
            "estimate is a map and a scalar term's is a number."
        )
    contrast[other_index] -= 1.0
    return term_name, contrast


def _term_covariance(model, term_name, foci):
    """Return the covariance block belonging to one term."""
    covariance = model.covariance(foci)
    term_slice = model.predictor.design.parameter_slices(model.predictor.n_bases)[term_name]
    return covariance[term_slice, term_slice]


def _spatial_statistics(model, block, contrast, covariance_block):
    """Return z-statistics and log p-values per voxel for a spatial contrast."""
    bases = model.predictor.bases
    n_columns, n_bases = block.n_columns, model.predictor.n_bases
    coefficients = np.atleast_2d(model.fitted_coefficients()[str(block.term)])

    estimate = (contrast @ coefficients) @ bases.T

    blocks = covariance_block.reshape(n_columns, n_bases, n_columns, n_bases)
    collapsed = np.einsum("a,apbq,b->pq", contrast, blocks, contrast, optimize=True)
    variance = np.einsum("vp,pq,vq->v", bases, collapsed, bases, optimize=True)

    standard_error = np.sqrt(np.maximum(variance, 0.0))
    z_values = estimate / np.where(standard_error > 0, standard_error, np.inf)
    return z_values, z_to_nlogp(z_values, tail="two")


def _joint_statistics(model, block, contrast_matrix, covariance_block):
    """Return chi-square, z and log p-values per voxel for a multi-row spatial contrast."""
    bases = model.predictor.bases
    n_columns, n_bases = block.n_columns, model.predictor.n_bases
    coefficients = np.atleast_2d(model.fitted_coefficients()[str(block.term)])

    estimate = (contrast_matrix @ coefficients) @ bases.T  # (n_rows, n_voxels)
    blocks = covariance_block.reshape(n_columns, n_bases, n_columns, n_bases)
    collapsed = np.einsum(
        "sa,apbq,tb->sptq", contrast_matrix, blocks, contrast_matrix, optimize=True
    )
    per_voxel = np.einsum("vp,sptq,vq->vst", bases, collapsed, bases, optimize=True)

    solved = np.linalg.solve(per_voxel, estimate.T[..., np.newaxis])
    chi_square = np.einsum("vs,vs->v", estimate.T, solved[..., 0], optimize=True)
    nlogp = chi2_to_nlogp(chi_square, contrast_matrix.shape[0])
    return chi_square, nlogp_to_z(nlogp, tail="two"), nlogp


def evaluate_hypotheses(model, hypotheses, foci, name=None):
    """Test one or more hypotheses against a fitted model.

    Named ``evaluate_`` rather than ``test_`` deliberately: pytest collects any module-level
    callable whose name begins with ``test``, so a public function called ``test_hypotheses``
    would be picked up as a broken test case in this repository and in any downstream suite that
    imports it.

    Parameters
    ----------
    model : :class:`~nimare.meta.cbmr.model.CBMRModel`
        Fitted model.
    hypotheses : :obj:`str` or :obj:`list` of :obj:`str`
        Hypotheses such as ``"diagnosis[a] = diagnosis[b]"`` or ``"s(age) = 0"``. A list is
        tested jointly, as a generalized linear hypothesis.
    foci : array_like
        The foci the model was fitted to, needed to re-evaluate the information matrix.
    name : :obj:`str`, optional
        Label for the emitted keys. Defaults to the hypotheses joined by ``";"``.

    Returns
    -------
    :obj:`dict`
        Maps and tables keyed as ``{"maps": ..., "tables": ...}``, using the same ``z_``,
        ``p_``, ``logp_`` and ``chiSquare_`` vocabulary as the rest of CBMR.
    """
    statements = [hypotheses] if isinstance(hypotheses, str) else list(hypotheses)
    if not statements:
        raise ContrastError("No hypotheses given.")

    bound_design = model.predictor.design

    parsed = [_parse_statement(statement, bound_design) for statement in statements]
    term_names = {term_name for term_name, _ in parsed}
    if len(term_names) > 1:
        raise ContrastError(
            "A joint hypothesis must stay within one term, but these span "
            f"{sorted(term_names)}."
        )
    term_name = parsed[0][0]
    block = next(b for b in bound_design.blocks if str(b.term) == term_name)
    contrast_matrix = np.vstack([contrast for _, contrast in parsed])
    covariance_block = _term_covariance(model, term_name, foci)

    label = name or ";".join(statements)
    maps, tables = {}, {}

    if not block.term.spatial:
        estimate = contrast_matrix @ np.atleast_1d(model.fitted_coefficients()[term_name])
        variance = contrast_matrix @ covariance_block @ contrast_matrix.T
        if contrast_matrix.shape[0] == 1:
            standard_error = float(np.sqrt(max(variance[0, 0], 0.0)))
            z_value = float(estimate[0] / standard_error) if standard_error > 0 else 0.0
            nlogp = float(z_to_nlogp(np.array([z_value]), tail="two")[0])
            tables[f"contrast_{label}"] = _scalar_table(
                estimate[0], standard_error, z_value, nlogp
            )
        else:
            chi_square = float(estimate @ np.linalg.solve(variance, estimate))
            nlogp = float(chi2_to_nlogp(np.array([chi_square]), contrast_matrix.shape[0])[0])
            z_value = float(nlogp_to_z(np.array([nlogp]), tail="two")[0])
            tables[f"contrast_{label}"] = _scalar_table(
                np.nan, np.nan, z_value, nlogp, chi_square=chi_square
            )
        return {"maps": maps, "tables": tables}

    if contrast_matrix.shape[0] == 1:
        z_values, nlogp = _spatial_statistics(model, block, contrast_matrix[0], covariance_block)
    else:
        chi_square, z_values, nlogp = _joint_statistics(
            model, block, contrast_matrix, covariance_block
        )
        maps[f"chiSquare_{label}"] = chi_square

    maps[f"z_{label}"] = z_values
    maps[f"p_{label}"] = _clip_p_values(np.exp(nlogp), dtype=DEFAULT_FLOAT_DTYPE)
    maps[f"logp_{label}"] = _nlogp_to_logp_values(nlogp)
    return {"maps": maps, "tables": tables}


def _scalar_table(estimate, standard_error, z_value, nlogp, chi_square=None):
    """Build the one-row table a scalar contrast reports."""
    row = {
        "estimate": [estimate],
        "standard_error": [standard_error],
        "z": [z_value],
        "p": [float(np.exp(nlogp))],
        "logp": [float(_nlogp_to_logp_values(np.array([nlogp]), dtype=np.float64)[0])],
    }
    if chi_square is not None:
        row["chi_square"] = [chi_square]
    return pd.DataFrame(row)
