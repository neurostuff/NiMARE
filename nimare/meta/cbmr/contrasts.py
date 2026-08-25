"""Hypothesis tests on a fitted term-based CBMR model.

A hypothesis is written over the *levels* of a term, in the notation the result maps already
use::

    result.test("schizophrenia = depression")            # one contrast
    result.test(["a = b", "b = c"])                      # tested jointly, as a GLH
    result.test("s(avg_age) = 0")
    result.test("2 * a = b + c")                          # arithmetic
    result.test(term="diagnosis", method="pairwise")     # every pair at once

Two ideas from the R ecosystem shape this.

Levels, not coefficients
    A term's coefficients are not always its levels. An ``sz()`` factor is reparameterized, so its
    coefficients are contrasts *among* levels and a hypothesis stated over them would be
    uninterpretable. The R package ``hypr`` exists to make this distinction explicit: a hypothesis
    matrix and a contrast matrix are different objects, related by a matrix operation rather than
    a relabelling. So hypotheses are parsed over ``TermBlock.level_names`` and pushed through
    ``TermBlock.level_map``, which is the identity whenever the two coincide.

Enumerate, don't ask
    "Which groups differ?" is a request for *all* pairwise comparisons, not an invitation to write
    six of them. ``method=`` generates them, after ``emmeans``' named contrast families and
    ``gratia::difference_smooths``.

Parsing is :meth:`patsy.DesignInfo.linear_constraint` -- the same parser ``statsmodels`` uses for
``t_test`` and ``wald_test``. It handles arithmetic, bare difference expressions, non-zero
right-hand sides and multiple comma-separated constraints, and returns the contrast matrix
directly. Writing that grammar again by hand would be strictly worse.
"""

import re

import numpy as np
import pandas as pd
import patsy

from nimare.transforms import chi2_to_nlogp, nlogp_to_z, z_to_nlogp
from nimare.utils import DEFAULT_FLOAT_DTYPE, _clip_p_values, _nlogp_to_logp_values

CONTRAST_METHODS = ("pairwise", "reference", "consecutive", "zero")

#: Separator between the two sides of a contrast label. Not "-": a level label for an interaction
#: term is itself hyphenated, so "depression-No-depression-Yes" would not say where one side ends.
VERSUS = "_vs_"

# Matches a coefficient reference: "n", "dx[a]", "dx[a]:drug[y]", or a bare label like
# "schizophrenia-Yes". Deliberately greedy over hyphens so composite labels resolve as one
# token, which is why binary operators need surrounding spaces -- see the module docstring.
_REFERENCE = re.compile(r"[A-Za-z_][\w.\-]*(?:\[[^\]]*\])?(?::[A-Za-z_][\w.\-]*(?:\[[^\]]*\])?)*")


class ContrastError(ValueError):
    """Raised when a hypothesis cannot be interpreted against a design."""


def _bare_label(name):
    """Return the readable form of a coefficient name, as the result maps use."""
    parts = []
    for piece in str(name).split(":"):
        match = re.search(r"\[(?:T\.)?([^\]]+)\]", piece)
        parts.append(match.group(1) if match else piece)
    return "-".join(parts)


def _level_index(bound_design):
    """Return the level vocabulary a hypothesis may be written over.

    Returns ``(names, term_of, aliases)``: every term's level names, which term each belongs to,
    and the bare-label aliases users see in map keys. Ambiguous aliases are dropped rather than
    guessed at, so a collision surfaces as "unknown coefficient" with the full names listed.
    """
    names, term_of, seen = [], {}, {}
    for block in bound_design.blocks:
        if block.term.exposure:
            # An exposure owns no coefficient, so it contributes no level to write a hypothesis
            # over. Naming one is caught by build_contrast, which can say so specifically.
            continue
        for level in block.level_names:
            if level in term_of:
                raise ContrastError(
                    f"Two terms both contribute a coefficient named {level!r}; the design is "
                    "ambiguous."
                )
            names.append(level)
            term_of[level] = str(block.term)
            alias = _bare_label(level)
            if alias != level:
                seen.setdefault(alias, []).append(level)

    aliases = {alias: levels[0] for alias, levels in seen.items() if len(levels) == 1}
    return names, term_of, aliases


def _canonicalize(statement, names, aliases):
    """Rewrite bare labels in ``statement`` into full coefficient names."""
    known = set(names)

    def replace(match):
        token = match.group(0)
        if token in known:
            return token
        return aliases.get(token, token)

    return _REFERENCE.sub(replace, statement)


def _derive_label(statements):
    """Name a contrast after the hypothesis it tests.

    A single equality becomes ``left_vs_right``, so a hand-written ``"a = b"`` is labelled exactly
    as ``method="pairwise"`` would label it. Anything else -- arithmetic, a bare difference
    expression, several statements -- keeps its own text with whitespace collapsed. That is not
    always pretty, but it is what the user wrote, so it cannot mislead about which contrast a map
    holds. Pass ``name=`` to override.
    """
    if len(statements) == 1 and statements[0].count("=") == 1:
        left, right = (side.strip() for side in statements[0].split("="))
        if right in ("0", "0.0", ""):
            return left.replace(" ", "")
        return f"{left}{VERSUS}{right}".replace(" ", "")
    return ";".join(s.replace(" ", "") for s in statements)


def _reject_exposure_references(bound_design, statements):
    """Say so plainly when a hypothesis names the exposure, rather than "unknown coefficient"."""
    exposure_names = {
        level
        for block in bound_design.blocks
        if block.term.exposure
        for level in (block.level_names + (block.term.expr, str(block.term)))
    }
    if not exposure_names:
        return
    for statement in statements:
        for token in _REFERENCE.findall(str(statement)):
            if token in exposure_names:
                term = next(b.term for b in bound_design.blocks if b.term.exposure)
                raise ContrastError(
                    f"{token!r} is the exposure {term}, which has no coefficient to test. Its "
                    "value is fixed at 1 by construction rather than estimated, so there is no "
                    "hypothesis to state about it. What the exposure changes is what every "
                    "*other* term means: each spatial term is a distribution over voxels rather "
                    "than a rate, so a contrast between them compares where foci fall rather "
                    "than how many."
                )


def build_contrast(bound_design, hypotheses):
    """Translate hypotheses into a contrast over one term's coefficients.

    Returns ``(term name, contrast matrix, constants, label)``. The matrix is over the term's
    *coefficients*, having been pushed through ``level_map`` from the levels the hypothesis was
    written in.
    """
    statements = [hypotheses] if isinstance(hypotheses, str) else list(hypotheses)
    if not statements:
        raise ContrastError("No hypotheses given.")

    names, term_of, aliases = _level_index(bound_design)
    _reject_exposure_references(bound_design, statements)
    canonical = [_canonicalize(str(s), names, aliases) for s in statements]

    try:
        constraint = patsy.DesignInfo(names).linear_constraint(canonical)
    except patsy.PatsyError as error:
        raise ContrastError(
            f"Could not interpret {statements if len(statements) > 1 else statements[0]!r}: "
            f"{error}. Hypotheses are written over coefficient names, of which this design has: "
            f"{', '.join(names)}."
        ) from error

    coefficients = np.atleast_2d(np.asarray(constraint.coefs, dtype=float))
    constants = np.asarray(constraint.constants, dtype=float).reshape(-1)

    touched = {term_of[name] for name, column in zip(names, coefficients.any(axis=0)) if column}
    if len(touched) > 1:
        raise ContrastError(
            f"This hypothesis spans the terms {sorted(touched)}. A contrast must stay within one "
            "term, since a spatial term's estimate is a map and a scalar term's is a number, and "
            "the two have no common scale."
        )

    term_name = touched.pop()
    block = next(b for b in bound_design.blocks if str(b.term) == term_name)
    positions = [index for index, name in enumerate(names) if term_of[name] == term_name]
    over_levels = coefficients[:, positions]

    # hypr's point: a hypothesis over levels is not a contrast over coefficients unless the two
    # coincide. level_map carries it across, and is the identity when they do.
    return term_name, over_levels @ block.level_map, constants, _derive_label(statements)


def generate_hypotheses(bound_design, term, method):
    """Generate the hypotheses a named contrast family stands for.

    Returns ``[(label, statement), ...]``. After ``emmeans``' contrast families: enumerating the
    comparisons is almost always what "which levels differ?" means, and writing them out by hand
    is both tedious and a place to make a mistake.
    """
    if method not in CONTRAST_METHODS:
        raise ContrastError(f"method must be one of {CONTRAST_METHODS}, got {method!r}.")

    candidates = [b for b in bound_design.blocks if str(b.term) == term or b.term.expr == term]
    if not candidates:
        available = sorted(str(b.term) for b in bound_design.blocks)
        raise ContrastError(f"No term named {term!r}. This design has: {', '.join(available)}.")
    block = candidates[0]
    levels = list(block.level_names)

    if method == "zero":
        return [(_bare_label(level), f"{level} = 0") for level in levels]

    if len(levels) < 2:
        raise ContrastError(
            f"method={method!r} compares levels, but {str(block.term)!r} has only one "
            f"coefficient. Use method='zero' to test it against zero."
        )

    if method == "pairwise":
        pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i + 1 :]]
    elif method == "reference":
        pairs = [(level, levels[0]) for level in levels[1:]]
    else:  # consecutive
        pairs = list(zip(levels[1:], levels[:-1]))

    return [(f"{_bare_label(a)}{VERSUS}{_bare_label(b)}", f"{a} = {b}") for a, b in pairs]


def _statistics(model, block, contrast, constants, covariance_block, bases):
    """Return the statistics for one contrast, spatial or scalar.

    Returns ``(estimate, standard_error, chi_square, z, nlogp)``. ``estimate`` and
    ``standard_error`` are None for a multi-row contrast, which has no single effect size --
    a joint hypothesis is a statement about a subspace, not about one number.
    """
    coefficients = np.atleast_2d(model.fitted_coefficients()[str(block.term)])
    n_rows = contrast.shape[0]

    if block.term.spatial:
        n_columns, n_bases = block.n_columns, bases.shape[1]
        blocks = covariance_block.reshape(n_columns, n_bases, n_columns, n_bases)
        estimate = (contrast @ coefficients) @ bases.T - constants[:, None]

        if n_rows == 1:
            collapsed = np.einsum("a,apbq,b->pq", contrast[0], blocks, contrast[0], optimize=True)
            variance = np.einsum("vp,pq,vq->v", bases, collapsed, bases, optimize=True)
            standard_error = np.sqrt(np.maximum(variance, 0.0))
            z = estimate[0] / np.where(standard_error > 0, standard_error, np.inf)
            return estimate[0], standard_error, None, z, z_to_nlogp(z, tail="two")

        collapsed = np.einsum("sa,apbq,tb->sptq", contrast, blocks, contrast, optimize=True)
        per_voxel = np.einsum("vp,sptq,vq->vst", bases, collapsed, bases, optimize=True)
        solved = np.linalg.solve(per_voxel, estimate.T[..., np.newaxis])
        chi_square = np.einsum("vs,vs->v", estimate.T, solved[..., 0], optimize=True)
        nlogp = chi2_to_nlogp(chi_square, n_rows)
        return None, None, chi_square, nlogp_to_z(nlogp, tail="two"), nlogp

    estimate = contrast @ np.atleast_1d(coefficients).reshape(-1) - constants
    variance = contrast @ covariance_block @ contrast.T
    if n_rows == 1:
        standard_error = float(np.sqrt(max(variance[0, 0], 0.0)))
        z = float(estimate[0] / standard_error) if standard_error > 0 else 0.0
        return (
            float(estimate[0]),
            standard_error,
            None,
            z,
            float(z_to_nlogp(np.array([z]), tail="two")[0]),
        )

    chi_square = float(estimate @ np.linalg.solve(variance, estimate))
    nlogp = float(chi2_to_nlogp(np.array([chi_square]), n_rows)[0])
    return None, None, chi_square, float(nlogp_to_z(np.array([nlogp]), tail="two")[0]), nlogp


def evaluate_hypotheses(
    model, hypotheses=None, foci=None, name=None, term=None, method=None, **covariance_kwargs
):
    """Test hypotheses about the fitted coefficients.

    Parameters
    ----------
    model : :class:`~nimare.meta.cbmr.model.CBMRModel`
        Fitted model.
    hypotheses : :obj:`str` or :obj:`list` of :obj:`str`, optional
        Hypotheses over level names. A list is tested jointly, as a generalized linear
        hypothesis, rather than one at a time.
    foci : array_like
        The foci the model was fitted to.
    name : :obj:`str`, optional
        Overrides the label derived from the hypothesis. Rarely needed.
    term, method : :obj:`str`, optional
        Generate a family of contrasts over one term instead of naming them: ``"pairwise"``,
        ``"reference"``, ``"consecutive"`` or ``"zero"``. Each is emitted under its own label.
    **covariance_kwargs
        Passed to :meth:`~nimare.meta.cbmr.model.CBMRModel.covariance`, so
        ``cov_type="sandwich"`` gives robust statistics. Named ``cov_type`` after statsmodels
        precisely so it cannot be confused with ``method``, which names a contrast family.

    Returns
    -------
    :obj:`dict`
        ``{"maps": ..., "tables": ...}``, using the ``est``, ``se``, ``z``, ``p``, ``logp`` and
        ``chiSquare`` vocabulary the rest of NiMARE uses.
    """
    if (term is None) != (method is None):
        raise ContrastError("term and method must be given together.")
    if term is None and hypotheses is None:
        raise ContrastError("Give either hypotheses, or term and method.")
    if term is not None and hypotheses is not None:
        raise ContrastError("Give either hypotheses, or term and method, not both.")

    bound_design = model.predictor.design
    if term is not None:
        requested = generate_hypotheses(bound_design, term, method)
    else:
        requested = [(name, hypotheses)]

    bases = model.predictor.bases
    slices = bound_design.parameter_slices(model.predictor.n_bases)
    covariance = model.covariance(foci, **covariance_kwargs)

    maps, tables = {}, {}
    for label, statement in requested:
        term_name, contrast, constants, derived = build_contrast(bound_design, statement)
        block = next(b for b in bound_design.blocks if str(b.term) == term_name)
        term_slice = slices[term_name]
        estimate, standard_error, chi_square, z, nlogp = _statistics(
            model, block, contrast, constants, covariance[term_slice, term_slice], bases
        )

        key = label or derived
        if block.term.spatial:
            if chi_square is not None:
                maps[f"chiSquare_{key}"] = chi_square
            else:
                maps[f"est_{key}"] = estimate
                maps[f"se_{key}"] = standard_error
            maps[f"z_{key}"] = z
            maps[f"p_{key}"] = _clip_p_values(np.exp(nlogp), dtype=DEFAULT_FLOAT_DTYPE)
            maps[f"logp_{key}"] = _nlogp_to_logp_values(nlogp)
        else:
            row = {"z": [z], "p": [float(np.exp(nlogp))]}
            row["logp"] = [float(_nlogp_to_logp_values(np.array([nlogp]), dtype=np.float64)[0])]
            if chi_square is None:
                row["est"], row["se"] = [estimate], [standard_error]
            else:
                row["chiSquare"] = [chi_square]
            tables[f"contrast_{key}"] = pd.DataFrame(row)

    return {"maps": maps, "tables": tables}
