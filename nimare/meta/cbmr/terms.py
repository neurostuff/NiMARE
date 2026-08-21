"""Term structure for CBMR models.

CBMR's linear predictor is a sum of outer products. Each term contributes an experiment-level
block and a voxel-level block::

    eta = sum_t  E_t  beta_t  V_t'

``E_t`` comes from the experiment annotations -- indicator columns for a factor, the column
itself for a covariate. ``V_t`` is either the spatial B-spline basis, making the term's
coefficient a map, or a single column of ones, making it a scalar. That choice is the *whole*
of the global-versus-voxelwise distinction, and it belongs to individual terms rather than to
the model: a model can perfectly well have one moderator of each kind.

Formula syntax marks it with ``s()`` (``spatial()`` is accepted as a synonym), after mgcv's
smooth terms::

    ~ 1                          one pooled spatial map
    ~ s(diagnosis)               one map per diagnosis
    ~ s(diagnosis) + n           plus a scalar coefficient for n, pooled across groups
    ~ s(diagnosis) + s(n)        plus a map for n
    ~ s(diagnosis) + n + s(age)  n scalar, age spatially varying

Notes on the intercept
----------------------
There is never a scalar intercept column. A cubic B-spline basis is a partition of unity, so it
already spans the constant and an intercept column would be collinear with it -- measured on
NiMARE's basis, the constant sits 98.6% inside its span. This is the documented behavior of
``splines::bs``, which drops a basis column, and of the ``cpr`` package, which prepends ``0 +``
to the formula for exactly this reason. Because near-collinearity does not trip a rank check,
it would degrade a fit quietly rather than fail.

So ``0 +`` is accepted and ignored, since it describes what already happens, and an explicit
``1 +`` alongside a cell-means spatial factor is rejected rather than silently reparameterized.
A spatial baseline is always present: implicit when nothing else spans it, and absorbed by a
cell-means factor term when one does.

Factors and covariates are not the same case here, which mgcv also distinguishes. A spatial
factor term expands to disjoint per-level blocks -- each level's coefficients see only that
level's experiments -- so it is identified without further constraint. A spatial covariate term
is identified by the covariate itself. Neither needs the sum-to-zero centering mgcv applies,
because neither competes with an intercept we do not emit.
"""

import re
from dataclasses import dataclass, field, replace

import numpy as np
import patsy

SPATIAL_MARKERS = ("s", "spatial")
INTERCEPT_TERM = "1"


class FormulaError(ValueError):
    """Raised when a CBMR formula cannot be interpreted."""


@dataclass(frozen=True)
class Term:
    """One term of a CBMR linear predictor.

    Attributes
    ----------
    expr : :obj:`str`
        Patsy expression for the experiment-level block, e.g. ``"1"``, ``"diagnosis"``,
        ``"diagnosis:drug_status"`` or ``"age"``.
    spatial : :obj:`bool`
        Whether the coefficient varies over voxels. ``True`` crosses the term with the spline
        basis; ``False`` gives it a single coefficient per experiment-level column.
    spacing : :obj:`int` or None, optional
        Overrides the model's ``spline_spacing`` for this term only. Ignored when
        ``spatial`` is ``False``. Default is None.
    """

    expr: str
    spatial: bool
    spacing: int = None

    def __post_init__(self):
        """Validate the term's fields."""
        if not self.expr:
            raise FormulaError("A term needs a non-empty expression.")
        if self.spacing is not None:
            if self.spatial is False:
                raise FormulaError(
                    f"spacing is meaningless for the non-spatial term {self.expr!r}, which has "
                    "no spline basis to space."
                )
            if int(self.spacing) <= 0:
                raise FormulaError(f"spacing must be positive, got {self.spacing!r}.")

    @property
    def is_intercept(self):
        """Whether this term is the spatial baseline."""
        return self.expr == INTERCEPT_TERM

    def __str__(self):
        """Render the term back into formula syntax."""
        if self.is_intercept:
            return INTERCEPT_TERM
        if not self.spatial:
            return self.expr
        inner = self.expr if self.spacing is None else f"{self.expr}, spacing={self.spacing}"
        return f"s({inner})"


def _split_top_level(text, separator="+"):
    """Split on a separator, ignoring occurrences nested inside parentheses."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise FormulaError(f"Unbalanced parentheses in {text!r}.")
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if depth != 0:
        raise FormulaError(f"Unbalanced parentheses in {text!r}.")
    parts.append("".join(current))
    return [part.strip() for part in parts]


def _parse_spatial_marker(piece):
    """Return ``(inner_expression, spacing)`` if ``piece`` is an ``s(...)`` call, else None."""
    match = re.fullmatch(r"(\w+)\s*\((.*)\)", piece, flags=re.DOTALL)
    if match is None or match.group(1) not in SPATIAL_MARKERS:
        return None

    arguments = _split_top_level(match.group(2), separator=",")
    expression, spacing = None, None
    for argument in arguments:
        if not argument:
            continue
        keyword = re.fullmatch(r"spacing\s*=\s*(\d+)", argument)
        if keyword is not None:
            spacing = int(keyword.group(1))
        elif expression is None:
            expression = argument
        else:
            raise FormulaError(
                f"{piece!r} takes one expression and an optional spacing=, got {arguments!r}."
            )
    if expression is None:
        raise FormulaError(f"{piece!r} needs an expression to make spatial, e.g. s(diagnosis).")
    return expression, spacing


@dataclass(frozen=True)
class Design:
    """An ordered set of CBMR terms, before it has seen any data.

    Parsing is deliberately separate from binding. A ``CBMR`` estimator is constructed with a
    formula but only meets the annotations at ``fit`` time, and whether a term is a factor --
    which decides how many columns it contributes and whether it absorbs the spatial baseline
    -- is a property of the data, not of the formula.
    """

    terms: tuple = field(default_factory=tuple)
    explicit_intercept: bool = False

    def __post_init__(self):
        """Normalize and validate the term sequence."""
        object.__setattr__(self, "terms", tuple(self.terms))
        seen = {}
        for term in self.terms:
            key = (term.expr, term.spatial)
            if key in seen:
                raise FormulaError(f"Term {term} appears more than once.")
            seen[key] = True
        if not self.terms:
            raise FormulaError("A design needs at least one term.")

    @classmethod
    def from_formula(cls, formula):
        """Parse a formula string into a :class:`Design`.

        Parameters
        ----------
        formula : :obj:`str`
            One-sided formula such as ``"~ s(diagnosis) + n + s(age)"``. The leading ``~`` is
            optional. ``0 +`` is accepted and ignored; see the module docstring.
        """
        if not isinstance(formula, str):
            raise FormulaError(f"A formula must be a string, got {type(formula).__name__}.")

        right_hand_side = formula.split("~", 1)[-1].strip()
        if not right_hand_side:
            raise FormulaError(f"Formula {formula!r} has no terms on the right of the '~'.")

        terms, explicit_intercept = [], False
        for piece in _split_top_level(right_hand_side):
            if not piece:
                raise FormulaError(f"Formula {formula!r} has an empty term.")
            if piece == "0":
                # No-op: there is never a scalar intercept to suppress. Accepted because it is
                # the correct idiom for a partition-of-unity basis elsewhere in the ecosystem.
                continue
            if piece == INTERCEPT_TERM:
                explicit_intercept = True
                terms.append(Term(expr=INTERCEPT_TERM, spatial=True))
                continue

            marker = _parse_spatial_marker(piece)
            if marker is not None:
                expression, spacing = marker
                if expression == INTERCEPT_TERM:
                    explicit_intercept = True
                terms.append(Term(expr=expression, spatial=True, spacing=spacing))
            else:
                terms.append(Term(expr=piece, spatial=False))

        if not terms:
            raise FormulaError(
                f"Formula {formula!r} has no terms; '0' only says there is no scalar intercept."
            )
        return cls(terms=tuple(terms), explicit_intercept=explicit_intercept)

    @property
    def spatial_terms(self):
        """Terms whose coefficients vary over voxels."""
        return tuple(term for term in self.terms if term.spatial)

    @property
    def global_terms(self):
        """Terms with a single coefficient per experiment-level column."""
        return tuple(term for term in self.terms if not term.spatial)

    def with_term(self, term):
        """Return a copy with one more term appended."""
        return replace(self, terms=self.terms + (term,))

    def __str__(self):
        """Render the design back into formula syntax."""
        return "~ " + " + ".join(str(term) for term in self.terms)


def formula_to_design(formula):
    """Parse ``formula`` if it is a string, or pass a :class:`Design` through unchanged."""
    if isinstance(formula, Design):
        return formula
    return Design.from_formula(formula)


def _experiment_block(expr, annotations):
    """Build one term's experiment-level design block.

    Uses patsy for everything except the intercept. Note the ``0 +``: patsy would otherwise add
    an intercept column and treatment-code a factor against a reference level, whereas CBMR
    wants one column per level -- see the module docstring on why there is no intercept column
    and why cell-means coding is the right default for a spatial factor.
    """
    if expr == INTERCEPT_TERM:
        return np.ones((len(annotations), 1), dtype=float), (INTERCEPT_TERM,)

    try:
        matrix = patsy.dmatrix(f"0 + {expr}", annotations, return_type="matrix")
    except patsy.PatsyError as error:
        available = ", ".join(sorted(map(str, annotations.columns)))
        raise FormulaError(
            f"Could not build term {expr!r} from the experiment annotations: {error}. "
            f"Available columns: {available}."
        ) from error

    names = tuple(str(name) for name in matrix.design_info.column_names)
    return np.asarray(matrix, dtype=float), names


def _spans_intercept(block):
    """Return whether an experiment-level block already spans the constant.

    True for cell-means indicators: every row loads on exactly one column, so the columns sum
    to the constant. That is what lets a spatial factor term serve as the baseline, and what
    makes adding a separate intercept redundant.
    """
    if block.size == 0:
        return False
    is_indicator = np.all((block == 0) | (block == 1))
    return bool(is_indicator and np.allclose(block.sum(axis=1), 1.0))


@dataclass(frozen=True)
class TermBlock:
    """One term bound to data: its experiment-level block and column names."""

    term: Term
    block: np.ndarray
    column_names: tuple

    @property
    def n_columns(self):
        """Number of experiment-level columns this term contributes."""
        return self.block.shape[1]

    def n_parameters(self, n_bases):
        """Return the number of coefficients this term owns, given the basis width."""
        return self.n_columns * (n_bases if self.term.spatial else 1)


@dataclass(frozen=True)
class BoundDesign:
    """A :class:`Design` resolved against experiment annotations.

    Owns the parameter layout, which is the piece covariance estimation needs. Coefficient
    blocks are not independent -- a pooled global moderator is correlated with every group's
    spatial coefficients -- so any covariance estimator has to know which slice belongs to
    which term in order to assemble the cross blocks. Handing out only per-block information
    is how standard errors end up being the inverse of a block rather than the block of an
    inverse.
    """

    blocks: tuple
    design: Design

    @property
    def terms(self):
        """The bound terms, in formula order."""
        return tuple(block.term for block in self.blocks)

    @property
    def spatial_blocks(self):
        """Bound terms whose coefficients vary over voxels."""
        return tuple(block for block in self.blocks if block.term.spatial)

    def n_parameters(self, n_bases):
        """Return the total number of coefficients, given the spline basis width."""
        return sum(block.n_parameters(n_bases) for block in self.blocks)

    def parameter_slices(self, n_bases):
        """Map each term to the slice of the flat coefficient vector it owns."""
        slices, offset = {}, 0
        for block in self.blocks:
            width = block.n_parameters(n_bases)
            slices[str(block.term)] = slice(offset, offset + width)
            offset += width
        return slices

    def describe(self, n_bases):
        """Return a human-readable per-term parameter budget.

        Worth putting in front of users: each ``s()`` term costs ``n_bases`` coefficients per
        column, which at the default spacing is 457 apiece -- as much as another group's whole
        baseline map. The old API hid that behind a single ``moderator_effect`` switch that
        promoted every moderator at once.
        """
        lines = []
        for block in self.blocks:
            lines.append(
                f"  {str(block.term):32} {block.n_columns:>4} column(s) x "
                f"{n_bases if block.term.spatial else 1:>5} = "
                f"{block.n_parameters(n_bases):>7} parameters"
            )
        lines.append(f"  {'total':32} {'':>4}            {self.n_parameters(n_bases):>13}")
        return "\n".join(lines)


def bind(design, annotations):
    """Resolve a :class:`Design` against experiment annotations.

    Where the intercept rules are enforced, because whether a term spans the constant depends
    on whether it is a factor, which only the data knows.
    """
    design = formula_to_design(design)

    blocks = []
    for term in design.terms:
        block, names = _experiment_block(term.expr, annotations)
        blocks.append(TermBlock(term=term, block=block, column_names=names))

    absorbing = [
        b
        for b in blocks
        if b.term.spatial and not b.term.is_intercept and _spans_intercept(b.block)
    ]

    if design.explicit_intercept and absorbing:
        raise FormulaError(
            "An explicit '1' cannot be combined with the cell-means spatial term "
            f"{absorbing[0].term} -- that term already gives every level its own map, so its "
            "columns span the constant and a separate baseline would be collinear with them. "
            f"Drop the '1' and write the design as '{design}' without it, or make the factor "
            "non-spatial if you wanted a shared baseline plus per-level offsets."
        )

    if not absorbing and not any(b.term.is_intercept for b in blocks):
        # Nothing spans the constant, so add the spatial baseline the same way R adds an
        # intercept. CBMR always estimates a spatial intensity; there is no model without one.
        intercept = Term(expr=INTERCEPT_TERM, spatial=True)
        block, names = _experiment_block(INTERCEPT_TERM, annotations)
        blocks.insert(0, TermBlock(term=intercept, block=block, column_names=names))
        design = Design(terms=(intercept,) + design.terms, explicit_intercept=False)

    return BoundDesign(blocks=tuple(blocks), design=design)
