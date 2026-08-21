"""Resolve an estimator's declared inputs through requirements and blocks.

Estimators and decoders declare what they need as ``_required_inputs``, a mapping
of name to ``(kind, field)``. That vocabulary is kept -- it is how every estimator
in the library already describes itself -- but it is now *translated* into
:mod:`nimare.studyset.requirements` and resolved in one pass, so:

* validity is intersected once and the studyset is narrowed once, rather than each
  item being fetched separately and the surviving positions intersected afterwards;
* every returned value is derived from the same narrowed view, so they cannot
  disagree about which analyses they describe.

The narrowed view is returned alongside, because that -- not the dict -- is the
thing worth passing to a block-based inner loop.
"""

from __future__ import annotations

from nimare.studyset import requirements as _req

__all__ = ["collect_inputs", "requirement_for"]


def requirement_for(name, spec):
    """Translate one ``_required_inputs`` entry into a requirement."""
    kind, field = spec
    if kind == "coordinates":
        return _req.Coordinates(name=name)
    if kind == "metadata":
        return _req.PerAnalysis(field, name=name)
    if kind == "image":
        return _req.Images(field, name=name)
    if kind == "annotations":
        return _req.Labels(name=name)
    if kind == "text":
        return _req.Texts(field or "abstract", name=name)
    raise ValueError(f"Input {kind!r} not understood.")


def collect_inputs(studyset, required_inputs, drop_invalid=True):
    """Return ``(narrowed studyset, inputs dict, blocks dict)``.

    One pass: the ``_required_inputs`` vocabulary is translated to requirements
    once, resolved once, and each requirement renders its own ``inputs_`` shape
    from the block it produced. There is no second switch over the same strings,
    and nothing that was built is discarded -- ``blocks`` is returned so an inner
    loop can take the aligned form instead of re-deriving it from the frame.
    """
    if not required_inputs:
        return studyset, {}, {}

    reqs = tuple(requirement_for(name, spec) for name, spec in dict(required_inputs).items())
    narrowed, blocks = studyset.resolve(reqs, drop_invalid=drop_invalid)

    out = {"id": list(narrowed.ids)}
    for req in reqs:
        value = req.as_input(narrowed.view, blocks[req.name])
        if value is None:
            raise ValueError(
                f"The collection must contain {req.name}, but no matching data were found."
            )
        out[req.name] = value
    return narrowed, out, blocks
