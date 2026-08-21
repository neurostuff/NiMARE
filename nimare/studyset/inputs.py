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
    """Return ``(narrowed studyset, inputs dict)`` for ``required_inputs``.

    The dict holds the shapes estimators already expect: a coordinates frame, a
    list of image paths, a list of metadata values, an annotations frame, and the
    retained ids under ``"id"``.
    """
    if not required_inputs:
        return studyset, {}

    specs = dict(required_inputs)
    reqs = tuple(requirement_for(name, spec) for name, spec in specs.items())
    narrowed, _blocks = studyset.resolve(reqs, drop_invalid=drop_invalid)

    out = {"id": list(narrowed.ids)}
    for name, (kind, field) in specs.items():
        if kind == "coordinates":
            value = narrowed.coordinates
        elif kind == "metadata":
            value = narrowed.get_metadata(field=field)
        elif kind == "image":
            value = narrowed.get_images(imtype=field)
        elif kind == "annotations":
            value = narrowed.annotations_df
        elif kind == "text":
            value = narrowed.get_texts(text_type=field)
        else:  # pragma: no cover - requirement_for already rejected it
            raise ValueError(f"Input {kind!r} not understood.")
        if value is None:
            raise ValueError(
                f"The collection must contain {name}, but no matching data were found."
            )
        out[name] = value
    return narrowed, out
