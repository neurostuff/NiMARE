"""Columnar studyset: an immutable store, views over it, and typed blocks.

A NIMADS studyset is a fixed-depth forest -- every point, image and condition
names exactly one parent analysis, and every analysis exactly one study -- so it
is a set of columnar tables joined by CSR offset arrays, not a graph.

The layering, which the module boundaries enforce:

``store``, ``columns``
    the data. Immutable, numpy only. Knows nothing about selections, algorithms,
    pandas or files.
``layout``
    the rules that keep parent links and offsets consistent.
``view``
    a selection: ``(store, analysis index, point mask)`` plus execution context.
``requirements``, ``blocks``
    what an algorithm declares it needs, and the shapes it gets back.
``edit``
    copy-on-write growth: ``store -> store``.
``io``
    NIMADS and parquet, in and out.

Data flows store -> view -> block and never back.
"""

from nimare.studyset._legacy_normalize import normalize_collection
from nimare.studyset.blocks import (
    Comparison,
    CoordinateBlock,
    DesignBlock,
    ImageBlock,
    LabelBlock,
    TextBlock,
)
from nimare.studyset.columns import AnnotationSet, ColumnStore, Dict8
from nimare.studyset.edit import (
    with_annotation,
    with_images,
    with_metadata,
    with_points,
)
from nimare.studyset.io import (
    convert_neurostore_json_to_parquet,
    from_nimads,
    from_parquet,
    to_nimads_dict,
    write_nimads,
    write_parquet,
)
from nimare.studyset.layout import check_invariants
from nimare.studyset.requirements import (
    Coordinates,
    Images,
    Labels,
    PerAnalysis,
    Texts,
)
from nimare.studyset.store import StudysetStore
from nimare.studyset.studyset import Studyset
from nimare.studyset.view import Context, View

__all__ = [
    "AnnotationSet",
    "ColumnStore",
    "Comparison",
    "Context",
    "CoordinateBlock",
    "Coordinates",
    "DesignBlock",
    "Dict8",
    "ImageBlock",
    "Images",
    "LabelBlock",
    "Labels",
    "PerAnalysis",
    "Studyset",
    "StudysetStore",
    "TextBlock",
    "Texts",
    "View",
    "check_invariants",
    "convert_neurostore_json_to_parquet",
    "from_nimads",
    "from_parquet",
    "normalize_collection",
    "to_nimads_dict",
    "with_annotation",
    "with_images",
    "with_metadata",
    "with_points",
    "write_nimads",
    "write_parquet",
]
