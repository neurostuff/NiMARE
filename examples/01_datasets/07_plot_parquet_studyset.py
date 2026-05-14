"""Loading Studysets from parquet.

.. _parquet_studyset:

===============================
Loading Studysets from parquet
===============================

NiMARE can load a :class:`~nimare.nimads.Studyset` from a directory of
parquet files. This table-backed format is useful for large NeuroStore
Studysets because it avoids parsing a single large nested JSON file and keeps
the Studyset lazy until nested Study/Analysis objects are explicitly needed.

The main use case for this format will be distributed Studyset releases from
https://www.neurostore.org/api/neurostore-studyset-releases/. Release archives
are expected to contain the same manifest and table layout demonstrated here.
"""

from pathlib import Path

import pandas as pd

from nimare.nimads import Studyset
from nimare.utils import get_resource_path

###############################################################################
# Find the example parquet Studyset
# -----------------------------------------------------------------------------
# A parquet Studyset directory contains a ``studyset.json`` manifest and one
# parquet file per canonical Studyset table. This example uses a small packaged
# slice of a NeuroStore release.

parquet_dir = Path(get_resource_path()) / "neurostore_parquet_studyset"
if not parquet_dir.exists():
    # Support running this example directly from a source checkout before the
    # new packaged resource has been installed into the active environment.
    parquet_dir = (
        Path(__file__).resolve().parents[2]
        / "nimare"
        / "resources"
        / "neurostore_parquet_studyset"
    )
print(sorted(path.name for path in parquet_dir.iterdir()))

###############################################################################
# Inspect the manifest
# -----------------------------------------------------------------------------
# The manifest records the Studyset id/name, schema version, annotation ids, and
# table filenames.

print((parquet_dir / "studyset.json").read_text())

###############################################################################
# Inspect the parquet table shapes
# -----------------------------------------------------------------------------
# The table layout is:
#
# - ``studies.parquet``: one row per study, with ``study_id``, ``name``,
#   ``description``, ``authors``, and ``publication``.
# - ``analyses.parquet``: one row per analysis, with the full analysis ``id``.
# - ``coordinates.parquet``: coordinate rows keyed by analysis id.
# - ``metadata.parquet``: one row per analysis with metadata descriptors.
# - ``annotations.parquet``: one row per analysis with annotation feature columns.
# - ``images.parquet``: image references keyed by analysis id.
# - ``texts.parquet``: text fields keyed by analysis id.

for table_file in sorted(parquet_dir.glob("*.parquet")):
    table = pd.read_parquet(table_file)
    print(f"{table_file.name}: {table.shape}")

###############################################################################
# Load the Studyset
# -----------------------------------------------------------------------------
# The constructor recognizes a parquet Studyset directory and returns a
# table-backed Studyset. The nested Study/Analysis object graph is not
# materialized during loading.

studyset = Studyset(parquet_dir)
print(studyset)
print(f"Studyset ID: {studyset.id}")
print(f"Number of studies: {len(studyset.study_ids)}")
print(f"Number of analyses: {len(studyset.ids)}")
print(f"Materialized nested objects? {studyset.is_materialized}")

###############################################################################
# Work with table-backed views
# -----------------------------------------------------------------------------
# The standard Studyset table views are available immediately.

print(studyset.coordinates.head())
print(studyset.metadata.head())

annotation_columns = [
    column
    for column in studyset.annotations_df.columns
    if column not in {"id", "study_id", "contrast_id"}
]
print(f"Annotation feature columns: {len(annotation_columns)}")
print(studyset.annotations_df[["id"] + annotation_columns[:5]].head())

###############################################################################
# Materialize only when needed
# -----------------------------------------------------------------------------
# Accessing ``studyset.studies`` reconstructs nested Study, Analysis, and Point
# objects from the parquet-backed tables. Most Studyset-aware NiMARE workflows
# can use the table-backed views without this step.

first_study = studyset.studies[0]
print(f"First study: {first_study.id}")
print(f"Analyses in first study: {len(first_study.analyses)}")
print(f"Materialized nested objects? {studyset.is_materialized}")
