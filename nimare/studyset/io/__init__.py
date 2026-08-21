"""Boundary adapters: wire formats in and out.

Deliberately not methods on the store. Reading and writing a serialization is an
adapter's job; the store's job is to hold the data.
"""

from nimare.studyset.io.nimads import (
    from_nimads,
    to_nimads_dict,
    write_nimads,
)
from nimare.studyset.io.parquet import (
    convert_neurostore_json_to_parquet,
    from_parquet,
    write_parquet,
)

__all__ = [
    "convert_neurostore_json_to_parquet",
    "from_nimads",
    "from_parquet",
    "to_nimads_dict",
    "write_nimads",
    "write_parquet",
]
