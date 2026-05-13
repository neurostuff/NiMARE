"""
.. _ma_feature_reduction:

============================================
Masked activation feature reduction workflow
============================================

This example documents the public API that will support masked activation
feature reduction workflows.

The implementation is still a scaffold, so this example intentionally avoids
calling any constructor or reducer that would raise ``NotImplementedError``.
It remains executable so Sphinx-Gallery can build the page once the public
API lands.
"""

from nimare.ml import MAFeatureDataset, MAFeatureExtractor, make_map_reducer

###############################################################################
# The scaffold is intentionally lightweight and import-only for now.
# -----------------------------------------------------------------------------
# These symbols are exposed so downstream examples and docs can import the
# future masked activation feature reduction workflow from the public module.
public_api = (
    MAFeatureDataset.__name__,
    MAFeatureExtractor.__name__,
    make_map_reducer.__name__,
)

print("Masked activation feature reduction public API scaffold:")
print("\n".join(public_api))
