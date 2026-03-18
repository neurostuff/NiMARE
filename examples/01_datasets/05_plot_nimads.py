"""

.. _nimads_object:

==========================
The NiMARE Studyset object
==========================

This example demonstrates the key functionality of the NeuroImaging Meta-Analysis Data Structure
(NIMADS) with NiMARE, including working with Studysets, annotations, coordinates, metadata,
and native Studyset-backed meta-analysis.
"""

from pprint import pprint
from requests import request

from nimare.meta.cbma import ALE
from nimare.nimads import Studyset


###############################################################################
# Download Data from NeuroStore
# -----------------------------------------------------------------------------


def download_file(url):
    """Download a file from NeuroStore."""
    response = request("GET", url)
    return response.json()


# Download a studyset and its annotation
nimads_studyset = download_file("https://neurostore.org/api/studysets/Cv2LLUqG76W9?nested=true")
nimads_annotation = download_file("https://neurostore.org/api/annotations/76PyNqoTNEsE")


###############################################################################
# Create and Explore Studyset
# -----------------------------------------------------------------------------
# Load the data into a NiMADS Studyset object and explore its contents

studyset = Studyset(nimads_studyset, annotations=nimads_annotation)

# Display basic information about the studyset
print("\nStudyset Information:")
print("-" * 50)
print(f"ID: {studyset.id}")
print(f"Name: {studyset.name}")
print(f"Number of studies: {len(studyset.studies)}")
print(f"Number of annotations: {len(studyset.annotations)}")


###############################################################################
# Explore Studies and Analyses
# -----------------------------------------------------------------------------
# Look at the first study and its analyses in detail

first_study = studyset.studies[0]
print("\nFirst Study Details:")
print("-" * 50)
print(f"Study ID: {first_study.id}")
print(f"Title: {first_study.name}")
print(f"Authors: {first_study.authors}")
print(f"Publication: {first_study.publication}")
print(f"Number of analyses: {len(first_study.analyses)}")

# Show details of the first analysis
first_analysis = first_study.analyses[0]
print("\nFirst Analysis Details:")
print("-" * 50)
print(f"Analysis ID: {first_analysis.id}")
print(f"Analysis Name: {first_analysis.name}")
print(f"Number of coordinates: {len(first_analysis.points)}")
print(f"Number of conditions: {len(first_analysis.conditions)}")


###############################################################################
# Working with Coordinates
# -----------------------------------------------------------------------------
# Demonstrate coordinate-based queries

# Example coordinate in MNI space
example_coord = [-42, -58, -15]  # MNI coordinates
print("\nCoordinate Search Results:")
print("-" * 50)
print(f"Searching near coordinate: {example_coord}")

# Find analyses with coordinates within 10mm
nearby_analyses = studyset.get_analyses_by_coordinates(example_coord, r=10)
print(f"\nFound {len(nearby_analyses)} analyses within 10mm")

# Find 5 closest analyses
closest_analyses = studyset.get_analyses_by_coordinates(example_coord, n=5)
print(f"\nClosest 5 analyses: {closest_analyses}")


###############################################################################
# Working with Annotations
# -----------------------------------------------------------------------------
# Demonstrate how to work with study annotations

print("\nAnnotation Information:")
print("-" * 50)
for annotation in studyset.annotations:
    print(f"\nAnnotation ID: {annotation.id}")
    print(f"Annotation Name: {annotation.name}")
    print(f"Number of notes: {len(annotation.notes)}")


###############################################################################
# Query Metadata
# -----------------------------------------------------------------------------
# Show how to query analyses based on metadata

# Get all analyses that have a specific metadata field
metadata_results = studyset.get_analyses_by_metadata("contrast_type")
print("\nAnalyses with contrast_type metadata:")
print("-" * 50)
pprint(metadata_results)


###############################################################################
# Run a meta-analysis directly on the Studyset
# -----------------------------------------------------------------------------
# Studysets are now accepted directly by NiMARE estimators and workflows.

results = ALE(null_method="approximate").fit(studyset)
print("\nMeta-analysis output maps:")
print("-" * 50)
print(sorted(results.maps))

###############################################################################
# Interoperate with the legacy Dataset class when needed
# -----------------------------------------------------------------------------
# Some older utilities still operate on :class:`~nimare.dataset.Dataset` tables.
# When you need that older interface, you can still convert explicitly.

nimare_dset = studyset.to_dataset()
print("\nLegacy Dataset coordinates preview:")
print("-" * 50)
print(nimare_dset.coordinates.head())
