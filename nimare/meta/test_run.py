"""Ad hoc script for running a small decoding workflow against Neurosynth data."""

from nimare.decode import CorrelationDecoder
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset


def select_features_for_pairwise_decoder(dset, feature_group, frequency_threshold):
    """Return feature columns with at least one and not all studies above threshold."""
    prefix = f"{feature_group.rstrip('_')}__"
    feature_cols = [c for c in dset.annotations.columns if c.startswith(prefix)]
    if not feature_cols:
        raise ValueError(f"No annotation columns found for feature_group '{feature_group}'.")

    coord_ids = set(dset.coordinates["id"].unique())
    annot_ids = set(dset.annotations["id"].unique())
    valid_ids = sorted(coord_ids.intersection(annot_ids))
    if not valid_ids:
        raise ValueError("No studies have both coordinates and annotations.")

    annot = dset.annotations.loc[dset.annotations["id"].isin(valid_ids), feature_cols]
    counts = (annot > frequency_threshold).sum(0)
    n_ids = len(valid_ids)
    valid_features = counts[(counts > 0) & (counts < n_ids)].index.tolist()
    return valid_features


# 1. Fetch Neurosynth data with LDA topics
files = fetch_neurosynth(
    data_dir="./neurosynth_data/",
    version="7",
    source="abstract",
    vocab="LDA50",
    type="weight",
)
neurosynth_db = files[0]

# 2. Convert to NiMARE Dataset
dset = convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db["coordinates"],
    metadata_file=neurosynth_db["metadata"],
    annotations_files=neurosynth_db["features"],
)

# 3. Fit CorrelationDecoder - this generates meta-analytic maps for each topic
feature_group = "LDA50_abstract_weight"
frequency_threshold = 0.01
valid_features = select_features_for_pairwise_decoder(
    dset,
    feature_group=feature_group,
    frequency_threshold=frequency_threshold,
)
print(f"[decoder] Using {len(valid_features)} features after filtering for pairwise fit.")

decoder = CorrelationDecoder(
    feature_group=None,  # Explicit list of features provided
    features=valid_features,
    frequency_threshold=frequency_threshold,
    n_cores=4,  # Parallelize map generation
)
decoder.fit(dset)

# 4. Decode your image
results = decoder.transform("your_unthresholded_map.nii.gz")
print(results.sort_values("r", ascending=False).head(10))
