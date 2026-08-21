"""Tests for Studyset-native execution paths."""

import os

import numpy as np
import pandas as pd
import pytest

from nimare import annotate
from nimare.correct import FDRCorrector
from nimare.decode import continuous, discrete
from nimare.diagnostics import FocusFilter
from nimare.generate import create_coordinate_studyset
from nimare.io import convert_dataset_to_nimads_dict, convert_nimads_to_dataset
from nimare.meta.cbma import ALE
from nimare.meta.ibma import Stouffers
from nimare.meta.kernel import MKDAKernel
from nimare.nimads import Studyset
from nimare.reports.base import run_reports
from nimare.studyset import normalize_collection
from nimare.transforms import ImageTransformer
from nimare.utils import get_template, mni2tal
from nimare.workflows import CBMAWorkflow, IBMAWorkflow, PairwiseCBMAWorkflow


def _make_mixed_space_studyset_payload(dataset):
    """Convert alternating analyses to Talairach coordinates in a Studyset payload."""
    payload = convert_dataset_to_nimads_dict(dataset)
    converted_analyses = 0
    analysis_index = 0

    for study in payload["studies"]:
        for analysis in study["analyses"]:
            if not analysis["points"]:
                continue

            if analysis_index % 2 == 0:
                coords = np.asarray(
                    [point["coordinates"] for point in analysis["points"]], dtype=float
                )
                tal_coords = mni2tal(coords)
                for point, tal_coord in zip(analysis["points"], tal_coords):
                    point["space"] = "TAL"
                    point["coordinates"] = tal_coord.tolist()
                converted_analyses += 1

            analysis_index += 1

    assert converted_analyses > 0
    return payload


def test_ale_studyset_parity(testdata_cbma):
    """ALE should accept Studysets and match Dataset outputs."""
    dset = testdata_cbma.slice(testdata_cbma.ids[:5])
    studyset = Studyset.from_dataset(dset)

    res_dset = ALE(null_method="approximate").fit(dset)
    res_studyset = ALE(null_method="approximate").fit(studyset)

    np.testing.assert_allclose(
        res_dset.get_map("stat", return_type="array"), res_studyset.maps["stat"]
    )
    np.testing.assert_allclose(res_dset.get_map("p", return_type="array"), res_studyset.maps["p"])
    np.testing.assert_allclose(res_dset.get_map("z", return_type="array"), res_studyset.maps["z"])


def test_ale_accepts_singular_sample_size_metadata(testdata_cbma):
    """ALE should accept Studysets that expose per-analysis ``sample_size`` metadata."""
    studyset = Studyset.from_dataset(testdata_cbma.slice(testdata_cbma.ids[:5]))

    for study in studyset.studies:
        study.metadata.pop("sample_sizes", None)
        for analysis in study.analyses:
            sample_sizes = analysis.metadata.pop("sample_sizes", None)
            if sample_sizes:
                analysis.metadata["sample_size"] = int(np.mean(sample_sizes))

    result = ALE(null_method="approximate").fit(studyset)

    assert "stat" in result.maps


def test_studyset_metadata_coerces_sample_size_fields():
    """Sample sizes are normalised from whichever level and shape declares them."""
    def one(study_metadata, analysis_metadata):
        return Studyset(
            {
                "id": "s",
                "name": "s",
                "studies": [
                    {
                        "id": "study-1",
                        "metadata": study_metadata,
                        "analyses": [
                            {
                                "id": "analysis-1",
                                "metadata": analysis_metadata,
                                "points": [{"coordinates": [1, 2, 3], "space": "MNI"}],
                            }
                        ],
                    }
                ],
            }
        )

    # A scalar under `sample_sizes` is not usable, so the study's string-valued
    # `sample_size` is what survives.
    studyset = one({"sample_size": "12"}, {"sample_sizes": 5})
    row = studyset.metadata.iloc[0]
    assert row["sample_sizes"] == [12]
    assert "sample_size" not in studyset.metadata.columns

    studyset = one({"sample_size": "12"}, {"sample_sizes": ["6", "7.5"]})
    assert studyset.metadata.iloc[0]["sample_sizes"] == [6, 7.5]


def test_stouffers_studyset_parity(testdata_ibma):
    """IBMA estimators should accept Studysets and match Dataset outputs."""
    dset = testdata_ibma.slice(testdata_ibma.ids[:5])
    studyset = Studyset.from_dataset(dset)

    res_dset = Stouffers().fit(dset)
    res_studyset = Stouffers().fit(studyset)

    np.testing.assert_allclose(res_dset.get_map("z", return_type="array"), res_studyset.maps["z"])
    np.testing.assert_allclose(res_dset.get_map("p", return_type="array"), res_studyset.maps["p"])


def test_cbma_workflow_accepts_studyset(tmp_path_factory, testdata_cbma_full):
    """CBMA workflow should run when passed a Studyset."""
    studyset = Studyset.from_dataset(testdata_cbma_full.slice(testdata_cbma_full.ids[:8]))
    tmpdir = tmp_path_factory.mktemp("test_cbma_workflow_accepts_studyset")

    workflow = CBMAWorkflow(
        estimator="ale",
        corrector="bonferroni",
        diagnostics=[],
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset)
    assert "z" in result.maps


def test_ibma_workflow_accepts_studyset(tmp_path_factory, testdata_ibma):
    """IBMA workflow should run when passed a Studyset."""
    studyset = Studyset.from_dataset(testdata_ibma.slice(testdata_ibma.ids[:5]))
    tmpdir = tmp_path_factory.mktemp("test_ibma_workflow_accepts_studyset")

    workflow = IBMAWorkflow(
        estimator="stouffers",
        corrector="bonferroni",
        diagnostics=[],
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset)
    assert "z" in result.maps


def test_pairwise_cbma_workflow_accepts_studyset(tmp_path_factory, testdata_cbma_full):
    """Pairwise CBMA workflow should run when passed two Studysets."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[5:10])
    studyset1 = Studyset.from_dataset(dset1)
    studyset2 = Studyset.from_dataset(dset2)
    tmpdir = tmp_path_factory.mktemp("test_pairwise_cbma_workflow_accepts_studyset")

    workflow = PairwiseCBMAWorkflow(
        estimator="mkdachi2",
        corrector="bonferroni",
        diagnostics=[],
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset1, studyset2)
    assert "z_desc-uniformity" in result.maps


def test_dataset_studyset_roundtrip_preserves_core_tables(testdata_ibma):
    """Dataset->Studyset->Dataset conversion should preserve images/metadata/labels/texts."""
    dset = testdata_ibma.slice(testdata_ibma.ids[:5])
    dset.metadata = dset.metadata.assign(custom_meta="meta_value")
    dset.annotations = dset.annotations.assign(custom_label=1.23)
    dset.texts = dset.texts.assign(custom_text="hello world")
    studyset = Studyset.from_dataset(dset)
    reloaded_studyset = Studyset(studyset.to_dict())
    roundtrip = convert_nimads_to_dataset(reloaded_studyset)
    if dset.basepath:
        roundtrip.update_path(dset.basepath)

    id_cols = {"id", "study_id", "contrast_id", "space"}
    orig_cols = {
        col for col in dset.images.columns if col not in id_cols and not col.endswith("__relative")
    }
    new_cols = {
        col
        for col in roundtrip.images.columns
        if col not in id_cols and not col.endswith("__relative")
    }

    assert orig_cols.issubset(new_cols)
    for col in orig_cols:
        if dset.images[col].notnull().any():
            assert roundtrip.images[col].notnull().any()

    assert "custom_meta" in roundtrip.metadata.columns
    assert "custom_label" in roundtrip.annotations.columns
    assert "custom_text" in roundtrip.texts.columns


def test_studyset_from_dataset_caches_independent_tables(testdata_ibma):
    """Studyset.from_dataset should snapshot Dataset tables for native execution."""
    dset = testdata_ibma.slice(testdata_ibma.ids[:5])
    image_cols = [
        col
        for col in dset.images.columns
        if col not in {"id", "study_id", "contrast_id", "space"} and not col.endswith("__relative")
    ]
    image_col = next((col for col in image_cols if dset.images[col].notnull().any()), None)
    assert image_col is not None

    studyset = Studyset.from_dataset(dset)
    dset.images.loc[:, image_col] = None

    view = normalize_collection(studyset)
    assert image_col in view.images.columns
    assert view.images[image_col].notnull().any()


def test_image_transformer_accepts_studyset(testdata_ibma):
    """Ensure ImageTransformer accepts Studyset inputs."""
    studyset = Studyset.from_dataset(testdata_ibma)
    transformed = ImageTransformer(target="z").transform(studyset)
    assert "z" in transformed.images.columns


def test_image_transformer_records_generated_images_in_studyset(testdata_ibma):
    """Image transformer should write generated image paths back into Studyset.images."""
    studyset = Studyset.from_dataset(testdata_ibma)

    original_varcope = studyset.images["varcope"].tolist()
    assert not all(isinstance(value, str) for value in original_varcope)

    transformed = ImageTransformer(target=["varcope", "p"]).transform(studyset)

    assert isinstance(transformed, Studyset)
    assert all(isinstance(value, str) for value in transformed.images["varcope"].tolist())
    assert all(isinstance(value, str) for value in transformed.images["p"].tolist())
    assert all(transformed.images["p"].map(os.path.isfile))

    # The transformer should return a copy rather than mutating the original Studyset in-place.
    assert not all(isinstance(value, str) for value in studyset.images["varcope"].tolist())


def test_kernel_transformer_studyset_parity(testdata_cbma):
    """Kernel transformers should accept Studysets and match Dataset outputs."""
    dset = testdata_cbma.slice(testdata_cbma.ids[:5])
    studyset = Studyset.from_dataset(dset)
    kernel = MKDAKernel()

    dset_array = kernel.transform(dset, return_type="array")
    studyset_array = kernel.transform(studyset, return_type="array")

    np.testing.assert_allclose(dset_array, studyset_array)

    dset_summary = kernel.transform(dset, return_type="summary_array")
    studyset_summary = kernel.transform(studyset, return_type="summary_array")

    np.testing.assert_allclose(dset_summary, studyset_summary)


def test_kernel_transformer_dataset_fast_path(monkeypatch, testdata_cbma):
    """Kernel transformers should keep Dataset inputs on the direct fast path."""
    dset = testdata_cbma.slice(testdata_cbma.ids[:5])
    kernel = MKDAKernel()

    def _fail(dataset):
        raise AssertionError("Dataset inputs should not be normalized through a Studyset wrapper")

    monkeypatch.setattr("nimare.meta.kernel.normalize_collection", _fail)

    output = kernel.transform(dset, return_type="summary_array")

    assert output.ndim == 1
    assert output.size > 0


def test_studyset_slice_accepts_analysis_ids(testdata_cbma):
    """Studyset.slice should accept analysis-level IDs."""
    studyset = Studyset.from_dataset(testdata_cbma.slice(testdata_cbma.ids[:5]))
    target_ids = [study.analyses[0].id for study in studyset.studies[:2]]

    sliced = studyset.slice(target_ids)

    assert {analysis.id for study in sliced.studies for analysis in study.analyses} == set(
        target_ids
    )


def test_studyset_filter_annotations_returns_executable_subset(testdata_cbma):
    """Annotation filtering returns a studyset that can be fitted directly."""
    studyset = Studyset.from_dataset(testdata_cbma.slice(testdata_cbma.ids[:5]))
    with_points = [bool(analysis.points) for analysis in studyset.analyses]
    keep_rows = [i for i, has in enumerate(with_points) if has][:3]
    include = np.array(
        [[1.0] if i in keep_rows else [0.0] for i in range(len(studyset.ids))]
    )
    annotated = studyset.with_annotation("curation", ["include"], include)

    filtered = annotated.filter_annotations("include", threshold=0.5)

    assert set(filtered.ids) == set(np.asarray(studyset.ids)[keep_rows])
    assert filtered.masker is not None
    result = ALE(null_method="approximate").fit(filtered)
    assert "stat" in result.maps


def test_studyset_filter_metadata_returns_executable_subset(testdata_cbma):
    """Metadata filtering returns a studyset that can be fitted directly."""
    studyset = Studyset.from_dataset(testdata_cbma.slice(testdata_cbma.ids[:5]))
    # Only analyses with foci: an image-only analysis has nothing for a
    # coordinate-based estimator to fit.
    with_points = [bool(analysis.points) for analysis in studyset.analyses]
    keep_rows = [i for i, has in enumerate(with_points) if has][:2]
    groups = ["keep" if i in keep_rows else "drop" for i in range(len(studyset.ids))]
    labelled = studyset.with_metadata("group", groups)

    filtered = labelled.filter_metadata("group", "==", "keep")

    assert set(filtered.ids) == set(np.asarray(studyset.ids)[keep_rows])
    assert filtered.masker is not None
    result = ALE(null_method="approximate").fit(filtered)
    assert "stat" in result.maps


def test_decoder_accepts_studyset(testdata_laird):
    """Discrete decoders should accept raw Studyset inputs."""
    studyset = Studyset.from_dataset(testdata_laird.slice(testdata_laird.ids[:20]))
    selected_ids = studyset.get_studies_by_mask(studyset.masker.mask_img)
    decoder = discrete.NeurosynthDecoder(feature_group="Neurosynth_TFIDF")

    decoder.fit(studyset)
    decoded_df = decoder.transform(ids=selected_ids[:3])

    assert isinstance(decoded_df, pd.DataFrame)
    assert not decoded_df.empty


def test_correlation_decoder_accepts_studyset(testdata_laird):
    """CorrelationDecoder should run on a studyset converted from a Dataset."""
    dset = testdata_laird.slice(testdata_laird.ids[:5])
    studyset = Studyset.from_dataset(dset)
    features = next(
        (dset.get_labels(ids=id_)[:3] for id_ in dset.ids if dset.get_labels(ids=id_)), []
    )
    assert features

    decoder = continuous.CorrelationDecoder(features=features, n_cores=1)
    decoder.fit(studyset)

    assert set(decoder.results_.maps.keys()) == set(features)


def test_lda_accepts_studyset(testdata_laird):
    """LDA should return a Studyset with tabular annotations attached."""
    studyset = Studyset.from_dataset(testdata_laird.slice(testdata_laird.ids[:20]))
    model = annotate.lda.LDAModel(n_topics=3, max_iter=10, text_column="abstract")

    annotated = model.fit(studyset)

    assert isinstance(annotated, Studyset)
    topic_columns = [col for col in annotated.annotations_df.columns if col.startswith("LDA")]
    assert len(topic_columns) == 3


def test_focus_filter_accepts_studyset(testdata_cbma):
    """Ensure FocusFilter accepts Studysets and returns a filtered Studyset."""
    studyset = Studyset.from_dataset(testdata_cbma.slice(testdata_cbma.ids[:5]))

    filtered = FocusFilter().transform(studyset)

    assert isinstance(filtered, Studyset)
    assert set(filtered.coordinates["id"].unique()).issubset(set(studyset.ids))


def test_reports_accept_studyset_results(tmp_path_factory, testdata_cbma):
    """Reports should run for Studyset-backed estimator results."""
    studyset = Studyset.from_dataset(testdata_cbma.slice(testdata_cbma.ids[:5]))
    result = ALE(null_method="approximate").fit(studyset)
    result = FDRCorrector(method="indep", alpha=0.05).transform(result)

    out_dir = tmp_path_factory.mktemp("test_reports_accept_studyset_results")
    run_reports(result, out_dir)
    assert (out_dir / "report.html").is_file()


def test_studyset_constructor_target_harmonizes_mixed_coordinate_spaces(testdata_cbma_full):
    """Studyset(target=...) should transform mixed TAL/MNI points into one execution space."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:6])
    studyset = Studyset(_make_mixed_space_studyset_payload(dset), target="mni152_2mm")

    expected = dset.coordinates.copy()
    actual = studyset.coordinates.copy()

    assert studyset.space == "mni152_2mm"
    assert studyset.masker is not None
    assert set(actual["space"].unique()) == {"mni152_2mm"}
    assert set(actual["id"].unique()) == set(expected["id"].unique())
    for id_ in expected["id"].unique():
        expected_coords = sorted(
            map(
                tuple,
                np.round(
                    expected.loc[expected["id"] == id_, ["x", "y", "z"]].to_numpy(),
                    decimals=8,
                ),
            )
        )
        actual_coords = sorted(
            map(
                tuple,
                np.round(
                    actual.loc[actual["id"] == id_, ["x", "y", "z"]].to_numpy(),
                    decimals=8,
                ),
            )
        )
        assert actual_coords == expected_coords
    assert all(
        point.space == "mni152_2mm"
        for study in studyset.studies
        for analysis in study.analyses
        for point in analysis.points
    )


def test_studyset_constructor_target_none_preserves_mixed_coordinate_spaces(testdata_cbma_full):
    """Studyset(target=None) should preserve source spaces and not auto-pick a target."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:6])
    studyset = Studyset(_make_mixed_space_studyset_payload(dset), target=None)

    assert studyset.space is None
    assert studyset.masker is None
    assert set(studyset.coordinates["space"].unique()) == {"MNI", "TAL"}


def test_studyset_reports_coordinates_in_the_space_it_is_read_in(testdata_cbma_full):
    """Harmonisation is derived, so the raw spaces remain available."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:6])
    payload = _make_mixed_space_studyset_payload(dset)

    raw = Studyset(payload, target=None)
    assert set(raw.coordinates["space"].unique()) == {"MNI", "TAL"}

    projected = Studyset(payload, target="mni152_2mm")
    assert projected.space == "mni152_2mm"
    assert projected.masker is not None
    assert set(projected.coordinates["space"].unique()) == {"mni152_2mm"}

    # The store was never rewritten, so the same studyset still reports raw
    # coordinates when read without a target.
    assert set(projected.with_context(space=None).coordinates["space"].unique()) == {
        "MNI",
        "TAL",
    }


def test_ale_accepts_fresh_mixed_space_studyset_with_explicit_target(testdata_cbma_full):
    """ALE.fit should run on a freshly constructed mixed-space Studyset with an explicit target."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    studyset = Studyset(_make_mixed_space_studyset_payload(dset), target="mni152_2mm")

    result = ALE(null_method="approximate").fit(studyset)

    assert "stat" in result.maps


def test_cbma_workflow_accepts_fresh_mixed_space_studyset(tmp_path_factory, testdata_cbma_full):
    """CBMAWorkflow.fit should run directly on a freshly constructed mixed-space Studyset."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:6])
    studyset = Studyset(_make_mixed_space_studyset_payload(dset), target="mni152_2mm")
    tmpdir = tmp_path_factory.mktemp("test_cbma_workflow_accepts_fresh_mixed_space_studyset")

    workflow = CBMAWorkflow(
        estimator="ale",
        corrector="bonferroni",
        diagnostics=[],
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset)
    assert "z" in result.maps


def test_pairwise_cbma_workflow_accepts_fresh_mixed_space_studysets(
    tmp_path_factory, testdata_cbma_full
):
    """PairwiseCBMAWorkflow.fit should run directly on fresh mixed-space Studysets."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[5:10])
    studyset1 = Studyset(_make_mixed_space_studyset_payload(dset1), target="mni152_2mm")
    studyset2 = Studyset(_make_mixed_space_studyset_payload(dset2), target="mni152_2mm")
    tmpdir = tmp_path_factory.mktemp(
        "test_pairwise_cbma_workflow_accepts_fresh_mixed_space_studysets"
    )

    workflow = PairwiseCBMAWorkflow(
        estimator="mkdachi2",
        corrector="bonferroni",
        diagnostics=[],
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset1, studyset2)
    assert "z_desc-uniformity" in result.maps


def test_studyset_constructor_preserves_execution_context(example_nimads_studyset):
    """Studyset constructor target/mask arguments should drive direct execution context."""
    mask = get_template("mni152_2mm", mask="brain")
    studyset = Studyset(example_nimads_studyset, target="mni152_2mm", mask=mask)
    normalized = normalize_collection(studyset)

    assert normalized is studyset
    assert studyset.space == "mni152_2mm"
    assert studyset.masker is not None
    assert np.array_equal(studyset.masker.mask_img.affine, mask.affine)


def test_studyset_handles_empty_annotations_and_texts():
    """Empty Studysets should expose Dataset-like empty annotation/text tables."""
    studyset = Studyset({"id": "empty", "name": "", "studies": []})

    assert studyset.annotations_df.empty
    assert list(studyset.annotations_df.columns) == ["id", "study_id", "contrast_id"]

    assert studyset.texts.empty
    assert list(studyset.texts.columns) == ["id", "study_id", "contrast_id"]


def test_studyset_view_slice_preserves_materialized_tables(testdata_ibma):
    """Slicing a Studyset should retain materialized projected tables."""
    studyset = Studyset.from_dataset(testdata_ibma.slice(testdata_ibma.ids[:5]))
    target_ids = set(studyset.ids[:2])

    _ = studyset.images
    _ = studyset.metadata
    _ = studyset.texts
    _ = studyset.annotations_df

    sliced = studyset.slice(sorted(target_ids))

    assert set(sliced.ids) == target_ids
    assert set(sliced.images["id"].unique()).issubset(target_ids)
    assert set(sliced.metadata["id"].unique()).issubset(target_ids)


def test_normalize_collection_passes_through_studyset(testdata_cbma):
    """normalize_collection should pass Studyset inputs through directly."""
    studyset = Studyset.from_dataset(testdata_cbma.slice(testdata_cbma.ids[:5]))
    normalized = normalize_collection(studyset)
    assert normalized is studyset


def test_cbmr_accepts_studyset_smoke():
    """CBMR should accept Studyset inputs."""
    pytest.importorskip("torch")
    from nimare.meta import models
    from nimare.meta.cbmr import CBMREstimator

    _, studyset = create_coordinate_studyset(
        foci=5,
        sample_size=(20, 30),
        n_studies=12,
        seed=13,
    )
    annotations_df = studyset.annotations_df.copy()
    n_rows = annotations_df.shape[0]
    annotations_df["diagnosis"] = [
        "schizophrenia" if i % 2 == 0 else "depression" for i in range(n_rows)
    ]
    annotations_df["drug_status"] = ["Yes" if i % 2 == 0 else "No" for i in range(n_rows)]
    studyset.annotations_df = annotations_df
    cbmr = CBMREstimator(
        group_categories=["diagnosis", "drug_status"],
        spline_spacing=100,
        model=models.PoissonEstimator,
        n_iter=10,
        lr=1,
        tol=1e4,
        device="cpu",
    )
    result = cbmr.fit(studyset)
    assert result.maps
