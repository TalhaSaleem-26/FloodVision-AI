import hashlib
import json

import numpy as np
import pytest
import rasterio

from floodvision.inference.exceptions import GeospatialError, InputValidationError
from floodvision.inference.pipeline import FloodInferencePipeline
from floodvision.inference.pipeline_config import PipelineConfig


@pytest.fixture(scope="module")
def pipeline(production_bundle_dir):
    return FloodInferencePipeline(production_bundle_dir, config=PipelineConfig(device="cpu"))


@pytest.mark.integration
@pytest.mark.requires_model
def test_pipeline_publishes_consistent_rasters_vectors_and_manifest(
    pipeline, raster_factory, tmp_path
):
    image = np.full((2, 32, 48), -20, dtype=np.float32)
    image[:, :2, :3] = np.nan
    source = raster_factory(image)
    output = tmp_path / "analysis"

    result = pipeline.predict(source, output)

    assert result.status == "completed"
    assert result.statistics["valid_pixels"] == 32 * 48 - 6
    assert result.geometry_fidelity["original"]["passed"]
    with rasterio.open(result.get_artifact_path("water_mask")) as dataset:
        mask = dataset.read(1)
    with rasterio.open(result.get_artifact_path("water_probability")) as dataset:
        probabilities = dataset.read(1)
    valid = np.isfinite(image).all(axis=0)
    np.testing.assert_array_equal(mask == -1, ~valid)
    np.testing.assert_array_equal(np.isnan(probabilities), ~valid)
    np.testing.assert_array_equal(mask[valid], probabilities[valid] >= result.threshold)
    assert result.statistics["water_pixels"] == np.count_nonzero(mask == 1)
    assert result.area_analytics["invalid_area_m2"] > 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_id"] == result.analysis_id
    assert manifest["files"] == {
        name: artifact.to_dict() for name, artifact in result.artifacts.items()
        if name != "manifest"
    }
    for record in manifest["files"].values():
        payload = (output / record["relative_path"]).read_bytes()
        assert record["size_bytes"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    geojson_path = result.get_artifact_path("water_regions_original")
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == result.vectorization["region_count"]


@pytest.mark.integration
@pytest.mark.requires_model
def test_pipeline_discards_staged_outputs_when_raster_validation_fails(
    pipeline, raster_factory, tmp_path
):
    source = raster_factory(np.zeros((1, 4, 4), dtype=np.float32))
    output = tmp_path / "analysis"

    with pytest.raises(InputValidationError, match="Expected 2 channels"):
        pipeline.predict(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".floodvision_*"))


@pytest.mark.integration
@pytest.mark.requires_model
def test_pipeline_all_invalid_disk_run_reports_missing_coverage(
    production_bundle_dir, raster_factory, tmp_path
):
    source = raster_factory(np.full((2, 8, 12), np.nan, dtype=np.float32))
    work_dir = tmp_path / "scratch"
    pipeline = FloodInferencePipeline(
        production_bundle_dir,
        config=PipelineConfig(device="cpu", storage="disk", work_dir=work_dir),
    )

    result = pipeline.predict(
        source, tmp_path / "analysis", export_rasters=False, export_vectors=False
    )

    assert result.statistics["valid_pixels"] == 0
    assert result.water_coverage is None
    assert result.tiling["processed_tiles"] == 0
    assert result.tiling["skipped_tiles"] == 1
    assert result.area_analytics["invalid_area_m2"] == result.area_analytics["total_area_m2"]
    assert set(result.artifacts) == {"analysis_report", "manifest"}
    assert "Input raster contains no valid pixels." in result.warnings
    assert not list(work_dir.iterdir())


@pytest.mark.integration
@pytest.mark.requires_model
def test_pipeline_rolls_back_written_rasters_after_later_export_error(
    pipeline, raster_factory, tmp_path, monkeypatch
):
    source = raster_factory(np.full((2, 8, 12), -18, dtype=np.float32))
    output = tmp_path / "analysis"

    def fail_vector_export(result, output_path, source_file):
        assert output_path.with_name("water_mask.tif").is_file()
        assert output_path.with_name("water_probability.tif").is_file()
        assert not output.exists()
        raise GeospatialError("Injected vector export failure")

    monkeypatch.setattr(pipeline.vector_exporter, "export", fail_vector_export)
    with pytest.raises(GeospatialError, match="Injected vector export failure"):
        pipeline.predict(source, output, calculate_area=False)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]
