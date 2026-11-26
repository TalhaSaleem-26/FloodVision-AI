import json
from dataclasses import replace

import pytest

from floodvision.inference.exceptions import InputValidationError
from floodvision.inference.pipeline_config import PipelineConfig
from floodvision.inference.pipeline_result import ArtifactInfo, PipelineResult


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"device": None}, "Invalid device"),
        ({"memory_budget_mib": True}, "must be an integer"),
        ({"max_tiles": 0}, "must be positive"),
        ({"tile_size": 256}, "requires 512"),
        ({"overlap": 512}, "Invalid tile overlap"),
        ({"min_blend_weight": 0}, "Invalid minimum blending weight"),
        ({"storage": "cloud"}, "Unsupported storage"),
        ({"max_pixels": 100}, "Vector pixel limit exceeds raster limit"),
        ({"connectivity": 6}, "Connectivity must be"),
        ({"simplification_tolerance_m": -1}, "Invalid simplification tolerance"),
        ({"max_area_change_percentage": 101}, "Invalid maximum area change"),
        ({"min_geometry_iou": 0}, "Invalid minimum geometry IoU"),
        ({"max_invalid_encroachment": 1}, "Invalid pixel encroachment"),
        ({"raster_block_size": 31}, "positive multiple of 16"),
    ],
)
def test_pipeline_config_rejects_invalid_options(options, message):
    with pytest.raises(InputValidationError, match=message):
        PipelineConfig(**options)


def test_pipeline_config_resolves_path_and_retains_tiling_options(tmp_path):
    scratch = tmp_path / "work"
    config = PipelineConfig(work_dir=str(scratch), overlap=64, min_blend_weight=0.01)

    assert config.work_dir == scratch.resolve()
    assert json.loads(json.dumps(config.to_dict()))["work_dir"] == str(scratch)
    tiling = config.get_tiling_config()
    assert tiling.stride == 448
    assert tiling.min_blend_weight == 0.01
    assert not scratch.exists()


def test_pipeline_result_serializes_nullable_fields_and_confines_artifact_paths(tmp_path):
    artifact = ArtifactInfo("mask.tif", "mask.tif", "image/tiff", 100, "digest")
    result = PipelineResult(
        analysis_id="id", source_file="scene.tif", model_version="model",
        device="cpu", threshold=0.5, output_dir=tmp_path / "analysis",
        geospatial_metadata={"crs": "EPSG:4326"}, statistics={"water_coverage": None},
        tiling={"total_tiles": 1}, timing={"processing_ms": 2},
        artifacts={"mask": artifact},
    )

    payload = json.loads(result.to_json())
    assert payload["statistics"]["water_coverage"] is None
    assert payload["area_analytics"] is None
    assert payload["artifacts"]["mask"] == artifact.to_dict()
    assert result.processing_time_ms == 2
    assert result.get_artifact_path("mask") == result.output_dir / "mask.tif"
    with pytest.raises(KeyError, match="Artifact not found"):
        result.get_artifact_path("missing")
    escaping = replace(artifact, relative_path="../outside.tif")
    with pytest.raises(ValueError, match="exceeds output directory"):
        replace(result, artifacts={"mask": escaping}).get_artifact_path("mask")
    with pytest.raises(ValueError, match="Out of range float"):
        replace(result, statistics={"water_coverage": float("nan")}).to_json()
