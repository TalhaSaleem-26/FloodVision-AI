import pytest

from floodvision.inference.pipeline import FloodInferencePipeline
from floodvision.inference.pipeline_config import PipelineConfig


@pytest.mark.regression
@pytest.mark.requires_model
@pytest.mark.requires_data
def test_mekong_reference_prediction_matches_validated_notebook(
    ml_dir, production_bundle_dir, tmp_path
):
    # Reference values from 14_unified_pipeline_validation.ipynb.
    source = ml_dir / "data/raw/Sen1Floods11/S1Hand/Mekong_1111068_S1Hand.tif"
    if not source.is_file():
        pytest.skip("Mekong_1111068 reference raster unavailable.")

    pipeline = FloodInferencePipeline(
        production_bundle_dir, config=PipelineConfig(device="cpu", storage="memory")
    )
    result = pipeline.predict(
        source, tmp_path / "reference", export_rasters=False, export_vectors=False
    )

    assert result.model_version == "1.0.0"
    assert result.statistics["valid_pixels"] == 262144
    assert result.statistics["water_pixels"] == 66106
    assert result.statistics["non_water_pixels"] == 196038
    assert result.statistics["mean_water_probability"] == pytest.approx(0.24564292287400458)
    assert result.water_area_km2 == pytest.approx(6.4259331394886825)
