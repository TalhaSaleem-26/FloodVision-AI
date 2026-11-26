import json
from dataclasses import replace

import pytest
from shapely.geometry import Polygon, box, shape

from floodvision.inference.exceptions import GeospatialError
from floodvision.inference.geometry_fidelity import GeometryFidelityValidator
from floodvision.inference.geometry_processor import GeometryProcessor
from floodvision.inference.vector_exporter import VectorExporter
from floodvision.inference.vectorizer import VectorizationResult, WaterRegion


@pytest.fixture
def polygon_result():
    geometry = Polygon(
        [(73, 33), (73, 33.0002), (73, 33.0004), (73.0004, 33.0004),
         (73.0004, 33), (73, 33)]
    )
    area = GeometryProcessor().calculate_area(geometry)
    return VectorizationResult([WaterRegion(4, geometry, area)], "EPSG:4326", 4)


def test_simplification_keeps_region_identity_area_budget_and_validity(polygon_result):
    processor = GeometryProcessor(tolerance_m=1, max_area_change_percentage=1)
    processed = processor.process(polygon_result)
    (region,) = processed.regions

    assert region.region_id == 4
    assert region.geometry.is_valid
    assert region.simplified
    assert region.processed_vertices < region.original_vertices
    assert abs(region.area_change_percentage) <= 1
    assert processed.original_area_m2 == polygon_result.total_area_m2
    assert processed.to_summary()["region_count"] == 1


def test_geometry_processing_rejects_inconsistent_original_area(polygon_result):
    bad_region = replace(polygon_result.regions[0], area_m2=1)
    with pytest.raises(GeospatialError, match="repair exceeded area budget"):
        GeometryProcessor().process(replace(polygon_result, regions=[bad_region]))


def test_vector_export_preserves_canonical_area_and_orients_polygon(polygon_result, tmp_path):
    processed = GeometryProcessor(tolerance_m=1).process(polygon_result)
    path = tmp_path / "water.geojson"
    exporter = VectorExporter(model_version="test-model", threshold=0.65)
    assert exporter.export(processed, path, "input/scene.tif") == path

    data = json.loads(path.read_text(encoding="utf-8"))
    (feature,) = data["features"]
    assert feature["id"] == 4
    assert shape(feature["geometry"]).exterior.is_ccw
    assert feature["properties"]["area_m2"] == polygon_result.total_area_m2
    assert feature["properties"]["display_area_m2"] == processed.processed_area_m2
    assert data["metadata"]["source_file"] == "scene.tif"
    assert data["metadata"]["threshold"] == 0.65
    assert data["metadata"]["geometry_variant"] == "processed"
    assert data["bbox"] == list(shape(feature["geometry"]).bounds)
    original = path.read_bytes()
    with pytest.raises(GeospatialError, match="Output already exists"):
        exporter.export(polygon_result, path, "different.tif")
    assert path.read_bytes() == original


def test_vector_export_rejects_duplicate_region_ids(polygon_result, tmp_path):
    duplicate = replace(polygon_result, regions=polygon_result.regions * 2)
    path = tmp_path / "water.geojson"
    with pytest.raises(GeospatialError, match="Duplicate region identifier"):
        VectorExporter().export(duplicate, path, "scene.tif")
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_fidelity_measures_overlap_in_projected_area():
    left = box(73, 33, 73.0002, 33.0002)
    disjoint = box(73.0003, 33, 73.0004, 33.0002)
    overlapping = box(73.0001, 33, 73.0003, 33.0002)
    validator = GeometryFidelityValidator()

    assert validator.calculate_overlap([left, disjoint]) == pytest.approx(0, abs=1e-7)
    assert validator.calculate_overlap([left, overlapping]) > 100
