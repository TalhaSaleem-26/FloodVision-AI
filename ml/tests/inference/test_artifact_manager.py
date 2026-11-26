import hashlib
import json

import pytest

from floodvision.inference.artifact_manager import ArtifactManager
from floodvision.inference.exceptions import InferenceError


def test_manifest_matches_published_artifact(tmp_path):
    output = tmp_path / "analysis"

    with ArtifactManager(output) as manager:
        staged = manager.write_json("result.json", {"water_pixels": 5})
        info = manager.register("result", "result.json")
        manager.write_manifest("analysis-id")
        assert manager.verify()
        assert not output.exists()
        assert manager.commit() == output

    assert not staged.exists()
    result = output / "result.json"
    assert json.loads(result.read_text(encoding="utf-8")) == {"water_pixels": 5}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_id"] == "analysis-id"
    assert manifest["files"]["result"] == info.to_dict()
    assert info.sha256 == hashlib.sha256(result.read_bytes()).hexdigest()


def test_same_size_tampering_prevents_publication_and_cleans_staging(tmp_path):
    output = tmp_path / "analysis"
    with pytest.raises(InferenceError, match="hash mismatch"):
        with ArtifactManager(output) as manager:
            staged = manager.get_path("data.csv")
            staged.write_bytes(b"old")
            manager.register("data", "data.csv")
            staged.write_bytes(b"new")
            manager.commit()

    assert not output.exists()
    assert not staged.parent.exists()


def test_existing_output_directory_is_preserved(tmp_path):
    output = tmp_path / "analysis"
    output.mkdir()
    sentinel = output / "existing.json"
    sentinel.write_bytes(b"{}")

    with pytest.raises(InferenceError, match="already exists"):
        with ArtifactManager(output):
            pytest.fail("Existing output was accepted")

    assert sentinel.read_bytes() == b"{}"
