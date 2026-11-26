import hashlib
import json

import pytest
import torch

from floodvision.inference.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    DeviceUnavailableError,
    ModelCompatibilityError,
)
from floodvision.inference.model_loader import ModelLoader


def test_missing_manifest_reports_missing_artifact(tmp_path):
    with pytest.raises(ArtifactNotFoundError, match="manifest.json is missing"):
        ModelLoader(tmp_path).verify_integrity()


def test_manifest_detects_same_size_artifact_change(tmp_path):
    payload = b"fixture"
    files = {}
    for filename in ModelLoader.REQUIRED_FILES:
        (tmp_path / filename).write_bytes(payload)
        files[filename] = {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {"bundle_version": "1.0.0", "hash_algorithm": "SHA-256", "files": files}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loader = ModelLoader(tmp_path)

    result = loader.verify_integrity()
    assert result["status"] == "verified"
    assert set(result["verified_files"]) == set(files)

    (tmp_path / ModelLoader.MODEL_FILENAME).write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        loader.verify_integrity()


@pytest.mark.requires_model
def test_production_bundle_loads_on_cpu_in_eval_mode(production_bundle_dir):
    loader = ModelLoader(production_bundle_dir)
    model = loader.load(device="cpu")

    assert loader.is_loaded
    assert not model.training
    assert next(model.parameters()).device == torch.device("cpu")
    assert loader.get_model_info()["model_version"] == "1.0.0"
    assert loader.verify_integrity()["total_files"] == 4
    assert loader.load(device="cpu") is model


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format_version": 0}, "Unsupported artifact format"),
        ({"model_version": "unknown"}, "Unsupported model version"),
        ({"model_config": {}}, "Incompatible model configuration"),
        ({"inference_config": {}}, "Incompatible inference configuration"),
        ({"model_state_dict": []}, "Invalid model state dictionary"),
    ],
)
def test_loader_rejects_incompatible_artifact_metadata(tmp_path, overrides, message):
    loader = ModelLoader(tmp_path)
    artifact = {
        "format_version": loader.SUPPORTED_FORMAT_VERSION,
        "model_version": loader.SUPPORTED_MODEL_VERSION,
        "model_config": loader.EXPECTED_MODEL_CONFIG,
        "inference_config": loader.EXPECTED_INFERENCE_CONFIG,
        "model_state_dict": {},
        "source_experiment": "synthetic",
        "checkpoint_epoch": 1,
        "validation_iou": 0.5,
    }
    with pytest.raises(ModelCompatibilityError, match=message):
        loader.validate_artifact(artifact | overrides)


def test_loader_auto_falls_back_to_cpu_and_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ModelLoader.resolve_device("auto") == torch.device("cpu")
    with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
        ModelLoader.resolve_device("cuda")
