
import hashlib
import json
from pathlib import Path
from threading import RLock

import torch
import yaml

from floodvision.models.factory import build_model

from .exceptions import (
    ArtifactNotFoundError,
    ArtifactIntegrityError,
    ModelCompatibilityError,
    ModelLoadError,
    DeviceUnavailableError,
)


class ModelLoader:
    """Load and manage the FloodVision production model."""

    MODEL_FILENAME = "floodvision_mobilenetv3_v1.pt"

    REQUIRED_FILES = (
        MODEL_FILENAME,
        "preprocessing.yaml",
        "metadata.json",
        "MODEL_CARD.md",
    )

    SUPPORTED_BUNDLE_VERSION = "1.0.0"
    SUPPORTED_MODEL_VERSION = "1.0.0"
    SUPPORTED_FORMAT_VERSION = 1
    EXPECTED_PARAMETERS = 1034057

    EXPECTED_MODEL_CONFIG = {
        "name": "mobilenet_unet",
        "in_channels": 2,
        "out_channels": 1,
        "pretrained": False,
    }

    EXPECTED_INFERENCE_CONFIG = {
        "input_channels": 2,
        "input_bands": ["VV", "VH"],
        "tile_height": 512,
        "tile_width": 512,
        "threshold": 0.5,
        "output_type": "logits",
        "task": "binary_water_segmentation",
        "ignore_index": -1,
    }

    def __init__(self, bundle_dir):

        self.bundle_dir = Path(bundle_dir).resolve()

        if not self.bundle_dir.is_dir():
            raise ArtifactNotFoundError(
                f"Bundle directory not found: {self.bundle_dir}"
            )

        self.manifest_path = (
            self.bundle_dir / "manifest.json"
        )

        self.model_path = (
            self.bundle_dir / self.MODEL_FILENAME
        )

        self.manifest = None
        self.model = None
        self.device = None
        self.metadata = None
        self.preprocessing_config = None
        self.inference_config = None

        # Prevent concurrent model initialization.
        self._lock = RLock()

    # =====================================================
    # 1. SHA-256 CHECKSUM
    # =====================================================

    @staticmethod
    def calculate_sha256(file_path):

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    # =====================================================
    # 2. ARTIFACT INTEGRITY VERIFICATION
    # =====================================================

    def verify_integrity(self):

        if not self.manifest_path.is_file():
            raise ArtifactNotFoundError(
                "manifest.json is missing."
            )

        try:
            with open(
                self.manifest_path,
                "r",
                encoding="utf-8"
            ) as file:

                manifest = json.load(file)

        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "Unable to read manifest."
            ) from exc

        if not isinstance(manifest, dict):
            raise ArtifactIntegrityError(
                "Invalid manifest structure."
            )

        if (
            manifest.get("bundle_version")
            != self.SUPPORTED_BUNDLE_VERSION
        ):
            raise ModelCompatibilityError(
                "Unsupported bundle version."
            )

        if manifest.get("hash_algorithm") != "SHA-256":
            raise ArtifactIntegrityError(
                "Unsupported hash algorithm."
            )

        files = manifest.get("files")

        if not isinstance(files, dict):
            raise ArtifactIntegrityError(
                "Invalid manifest file entries."
            )

        verified_files = []

        for filename in self.REQUIRED_FILES:

            if filename not in files:
                raise ArtifactIntegrityError(
                    f"Manifest entry missing: {filename}"
                )

            expected = files[filename]

            if not isinstance(expected, dict):
                raise ArtifactIntegrityError(
                    f"Invalid manifest entry: {filename}"
                )

            path = self.bundle_dir / filename

            if not path.is_file():
                raise ArtifactNotFoundError(
                    f"Missing artifact: {filename}"
                )

            try:
                actual_size = path.stat().st_size

                if actual_size != expected.get("size_bytes"):
                    raise ArtifactIntegrityError(
                        f"Size mismatch: {filename}"
                    )

                actual_hash = self.calculate_sha256(path)

            except OSError as exc:
                raise ArtifactIntegrityError(
                    f"Unable to read artifact: {filename}"
                ) from exc

            if actual_hash != expected.get("sha256"):
                raise ArtifactIntegrityError(
                    f"SHA-256 mismatch: {filename}"
                )

            verified_files.append(filename)

        self.manifest = manifest

        return {
            "status": "verified",
            "bundle_version": manifest["bundle_version"],
            "verified_files": verified_files,
            "total_files": len(verified_files),
        }

    # =====================================================
    # 3. DEVICE MANAGEMENT
    # =====================================================

    @staticmethod
    def resolve_device(device="auto"):

        if device == "auto":
            return torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )

        try:
            resolved = torch.device(device)

        except (RuntimeError, ValueError, TypeError) as exc:
            raise DeviceUnavailableError(
                f"Invalid device: {device}"
            ) from exc

        if resolved.type not in ("cpu", "cuda"):
            raise DeviceUnavailableError(
                f"Unsupported device: {resolved}"
            )

        if resolved.type == "cpu":
            return torch.device("cpu")

        if resolved.type == "cuda":

            if not torch.cuda.is_available():
                raise DeviceUnavailableError(
                    "CUDA is not available."
                )

            if (
                resolved.index is not None
                and resolved.index >= torch.cuda.device_count()
            ):
                raise DeviceUnavailableError(
                    f"GPU not found: {resolved}"
                )

            if resolved.index is None:
                resolved = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                )

        return resolved

    # =====================================================
    # 4. MODEL COMPATIBILITY VALIDATION
    # =====================================================

    def validate_artifact(self, artifact):

        if not isinstance(artifact, dict):
            raise ModelCompatibilityError(
                "Artifact must be a dictionary."
            )

        required_keys = {
            "format_version",
            "model_version",
            "model_config",
            "model_state_dict",
            "inference_config",
            "source_experiment",
            "checkpoint_epoch",
            "validation_iou",
        }

        missing = required_keys - artifact.keys()

        if missing:
            raise ModelCompatibilityError(
                f"Missing artifact keys: {sorted(missing)}"
            )

        if (
            artifact["format_version"]
            != self.SUPPORTED_FORMAT_VERSION
        ):
            raise ModelCompatibilityError(
                "Unsupported artifact format."
            )

        if (
            artifact["model_version"]
            != self.SUPPORTED_MODEL_VERSION
        ):
            raise ModelCompatibilityError(
                "Unsupported model version."
            )

        if (
            artifact["model_config"]
            != self.EXPECTED_MODEL_CONFIG
        ):
            raise ModelCompatibilityError(
                "Incompatible model configuration."
            )

        if (
            artifact["inference_config"]
            != self.EXPECTED_INFERENCE_CONFIG
        ):
            raise ModelCompatibilityError(
                "Incompatible inference configuration."
            )

        if not isinstance(
            artifact["model_state_dict"], dict
        ):
            raise ModelCompatibilityError(
                "Invalid model state dictionary."
            )

        return True

    # =====================================================
    # 5. LOAD BUNDLE CONFIGURATION
    # =====================================================

    def _load_bundle_configuration(self):

        try:

            with open(
                self.bundle_dir / "metadata.json",
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(file)

            with open(
                self.bundle_dir / "preprocessing.yaml",
                "r",
                encoding="utf-8"
            ) as file:

                preprocessing_config = yaml.safe_load(file)

        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ModelLoadError(
                "Unable to load bundle configuration."
            ) from exc

        if not isinstance(metadata, dict):
            raise ModelCompatibilityError(
                "Invalid metadata."
            )

        if not isinstance(
            preprocessing_config, dict
        ):
            raise ModelCompatibilityError(
                "Invalid preprocessing configuration."
            )

        if (
            metadata.get("model_version")
            != self.SUPPORTED_MODEL_VERSION
        ):
            raise ModelCompatibilityError(
                "Metadata version mismatch."
            )

        if not isinstance(
            metadata.get("model_name"), str
        ):
            raise ModelCompatibilityError(
                "Model name missing from metadata."
            )

        if not isinstance(
            preprocessing_config.get("preprocessing"),
            dict
        ):
            raise ModelCompatibilityError(
                "Missing preprocessing configuration."
            )

        return metadata, preprocessing_config

    # =====================================================
    # 6. MODEL LOADING AND CACHING
    # =====================================================

    def load(self, device="auto"):

        resolved_device = self.resolve_device(device)

        with self._lock:

            # Reuse the cached model.
            if (
                self.model is not None
                and self.device == resolved_device
            ):
                return self.model

            # Verify artifact integrity before loading.
            self.verify_integrity()

            try:

                artifact = torch.load(
                    self.model_path,
                    map_location="cpu",
                    weights_only=True
                )

            except Exception as exc:
                raise ModelLoadError(
                    "Unable to deserialize model artifact."
                ) from exc

            # Validate checkpoint compatibility.
            self.validate_artifact(artifact)

            # Load metadata and preprocessing configuration.
            metadata, preprocessing_config = (
                self._load_bundle_configuration()
            )

            try:

                model_config = (
                    artifact["model_config"].copy()
                )

                # Prevent pretrained weight downloads.
                model_config["pretrained"] = False

                # Reconstruct model architecture.
                model = build_model(model_config)

                # Restore trained weights.
                model.load_state_dict(
                    artifact["model_state_dict"],
                    strict=True
                )

                parameters = sum(
                    p.numel()
                    for p in model.parameters()
                )

                if parameters != self.EXPECTED_PARAMETERS:
                    raise ModelCompatibilityError(
                        "Unexpected model parameter count."
                    )

                # Freeze all model parameters.
                model.requires_grad_(False)

                # Move model to selected device.
                model = model.to(resolved_device)

                # Enable inference behavior.
                model.eval()

            except ModelCompatibilityError:
                raise

            except Exception as exc:
                raise ModelLoadError(
                    "Model initialization failed."
                ) from exc

            # Cache only after successful initialization.
            self.model = model
            self.device = resolved_device
            self.metadata = metadata

            self.preprocessing_config = (
                preprocessing_config
            )

            self.inference_config = (
                artifact["inference_config"].copy()
            )

            return self.model

    # =====================================================
    # 7. MODEL INFORMATION
    # =====================================================

    def get_model_info(self):

        with self._lock:

            if self.model is None:
                raise ModelLoadError(
                    "Model has not been loaded."
                )

            return {
                "model_name": self.metadata["model_name"],
                "model_version": self.metadata["model_version"],
                "device": str(self.device),

                "parameters": sum(
                    p.numel()
                    for p in self.model.parameters()
                ),

                "trainable_parameters": sum(
                    p.numel()
                    for p in self.model.parameters()
                    if p.requires_grad
                ),

                "training_mode": self.model.training,

                "threshold": (
                    self.inference_config["threshold"]
                ),

                "input_bands": list(
                    self.inference_config["input_bands"]
                ),

                "tile_height": (
                    self.inference_config["tile_height"]
                ),

                "tile_width": (
                    self.inference_config["tile_width"]
                ),

                "status": "ready",
            }

    # =====================================================
    # 8. MODEL LOADED STATUS
    # =====================================================

    @property
    def is_loaded(self):

        with self._lock:
            return self.model is not None
