
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ArtifactInfo:

    filename: str
    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str

    def to_dict(self):

        return asdict(self)


@dataclass
class PipelineResult:

    analysis_id: str
    source_file: str
    model_version: str
    device: str
    threshold: float

    output_dir: Path

    geospatial_metadata: dict[str, Any]

    statistics: dict[str, Any]

    tiling: dict[str, Any]

    timing: dict[str, Any]

    area_analytics: dict[str, Any] | None = None

    vectorization: dict[str, Any] | None = None

    geometry_processing: dict[str, Any] | None = None

    geometry_fidelity: dict[str, Any] | None = None

    artifacts: dict[str, ArtifactInfo] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    status: str = "completed"

    task: str = "binary_water_segmentation"

    deployment_status: str = "research_prototype"

    def to_dict(self):

        return {
            "project": "FloodVision AI",

            "analysis_id": self.analysis_id,

            "status": self.status,

            "task": self.task,

            "deployment_status": self.deployment_status,

            "source_file": self.source_file,

            "model": {
                "version": self.model_version,
                "device": self.device,
                "threshold": self.threshold
            },

            "geospatial_metadata":
                self.geospatial_metadata,

            "statistics":
                self.statistics,

            "tiling":
                self.tiling,

            "area_analytics":
                self.area_analytics,

            "vectorization":
                self.vectorization,

            "geometry_processing":
                self.geometry_processing,

            "geometry_fidelity":
                self.geometry_fidelity,

            "timing":
                self.timing,

            "artifacts": {
                name: artifact.to_dict()
                for name, artifact
                in self.artifacts.items()
            },

            "warnings": list(self.warnings),

            "limitations": [
                "Predicted water is not confirmed flood extent.",
                "Model predictions may contain false positives.",
                "Results are intended for research use."
            ]
        }

    def to_json(self, indent=4):

        return json.dumps(
            self.to_dict(),
            indent=indent,
            allow_nan=False
        )

    def get_artifact_path(self, name):

        if name not in self.artifacts:

            raise KeyError(
                f"Artifact not found: {name}"
            )

        root = self.output_dir.resolve()

        path = (
            root /
            self.artifacts[name].relative_path
        ).resolve()

        if not path.is_relative_to(root):

            raise ValueError(
                "Artifact path exceeds output directory."
            )

        return path

    @property
    def water_coverage(self):

        return self.statistics.get(
            "water_coverage"
        )

    @property
    def water_area_km2(self):

        if self.area_analytics is None:
            return None

        return self.area_analytics.get(
            "water_area_km2"
        )

    @property
    def processing_time_ms(self):

        return self.timing.get(
            "processing_ms"
        )
