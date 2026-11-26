
import hashlib
import json
import os
import shutil
import tempfile

from pathlib import Path

from .exceptions import InferenceError
from .pipeline_result import ArtifactInfo


class ArtifactManager:

    MEDIA_TYPES = {
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".geojson": "application/geo+json",
        ".json": "application/json",
        ".csv": "text/csv"
    }

    def __init__(self, output_dir):

        self.output_dir = Path(
            output_dir
        ).expanduser().resolve()

        self.staging_dir = None
        self.artifacts = {}
        self.committed = False

    def __enter__(self):

        if self.output_dir.exists():
            raise InferenceError(
                f"Output directory already exists: "
                f"{self.output_dir}"
            )

        self.output_dir.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.staging_dir = Path(
            tempfile.mkdtemp(
                prefix=".floodvision_",
                dir=self.output_dir.parent
            )
        )

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if (
            self.staging_dir is not None
            and self.staging_dir.exists()
        ):
            shutil.rmtree(self.staging_dir)

        return False

    @staticmethod
    def calculate_sha256(path):

        digest = hashlib.sha256()

        with open(path, "rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                digest.update(chunk)

        return digest.hexdigest()

    def get_path(self, filename):

        if self.staging_dir is None or self.committed:
            raise InferenceError(
                "Artifact manager is not active."
            )

        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in (".", "..")
            or "/" in filename
            or "\\" in filename
        ):
            raise InferenceError(
                "Invalid artifact filename."
            )

        return self.staging_dir / filename

    def register(self, name, filename):

        path = self.get_path(filename)

        if name in self.artifacts:
            raise InferenceError(
                f"Duplicate artifact name: {name}"
            )

        if any(
            item.relative_path == filename
            for item in self.artifacts.values()
        ):
            raise InferenceError(
                "Artifact file already registered."
            )

        if path.is_symlink() or not path.is_file():
            raise InferenceError(
                f"Artifact not found: {filename}"
            )

        if path.stat().st_size <= 0:
            raise InferenceError(
                f"Empty artifact: {filename}"
            )

        artifact = ArtifactInfo(
            filename=filename,
            relative_path=filename,
            media_type=self.MEDIA_TYPES.get(
                path.suffix.lower(),
                "application/octet-stream"
            ),
            size_bytes=path.stat().st_size,
            sha256=self.calculate_sha256(path)
        )

        self.artifacts[name] = artifact

        return artifact

    def write_json(self, filename, data):

        path = self.get_path(filename)

        with open(
            path,
            "x",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

            file.flush()
            os.fsync(file.fileno())

        return path

    def verify(self):

        for name, artifact in self.artifacts.items():

            path = self.get_path(
                artifact.relative_path
            )

            if not path.is_file():
                raise InferenceError(
                    f"Missing artifact: {name}"
                )

            if path.stat().st_size != artifact.size_bytes:
                raise InferenceError(
                    f"Artifact size mismatch: {name}"
                )

            if self.calculate_sha256(path) != artifact.sha256:
                raise InferenceError(
                    f"Artifact hash mismatch: {name}"
                )

        return True

    def write_manifest(self, analysis_id):

        manifest = {
            "manifest_version": "1.0.0",
            "analysis_id": analysis_id,
            "hash_algorithm": "SHA-256",
            "files": {
                name: artifact.to_dict()
                for name, artifact in self.artifacts.items()
            }
        }

        return self.write_json(
            "manifest.json",
            manifest
        )

    def commit(self):

        if self.committed:
            raise InferenceError(
                "Artifacts already committed."
            )

        self.verify()

        if self.output_dir.exists():
            raise InferenceError(
                "Output directory already exists."
            )

        os.rename(
            self.staging_dir,
            self.output_dir
        )

        self.committed = True

        return self.output_dir
