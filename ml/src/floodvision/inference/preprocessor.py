
from pathlib import Path

import numpy as np
import torch
import yaml

from floodvision.preprocessing.sar import (
    read_sar_image
)

from floodvision.preprocessing.normalization import (
    SARNormalizer
)

from .exceptions import (
    PreprocessingError,
    InputValidationError
)

from .schemas import (
    SARInput,
    PreprocessedInput
)

from .validator import InputValidator


class SARPreprocessor:
    """Training-compatible SAR preprocessing for inference."""

    EXPECTED_BANDS = ("VV", "VH")

    def __init__(
        self,
        config_path,
        require_crs=True,
        allow_unverified_band_order=False
    ):

        self.config_path = Path(
            config_path
        ).resolve()

        if not self.config_path.is_file():
            raise PreprocessingError(
                f"Configuration not found: {self.config_path}"
            )

        self.config = self._load_config()

        self._validate_config()

        try:

            self.normalizer = SARNormalizer(
                self.config_path
            )

        except Exception as exc:
            raise PreprocessingError(
                "Unable to initialize SAR normalizer."
            ) from exc

        self.validator = InputValidator(
            require_crs=require_crs,
            allow_unverified_band_order=(
                allow_unverified_band_order
            )
        )

    # =====================================================
    # 1. LOAD PREPROCESSING CONFIGURATION
    # =====================================================

    def _load_config(self):

        try:

            with open(
                self.config_path,
                "r",
                encoding="utf-8"
            ) as file:

                config = yaml.safe_load(file)

        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise PreprocessingError(
                "Unable to read preprocessing configuration."
            ) from exc

        if not isinstance(config, dict):
            raise PreprocessingError(
                "Invalid preprocessing configuration."
            )

        preprocessing = config.get("preprocessing")

        if not isinstance(preprocessing, dict):
            raise PreprocessingError(
                "Missing preprocessing configuration."
            )

        return preprocessing

    # =====================================================
    # 2. CONFIGURATION VALIDATION
    # =====================================================

    def _validate_config(self):

        if (
            self.config.get("method")
            != "percentile_zscore"
        ):
            raise PreprocessingError(
                "Unsupported preprocessing method."
            )

        if (
            self.config.get("fit_split")
            != "geographical/train"
        ):
            raise PreprocessingError(
                "Expected training-only normalization statistics."
            )

        if self.config.get("ignore_index") != -1:
            raise PreprocessingError(
                "Expected ignore index -1."
            )

        if self.config.get("invalid_fill_value") != 0.0:
            raise PreprocessingError(
                "Expected invalid fill value 0.0."
            )

        epsilon = self.config.get("epsilon")

        if (
            not isinstance(epsilon, (int, float))
            or not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise PreprocessingError(
                "Invalid normalization epsilon."
            )

        channels = self.config.get("channels")

        if not isinstance(channels, dict):
            raise PreprocessingError(
                "Missing channel statistics."
            )

        for band in self.EXPECTED_BANDS:

            stats = channels.get(band)

            if not isinstance(stats, dict):
                raise PreprocessingError(
                    f"Missing statistics for {band}."
                )

            required_keys = (
                "clip_min",
                "clip_max",
                "mean",
                "std"
            )

            for key in required_keys:

                value = stats.get(key)

                if (
                    not isinstance(value, (int, float))
                    or not np.isfinite(value)
                ):
                    raise PreprocessingError(
                        f"Invalid {band} statistic: {key}"
                    )

            if stats["clip_min"] >= stats["clip_max"]:
                raise PreprocessingError(
                    f"Invalid clipping range for {band}."
                )

            if stats["std"] <= 0:
                raise PreprocessingError(
                    f"Invalid standard deviation for {band}."
                )

    # =====================================================
    # 3. READ RAW SAR IMAGE
    # =====================================================

    def read(self, path):

        metadata = self.validator.validate_raster(
            path,
            require_tile_size=True
        )

        try:

            image, valid_mask, read_metadata = (
                read_sar_image(metadata.path)
            )

        except Exception as exc:
            raise PreprocessingError(
                f"Unable to read SAR image: {metadata.path}"
            ) from exc

        if (
            read_metadata["crs"] != metadata.crs
            or read_metadata["transform"] != metadata.transform
        ):
            raise InputValidationError(
                "Raster metadata changed during reading."
            )

        sar_input = SARInput(
            image=image,
            valid_mask=valid_mask,
            metadata=metadata
        )

        sar_input.valid_mask = (
            self.validator.validate_input(sar_input)
        )

        return sar_input

    # =====================================================
    # 4. NORMALIZATION AND TENSOR CONVERSION
    # =====================================================

    def preprocess(
        self,
        sar_input: SARInput
    ):

        valid_mask = self.validator.validate_input(
            sar_input
        )

        image = sar_input.image.astype(
            np.float32,
            copy=False
        )

        # Recheck validity after float32 conversion.
        valid_mask = self.validator.validate_array(
            image,
            valid_mask
        )

        try:

            normalized = self.normalizer(
                image,
                valid_mask
            )

        except Exception as exc:
            raise PreprocessingError(
                "SAR normalization failed."
            ) from exc

        if normalized.shape != image.shape:
            raise PreprocessingError(
                "Normalization changed image dimensions."
            )

        if not np.isfinite(normalized).all():
            raise PreprocessingError(
                "Normalized image contains NaN or Inf."
            )

        fill_value = self.config["invalid_fill_value"]

        if not np.all(
            normalized[:, ~valid_mask] == fill_value
        ):
            raise PreprocessingError(
                "Invalid pixels were not correctly filled."
            )

        # CHW -> NCHW
        tensor = torch.from_numpy(
            np.ascontiguousarray(normalized)
        ).float().unsqueeze(0)

        expected_shape = (
            1,
            2,
            512,
            512
        )

        if tuple(tensor.shape) != expected_shape:
            raise PreprocessingError(
                f"Unexpected tensor shape: {tensor.shape}"
            )

        return PreprocessedInput(
            tensor=tensor,
            valid_mask=valid_mask.copy(),
            metadata=sar_input.metadata
        )

    # =====================================================
    # 5. END-TO-END PREPROCESSING
    # =====================================================

    def __call__(self, path):

        sar_input = self.read(path)

        return self.preprocess(sar_input)
