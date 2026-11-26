
import time
import math

import numpy as np
import torch

from .exceptions import PostprocessingError

from .schemas import (
    RawPrediction,
    PredictionResult,
    PredictionStatistics,
)


class SegmentationPostprocessor:
    """Convert raw logits into water segmentation results."""

    WATER_LABEL = 1
    NON_WATER_LABEL = 0
    INVALID_LABEL = -1

    def __init__(self, threshold=0.5):

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 < threshold < 1
        ):
            raise PostprocessingError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = float(threshold)

    # =====================================================
    # 1. VALIDATE RAW PREDICTION
    # =====================================================

    def validate_prediction(
        self,
        raw: RawPrediction
    ):

        if not isinstance(raw, RawPrediction):
            raise PostprocessingError(
                "Expected RawPrediction object."
            )

        if not isinstance(raw.logits, torch.Tensor):
            raise PostprocessingError(
                "Logits must be a tensor."
            )

        if not torch.is_floating_point(raw.logits):
            raise PostprocessingError(
                "Logits must be floating-point values."
            )

        expected_shape = (
            1,
            1,
            raw.metadata.height,
            raw.metadata.width
        )

        if tuple(raw.logits.shape) != expected_shape:
            raise PostprocessingError(
                f"Invalid logits shape: "
                f"{tuple(raw.logits.shape)}"
            )

        if not torch.isfinite(raw.logits).all().item():
            raise PostprocessingError(
                "Logits contain NaN or Inf."
            )

        if not isinstance(raw.valid_mask, np.ndarray):
            raise PostprocessingError(
                "Validity mask must be a NumPy array."
            )

        if raw.valid_mask.shape != expected_shape[2:]:
            raise PostprocessingError(
                "Invalid validity mask dimensions."
            )

        if raw.valid_mask.dtype != np.bool_:
            raise PostprocessingError(
                "Validity mask must have boolean dtype."
            )

        if not raw.valid_mask.any():
            raise PostprocessingError(
                "No valid prediction pixels."
            )

        return True

    # =====================================================
    # 2. COMPUTE PREDICTION STATISTICS
    # =====================================================

    def calculate_statistics(
        self,
        mask,
        probabilities,
        valid_mask
    ) -> PredictionStatistics:

        total_pixels = int(mask.size)

        valid_pixels = int(valid_mask.sum())

        invalid_pixels = int(
            total_pixels - valid_pixels
        )

        water_pixels = int(
            (mask == self.WATER_LABEL).sum()
        )

        non_water_pixels = int(
            (mask == self.NON_WATER_LABEL).sum()
        )

        if (
            water_pixels + non_water_pixels
            != valid_pixels
        ):
            raise PostprocessingError(
                "Pixel count consistency check failed."
            )

        if valid_pixels == 0:
            raise PostprocessingError(
                "Cannot calculate coverage without valid pixels."
            )

        water_coverage = (
            water_pixels / valid_pixels
        ) * 100

        non_water_coverage = (
            non_water_pixels / valid_pixels
        ) * 100

        valid_percentage = (
            valid_pixels / total_pixels
        ) * 100

        mean_water_probability = float(
            probabilities[valid_mask].mean(
                dtype=np.float64
            )
        )

        return PredictionStatistics(
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
            invalid_pixels=invalid_pixels,

            water_pixels=water_pixels,
            non_water_pixels=non_water_pixels,

            water_coverage=water_coverage,
            non_water_coverage=non_water_coverage,
            valid_percentage=valid_percentage,

            mean_water_probability=(
                mean_water_probability
            )
        )

    # =====================================================
    # 3. POSTPROCESS MODEL OUTPUT
    # =====================================================

    def process(
        self,
        raw: RawPrediction
    ) -> PredictionResult:

        start_time = time.perf_counter()

        self.validate_prediction(raw)

        try:

            with torch.inference_mode():

                probabilities = (
                    torch.sigmoid(raw.logits)
                    .squeeze(0)
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                    .copy()
                )

            valid_mask = raw.valid_mask.copy()

            mask = (
                probabilities >= self.threshold
            ).astype(np.int8)

            # Restore invalid pixels.
            mask[~valid_mask] = self.INVALID_LABEL

            # Invalid probabilities are undefined.
            probabilities[~valid_mask] = np.nan

            if not np.isin(
                mask,
                [
                    self.INVALID_LABEL,
                    self.NON_WATER_LABEL,
                    self.WATER_LABEL
                ]
            ).all():
                raise PostprocessingError(
                    "Unexpected segmentation labels."
                )

            if not np.array_equal(
                mask == self.INVALID_LABEL,
                ~valid_mask
            ):
                raise PostprocessingError(
                    "Invalid pixel restoration failed."
                )

            if not np.isfinite(
                probabilities[valid_mask]
            ).all():
                raise PostprocessingError(
                    "Invalid probability values."
                )

            statistics = self.calculate_statistics(
                mask=mask,
                probabilities=probabilities,
                valid_mask=valid_mask
            )

        except PostprocessingError:
            raise

        except Exception as exc:
            raise PostprocessingError(
                "Segmentation postprocessing failed."
            ) from exc

        postprocessing_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return PredictionResult(
            mask=mask,
            probabilities=probabilities,

            valid_mask=valid_mask,
            metadata=raw.metadata,

            statistics=statistics,

            model_version=raw.model_version,
            threshold=self.threshold,
            device=raw.device,

            inference_time_ms=raw.inference_time_ms,

            postprocessing_time_ms=(
                postprocessing_time_ms
            )
        )

    # =====================================================
    # 4. CALLABLE INTERFACE
    # =====================================================

    def __call__(
        self,
        raw: RawPrediction
    ) -> PredictionResult:

        return self.process(raw)
