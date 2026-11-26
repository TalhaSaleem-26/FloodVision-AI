
import time

import numpy as np
import torch

from .exceptions import PredictionError

from .model_loader import ModelLoader

from .schemas import (
    PreprocessedInput,
    RawPrediction,
)


class FloodPredictor:
    """Execute the production water segmentation model."""

    def __init__(
        self,
        loader: ModelLoader,
        device="auto"
    ):

        self.loader = loader

        self.model = self.loader.load(
            device=device
        )

        self.device = self.loader.device

        self.config = (
            self.loader.inference_config.copy()
        )

        self.model_version = (
            self.loader.metadata["model_version"]
        )

    # =====================================================
    # 1. INPUT VALIDATION
    # =====================================================

    def validate_input(
        self,
        processed: PreprocessedInput
    ):

        if not isinstance(
            processed,
            PreprocessedInput
        ):
            raise PredictionError(
                "Expected PreprocessedInput object."
            )

        tensor = processed.tensor

        if not isinstance(tensor, torch.Tensor):
            raise PredictionError(
                "Input must be a PyTorch tensor."
            )

        expected_shape = (
            1,
            self.config["input_channels"],
            self.config["tile_height"],
            self.config["tile_width"]
        )

        if tuple(tensor.shape) != expected_shape:
            raise PredictionError(
                f"Expected shape {expected_shape}, "
                f"received {tuple(tensor.shape)}."
            )

        if tensor.dtype != torch.float32:
            raise PredictionError(
                "Expected float32 input tensor."
            )

        if not torch.isfinite(tensor).all().item():
            raise PredictionError(
                "Input tensor contains NaN or Inf."
            )

        if not isinstance(
            processed.valid_mask,
            np.ndarray
        ):
            raise PredictionError(
                "Validity mask must be a NumPy array."
            )

        if processed.valid_mask.shape != expected_shape[2:]:
            raise PredictionError(
                "Invalid validity mask dimensions."
            )

        if processed.valid_mask.dtype != np.bool_:
            raise PredictionError(
                "Validity mask must have boolean dtype."
            )

        if not processed.valid_mask.any():
            raise PredictionError(
                "Input contains no valid pixels."
            )

        if (
            processed.metadata.height,
            processed.metadata.width
        ) != expected_shape[2:]:
            raise PredictionError(
                "Raster metadata dimensions mismatch."
            )

        return True

    # =====================================================
    # 2. OUTPUT VALIDATION
    # =====================================================

    def validate_output(self, logits):

        if not isinstance(logits, torch.Tensor):
            raise PredictionError(
                "Model output must be a tensor."
            )

        expected_shape = (
            1,
            1,
            self.config["tile_height"],
            self.config["tile_width"]
        )

        if tuple(logits.shape) != expected_shape:
            raise PredictionError(
                f"Unexpected output shape: "
                f"{tuple(logits.shape)}"
            )

        if not torch.is_floating_point(logits):
            raise PredictionError(
                "Expected floating-point logits."
            )

        if not torch.isfinite(logits).all().item():
            raise PredictionError(
                "Model output contains NaN or Inf."
            )

        return True

    # =====================================================
    # 3. DEVICE SYNCHRONIZATION
    # =====================================================

    def _synchronize(self):

        if self.device.type == "cuda":

            torch.cuda.synchronize(
                self.device
            )

    # =====================================================
    # 4. MODEL PREDICTION
    # =====================================================

    def predict(
        self,
        processed: PreprocessedInput
    ) -> RawPrediction:

        self.validate_input(processed)

        if self.model.training:
            raise PredictionError(
                "Model must be in evaluation mode."
            )

        try:

            tensor = processed.tensor.to(
                self.device
            )

            self._synchronize()

            start_time = time.perf_counter()

            with torch.inference_mode():

                logits = self.model(tensor)

            self._synchronize()

            inference_time_ms = (
                time.perf_counter() - start_time
            ) * 1000

            self.validate_output(logits)

            logits = logits.detach().cpu()

        except PredictionError:
            raise

        except Exception as exc:
            raise PredictionError(
                "Model inference failed."
            ) from exc

        return RawPrediction(
            logits=logits,

            valid_mask=processed.valid_mask.copy(),
            metadata=processed.metadata,

            model_version=self.model_version,
            device=str(self.device),

            inference_time_ms=inference_time_ms
        )

    # =====================================================
    # 5. CALLABLE INTERFACE
    # =====================================================

    def __call__(
        self,
        processed: PreprocessedInput
    ) -> RawPrediction:

        return self.predict(processed)
