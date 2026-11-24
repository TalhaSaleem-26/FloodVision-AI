from pathlib import Path

import numpy as np
import yaml


class SARNormalizer:

    def __init__(self, config_path: str | Path):

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as file:
            config = yaml.safe_load(file)["preprocessing"]

        self.channels = config["channels"]
        self.epsilon = config["epsilon"]
        self.fill_value = config["invalid_fill_value"]

        self.band_names = ["VV", "VH"]

    def __call__(
        self,
        image: np.ndarray,
        valid_mask: np.ndarray
    ):

        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError("Expected image shape (2, H, W)")

        if valid_mask.shape != image.shape[1:]:
            raise ValueError("Invalid mask dimensions")

        normalized = np.full(
            image.shape,
            self.fill_value,
            dtype=np.float32
        )

        valid_mask = (
            valid_mask
            & np.isfinite(image).all(axis=0)
        )

        for index, band in enumerate(self.band_names):

            stats = self.channels[band]

            values = image[index][valid_mask]

            clipped = np.clip(
                values,
                stats["clip_min"],
                stats["clip_max"]
            )

            scaled = (
                clipped - stats["mean"]
            ) / max(stats["std"], self.epsilon)

            normalized[index][valid_mask] = scaled

        return normalized