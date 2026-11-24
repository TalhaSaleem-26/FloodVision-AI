from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import Dataset

from floodvision.preprocessing.sar import (
    read_sar_image,
    read_target_mask
)

from floodvision.preprocessing.normalization import (
    SARNormalizer
)


class FloodDataset(Dataset):

    def __init__(
        self,
        dataset_config_path,
        preprocessing_config_path,
        split_path,
        transforms=None
    ):

        dataset_config_path = Path(
            dataset_config_path
        ).resolve()

        self.base_dir = dataset_config_path.parents[1]

        with open(
            dataset_config_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = yaml.safe_load(file)["dataset"]

        self.data_dir = (
            self.base_dir / self.config["root_dir"]
        )

        self.image_dir = (
            self.data_dir / self.config["image_dir"]
        )

        self.mask_dir = (
            self.data_dir / self.config["mask_dir"]
        )

        self.samples = pd.read_csv(split_path)

        self.normalizer = SARNormalizer(
            preprocessing_config_path
        )

        self.transforms = transforms

        self.ignore_index = self.config["ignore_index"]

        if self.samples.empty:
            raise ValueError("Dataset split is empty")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample_id = self.samples.iloc[index]["sample_id"]

        image_path = (
            self.image_dir / f"{sample_id}_S1Hand.tif"
        )

        mask_path = (
            self.mask_dir / f"{sample_id}_LabelHand.tif"
        )

        # Load raw satellite image
        image, input_valid, _ = read_sar_image(
            image_path
        )

        # Load ground truth
        mask = read_target_mask(mask_path)

        # Normalize image
        image = self.normalizer(
            image,
            input_valid
        )

        # Ignore invalid input pixels in target
        mask[~input_valid] = self.ignore_index

        # CHW -> HWC for Albumentations
        image = np.transpose(image, (1, 2, 0))

        # Apply synchronized augmentation
        if self.transforms is not None:

            augmented = self.transforms(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        # HWC -> CHW for PyTorch
        image = np.transpose(image, (2, 0, 1))

        # Convert NumPy arrays to tensors
        image = torch.from_numpy(
            np.ascontiguousarray(image)
        ).float()

        mask = torch.from_numpy(
            np.ascontiguousarray(mask)
        ).float()

        # Add output channel dimension
        mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "sample_id": sample_id
        }