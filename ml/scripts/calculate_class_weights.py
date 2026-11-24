from pathlib import Path

import yaml
from tqdm import tqdm

from floodvision.datasets.flood_dataset import FloodDataset


BASE_DIR = Path(__file__).resolve().parents[1]


def main():

    # Initialize training dataset

    dataset = FloodDataset(
        dataset_config_path=(
            BASE_DIR / "configs/dataset.yaml"
        ),
        preprocessing_config_path=(
            BASE_DIR / "configs/preprocessing.yaml"
        ),
        split_path=(
            BASE_DIR / "data/splits/geographical/train.csv"
        ),
        transforms=None
    )

    water_pixels = 0
    non_water_pixels = 0
    invalid_pixels = 0

    # Count pixel classes

    for index in tqdm(
        range(len(dataset)),
        desc="Calculating class weights"
    ):

        sample = dataset[index]

        mask = sample["mask"]

        water_pixels += (
            mask == 1
        ).sum().item()

        non_water_pixels += (
            mask == 0
        ).sum().item()

        invalid_pixels += (
            mask == -1
        ).sum().item()

    valid_pixels = (
        water_pixels + non_water_pixels
    )

    if water_pixels == 0 or non_water_pixels == 0:

        raise ValueError(
            "Both classes must exist in training data"
        )

    # Calculate positive class weight

    pos_weight = non_water_pixels / water_pixels

    results = {
        "split": "train",
        "samples": len(dataset),
        "water_pixels": water_pixels,
        "non_water_pixels": non_water_pixels,
        "invalid_pixels": invalid_pixels,
        "valid_pixels": valid_pixels,
        "water_percentage": (
            water_pixels / valid_pixels * 100
        ),
        "non_water_percentage": (
            non_water_pixels / valid_pixels * 100
        ),
        "pos_weight": float(pos_weight)
    }

    print("\n" + "=" * 50)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 50)

    for name, value in results.items():

        print(f"{name}: {value}")

    # Save class weights

    output_path = (
        BASE_DIR / "configs/class_weights.yaml"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            results,
            file,
            sort_keys=False
        )

    print("\nClass weights saved successfully!")
    print("Output:", output_path)


if __name__ == "__main__":

    main()