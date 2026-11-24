from pathlib import Path

import pandas as pd
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "configs" / "split.yaml"

STATS_PATH = (
    BASE_DIR / "data" / "processed"
    / "dataset_statistics.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "splits" / "geographical"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["split"]


def create_splits(df, config):

    train_regions = set(config["train_regions"])
    val_regions = set(config["val_regions"])
    test_regions = set(config["test_regions"])

    # Prevent geographical leakage
    assert not train_regions & val_regions
    assert not train_regions & test_regions
    assert not val_regions & test_regions

    all_regions = set(df["region"])

    assigned_regions = (
        train_regions | val_regions | test_regions
    )

    assert assigned_regions == all_regions

    train_df = df[
        df["region"].isin(train_regions)
    ].copy()

    val_df = df[
        df["region"].isin(val_regions)
    ].copy()

    test_df = df[
        df["region"].isin(test_regions)
    ].copy()

    return train_df, val_df, test_df


def validate_splits(train_df, val_df, test_df, df):

    assert df["sample_id"].is_unique

    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])

    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids

    assert (
        train_ids | val_ids | test_ids
    ) == set(df["sample_id"])

    print("Split validation successful!")


def main():

    config = load_config()
    df = pd.read_csv(STATS_PATH)

    train_df, val_df, test_df = create_splits(
        df, config
    )

    validate_splits(
        train_df, val_df, test_df, df
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    for name, split_df in splits.items():

        split_df.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False
        )

        percentage = len(split_df) / len(df) * 100

        print(f"\n{name.upper()} SET")
        print(f"Samples: {len(split_df)}")
        print(f"Percentage: {percentage:.2f}%")
        print("Regions:", split_df["region"].unique())

    print("\nAll splits saved successfully!")


if __name__ == "__main__":
    main()