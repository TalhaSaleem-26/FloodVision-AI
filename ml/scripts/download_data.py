from pathlib import Path

import yaml
from google.cloud import storage
from tqdm import tqdm


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "dataset.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["dataset"]


def download_folder(client, bucket_name, prefix, destination, extension):
    destination.mkdir(parents=True, exist_ok=True)

    blobs = [
        blob
        for blob in client.list_blobs(bucket_name, prefix=prefix)
        if blob.name.endswith(extension)
    ]

    if not blobs:
        raise RuntimeError(f"No files found: {prefix}")

    print(f"\nFolder: {destination.name}")
    print(f"Total files: {len(blobs)}")

    for blob in tqdm(blobs, desc=destination.name):
        filename = Path(blob.name).name
        file_path = destination / filename

        # Skip previously downloaded files
        if file_path.exists() and file_path.stat().st_size == blob.size:
            continue

        temp_path = file_path.with_suffix(
            file_path.suffix + ".part"
        )

        try:
            blob.download_to_filename(str(temp_path), timeout=120)

            if temp_path.stat().st_size != blob.size:
                raise IOError(f"Incomplete download: {filename}")

            temp_path.replace(file_path)

        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    return len(blobs)


def main():
    config = load_config()

    client = storage.Client.create_anonymous_client()

    bucket = config["bucket"]
    prefix = config["remote_prefix"]
    root_dir = BASE_DIR / config["root_dir"]

    image_dir = config["image_dir"]
    mask_dir = config["mask_dir"]

    print("Starting Sen1Floods11 dataset download...")

    image_count = download_folder(
        client,
        bucket,
        f"{prefix}/{image_dir}/",
        root_dir / image_dir,
        ".tif"
    )

    mask_count = download_folder(
        client,
        bucket,
        f"{prefix}/{mask_dir}/",
        root_dir / mask_dir,
        ".tif"
    )

    # Download official dataset splits
    split_count = download_folder(
        client,
        bucket,
        f"v{config['version']}/splits/flood_handlabeled/",
        BASE_DIR / "data" / "splits" / "official",
        ".csv"
    )

    print("\n========== Download Summary ==========")
    print(f"Satellite images: {image_count}")
    print(f"Segmentation masks: {mask_count}")
    print(f"Official split files: {split_count}")
    print(f"Dataset location: {root_dir}")

    if image_count == mask_count == 446:
        print("Dataset file counts verified!")
    else:
        raise RuntimeError("Unexpected dataset file counts")


if __name__ == "__main__":
    main()