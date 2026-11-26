import numpy as np
import pytest
import torch
from rasterio.errors import RasterioIOError

from floodvision.datasets.flood_dataset import FloodDataset
from floodvision.preprocessing.transforms import get_train_transforms, get_val_transforms


@pytest.fixture
def dataset_files(tmp_path, raster_factory):
    config = tmp_path / "config" / "dataset.yaml"
    config.parent.mkdir()
    config.write_text(
        "dataset: {root_dir: ., image_dir: ., mask_dir: ., ignore_index: -1}",
        encoding="utf-8",
    )
    split = tmp_path / "split.csv"
    split.write_text("sample_id\nscene\n", encoding="utf-8")
    image = np.full((2, 2, 3), -18, dtype=np.float32)
    image[0, 0, 0] = -6
    image[1, 1, 2] = np.nan
    raster_factory(image, filename="scene_S1Hand.tif")
    mask = np.zeros((1, 2, 3), dtype=np.float32)
    mask[0, 0, 0] = 1
    raster_factory(mask, filename="scene_LabelHand.tif")
    return config, split


def test_dataset_loads_normalized_tensors_and_ignores_invalid_input(
    dataset_files, preprocessing_config_path
):
    config, split = dataset_files
    dataset = FloodDataset(config, preprocessing_config_path, split, get_val_transforms())

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["sample_id"] == "scene"
    assert sample["image"].dtype == sample["mask"].dtype == torch.float32
    torch.testing.assert_close(sample["mask"], torch.tensor([[[1., 0., 0.], [0., 0., -1.]]]))
    torch.testing.assert_close(sample["image"][:, 1, 2], torch.zeros(2))
    torch.testing.assert_close(sample["image"][:, 0, 0], torch.tensor([1., -0.2]))


def test_seeded_augmentation_keeps_image_and_target_aligned(
    dataset_files, preprocessing_config_path
):
    config, split = dataset_files
    first = FloodDataset(config, preprocessing_config_path, split, get_train_transforms(seed=42))
    second = FloodDataset(config, preprocessing_config_path, split, get_train_transforms(seed=42))

    original = torch.tensor([[[1., 0., 0.], [0., 0., -1.]]])
    masks = []
    for _ in range(8):
        sample = first[0]
        torch.testing.assert_close(sample["image"], second[0]["image"])
        torch.testing.assert_close(sample["mask"] == 1, sample["image"][:1] == 1)
        torch.testing.assert_close(sample["mask"] == -1, (sample["image"] == 0).all(0, keepdim=True))
        masks.append(sample["mask"])
    assert any(not mask.equal(original) for mask in masks)


def test_dataset_rejects_empty_split(dataset_files, preprocessing_config_path):
    config, split = dataset_files
    split.write_text("sample_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset split is empty"):
        FloodDataset(config, preprocessing_config_path, split)


def test_dataset_rejects_missing_raster(dataset_files, preprocessing_config_path):
    config, split = dataset_files
    split.write_text("sample_id\nmissing\n", encoding="utf-8")
    dataset = FloodDataset(config, preprocessing_config_path, split)
    with pytest.raises(RasterioIOError):
        dataset[0]


@pytest.mark.parametrize(
    ("filename", "shape", "label", "message"),
    [("scene_S1Hand.tif", (1, 2, 3), 0, "Expected 2 SAR channels"),
     ("scene_LabelHand.tif", (1, 2, 3), 2, "Unexpected mask labels")],
)
def test_dataset_rejects_malformed_raster(
    dataset_files, preprocessing_config_path, raster_factory, filename, shape, label, message
):
    config, split = dataset_files
    raster_factory(np.full(shape, label, np.float32), filename=filename)
    dataset = FloodDataset(config, preprocessing_config_path, split)
    with pytest.raises(ValueError, match=message):
        dataset[0]
