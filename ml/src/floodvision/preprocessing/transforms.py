import albumentations as A


def get_train_transforms(seed=42):

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ],
        seed=seed
    )


def get_val_transforms():

    return None