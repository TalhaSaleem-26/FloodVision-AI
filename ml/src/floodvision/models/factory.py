from .baseline_cnn import BaselineCNN
from .unet import UNet
from .resnet_unet import ResNetUNet


MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    "unet": UNet,
    "resnet_unet": ResNetUNet
}


def build_model(config):

    config = config.copy()

    name = config.pop("name")

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {name}"
        )

    return MODEL_REGISTRY[name](**config)