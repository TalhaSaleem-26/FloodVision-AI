from .baseline_cnn import BaselineCNN
from .unet import UNet
from .resnet_unet import ResNetUNet
from .mobilenet_unet import MobileNetUNet


MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    "unet": UNet,
    "resnet_unet": ResNetUNet,
    "mobilenet_unet": MobileNetUNet
}


def build_model(config):

    config = config.copy()

    model_name = config.pop("name")

    if model_name not in MODEL_REGISTRY:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model_class = MODEL_REGISTRY[model_name]

    return model_class(**config)