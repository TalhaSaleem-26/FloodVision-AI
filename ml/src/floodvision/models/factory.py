from .baseline_cnn import BaselineCNN
from .unet import UNet


MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    "unet": UNet,
}


def build_model(config):

    config = config.copy()

    model_name = config.pop("name")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]

    return model_class(**config)