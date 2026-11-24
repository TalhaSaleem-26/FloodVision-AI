from floodvision.models.baseline_cnn import BaselineCNN


MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN
}


def build_model(config):

    config = config.copy()

    name = config.pop("name")

    if name not in MODEL_REGISTRY:

        raise ValueError(
            f"Unsupported model: {name}"
        )

    model_class = MODEL_REGISTRY[name]

    return model_class(**config)