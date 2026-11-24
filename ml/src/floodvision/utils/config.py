from pathlib import Path

import yaml


def load_config(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    if not isinstance(config, dict):

        raise ValueError(
            "Invalid configuration"
        )

    return config