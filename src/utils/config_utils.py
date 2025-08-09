"""
Utilities for loading and handling configuration files.
"""

import os
import yaml


def load_config(config_path: str = "config/config.yaml", verbose: bool = False):
    """
    Loads configuration settings from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file. Defaults to "config/config.yaml".
        verbose (bool): If True, prints logging information. Defaults to False.
    Returns:
        dict: Configuration parameters loaded from the YAML file.
    Raises:
        FileNotFoundError: If the configuration file does not exist at the specified path.
        ValueError: If there is an error parsing the YAML file.
    """

    if verbose:
        print(f"\n[INFO] Loading Configurations from {config_path}...")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}") from e


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split(".")[0]}` Module.\n")
    config = load_config(verbose=True)
    print(f"Config Type: {type(config)}")
    print(f"Config Items:\n{config.items()}")


if __name__ == "__main__":
    main()
