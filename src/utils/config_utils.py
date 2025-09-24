"""
Utilities for loading and handling configuration files.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field, ValidationError, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from utils.logging_utils import setup_logger

logger = setup_logger(log_file=__file__)


class Settings(BaseSettings):
    """Pydantic BaseSettings container for application config."""

    # .env
    GH_PAT: Optional[SecretStr] = Field(None)

    # .yaml
    PATH_DATA_ROOT: str = Field(...)
    PATH_TRACK_ANNOT_ROOT: str = Field(...)
    PATH_VIDEOS_ROOT: str = Field(...)
    PATH_ASSETS: str = Field(...)
    PATH_METRICS: str = Field(...)
    PATH_MODELS: str = Field(...)
    PATH_MODELS_CHECKPOINTS: str = Field(...)
    PATH_LOGS: str = Field(...)
    PATH_DATA_PROCESSING_MODULE: str = Field(...)
    PATH_UTILS_MODULE: str = Field(...)

    NUM_ACTIVITY_LABELS: int = Field(...)
    NUM_ACTION_LABELS: int = Field(...)
    EPOCHS: int = Field(...)
    MODEL_MODE: str = Field(...)
    TRAIN_IDS: List[int] = Field(...)
    VALIDATION_IDS: List[int] = Field(...)
    TEST_IDS: List[int] = Field(...)

    # Baseline 1
    B1_PATH_MODEL: str = Field(...)
    B1_FEATURES_SHAPE_0: int = Field(...)
    B1_FEATURES_SHAPE_1: int = Field(...)
    B1_LEFT_FRAMES: int = Field(...)
    B1_RIGHT_FRAMES: int = Field(...)
    B1_TRAIN_EPOCHS: int = Field(...)
    B1_TRAIN_BATCH_SIZE: int = Field(...)
    B1_EVAL_BATCH_SIZE: int = Field(...)
    B1_LR: float = Field(...)
    B1_FREEZE_BACKBONE: bool = Field(...)
    B1_NUM_WORKERS: int = Field(...)
    B1_OPTIMIZER: str = Field(...)
    B1_WEIGHT_DECAY: float = Field(...)
    B1_LOSS_FN: str = Field(...)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Customize the configuration sources for loading settings.

        Args:
            settings_cls (BaseSettings): The settings class reference.
            init_settings (dict): Settings provided at initialization.
            env_settings (dict): Environment variable settings.
            dotenv_settings (dict): Settings loaded from `.env` file.
            file_secret_settings (dict): Settings loaded from secret files.

        Returns:
            tuple: Ordered configuration sources for pydantic to process.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=Path("config/config.yaml"),
                yaml_file_encoding="utf-8",
            ),
            file_secret_settings,
        )


def get_settings(verbose: bool = False) -> Settings:
    """Retrieve application settings.

    Args:
        verbose (bool): If True, log the loaded settings (with secrets masked).

    Returns:
        Settings: An initialized settings object.
    """
    logger.info("Loading App Settings")

    try:
        settings = Settings()  # type: ignore
    except (ValidationError, FileNotFoundError, OSError) as exc:
        logger.exception("Failed to initialize settings: %s", exc)
        raise

    if verbose:
        dump = getattr(settings, "model_dump", None)
        values = dump() if callable(dump) else getattr(settings, "dict", lambda: {})()

        logger.info("Loaded settings: %s", values)

    return settings


def main():
    """Entry point for the program.

    Demonstrates how to load and access configuration values
    from the settings object.
    """
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.\n")

    # Usage
    settings = get_settings(verbose=True)

    print(settings.PATH_DATA_ROOT)  # From config.yaml
    print(settings.PATH_VIDEOS_ROOT)  # From config.yaml
    print(settings.GH_PAT)  # From .env


if __name__ == "__main__":
    main()
