"""
Utilities for loading and handling configuration files.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class Settings(BaseSettings):
    # .env
    GH_PAT: Optional[str] = Field(None)
    CLONING: Optional[str] = Field(None)

    # .yaml
    PATH_DATA_ROOT: str = Field(...)
    PATH_TRACK_ANNOT_ROOT: str = Field(...)
    PATH_VIDEOS_ROOT: str = Field(...)
    NUM_ACTIVITY_LABELS: int = Field(...)
    NUM_ACTION_LABELS: int = Field(...)
    EPOCHS: int = Field(...)
    PATH_MODELS: str = Field(...)
    PATH_MODELS_CHECKPOINTS: str = Field(...)
    MODEL_MODE: str = Field(...)
    TRAIN_IDS: List[int] = Field(...)
    VALIDATION_IDS: List[int] = Field(...)
    TEST_IDS: List[int] = Field(...)
    PATH_ASSETS: str = Field(...)
    PATH_METRICS: str = Field(...)

    # Baseline 1
    B1_PATH_B1MODEL: str = Field(...)
    B1_FEATURES_SHAPE_0: int
    B1_FEATURES_SHAPE_1: int
    B1_LEFT_FRAMES: int
    B1_RIGHT_FRAMES: int
    B1_TRAIN_EPOCHS: int
    B1_BATCH_SIZE: int
    B1_LR: float
    B1_FREEZE_BACKBONE: bool
    B1_NUM_WORKERS: int
    B1_OPTIMIZER: str
    B1_WEIGHT_DECAY: float


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


def get_settings():
    """Retrieve application settings.

    Returns:
        Settings: An initialized settings object.
    """
    return Settings()  # type: ignore


def main():
    """Entry point for the program.

    Demonstrates how to load and access configuration values
    from the settings object.
    """
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.\n")

    # Usage
    settings = get_settings()

    print(settings.PATH_DATA_ROOT)  # From config.yaml
    print(settings.PATH_VIDEOS_ROOT)  # From config.yaml
    print(settings.GH_PAT)  # From .env


if __name__ == "__main__":
    main()
