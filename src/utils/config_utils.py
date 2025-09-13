"""
Utilities for loading and handling configuration files.
"""

import os
from pathlib import Path
from typing import Optional
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
        return (
            init_settings,  # kwargs passed directly to Settings()
            env_settings,  # Environment variables
            dotenv_settings,  # .env file
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=Path("config/config.yaml"),
                yaml_file_encoding="utf-8",
            ),  # YAML file
            file_secret_settings,  # Secrets from files
        )


def get_settings():
    return Settings()  # type: ignore


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.\n")
    # Usage

    settings = get_settings()

    print(settings.PATH_DATA_ROOT)  # From config.yaml
    print(settings.PATH_VIDEOS_ROOT)  # From config.yaml
    print(settings.GH_PAT)  # From .env


if __name__ == "__main__":
    main()
