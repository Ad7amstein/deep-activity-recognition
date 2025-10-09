"""
Base controller module for managing application setup, logging, and reproducibility.
"""

import os
from typing import Optional
import random
import string
import numpy as np
import torch
from utils.config_utils import get_settings
from utils.logging_utils import setup_logger
from models.enums import PathPrefixEnum

app_settings = get_settings()
class BaseController:
    """Base controller class for managing application setup and reproducibility.

    Attributes:
        verbose (bool): Whether to enable verbose logging output.
        app_settings (Any): Application settings loaded from configuration.
        logger (logging.Logger): Configured logger instance.
        base_dir (str): Base directory of the project.
        files_dir (str): Directory path for asset files.
    """

    def __init__(
        self, seed_value: int = 42, set_seeds: bool = True, verbose: bool = True
    ) -> None:
        """Initialize the BaseController.

        Args:
            seed_value (int, optional): The seed value for reproducibility.
                Defaults to 42.
            set_seeds (bool, optional): Whether to set all seeds during initialization.
                Defaults to True.
            verbose (bool, optional): Whether to enable verbose logging.
                Defaults to True.
        """

        self.verbose = verbose
        self.app_settings = app_settings
        self.baseline_root = BaseController.get_baseline_root()
        self.logger = setup_logger(
            logger_name=__name__,
            log_file=__file__,
            log_dir=os.path.join(self.app_settings.PATH_LOGS, self.baseline_root),
            log_to_console=self.verbose,
        )
        self.logger.info("Initializing BaseController Module...")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.files_dir = os.path.join(self.base_dir, "assets", "files")
        if set_seeds:
            self.set_all_seeds(seed_value=seed_value)

    def set_all_seeds(self, seed_value: int, verbose: Optional[bool] = None) -> None:
        """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

        Args:
            seed_value (int): The seed value to set.
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                If None, falls back to the instance-level verbosity setting. Defaults to None.
        """

        self.logger.info("Setting all seeds (seed_value=%s)...", seed_value)
        verbose = verbose if verbose is not None else self.verbose
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_random_string(self, length: int = 12) -> str:
        """Generate a random alphanumeric string.

        Args:
            length (int, optional): Length of the generated string. Defaults to 12.

        Returns:
            str: A random string containing lowercase letters and digits.
        """

        return "".join(
            random.choices("".join([string.ascii_lowercase, string.digits]), k=length)
        )

    @staticmethod
    def get_baseline_root(baseline_num: int = app_settings.BASELINE_NUM, stage_num: Optional[int] = app_settings.STAGE_NUM, exp_num: int = app_settings.EXPERIMENT_NUM) -> str:
        """Generate the baseline directory path based on provided identifiers.

        Args:
            baseline_num (int): The baseline number used to form the root directory name.
            stage_num (Optional[int]): The stage number to include in the path. Defaults to None.
            exp_num (int): The experiment number to include in the path if greater than 0. Defaults to 0.

        Returns:
            str: The constructed baseline path string combining baseline, stage, and experiment prefixes.
        """

        baseline_root = f"{PathPrefixEnum.baseline_prefix.value}{baseline_num}"

        if stage_num is not None:
            baseline_root = os.path.join(
                baseline_root, f"{PathPrefixEnum.stage_prefix.value}{stage_num}"
            )

        if exp_num > 0:
            baseline_root = os.path.join(
                baseline_root, f"{PathPrefixEnum.experiment_prefix.value}{exp_num}"
            )

        return baseline_root


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )
    BaseController()


if __name__ == "__main__":
    main()
