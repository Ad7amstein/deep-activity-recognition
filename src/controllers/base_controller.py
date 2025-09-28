import os
from typing import Optional
import random
import numpy as np
import torch
from utils.config_utils import get_settings
from utils.logging_utils import setup_logger


class BaseController:
    def __init__(
        self, seed_value: int = 42, set_seeds: bool = True, verbose: bool = True
    ) -> None:
        self.verbose = verbose
        self.app_settings = get_settings()
        self.logger = setup_logger(
            logger_name=__name__,
            log_file=__file__,
            log_dir=self.app_settings.PATH_LOGS,
            log_to_console=self.verbose,
        )
        self.logger.info("Initializing BaseController Module...")
        if set_seeds:
            self.set_all_seeds(seed_value=seed_value)

    def set_all_seeds(self, seed_value: int, verbose: Optional[bool] = None) -> None:
        self.logger.info("Setting all seeds...")
        verbose = verbose if verbose is not None else self.verbose
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )
    BaseController()


if __name__ == "__main__":
    main()
