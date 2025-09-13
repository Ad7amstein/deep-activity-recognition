import os
from utils.config_utils import get_settings
from torch.utils.data import Dataset


class BaseController:
    def __init__(self) -> None:
        self.app_settings = get_settings()


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
