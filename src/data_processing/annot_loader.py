import os
from data_processing.box_info import BoxInfo
from utils.config_utils import load_config

CONFIG = load_config()


class AnnotationLoader:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{__class__.__name__}()"


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split(".")[0]}` Module.\n")

    annot_loader = AnnotationLoader()
    print(annot_loader)


if __name__ == "__main__":
    main()
