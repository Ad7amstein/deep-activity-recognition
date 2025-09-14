import os
from enum import Enum


class OptimizerEnum(Enum):
    ADAMW = "AdamW"
    SGD = "SGD"

class LossFNEnum(Enum):
    CROSS_ENTROPY_LOSS = "cross_entropy_loss"
    BCE_LOSS = "bce_loss"


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
