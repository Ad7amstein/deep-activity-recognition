"""Configuration enumeration for baseline model parameters."""

import os
from enum import Enum


class B1Enum(Enum):
    """Enumeration of baseline model configuration parameters.

    Attributes:
        FEATURES_SHAPE (tuple): Expected shape of input features (height, width).
        RIGHT_FRAMES (int): Number of right frames to include (default 0).
        LEFT_FRAMES (int): Number of left frames to include (default 0).
        TRAIN_EPOCHS (int): Number of training epochs (default 5).
    """

    FEATURES_SHAPE = (224, 224)
    RIGHT_FRAMES = 5
    LEFT_FRAMES = 4
    TRAIN_EPOCHS = 10
    BATCH_SIZE = 64
    LR = 0.1
    FREEZE_BACKBONE = False


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
