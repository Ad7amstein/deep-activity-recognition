"""
This module defines enumerations for machine learning components.
"""

import os
from enum import Enum


class OptimizerEnum(Enum):
    """
    Enumeration of supported optimizer algorithms.

    Attributes:
        ADAMW (str): AdamW optimizer, commonly used for transformer models
                     and regularized training.
        SGD (str): Stochastic Gradient Descent optimizer.
    """

    ADAMW = "AdamW"
    SGD = "SGD"


class LossFNEnum(Enum):
    """
    Enumeration of supported loss functions.

    Attributes:
        CROSS_ENTROPY_LOSS (str): Cross-entropy loss, typically used for
                                  multi-class classification.
        BCE_LOSS (str): Binary cross-entropy loss, used for binary classification tasks.
    """

    CROSS_ENTROPY_LOSS = "cross_entropy_loss"
    BCE_LOSS = "bce_loss"


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
