"""Enumeration of model training and evaluation result keys."""

import os
from enum import Enum


class ModelResults(Enum):
    """String enumeration of training and evaluation result keys.

    Attributes:
        TRAIN_LOSS (str): Key for training loss values.
        TRAIN_ACCURACY (str): Key for training accuracy values.
        TEST_LOSS (str): Key for test loss values.
        TEST_ACCURACY (str): Key for test accuracy values.
        TEST_PRECISION (str): Key for test precision values.
        TEST_RECALL (str): Key for test recall values.
        TEST_F1_SCORE (str): Key for test F1-score values.
    """

    TRAIN_LOSS = "train_loss"
    TRAIN_ACCURACY = "train_acc"
    TEST_LOSS = "test_loss"
    TEST_ACCURACY = "test_acc"
    TEST_PRECISION = "test_precision"
    TEST_RECALL = "test_recall"
    TEST_F1_SCORE = "test_f1_score"
    CONFUSION_MATRIX = "test_confmat"


class ModelMode(Enum):
    """Enumeration of model execution modes.

    Attributes:
        TRAIN (str): Mode for training the model.
        TEST (str): Mode for evaluating or testing the model.
    """

    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
