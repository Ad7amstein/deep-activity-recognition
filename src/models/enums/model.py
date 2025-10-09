"""Enumeration of model training and evaluation result keys."""

import os
from enum import Enum


class ModelResults(Enum):
    """
    String enumeration of training and evaluation result keys.

    Attributes:
        TRAIN_LOSS (str): Key for training loss values.
        TRAIN_ACCURACY (str): Key for training accuracy values.
        TEST_LOSS (str): Key for test loss values.
        TEST_ACCURACY (str): Key for test accuracy values.
        TEST_PRECISION (str): Key for test precision values.
        TEST_RECALL (str): Key for test recall values.
        TEST_F1_SCORE (str): Key for test F1-score values.
        CONFUSION_MATRIX (str): Key for test confusion matrix values.
        TOTAL_TRAIN_TIME (str): Key for total training time.
        TIME_PER_EPOCH (str): Key for average time per epoch.
    """

    TRAIN_LOSS = "train_loss"
    TRAIN_ACCURACY = "train_acc"
    TEST_LOSS = "test_loss"
    TEST_ACCURACY = "test_acc"
    TEST_PRECISION = "test_precision"
    TEST_RECALL = "test_recall"
    TEST_F1_SCORE = "test_f1_score"
    CONFUSION_MATRIX = "test_confmat"
    TOTAL_TRAIN_TIME = "total_train_time"
    TIME_PER_EPOCH = "time_per_epoch"


class ModelMode(Enum):
    """
    Enumeration of model operational modes.

    Attributes:
        TRAIN (str): Mode for training the model.
        VALIDATION (str): Mode for validating the model during training.
        TEST (str): Mode for evaluating the model on the test dataset.
        INFERENCE (str): Mode for running inference on unseen data.
    """

    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"
    INFERENCE = "inference"


class ModelBaseline(Enum):
    """
    Enumeration of baseline model identifiers.

    Attributes:
        BASELINE1 (int): Identifier for baseline model 1.
        BASELINE3 (int): Identifier for baseline model 3.
        BASELINE4 (int): Identifier for baseline model 4.
        BASELINE5 (int): Identifier for baseline model 5.
        BASELINE6 (int): Identifier for baseline model 6.
        BASELINE7 (int): Identifier for baseline model 7.
        BASELINE8 (int): Identifier for baseline model 8.
    """

    BASELINE1 = 1
    BASELINE3 = 3
    BASELINE4 = 4
    BASELINE5 = 5
    BASELINE6 = 6
    BASELINE7 = 7
    BASELINE8 = 8

class ModelStage(Enum):
    STAGE1 = 1
    STAGE2 = 2


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
