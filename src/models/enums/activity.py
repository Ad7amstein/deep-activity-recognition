"""Enumeration of activity categories and labels for classification."""

import os
from enum import Enum


class ActivityEnum(Enum):
    """Enumeration of activity categories and their corresponding labels.

    Attributes:
        RIGHT_SET_CATEGORY (str): String key for right-side set activity.
        RIGHT_SPIKE_CATEGORY (str): String key for right-side spike activity.
        RIGHT_PASS_CATEGORY (str): String key for right-side pass activity.
        RIGHT_WINPOINT_CATEGORY (str): String key for right-side win point activity.
        LEFT_SET_CATEGORY (str): String key for left-side set activity.
        LEFT_SPIKE_CATEGORY (str): String key for left-side spike activity.
        LEFT_PASS_CATEGORY (str): String key for left-side pass activity.
        LEFT_WINPOINT_CATEGORY (str): String key for left-side win point activity.
        RIGHT_SET_LABEL (int): Numeric label for right-side set activity.
        RIGHT_SPIKE_LABEL (int): Numeric label for right-side spike activity.
        RIGHT_PASS_LABEL (int): Numeric label for right-side pass activity.
        RIGHT_WINPOINT_LABEL (int): Numeric label for right-side win point activity.
        LEFT_SET_LABEL (int): Numeric label for left-side set activity.
        LEFT_SPIKE_LABEL (int): Numeric label for left-side spike activity.
        LEFT_PASS_LABEL (int): Numeric label for left-side pass activity.
        LEFT_WINPOINT_LABEL (int): Numeric label for left-side win point activity.
    """

    RIGHT_SET_CATEGORY = "r_set"
    RIGHT_SPIKE_CATEGORY = "r_spike"
    RIGHT_PASS_CATEGORY = "r-pass"
    RIGHT_WINPOINT_CATEGORY = "r_winpoint"

    LEFT_SET_CATEGORY = "l_set"
    LEFT_SPIKE_CATEGORY = "l-spike"
    LEFT_PASS_CATEGORY = "l-pass"
    LEFT_WINPOINT_CATEGORY = "l_winpoint"

    RIGHT_SET_LABEL = 0
    RIGHT_SPIKE_LABEL = 1
    RIGHT_PASS_LABEL = 2
    RIGHT_WINPOINT_LABEL = 3

    LEFT_SET_LABEL = 4
    LEFT_SPIKE_LABEL = 5
    LEFT_PASS_LABEL = 6
    LEFT_WINPOINT_LABEL = 7


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
