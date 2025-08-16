import os
from enum import Enum


class ActivityEnum(Enum):
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
