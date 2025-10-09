"""
This module defines standardized path prefixes used for naming conventions in experiments,
baselines, and stages.
"""

import os
from enum import Enum


class PathPrefixEnum(Enum):
    """Enumeration of path prefixes for different experiment components.

    Attributes:
        baseline_prefix (str): Prefix used for baseline-related paths.
        stage_prefix (str): Prefix used for stage-related paths.
        experiment_prefix (str): Prefix used for experiment-related paths.
    """

    baseline_prefix = "baseline_"
    stage_prefix = "stage_"
    experiment_prefix = "exp_"


def main():
    """Entry point for the program.

    Prints a simple message indicating that the module was executed directly.
    """
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
