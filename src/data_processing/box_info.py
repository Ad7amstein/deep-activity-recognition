"""
This module provides the BoxInfo class for parsing
and storing bounding box information from a formatted string.
"""

import os
from utils.logging_utils import setup_logger
from utils.config_utils import get_settings

app_settings = get_settings()


class BoxInfo:
    """
    Represents bounding box and player information parsed from a single line of data.

    Attributes:
        line (str): The original input line.
        category (str): The category label extracted from the line.
        player_id (int): The player ID.
        box (tuple): A tuple (x1, y1, x2, y2) representing the bounding box coordinates.
        frame_id (int): The frame ID.
        lost (int): Indicator if the object is lost.
        grouping (int): Grouping information.
        generated (int): Indicator if the box was generated.
    """

    def __init__(self, line: str, verbose: bool = False) -> None:
        """
        Initializes the class by parsing a line of box information.

        Args:
            line (str): A string containing box information, expected to be space-separated values.
            verbose (bool): If True, prints logging information. Defaults to False.
        """

        self.verbose = verbose
        self.logger = setup_logger(
            logger_name=__name__,
            log_file=__file__,
            log_dir=os.path.join(app_settings.PATH_LOGS, "app"),
            log_to_console=verbose,
        )
        if self.verbose:
            self.logger.info("Initializing %s Class...", __class__.__name__)
        self.line = line
        all_info = (self.line.strip()).split()
        self.category = all_info.pop()
        (
            self.player_id,
            x1,
            y1,
            x2,
            y2,
            self.frame_id,
            self.lost,
            self.grouping,
            self.generated,
        ) = [int(val) for val in all_info]
        self.box = x1, y1, x2, y2

    @property
    def line(self) -> str:
        """
        Returns the value of the `_line` attribute.

        Returns:
            str: The value stored in the `_line` attribute.
        """

        return self._line

    @line.setter
    def line(self, value: str) -> None:
        """
        Sets the value of the line attribute after validating its format.

        Args:
            value (str): A string expected to contain exactly 10 whitespace-separated values.

        Raises:
            TypeError: If the input is not of a string type.
            ValueError: If the input string does not contain exactly 10 values.
        """

        if not isinstance(value, str):
            raise TypeError("Line must be of type string (str).")

        if len(value.strip().split()) != 10:
            raise ValueError("Line must contain 10 values.")

        self._line = value

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.

        Returns:
            str: A formatted string containing the class name and the value of `self.line`.
        """
        return f'{__class__.__name__}(line="{self.line}")'


def main():
    """Entry Point for the Program."""

    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.\n")
    obj = BoxInfo("0 1002 436 1077 570 3586 0 1 0 digging ", verbose=True)
    print(obj)


if __name__ == "__main__":
    main()
