"""
This module defines asset-related enumerations.
"""

import os
from enum import Enum


class AssetTypeEnum(Enum):
    """
    Enumeration of supported asset types.

    Attributes:
        FILE (str): Represents a file asset type.
    """

    FILE = "file"


def main():
    """Entry Point for the Program."""

    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
