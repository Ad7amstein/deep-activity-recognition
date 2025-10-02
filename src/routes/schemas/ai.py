"""
This module defines the request schema for inference operations using Pydantic.
"""

import os
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """
    Schema for an inference request.

    Attributes:
        baseline_number (int): Identifier for the baseline model to use for inference.
        image_filename (str): Path or filename of the input image for inference.
    """

    baseline_number: int
    image_filename: str


def main():
    """Entry point for the program (for standalone execution)."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
