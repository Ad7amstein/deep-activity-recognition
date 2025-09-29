import os
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    baseline_number: int
    image_filename: str


def main():
    """Entry point for the program (for standalone execution)."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
