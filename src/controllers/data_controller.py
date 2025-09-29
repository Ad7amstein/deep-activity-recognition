"""
Data controller module for handling file uploads, validation, and storage.
"""

import os
import re
from typing import Tuple
from fastapi import UploadFile
from controllers.base_controller import BaseController
from controllers.project_controller import ProjectController
from models.enums import ResponseSignalEnum


class DataController(BaseController):
    """Controller for managing file upload validation and storage paths.

    Attributes:
        size_scale (int): Scaling factor for file size validation (bytes in one MB).
    """

    def __init__(self) -> None:
        """Initialize the DataController."""

        super().__init__()
        self.size_scale = 1024 * 1024

    def validate_uploaded_file(self, file: UploadFile) -> Tuple[bool, str]:
        """Validate the uploaded file against allowed types and maximum size.

        Args:
            file (UploadFile): The uploaded file to validate.

        Returns:
            tuple[bool, str]:
                - A boolean indicating whether the file is valid.
                - A response signal message describing the result.
        """

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignalEnum.FILE_TYPE_NOT_SUPPORTED.value
        if (
            file.size is not None
            and file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale
        ):
            return False, ResponseSignalEnum.FILE_SIZE_EXCEEDED.value

        return True, ResponseSignalEnum.FILE_UPLOAD_SUCCESS.value

    def clean_file_name(self, file_name: str) -> str:
        """Clean and sanitize a file name.

        Removes invalid characters and replaces spaces with underscores.

        Args:
            file_name (str): The original file name.

        Returns:
            str: The cleaned file name.
        """

        cleaned_file_name = re.sub(r"[^\w.]", "", file_name)
        cleaned_file_name = cleaned_file_name.replace(" ", "_")
        return cleaned_file_name

    def generate_unique_filepath(
        self, original_file_name: str, project_id: int
    ) -> Tuple[str, str]:
        """Generate a unique file path for the uploaded file within a project directory.

        Args:
            original_file_name (str): The original file name of the uploaded file.
            project_id (int): The ID of the project to associate the file with.

        Returns:
            tuple[str, str]:
                - The full path of the newly generated file.
                - The new unique file name.
        """

        project_dir = ProjectController().get_project_path(project_id)
        cleaned_file_name = self.clean_file_name(original_file_name)

        new_file_path = ""
        while True:
            random_key = self.generate_random_string()
            new_filename = f"{random_key}_{cleaned_file_name}"
            new_file_path = os.path.join(project_dir, new_filename)
            if not os.path.exists(new_file_path):
                break

        return new_file_path, new_filename


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
