"""
This module defines response signal enumerations.
"""

import os
from enum import Enum


class ResponseSignalEnum(Enum):
    """
    Enumeration of response signals for file and project operations.

    Attributes:
        FILE_VALIDATED_SUCCESS (str): Indicates file validation completed successfully.
        FILE_TYPE_NOT_SUPPORTED (str): Indicates the file type is not supported.
        FILE_SIZE_EXCEEDED (str): Indicates the file size exceeds allowed limits.
        FILE_UPLOAD_SUCCESS (str): Indicates the file was uploaded successfully.
        FILE_UPLOAD_FAILED (str): Indicates the file upload process failed.
        NO_FILES_ERROR (str): Indicates no files were found.
        FILE_ID_ERROR (str): Indicates no file was found with the given ID.
        PROJECT_NOT_FOUND_ERROR (str): Indicates the specified project was not found.
    """

    FILE_VALIDATED_SUCCESS = "file_validated_successfully"
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    NO_FILES_ERROR = "no_files_found"
    FILE_ID_ERROR = "no_file_found_with_this_id"
    PROJECT_NOT_FOUND_ERROR = "project_not_found"


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
