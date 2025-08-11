"""Path utilities for validating existing files and directories."""

import os


def check_dir(dir_path: str) -> None:
    """
    Checks if the provided path is a valid directory.
    Args:
        dir_path (str): The path to the directory to check.
    Raises:
        TypeError: If `dir_path` is not a string.
        ValueError: If `dir_path` does not exist or is not a directory.
    """

    if not isinstance(dir_path, str):
        raise TypeError(f"directory must be a string, got {type(dir_path).__name__}")
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} does not exist or is not a directory.")


def check_file(file_path: str) -> None:
    """
    Validates that the provided file_path path is a string and points to an existing file.
    Args:
        file_path (str): Path to the file.
    Raises:
        TypeError: If `file_path` is not a string.
        ValueError: If `file_path` does not exist or is not a file.
    """

    if not isinstance(file_path, str):
        raise TypeError(f"file must be a string, got {type(file_path).__name__}")
    if not os.path.isfile(file_path):
        raise ValueError(f"{file_path} does not exist or is not a file.")


def check_path(path: str, path_type: str) -> None:
    """
    Checks the validity of a given path based on the specified path type.
    Args:
        path (str): The path to check.
        path_type (str): The type of the path. Must be either "file" or "dir".
    Raises:
        Appropriate exceptions may be raised by `check_file` or `check_dir` if the path is invalid.
    """

    if path_type == "file":
        check_file(path)
    elif path_type == "dir":
        check_dir(path)
    else:
        print("[WARNING] Wrong path type is provided. Available path types (file, dir)")


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
