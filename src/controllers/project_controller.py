"""
Project controller module for managing project-related file directories.
"""

import os
from controllers.base_controller import BaseController


class ProjectController(BaseController):
    """Controller for managing project-specific directories.

    Attributes:
        Inherits all attributes from `BaseController`, including:
            files_dir (str): Base directory path for project files.
    """

    def __init__(self) -> None:
        """Initialize the ProjectController."""

        super().__init__()

    def get_project_path(self, project_id: int) -> str:
        """Get or create the directory path for a given project.

        Args:
            project_id (int): The unique identifier of the project.

        Returns:
            str: The absolute path to the project directory.
        """

        project_dir = os.path.join(self.files_dir, str(project_id))
        os.makedirs(project_dir, exist_ok=True)

        return project_dir


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
