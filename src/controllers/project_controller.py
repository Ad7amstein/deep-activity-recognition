import os
from controllers.base_controller import BaseController


class ProjectController(BaseController):
    def __init__(self) -> None:
        super().__init__()

    def get_project_path(self, project_id: int):
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
