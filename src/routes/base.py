"""
Base routes module for the deep-activity-recognition.
"""

import os
from fastapi import APIRouter, Request

base_router = APIRouter(prefix="/api/v1", tags=["api_v1"])


@base_router.get("/")
async def welcome(request: Request):
    """
    welcome route and entry point for testing connection.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        dict: A dictionary containing:
            app_name (str): The application name from settings.
            app_version (str): The application version from settings.
    """

    return {
        "app_name": request.app.state.settings.APP_NAME,
        "app_version": request.app.state.settings.APP_VERSION,
    }


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
