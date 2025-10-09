"""
This module defines routes for data management, including file upload
functionalities within a given project.
"""

import os
import aiofiles
from fastapi import APIRouter, UploadFile, status, Request
from fastapi.responses import JSONResponse
from controllers import DataController
from controllers.base_controller import BaseController
from utils.logging_utils import setup_logger
from models.enums import ResponseSignalEnum


data_router = APIRouter(prefix="/api/v1/data", tags=["api_v1", "data"])


@data_router.post("/upload/{project_id}")
async def upload_data(
    request: Request,
    project_id: int,
    file: UploadFile,
):
    """
    Upload a file to the server for a given project.

    Args:
        request (Request): FastAPI request object containing application state.
        project_id (int): Identifier of the project to which the file belongs.
        file (UploadFile): The file uploaded by the client.

    Returns:
        JSONResponse: A JSON response containing:
            - "signal" (str): Status of the upload (success or failure).
            - "file_name" (str): Name of the uploaded file (if successful).
            - "file_path" (str): Path where the file is saved (if successful).

        On failure, returns an error signal with HTTP 400 status.
    """
    app_settings = request.app.state.settings
    logger = setup_logger(
        logger_name=__name__,
        log_file=__file__,
        log_dir=os.path.join(app_settings.PATH_LOGS, BaseController.get_baseline_root()),
        log_to_console=True,
        use_tqdm=True,
        file_mode="a",
    )
    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file)

    if not is_valid:
        return JSONResponse(
            content={"signal": result_signal}, status_code=status.HTTP_400_BAD_REQUEST
        )

    file_path, file_name = data_controller.generate_unique_filepath(
        original_file_name=file.filename, project_id=project_id  # type: ignore
    )

    logger.info("File Upload in progress...")
    try:
        async with aiofiles.open(file_path, mode="wb") as f:  # type: ignore
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except OSError as exc:
        logger.error("Error while uploading file: %s", exc)
        return JSONResponse(
            content={"signal": ResponseSignalEnum.FILE_UPLOAD_FAILED.value},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    return JSONResponse(
        content={
            "signal": ResponseSignalEnum.FILE_UPLOAD_SUCCESS.value,
            "file_name": file_name,
            "file_path": file_path,
        },
        status_code=status.HTTP_200_OK,
    )


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
