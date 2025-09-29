"""
Main script for the application.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import base, data, ai
from utils.config_utils import get_settings
from utils.logging_utils import setup_logger

app_settings = get_settings()
logger = setup_logger(
    logger_name=__name__,
    log_file=__file__,
    log_dir=app_settings.PATH_LOGS,
    log_to_console=True,
    file_mode="a",
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=[W0621]
    """
    Asynchronous lifespan context manager for a FastAPI application.

    Args:
        app (fastapi.FastAPI): The FastAPI application instance whose lifespan is
            being managed.

    Yields:
        None: Yields control to the FastAPI application runtime after performing
        initialization. Cleanup is performed after the yield when the application
        is shutting down.
    """
    app.state.app_settings = app_settings

    logger.info("START: Application is started")
    yield
    logger.info("END: Application is ended")


app = FastAPI(lifespan=lifespan)
app.include_router(base.base_router)
app.include_router(ai.ai_router)
app.include_router(data.data_router)


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
