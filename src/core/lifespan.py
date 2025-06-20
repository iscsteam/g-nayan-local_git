from contextlib import asynccontextmanager
from fastapi import FastAPI

# Imports from our application structure
from src.core.config import get_app_settings
from src.ml.inference import load_all_models
from src.services.object_store import initialize_minio_client
from src.services.database import init_db_pool, close_db_pool # Added close_db_pool
from src.core.logger import setup_logging

# Logger setup will be handled by setup_logging() called in lifespan.
# We still need a logger instance for this file.
import logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize logging configuration first
    setup_logging()

    # Startup events
    logger.info("Application startup sequence initiated...")

    # 1. Load Application Settings
    # Although settings are loaded on demand by get_app_settings(),
    # it's good to call it once here if there are any initial checks
    # or to make it available if needed during startup.
    settings = get_app_settings()
    logger.info("Application settings loaded.")

    # 2. Initialize Database Pool
    # init_db_pool expects a config dictionary.
    # We'll pass the Pydantic model directly, or its dict representation.
    # The init_db_pool in services/database.py currently just prints.
    # It will be updated in a later step to set up a real pool.
    try:
        db_config_dict = settings.DB.model_dump()
        init_db_pool(config=db_config_dict) # Pass the dict
        logger.info("Database pool conceptually initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        # Depending on severity, you might want to raise the error and stop app startup
        # raise

    # 3. Initialize MinIO Client
    try:
        initialize_minio_client(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket_name=settings.MINIO_BUCKET_NAME,
            secure=settings.MINIO_SECURE
        )
        logger.info("MinIO client initialized and bucket checked/created.")
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client: {e}")
        # Decide if this is critical enough to stop the app
        # raise

    # 4. Load Machine Learning Models
    try:
        logger.info("Loading machine learning models...")
        load_all_models() # This function now gets device from settings internally
        logger.info("Machine learning models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load machine learning models: {e}")
        # This is likely critical, so consider raising to stop app startup
        raise # Models are critical for this app

    logger.info("Application startup sequence completed.")

    yield

    # Shutdown events
    logger.info("Application shutdown sequence initiated...")
    # Close the database pool
    close_db_pool()
    # Example: If db_pool had a close() method:
    # if db_pool:
    #     db_pool.close()
    #     logger.info("Database pool closed.")
    logger.info("Application shutdown sequence completed.")
