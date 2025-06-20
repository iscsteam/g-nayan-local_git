import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.sessions import SessionMiddleware # If still needed
from prometheus_fastapi_instrumentator import Instrumentator

# Application imports
from src.api.endpoints import router as api_router
from src.core.lifespan import lifespan # Manages startup/shutdown
# from src.core.config import get_app_settings # To get settings if needed for main app config (currently unused)

# Get app settings if needed for app initialization (e.g. title, version from config)
# settings = get_app_settings() # Uncomment if used for app = FastAPI(...)

app = FastAPI(
    title="FastAPI for DR Refactored",
    description="Diabetic Retinopathy Detection API - Refactored Structure",
    version="2.0.0",
    lifespan=lifespan # Attach the lifespan manager
)

# Middleware
# Consider if SessionMiddleware is truly needed for your API.
# If not, it can be removed. It adds overhead and is for stateful sessions.
# app.add_middleware(SessionMiddleware, secret_key=get_app_settings().FASTAPI_SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Configure this more restrictively for production
    allow_credentials=True, # Often False for public APIs if not using cookies/auth headers for CORS
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router, prefix="/api/v1") # Added a common prefix like /api/v1

# Prometheus Instrumentation
# Define which paths to instrument (copied from original main.py)
# These paths should now reflect the new /api/v1 prefix
ALLOWED_PATHS_FOR_INSTRUMENTATION = {
    "/api/v1/get_data_from_api_logs",
    "/api/v1/get_data_from_db",
    "/api/v1/infer_for_diabetic_retinopathy/upload_images"
    # Note: /submit_feedback_from_frontend/from_json_to_db was not in original list
}
METRICS_PATH = "/metrics"

def should_instrument_callback(request: Request) -> bool:
    # Do not instrument /metrics endpoint itself
    if request.url.path == METRICS_PATH:
        return False
    # Instrument allowed paths
    if request.url.path in ALLOWED_PATHS_FOR_INSTRUMENTATION:
        return True
    # Do not instrument any other paths by default
    return False

instrumentator = Instrumentator(
    should_instrument_handler_middleware=True, # Default, explicit
    # Other options can be explored if needed
)
instrumentator.instrument(
    app,
    should_instrument_hook=should_instrument_callback
).expose(app, endpoint=METRICS_PATH, include_in_schema=True)


# Root path for basic check
@app.get("/", tags=["Root"])
async def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Diabetic Retinopathy Detection API!"}


if __name__ == "__main__":
    # For development, Uvicorn can be run this way.
    # For production, prefer using Gunicorn with Uvicorn workers.
    # Example: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
