# ---- Build Stage ----
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build-time dependencies if any (e.g., compilers, build tools)
# For this project, pip itself is the main build tool for Python packages.

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy the rest of the application code
COPY src/ src/
COPY main.py .
# Copy model_paths if they are part of the application build context
# and not mounted or downloaded at runtime.
COPY model_paths/ model_paths/


# ---- Final Stage ----
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create a non-root user and group
# This needs to be done before copying files from builder if we want them owned by this user initially,
# or chown them after copying.
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Install system dependencies (runtime)
# These were identified from the original Dockerfile
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code from the builder stage
# Ensure the WORKDIR /app exists and appuser can write to it if api_hits.log is created there.
# The WORKDIR instruction creates the directory if it doesn't exist, owned by root.
# So, we copy code, then chown, then switch user.
COPY --from=builder /app/src/ src/
COPY --from=builder /app/main.py .
COPY --from=builder /app/model_paths/ model_paths/ # Ensure models are copied

# Set ownership for the app directory to the non-root user
# This allows the app to run and write logs (e.g., api_hits.log in /app)
RUN chown -R appuser:appgroup /app && chmod -R 755 /app
# Any other directories that need to be writable by appuser should be handled here.
# For instance, if api_hits.log was in /var/log/app_log.txt, that would need setup.
# Since it's relative "api_hits.log", it will be in /app.

# Switch to the non-root user
USER appuser

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
