# Use the official Python image as a base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to /app
COPY . .
# **Copy the .env file** into the container (optional)
COPY .env .


# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
