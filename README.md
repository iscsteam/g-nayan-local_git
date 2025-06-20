# Diabetic Retinopathy Detection API

This project provides a FastAPI-based API for Diabetic Retinopathy (DR) detection from fundus images. It uses machine learning models to classify images, determine DR stage, and identify left/right eyes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started / Local Development Setup](#getting-started--local-development-setup)
  - [Cloning the Repository](#cloning-the-repository)
  - [Python Environment Setup](#python-environment-setup)
  - [Environment Variables](#environment-variables)
- [Running the Application Locally](#running-the-application-locally)
- [API Documentation](#api-documentation)
  - [Overview](#overview)
  - [Interactive Docs](#interactive-docs)
  - [Key Endpoints](#key-endpoints)
    - [Inference Endpoint](#inference-endpoint)
    - [Feedback Endpoint](#feedback-endpoint)
    - [Data Retrieval Endpoints](#data-retrieval-endpoints)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
  - [Docker Compose](#docker-compose)
  - [Database Setup](#database-setup)
- [Monitoring](#monitoring)
  - [Prometheus](#prometheus)
  - [cAdvisor](#cadvisor)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

## Getting Started / Local Development Setup

### Cloning the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Python Environment Setup

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Environment Variables

The application uses environment variables for configuration. These are typically stored in a `.env` file in the project root. Create a `.env` file by copying the example:

```bash
cp .env.example .env
```

Review and update the variables in `.env` as needed. Key variables include:

- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`
- `FASTAPI_SECRET_KEY`
- Paths to ML models (`RESNET18_PATH`, `EFFINET_MODEL_V2_PATH`, `LEFT_RIGHT_MODEL_PATH`)

## Running the Application Locally

Once the environment is set up and dependencies are installed, you can run the FastAPI application using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at `http://localhost:8000`.

## API Documentation

### Overview

The API provides endpoints for:
- Running diabetic retinopathy inference on uploaded eye images.
- Submitting feedback on inference results.
- Retrieving API logs and DR data from the database.

All API endpoints are prefixed with `/api/v1`.

### Interactive Docs

FastAPI provides interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to explore and test the API endpoints directly from your browser.

### Key Endpoints

#### Inference Endpoint

- **POST** `/api/v1/infer_for_diabetic_retinopathy/upload_images`
  - **Description**: Uploads left and right eye fundus images for a patient and returns the DR stage classification, confidence, and other details.
  - **Request**: `multipart/form-data` with `patient_id` (query parameter), `left_image_upload` (file), and `right_image_upload` (file).
  - **Example Response** (`InferenceResponseSchema`):
    ```json
    {
      "left_eye": {
        "predicted_class": 0,
        "Stage": "No Diabetic Retinopathy",
        "confidence": 95.5,
        "explanation": "No signs of diabetic retinopathy detected.",
        "Note": "Your eye is in the safe zone.",
        "Risk_Factor": 5.0
      },
      "right_eye": {
        "predicted_class": 1,
        "Stage": "Mild Diabetic Retinopathy",
        "confidence": 85.0,
        "explanation": "Mild signs of diabetic retinopathy detected.",
        "Note": "Mild diabetic retinopathy detected. Risk factor is 15.0%. Please consult your doctor for further advice.",
        "Risk_Factor": 15.0
      }
    }
    ```

#### Feedback Endpoint

- **POST** `/api/v1/submit_feedback_from_frontend/from_json_to_db`
  - **Description**: Submits feedback and doctor's diagnosis for a previously processed patient's results.
  - **Request Body**: `FeedbackSchema` (JSON)
  - **Example Response** (`MessageResponse`):
    ```json
    {
      "message": "Feedback and results saved successfully for patient: example_patient_01"
    }
    ```

#### Data Retrieval Endpoints

- **GET** `/api/v1/get_data_from_api_logs`
  - **Description**: Retrieves API logs with pagination (`skip`, `limit`).
  - **Response**: List of `ApiLogEntrySchema`.
- **GET** `/api/v1/get_data_from_db`
  - **Description**: Retrieves diabetic retinopathy data entries with pagination (`skip`, `limit`).
  - **Response**: List of `DiabeticRetinopathyDbEntrySchema`.

## Running Tests

To run the automated tests (unit and integration):

```bash
pytest
```

Ensure you have any necessary test dependencies installed (usually part of `requirements.txt` or a separate `requirements-dev.txt`).

## Deployment

### Docker Compose

The application is designed to be deployed using Docker Compose. This setup includes the FastAPI application, MySQL database, MinIO object storage, Prometheus for monitoring, and Grafana for visualization.

To build and run the services:

```bash
docker-compose up --build
```

For potentially faster builds, especially if using BuildKit, you can set `COMPOSE_BAKE=true`:

```bash
COMPOSE_BAKE=true docker-compose up --build
```

The `docker-compose.yml` file defines the services:
- `fastapi`: The main application.
- `mysql`: MySQL database instance.
- `minio`: MinIO object storage.
- `prometheus`: Prometheus monitoring server.
- `grafana`: Grafana dashboard server.
- `cadvisor`: Container advisor for resource usage metrics.

All services are connected via the `app-network`. Persistent data for MySQL, MinIO, and Grafana is stored in Docker volumes.

### Database Setup

The MySQL service in Docker Compose will initialize based on the environment variables provided in your `.env` file (`MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_ROOT_PASSWORD`).

The necessary tables (`diabetic_retinopathy` for patient records and `api_logs` for logging) need to be created in the MySQL database. You can connect to the MySQL container to create them:

```bash
docker exec -it mysql-container mysql -u root -p
```
Enter the `MYSQL_ROOT_PASSWORD` when prompted.

Then, execute the following SQL commands:

```sql
CREATE TABLE diabetic_retinopathy (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Patient_ID VARCHAR(100) NOT NULL,
    Predicted_Class VARCHAR(50),
    Stage VARCHAR(50),
    Confidence FLOAT,
    Explanation TEXT,
    Note TEXT,
    Risk_Factor TEXT,
    Review TEXT,
    Feedback TEXT,
    Doctors_Diagnosis TEXT,
    email_id VARCHAR(255),
    timestamp DATETIME
);

CREATE TABLE api_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    level VARCHAR(10),
    message TEXT
);
```

## Monitoring

### Prometheus

Prometheus is configured to scrape metrics from the FastAPI application and cAdvisor.
- **FastAPI metrics endpoint**: `http://localhost:8000/metrics` (accessible from within the Docker network by Prometheus as `fastapi:8000/metrics`).
- **Prometheus UI**: `http://localhost:6001`

**Example Prometheus Queries:**

- **API Request Rate (per minute) for a specific endpoint:**
  `rate(fastapi_http_requests_total{handler="/api/v1/infer_for_diabetic_retinopathy/upload_images"}[1m])`
- **Total number of API hits for a specific endpoint:**
  `sum(fastapi_http_requests_total{handler="/api/v1/get_data_from_api_logs"})`
  (Note: `http_requests_total` might be the metric name from your instrumentator if different from `fastapi_http_requests_total`)
- **Container CPU Usage (percentage, 1-minute average for `fastapi` container):**
  `rate(container_cpu_usage_seconds_total{container_name="fastapi_local_dr"}[1m]) * 100`
  (Note: The original `deployment.md` query `rate(container_cpu_usage_seconds_total{container="fastapi"}[1m]) * 100` might need `container_name` instead of `container` depending on your cAdvisor/Prometheus setup. The `docker-compose.yml` names the service `fastapi` and container `fastapi_local_dr`.)

### Grafana

Grafana is available at `http://localhost:6002`. You can configure it to use Prometheus as a data source and build dashboards to visualize metrics. Default login is often `admin/admin`.

### cAdvisor

cAdvisor (Container Advisor) provides resource usage and performance metrics for Docker containers. It's accessible at `http://localhost:8080`. Prometheus scrapes metrics from cAdvisor.

## Project Structure

```
.
├── .env.example        # Example environment variables
├── docker-compose.yml  # Docker Compose configuration
├── Dockerfile          # Dockerfile for the FastAPI application
├── deployment.md       # Original deployment notes
├── main.py             # FastAPI application entry point
├── model_paths/        # Directory for ML model files (ensure actual models are here or volume mounted)
├── prometheus.yml      # Prometheus configuration
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── src/                # Source code
│   ├── api/            # API related modules (endpoints, schemas)
│   ├── core/           # Core application logic (config, lifespan, logger)
│   ├── ml/             # Machine learning models and inference logic
│   ├── processing/     # Image processing utilities
│   └── services/       # External services (database, object store)
└── tests/              # Automated tests
    ├── integration/
    └── unit/
```

## Troubleshooting

- **`.env` file not found/loaded**: Ensure you have copied `.env.example` to `.env` and that Docker Compose (or Uvicorn locally) is picking it up. For Docker Compose, variables in `.env` are automatically available to the services.
- **Model file not found**: Verify that the paths in your `.env` file for `RESNET18_PATH`, `EFFINET_MODEL_V2_PATH`, and `LEFT_RIGHT_MODEL_PATH` correctly point to the model files. When using Docker, ensure these models are either copied into the image via the `Dockerfile` or mounted as volumes in `docker-compose.yml`. The current `Dockerfile` and `docker-compose.yml` imply models should be in a `model_paths` directory within the build context or application directory.
- **Database connection issues**: Check MySQL container logs (`docker logs mysql-container`). Ensure the environment variables for MySQL connection in your `.env` file match those used to configure the MySQL service in `docker-compose.yml`.
- **MinIO connection issues**: Check MinIO container logs (`docker logs minio_container`). Verify MinIO credentials and endpoint in `.env`.

```
