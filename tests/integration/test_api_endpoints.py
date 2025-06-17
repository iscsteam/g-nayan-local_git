import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from PIL import Image, ImageFile
from io import BytesIO
import os

# Ensure main:app can be imported. This might require setting PYTHONPATH
# or structuring the project so 'main.py' and 'src' are discoverable.
# For TestClient, we usually import the 'app' object from main.
# Let's assume for now that we can import it.
# If not, this test would fail at import time.
from main import app # Import the FastAPI app instance from your main.py

client = TestClient(app)

# Helper to create a dummy image file-like object
def create_dummy_image_bytes(img_format="PNG", size=(100,100), color="red") -> BytesIO:
    img = Image.new("RGB", size, color)
    byte_io = BytesIO()
    img.save(byte_io, format=img_format)
    byte_io.seek(0)
    return byte_io

@pytest.fixture
def dummy_left_image_bytes():
    return create_dummy_image_bytes(color="blue")

@pytest.fixture
def dummy_right_image_bytes():
    return create_dummy_image_bytes(color="green")

# This is a very basic integration test. More comprehensive tests would:
# - Mock specific model prediction values to test different scenarios.
# - Verify the content of data saved to MinIO (if not fully mocked).
# - Test edge cases and error responses more thoroughly.

@patch("src.services.object_store.upload_image_to_minio")
@patch("src.services.object_store.save_results_to_minio")
@patch("src.ml.inference.predict_fundus_nonfundus")
@patch("src.ml.inference.predict_left_right_eye")
@patch("src.ml.inference.predict_dr_stage")
def test_run_inference_endpoint_success(
    mock_predict_dr_stage,
    mock_predict_left_right_eye,
    mock_predict_fundus,
    mock_save_results,
    mock_upload_image,
    dummy_left_image_bytes,
    dummy_right_image_bytes
):
    # Configure mocks to return valid "happy path" values
    mock_upload_image.return_value = "http://minio.example.com/dummy_url.jpg"
    mock_save_results.return_value = None # This function doesn't return anything significant

    mock_predict_fundus.return_value = "fundus" # Both are fundus images

    # Mock left/right eye detection: first call is left, second is right
    mock_predict_left_right_eye.side_effect = ["Left", "Right"]

    # Mock DR stage prediction: (class, confidence)
    # First call for left eye, second for right eye
    mock_predict_dr_stage.side_effect = [(0, 0.95), (1, 0.85)]

    files = {
        "left_image_upload": ("left.png", dummy_left_image_bytes, "image/png"),
        "right_image_upload": ("right.png", dummy_right_image_bytes, "image/png"),
    }
    response = client.post(
        "/api/v1/infer_for_diabetic_retinopathy/upload_images?patient_id=test_patient_001",
        files=files
    )

    assert response.status_code == 200
    response_json = response.json()

    assert "left_eye" in response_json
    assert "right_eye" in response_json
    assert response_json["left_eye"]["predicted_class"] == 0
    assert response_json["left_eye"]["Stage"] == "No Diabetic Retinopathy" # Based on CATEGORY_MAPPING
    assert response_json["right_eye"]["predicted_class"] == 1
    assert response_json["right_eye"]["Stage"] == "Mild Diabetic Retinopathy"

    # Check that mocks were called (basic check)
    assert mock_upload_image.call_count == 2
    assert mock_save_results.call_count == 1
    assert mock_predict_fundus.call_count == 2
    assert mock_predict_left_right_eye.call_count == 2
    assert mock_predict_dr_stage.call_count == 2

# TODO: Add tests for error cases (e.g., non-fundus image, eye mismatch, etc.)
# These would involve configuring mocks to return values that trigger those error conditions.
