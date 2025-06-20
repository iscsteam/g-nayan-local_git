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

@pytest.fixture
def invalid_image_bytes():
    byte_io = BytesIO(b"this is not an image")
    byte_io.seek(0)
    return byte_io

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


@patch("src.services.object_store.upload_image_to_minio") # Still need to mock this as it's called before fundus check
@patch("src.ml.inference.predict_fundus_nonfundus")
def test_run_inference_endpoint_non_fundus_error(
    mock_predict_fundus,
    mock_upload_image,
    dummy_left_image_bytes,
    dummy_right_image_bytes
):
    mock_upload_image.return_value = "http://minio.example.com/dummy_url.jpg"
    # Simulate one image being non-fundus
    mock_predict_fundus.side_effect = ["fundus", "non-fundus"]

    files = {
        "left_image_upload": ("left.png", dummy_left_image_bytes, "image/png"),
        "right_image_upload": ("right.png", dummy_right_image_bytes, "image/png"),
    }
    response = client.post(
        "/api/v1/infer_for_diabetic_retinopathy/upload_images?patient_id=test_patient_non_fundus",
        files=files
    )

    assert response.status_code == 400
    response_json = response.json()
    assert "detail" in response_json
    assert "non-fundus" in response_json["detail"].lower()
    assert "both must be fundus images" in response_json["detail"].lower()


@patch("src.services.object_store.upload_image_to_minio")
@patch("src.ml.inference.predict_fundus_nonfundus")
@patch("src.ml.inference.predict_left_right_eye")
def test_run_inference_endpoint_eye_mismatch_error_left_is_right(
    mock_predict_left_right_eye,
    mock_predict_fundus,
    mock_upload_image,
    dummy_left_image_bytes,
    dummy_right_image_bytes
):
    mock_upload_image.return_value = "http://minio.example.com/dummy_url.jpg"
    mock_predict_fundus.return_value = "fundus" # Both are fundus

    # Simulate left image detected as "Right", right image correctly as "Right"
    mock_predict_left_right_eye.side_effect = ["Right", "Right"]

    files = {
        "left_image_upload": ("left.png", dummy_left_image_bytes, "image/png"),
        "right_image_upload": ("right.png", dummy_right_image_bytes, "image/png"),
    }
    response = client.post(
        "/api/v1/infer_for_diabetic_retinopathy/upload_images?patient_id=test_patient_eye_mismatch",
        files=files
    )

    assert response.status_code == 400
    response_json = response.json()
    assert "detail" in response_json
    # This specific case: "Both uploaded images appear to be 'Right eye' images"
    assert "both uploaded images appear to be 'right eye' images" in response_json["detail"].lower()

@patch("src.services.object_store.upload_image_to_minio")
@patch("src.ml.inference.predict_fundus_nonfundus")
@patch("src.ml.inference.predict_left_right_eye")
def test_run_inference_endpoint_eye_mismatch_error_swapped(
    mock_predict_left_right_eye,
    mock_predict_fundus,
    mock_upload_image,
    dummy_left_image_bytes,
    dummy_right_image_bytes
):
    mock_upload_image.return_value = "http://minio.example.com/dummy_url.jpg"
    mock_predict_fundus.return_value = "fundus" # Both are fundus

    # Simulate left image detected as "Right", right image detected as "Left" (swapped)
    mock_predict_left_right_eye.side_effect = ["Right", "Left"]

    files = {
        "left_image_upload": ("left.png", dummy_left_image_bytes, "image/png"),
        "right_image_upload": ("right.png", dummy_right_image_bytes, "image/png"),
    }
    response = client.post(
        "/api/v1/infer_for_diabetic_retinopathy/upload_images?patient_id=test_patient_eye_mismatch_swapped",
        files=files
    )

    assert response.status_code == 400
    response_json = response.json()
    assert "detail" in response_json
    assert "image order mismatch" in response_json["detail"].lower()


def test_run_inference_endpoint_invalid_image_file_error(
    invalid_image_bytes, # Use the new fixture for invalid image data
    dummy_right_image_bytes # A valid image for the other one
):
    # No ML model mocks needed as this should fail at Image.open()
    # upload_image_to_minio might be called for the valid image if processing order allows,
    # but the error should occur before all ML steps.
    # For safety, if MinIO is called unconditionally first, it should be mocked.
    # Based on current endpoints.py, Image.open is called before MinIO for the *same* image.

    files = {
        "left_image_upload": ("invalid.txt", invalid_image_bytes, "text/plain"), # Invalid image
        "right_image_upload": ("right.png", dummy_right_image_bytes, "image/png"), # Valid image
    }
    response = client.post(
        "/api/v1/infer_for_diabetic_retinopathy/upload_images?patient_id=test_patient_invalid_file",
        files=files
    )

    assert response.status_code == 400 # Expecting UnidentifiedImageError leading to 400
    response_json = response.json()
    assert "detail" in response_json
    # Check for message related to "Left image: Invalid or corrupt image file"
    assert "left image: invalid or corrupt image file" in response_json["detail"].lower()

@patch("src.services.object_store.upload_image_to_minio")
def test_run_inference_endpoint_ioerror_image_file_error(
    mock_upload_image, # Mock MinIO as it might be called for the first valid image
    dummy_left_image_bytes,
    dummy_right_image_bytes
):
    # This test simulates an IOError that is not UnidentifiedImageError
    # It's harder to trigger deterministically without deeper PIL mocking.
    # Instead, we'll mock Image.open itself for one of the images to raise IOError.

    mock_upload_image.return_value = "http://minio.example.com/dummy_url.jpg"

    files = {
        "left_image_upload": ("left.png", dummy_left_image_bytes, "image/png"),
        "right_image_upload": ("right.png", dummy_right_image_bytes, "image/png"), # This one will raise IOError
    }

    with patch("PIL.Image.open") as mock_image_open:
        # First call (left_pil_image) is successful
        mock_left_pil = Image.new("RGB", (60, 30), color="red")

        # Second call (right_pil_image) raises IOError
        mock_image_open.side_effect = [mock_left_pil, IOError("Simulated IOError")]

        response = client.post(
            "/api/v1/infer_for_diabetic_retinopathy/upload_images?patient_id=test_patient_ioerror",
            files=files
        )

    assert response.status_code == 400
    response_json = response.json()
    assert "detail" in response_json
    assert "right image: could not read image file" in response_json["detail"].lower()

# TODO: Add tests for other error cases from the endpoint logic.
# Example:
# - What if MinIO upload fails? (Currently logged but doesn't cause HTTP error)
# - What if DR stage prediction returns unexpected values? (Less likely with current setup)
# - Test pagination on /get_data_from_api_logs and /get_data_from_db
# - Test /submit_feedback_from_frontend/from_json_to_db endpoint success & errors.

# Placeholder for pagination tests
def test_get_api_logs_pagination():
    # This test would require mocking data_from_db from src.services.database
    # and verifying the query and parameters (limit, skip) passed to it.
    pass

def test_get_dr_data_pagination():
    # Similar to above, mock data_from_db.
    pass

# Placeholder for feedback endpoint test
def test_submit_feedback_success():
    # Mock save_feedback_to_db
    pass
