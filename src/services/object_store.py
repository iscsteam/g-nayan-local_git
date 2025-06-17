import io
import json
from datetime import timedelta
from minio import Minio
from minio.error import S3Error # Keep S3Error for specific error handling if needed
from typing import Any # For BytesIO type hint
from io import BytesIO # Explicit import for clarity

# Global Minio client and bucket name, to be initialized in lifespan from config
minio_client: Minio = None # type: ignore
BUCKET_NAME: str = None # type: ignore

# Placeholder for initializing client and bucket name from config
# This will be called from the lifespan event
def initialize_minio_client(endpoint: str, access_key: str, secret_key: str, bucket_name: str, secure: bool = False):
    global minio_client, BUCKET_NAME
    minio_client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
    BUCKET_NAME = bucket_name

    try:
        found = minio_client.bucket_exists(BUCKET_NAME)
        if not found:
            minio_client.make_bucket(BUCKET_NAME)
            print(f"Bucket '{BUCKET_NAME}' created.")
        else:
            print(f"Bucket '{BUCKET_NAME}' already exists.")
    except S3Error as e:
        print(f"Error initializing MinIO bucket: {e}")
        # Potentially raise an error here to stop app startup if MinIO is critical
        raise

def upload_image_to_minio(patient_id: str, image_bytes: bytes, image_name: str) -> str:
    if not minio_client or not BUCKET_NAME:
        raise ConnectionError("MinIO client not initialized. Call initialize_minio_client first.")

    object_name = f"{patient_id}/{image_name}"

    try:
        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=BytesIO(image_bytes), # Use BytesIO here
            length=len(image_bytes),
            content_type='image/jpeg' # Assuming jpeg, adjust if other types are used
        )
        image_url = minio_client.presigned_get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            expires=timedelta(hours=1) # Configurable expiry?
        )
        return image_url
    except S3Error as e:
        print(f"MinIO S3Error during image upload: {e}")
        # Consider raising a custom exception or HTTPException if in API context
        raise Exception(f"Failed to upload image {image_name} to MinIO. {e}")


def save_results_to_minio(patient_id: str, results: dict, filename: str = 'results.json') -> None:
    if not minio_client or not BUCKET_NAME:
        raise ConnectionError("MinIO client not initialized. Call initialize_minio_client first.")

    object_name = f"{patient_id}/{filename}"
    json_data = json.dumps(results, indent=4)
    json_bytes = json_data.encode('utf-8')
    json_stream = BytesIO(json_bytes)

    try:
        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=json_stream,
            length=len(json_bytes),
            content_type="application/json"
        )
        print(f"Successfully saved results to {object_name} in bucket {BUCKET_NAME}")
    except S3Error as e:
        print(f"MinIO S3Error during results save: {e}")
        # Consider raising a custom exception
        raise Exception(f"Failed to save results for {patient_id} to MinIO. {e}")
