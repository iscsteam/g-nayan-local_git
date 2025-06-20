from fastapi import APIRouter, File, UploadFile, HTTPException, Request # Removed Depends
# from fastapi.responses import JSONResponse # Removed JSONResponse
from fastapi.concurrency import run_in_threadpool # For MinIO sync operations
from PIL import Image
from PIL import UnidentifiedImageError
from io import BytesIO
from typing import List # Removed Dict, Any

# Application imports
from src.api.schemas import (
    InferenceResponseSchema,
    FeedbackSchema,
    MessageResponse,
    ApiLogEntrySchema,
    DiabeticRetinopathyDbEntrySchema,
    # EyeResultBase, # Not directly used as response model here but by InferenceResponseSchema
)
from src.ml.inference import (
    predict_fundus_nonfundus,
    predict_left_right_eye,
    predict_dr_stage,
    generate_full_result_details,
)
from src.services.object_store import upload_image_to_minio, save_results_to_minio
from src.services.database import data_from_db, save_feedback_to_db
# from src.core.config import get_app_settings # For logger, potentially other settings (currently unused)
# from src.processing.image_utils import fetch_image_from_url # May not be needed if we pass PIL images (currently unused)

# Get a logger instance (can be more sophisticated later from src.core.logger)
import logging
logger = logging.getLogger(__name__)
# BasicConfig should be done once, ideally in lifespan or main.py for the whole app
# logging.basicConfig(level=logging.INFO) # Avoid re-configuring if already done in lifespan

router = APIRouter()

# Dependency to get settings, if needed directly in endpoints
# def get_settings_dependency():
#     return get_app_settings()

@router.post("/infer_for_diabetic_retinopathy/upload_images",
             response_model=InferenceResponseSchema,
             responses={
                 400: {"model": MessageResponse, "description": "Invalid image or image order"},
                 422: {"model": MessageResponse, "description": "Validation Error"},
                 500: {"model": MessageResponse, "description": "Internal Server Error"}
             })
async def run_inference_endpoint(
    patient_id: str,
    left_image_upload: UploadFile = File(..., description="Left eye fundus image"),
    right_image_upload: UploadFile = File(..., description="Right eye fundus image"),
    request: Request = None # For logging client host if needed
) -> InferenceResponseSchema:
    try:
        client_host = request.client.host if request else "unknown"
        logger.info(f"[START] Inference API hit for patient: {patient_id} from {client_host}")

        left_image_bytes = await left_image_upload.read()
        right_image_bytes = await right_image_upload.read()

        try:
            left_pil_image = Image.open(BytesIO(left_image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            logger.warning(f"Cannot identify left image format for patient {patient_id}. Uploaded filename: {left_image_upload.filename}")
            raise HTTPException(status_code=400, detail=f"Left image: Invalid or corrupt image file (e.g., not a JPG/PNG or file is damaged). Please upload a valid image.")
        except IOError as e: # Catch other PIL related IOErrors
            logger.warning(f"IOError opening left image for patient {patient_id} (filename: {left_image_upload.filename}): {e}")
            raise HTTPException(status_code=400, detail=f"Left image: Could not read image file. It might be truncated, corrupt, or an unsupported format.")

        try:
            right_pil_image = Image.open(BytesIO(right_image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            logger.warning(f"Cannot identify right image format for patient {patient_id}. Uploaded filename: {right_image_upload.filename}")
            raise HTTPException(status_code=400, detail=f"Right image: Invalid or corrupt image file (e.g., not a JPG/PNG or file is damaged). Please upload a valid image.")
        except IOError as e: # Catch other PIL related IOErrors
            logger.warning(f"IOError opening right image for patient {patient_id} (filename: {right_image_upload.filename}): {e}")
            raise HTTPException(status_code=400, detail=f"Right image: Could not read image file. It might be truncated, corrupt, or an unsupported format.")

        # Upload images to MinIO (run sync MinIO calls in threadpool)
        # The URLs are for record-keeping/future use; PIL images are used for current inference
        try:
            left_image_url = await run_in_threadpool(
                upload_image_to_minio, patient_id, left_image_bytes, f"{patient_id}_left_original.jpg"
            )
            right_image_url = await run_in_threadpool(
                upload_image_to_minio, patient_id, right_image_bytes, f"{patient_id}_right_original.jpg"
            )
            logger.info(f"Images for patient {patient_id} uploaded to MinIO. Left: {left_image_url}, Right: {right_image_url}")
        except Exception as e:
            logger.error(f"MinIO upload failed for patient {patient_id}: {e}")
            # Decide if this is critical. If URLs are not immediately needed, can proceed.
            # For now, let's treat it as non-critical for the inference flow itself if PIL images are available.
            # raise HTTPException(status_code=500, detail=f"Image upload to MinIO failed: {str(e)}")


        # 1. Fundus/Non-Fundus classification
        left_fundus_pred = predict_fundus_nonfundus(left_pil_image)
        right_fundus_pred = predict_fundus_nonfundus(right_pil_image)

        if left_fundus_pred == "non-fundus" or right_fundus_pred == "non-fundus":
            msg = f"Invalid image type. Left: {left_fundus_pred}, Right: {right_fundus_pred}. Both must be fundus images."
            logger.warning(msg)
            raise HTTPException(status_code=400, detail=msg)

        # 2. Left/Right Eye classification
        left_eye_detected_side = predict_left_right_eye(left_pil_image)
        right_eye_detected_side = predict_left_right_eye(right_pil_image)

        logger.info(f"Patient {patient_id}: Left image detected as {left_eye_detected_side}, Right image detected as {right_eye_detected_side}")

        # Check for mismatches (common error scenarios from original main.py)
        if left_eye_detected_side == "Right" and right_eye_detected_side == "Left":
            msg = "Image order mismatch: Left image slot contains a 'Right' eye, and Right image slot contains a 'Left' eye. Please re-upload correctly."
            logger.warning(f"Patient {patient_id}: {msg}")
            raise HTTPException(status_code=400, detail=msg)
        if left_eye_detected_side == "Right": #  and right_eye_detected_side == "Right" (implicit)
             msg = "Image content mismatch: The image uploaded for the 'Left eye' appears to be a 'Right eye' image. Please upload the correct left eye image."
             logger.warning(f"Patient {patient_id}: {msg}")
             # This case can be more complex: what if right_eye_detected_side is also Right?
             # Original logic had checks for (L,L) and (R,R) too.
             # If Left slot has Right image, and Right slot has Right image --> Missing Left
             if right_eye_detected_side == "Right":
                 msg = "Image content problem: Both uploaded images appear to be 'Right eye' images. A 'Left eye' image is required."
                 logger.warning(f"Patient {patient_id}: {msg}")
                 raise HTTPException(status_code=400, detail=msg)
             # If Left slot has Right image, and Right slot has Left image --> Covered by swap check.
             # Fallthrough for just Left slot having Right image, but Right slot has correct Right image (unlikely if user uploads pair)
             # For simplicity, if left_eye_detected_side is "Right", it's an issue.
             raise HTTPException(status_code=400, detail=msg)


        if right_eye_detected_side == "Left": # and left_eye_detected_side == "Left" (implicit)
            msg = "Image content mismatch: The image uploaded for the 'Right eye' appears to be a 'Left eye' image. Please upload the correct right eye image."
            logger.warning(f"Patient {patient_id}: {msg}")
            if left_eye_detected_side == "Left":
                 msg = "Image content problem: Both uploaded images appear to be 'Left eye' images. A 'Right eye' image is required."
                 logger.warning(f"Patient {patient_id}: {msg}")
                 raise HTTPException(status_code=400, detail=msg)
            raise HTTPException(status_code=400, detail=msg)


        # 3. DR Stage classification
        left_class, left_confidence = predict_dr_stage(left_pil_image)
        right_class, right_confidence = predict_dr_stage(right_pil_image)
        logger.info(f"Patient {patient_id} DR Stage - Left: class={left_class}, conf={left_confidence:.2f}; Right: class={right_class}, conf={right_confidence:.2f}")

        # 4. Generate detailed results
        left_result_details = generate_full_result_details(left_class, left_confidence)
        right_result_details = generate_full_result_details(right_class, right_confidence)

        final_results = {"left_eye": left_result_details, "right_eye": right_result_details}

        # 5. Save results to MinIO (run sync MinIO calls in threadpool)
        try:
            await run_in_threadpool(save_results_to_minio, patient_id, final_results, f"{patient_id}_inference_results.json")
            logger.info(f"Inference results for patient {patient_id} saved to MinIO.")
        except Exception as e:
            logger.error(f"Failed to save results to MinIO for patient {patient_id}: {e}")
            # Non-critical for returning result to user, but should be logged.

        logger.info(f"[SUCCESS] Processed inference for {patient_id}")
        return InferenceResponseSchema(**final_results)

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except ValueError as ve: # Catch specific errors like image opening if not caught by PIL
        logger.error(f"ValueError during inference for patient {patient_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during inference for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/submit_feedback_from_frontend/from_json_to_db", response_model=MessageResponse)
async def submit_feedback_endpoint(feedback: FeedbackSchema) -> MessageResponse:
    try:
        # The FeedbackSchema already validates the structure.
        # save_feedback_to_db expects feedback_data as a dict, patient_id, and email_id.
        # The schema has patient_id and email_id at the top level.
        await run_in_threadpool(
            save_feedback_to_db,
            feedback_data=feedback.model_dump(), # Pass the whole model dict
            patient_id=feedback.patient_id,
            email_id=feedback.email_id
            )
        return MessageResponse(message=f"Feedback and results saved successfully for patient: {feedback.patient_id}")
    except ValueError as ve: # For validation errors not caught by Pydantic (e.g. inside save_feedback_to_db)
        logger.warning(f"Feedback submission validation error for patient {feedback.patient_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error submitting feedback for patient {feedback.patient_id}: {e}", exc_info=True)
        # Check if it's a DB connection error specifically if possible
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.get("/get_data_from_api_logs", response_model=List[ApiLogEntrySchema])
async def get_api_logs_endpoint(skip: int = 0, limit: int = 100) -> List[ApiLogEntrySchema]:
    try:
        # Assuming data_from_db is synchronous
        query = "SELECT * FROM api_logs ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        logs = await run_in_threadpool(data_from_db, query, (limit, skip))
        return logs
    except Exception as e:
        logger.error(f"Error fetching API logs (skip={skip}, limit={limit}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve API logs.")


@router.get("/get_data_from_db", response_model=List[DiabeticRetinopathyDbEntrySchema])
async def get_dr_data_endpoint(skip: int = 0, limit: int = 100) -> List[DiabeticRetinopathyDbEntrySchema]:
    try:
        # Assuming data_from_db is synchronous
        query = "SELECT * FROM diabetic_retinopathy ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        dr_data = await run_in_threadpool(data_from_db, query, (limit, skip))
        return dr_data
    except Exception as e:
        logger.error(f"Error fetching DR data (skip={skip}, limit={limit}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve diabetic retinopathy data.")
