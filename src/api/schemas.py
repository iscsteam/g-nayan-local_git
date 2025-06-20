from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional # Removed Dict

class EyeResultBase(BaseModel):
    predicted_class: int
    Stage: str
    confidence: float = Field(..., ge=0, le=100) # Assuming confidence is 0-100
    explanation: str
    Note: Optional[str] = None
    Risk_Factor: float

class EyeResultWithFeedback(EyeResultBase):
    review: Optional[str] = None
    feedback: Optional[str] = None
    doctors_diagnosis: Optional[str] = None

class FeedbackSchema(BaseModel):
    patient_id: Optional[str] = "default_patient_id"
    email_id: Optional[EmailStr] = None
    left_eye: EyeResultWithFeedback
    right_eye: EyeResultWithFeedback

class InferenceResponseSchema(BaseModel):
    left_eye: EyeResultBase
    right_eye: EyeResultBase

# Generic message response for errors or simple messages
class MessageResponse(BaseModel):
    message: str

# Schema for individual image prediction (fundus/non-fundus, left/right)
# These are not directly part of request/response bodies but good for internal use
class ImageClassificationPrediction(BaseModel):
    prediction: str # e.g., "fundus", "non-fundus", "Left", "Right"
    # confidence: Optional[float] = None # If confidence is available


# Schema for API Log entry
class ApiLogEntrySchema(BaseModel):
    timestamp: datetime # Assuming it's fetched as datetime object from DB
    level: str
    message: str

    class Config:
        from_attributes = True # Compatibility if using ORM results directly (orm_mode deprecated)

# Schema for Diabetic Retinopathy DB entry
class DiabeticRetinopathyDbEntrySchema(BaseModel):
    Patient_ID: str
    Predicted_Class: Optional[int] = None # Made optional if they can be null in DB
    Stage: Optional[str] = None
    Confidence: Optional[float] = None
    Explanation: Optional[str] = None
    Note: Optional[str] = None
    Risk_Factor: Optional[float] = None
    Review: Optional[str] = None
    Feedback: Optional[str] = None
    Doctors_Diagnosis: Optional[str] = None
    email_id: Optional[EmailStr] = None # Already defined EmailStr
    timestamp: Optional[datetime] = None # Assuming it's fetched as datetime

    class Config:
        from_attributes = True # orm_mode deprecated
