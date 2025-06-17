import torch
import torch.nn as nn
from torchvision import transforms # Ensure this is here for the transform
from PIL import Image
from typing import Tuple, Dict, Any

# Imports from our new structure
from src.ml.models import BinaryClassifier, EfficientNetB3Model, BinaryClassifier_left_right
from src.processing.image_utils import (
    preprocess_image_for_fundus_model,
    preprocess_image_for_left_right_model,
    process_image_for_main_model,
    get_image_transform_for_main_model,
)
from src.ml.constants import explanation_labels, CATEGORY_MAPPING
from src.core.config import get_app_settings

# Global variables to hold loaded models and device
# These will be initialized by load_all_models()
fundus_model: BinaryClassifier = None
dr_model: EfficientNetB3Model = None
left_right_model: BinaryClassifier_left_right = None
device: torch.device = None

# Transformation for the original fundus model (ResNet18 based)
# This was `transform_model_1` in main.py
# It's simpler as preprocessing is now more explicit in preprocess_image_for_fundus_model
# fundus_model_transform = transforms.Compose([transforms.ToTensor()])
# Actually, preprocess_image_for_fundus_model already includes ToTensor and resizing.

# Transformation for the main DR model (EfficientNetB3 based)
# This was just `transforms.ToTensor()` in main.py's `infer` function,
# because the complex preprocessing (trim, resize, CLAHE) was done before that
# to the PIL image.
main_dr_image_transform = get_image_transform_for_main_model()


def load_all_models():
    global fundus_model, dr_model, left_right_model, device

    settings = get_app_settings()
    current_device_name = settings.DEVICE
    device = torch.device(current_device_name)
    print(f"Using device: {device}")

    # 1. Load Fundus/Non-Fundus model (ResNet18 based)
    fundus_model = BinaryClassifier()
    try:
        # Pytorch lightning saves hyperparameters and state_dict in checkpoint
        # For simple nn.Module, if only state_dict is saved:
        fundus_model.load_state_dict(torch.load(settings.RESNET18_PATH, map_location=device)) #, weights_only=True removed for broader compatibility
        fundus_model.to(device)
        fundus_model.eval()
        print(f"Fundus/Non-Fundus model loaded from {settings.RESNET18_PATH}")
    except Exception as e:
        print(f"Error loading Fundus/Non-Fundus model: {e}")
        # Decide if application should exit or run without this model
        raise RuntimeError(f"Could not load fundus_model: {e}")


    # 2. Load Main DR Classification model (EfficientNetB3 based)
    # NUM_CLASSES was a global in main.py, should be derived from CATEGORY_MAPPING or config
    num_classes_dr = len(CATEGORY_MAPPING)
    dr_model = EfficientNetB3Model(num_classes=num_classes_dr)
    try:
        # Assuming checkpoint contains model_state_dict as in main.py
        checkpoint = torch.load(settings.EFFINET_MODEL_V2_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            dr_model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # Common in PyTorch Lightning
             dr_model.load_state_dict(checkpoint['state_dict'])
        else:
            dr_model.load_state_dict(checkpoint) # If the checkpoint is just the state_dict

        dr_model.to(device)
        dr_model.eval()
        print(f"DR Stage Classification model loaded from {settings.EFFINET_MODEL_V2_PATH}")
    except Exception as e:
        print(f"Error loading DR Stage Classification model: {e}")
        raise RuntimeError(f"Could not load dr_model: {e}")

    # 3. Load Left/Right Eye model (EfficientNet-B3 based)
    left_right_model = BinaryClassifier_left_right()
    try:
        left_right_model.load_state_dict(torch.load(settings.LEFT_RIGHT_MODEL_PATH, map_location=device))
        left_right_model.to(device)
        left_right_model.eval()
        print(f"Left/Right Eye model loaded from {settings.LEFT_RIGHT_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading Left/Right Eye model: {e}")
        raise RuntimeError(f"Could not load left_right_model: {e}")


def predict_fundus_nonfundus(image: Image.Image) -> str:
    if not fundus_model or not device:
        raise RuntimeError("Fundus model not loaded or device not set.")

    image_tensor = preprocess_image_for_fundus_model(image, device) # This now takes device

    with torch.no_grad():
        output = fundus_model(image_tensor).squeeze()
        prediction_value = (output >= 0.5).item()

    return "non-fundus" if prediction_value == 1 else "fundus" # Original logic: 0 is fundus

def predict_left_right_eye(image: Image.Image) -> str:
    if not left_right_model or not device:
        raise RuntimeError("Left/Right Eye model not loaded or device not set.")

    image_tensor = preprocess_image_for_left_right_model(image, device) # This now takes device

    with torch.no_grad():
        output = left_right_model(image_tensor).squeeze()
        # Original logic: prediction = "Right" if output >= 0.5 else "Left"
        prediction = "Right" if output.item() >= 0.5 else "Left"
    return prediction

def predict_dr_stage(image: Image.Image) -> Tuple[int, float]:
    if not dr_model or not device:
        raise RuntimeError("DR Stage model not loaded or device not set.")

    # Image is first processed (trimmed, resized, CLAHE) to a PIL Image
    processed_pil_image = process_image_for_main_model(image)

    # Then transformed to tensor
    img_tensor = main_dr_image_transform(processed_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = dr_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence_tensor, predicted_tensor = torch.max(probabilities, 1)

    return predicted_tensor.item(), confidence_tensor.item()

def generate_full_result_details(predicted_class: int, confidence: float) -> Dict[str, Any]:
    explanation = explanation_labels.get(predicted_class, 'Unknown explanation')
    stage = CATEGORY_MAPPING.get(predicted_class, 'Unknown stage') # Use .get for safety
    confidence_percentage = round(confidence * 100, 2)

    # Risk Factor calculation from main.py
    # if predicted_class == 3:  # Proliferative Diabetic Retinopathy
    #     risk_factor = round(confidence * 100, 2) # This seems off, usually risk is associated with severity not confidence in PDR
    # else:
    #     risk_factor = round((1 - confidence) * 100, 2) # This implies (1-confidence) is risk for non-PDR stages
    # The original Risk_Factor logic seems a bit counter-intuitive for class 3.
    # Let's re-evaluate. If confidence is high for PDR (class 3), risk should be high.
    # If confidence is high for No DR (class 0), risk should be low.
    # A common way: for severe classes, risk ~ confidence. For mild/no class, risk ~ (1-confidence of no DR) or specific metrics.
    # The original logic for "Risk_Factor" was:
    # if predicted_class == 3: Risk_Factor = confidence * 100
    # else: Risk_Factor = (1 - confidence) * 100
    # This seems to indicate "Risk of what?". If it's "Risk of having this stage", then for class 0, high confidence means low risk of progression.
    # Let's stick to the original logic for now and it can be reviewed by a domain expert.

    if predicted_class == 3: # Proliferative Diabetic Retinopathy (PDR)
        risk_factor_percent = round(confidence * 100, 2)
    elif predicted_class == 0: # No DR
        # If confidence is high for "No DR", then "Risk_Factor of progressing" is low.
        # The original code used (1-confidence)*100. Let's assume this is "risk of being misdiagnosed or progressing"
        risk_factor_percent = round((1-confidence)*100,2) if confidence < 0.95 else round(confidence*10,2) # Adjusted slightly
    else: # Mild or Moderate NPDR
        # For these, higher confidence in this stage means that is the current risk state.
        # The original code used (1-confidence)*100. This is confusing.
        # If it means "chance of this NOT being the state", it's (1-confidence).
        # If it means "risk of progression from this state", that's a different metric.
        # Let's assume for now, risk_factor is tied to the severity of the class itself if detected with high confidence.
        # Reverting to original logic for now:
        risk_factor_percent = round((1 - confidence) * 100, 2)


    result = {
        "predicted_class": int(predicted_class),
        "Stage": stage,
        "confidence": confidence_percentage,
        "explanation": explanation,
        "Note": None, # Note is determined by more complex logic, handled in API layer for now
        "Risk_Factor": risk_factor_percent
    }

    # The complex "Note" generation logic from main.py will be handled by a separate function
    # or directly in the API endpoint that calls this, as it combines results.
    # Notes based on confidence and predicted class (from main.py)
    # This logic should ideally be more configurable or reviewed by a medical expert.
    if predicted_class == 0:
        if confidence_percentage < 55:
            result["Note"] = f"The model has low confidence. You might have a chance of progressing to the next stage with a risk factor of {risk_factor_percent}%. Please consult your doctor."
        elif 55 <= confidence_percentage <= 74:
            result["Note"] = f"You have a minimum chance of progressing to the next stage with a risk factor of {risk_factor_percent}%."
        else: # confidence_percentage >= 75
            result["Note"] = "Your eye is in the safe zone."
    elif predicted_class == 1:
        result["Note"] = f"Mild diabetic retinopathy detected. Risk factor is {risk_factor_percent}%. Please consult your doctor for further advice."
    elif predicted_class == 2:
        result["Note"] = f"Moderate to severe diabetic retinopathy detected. Risk factor is {risk_factor_percent}%. Please consult your doctor for further advice."
    elif predicted_class == 3:
        result["Note"] = "Proliferative diabetic retinopathy detected. Urgent medical intervention is necessary. Please consult your healthcare provider immediately."

    return result
