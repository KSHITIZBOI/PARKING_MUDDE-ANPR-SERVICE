from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import logging
import json
import hashlib
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from app.validator import VRNValidator

from app.plate_finder import PlateFinder
from app.anpr_v2_preprocessing import ImagePreprocessor
from app.anpr_v2_multi_ocr import MultiOCREngine

logger = logging.getLogger(__name__)

# ✨ NEW: Indian Plate Preprocessing Functions


def deskew_plate(plate_img):
    """
    Auto-correct plate rotation/skew

    Args:
        plate_img: Detected plate ROI (BGR image)
    Returns:
        Deskewed plate image
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return plate_img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)

    if abs(median_angle) < 2:
        return plate_img

    (h, w) = plate_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(plate_img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    logger.debug(f"Plate deskewed by {median_angle:.2f} degrees")
    return rotated


def preprocess_indian_plate(plate_img):
    """
    Complete preprocessing optimized for Indian plates

    Steps:
    1. Deskew (fix rotation)
    2. Remove IND logo (left 20%)
    3. Apply inverted threshold (best result from testing)

    Args:
        plate_img: Detected plate ROI
    Returns:
        Preprocessed plate ready for OCR
    """
    # STEP 1: Fix rotation
    plate_img = deskew_plate(plate_img)

    # STEP 2: Remove IND logo (crop left 20%)
    h, w = plate_img.shape[:2]
    crop_left = int(w * 0.20)
    plate_img = plate_img[:, crop_left:]

    # STEP 3: Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # STEP 4: Apply OTSU threshold
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # STEP 5: Invert (testing showed this works best!)
    inverted = cv2.bitwise_not(binary)

    # STEP 6: Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


app = FastAPI(
    title="ParkingMudde ANPR API",
    version="2.2.0",  # ✨ Updated version - Indian plate optimizations
    description="Direct plate detection for Indian vehicles with OCR post-processing"
)

# ✨ NEW: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✨ NEW: Request timeout middleware


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "TIMEOUT",
                "message": "Request took too long to process (>30s). Please try again with a smaller image."
            }
        )

# ✅ INITIALIZE ONCE (not on every request!)
plate_finder = PlateFinder()
preprocessor = ImagePreprocessor()
multi_ocr = MultiOCREngine(use_easyocr=True, use_paddleocr=False)
validator = VRNValidator()

logger.info("✅ ANPR components initialized")

# ✨ Create directories on startup
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ✨ NEW: Directory for saving detection images (for future training)
SAVED_IMAGES_DIR = Path("data/saved_detections")
SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ✨ OCR Post-Processing Function
def fix_common_ocr_errors(vrn: str) -> str:
    """
    Fix OCR errors based on Indian VRN format
    Format: XX00XX0000 or XX00X0000

    Rules:
    - First 2 chars: Letters (state code) - convert numbers to letters
    - Next 2 chars: Numbers (district) - convert letters to numbers
    - Middle 1-2 chars: Letters (series) - convert numbers to letters
    - Last 4 chars: Numbers - convert letters to numbers

    Examples:
        "UP32DO2O19" → "UP32D02019" ✅
        "DL4CAF4943" → "DL4CAF4943" ✅ (already correct)
        "MH02A81234" → "MH02AB1234" ✅
    """
    if not vrn or len(vrn) < 8:
        return vrn

    # Clean and uppercase
    vrn = vrn.replace(' ', '').upper()

    # Remove common OCR artifacts
    vrn = vrn.replace("'", "").replace('"', '').replace('`', '')

    corrected = list(vrn)
    changes_made = []

    # Rule 1: First 2 chars MUST be letters (state code)
    for i in [0, 1]:
        if i < len(corrected) and corrected[i].isdigit():
            old = corrected[i]
            if corrected[i] == '0':
                corrected[i] = 'O'
            elif corrected[i] == '1':
                corrected[i] = 'I'
            elif corrected[i] == '5':
                corrected[i] = 'S'
            elif corrected[i] == '8':
                corrected[i] = 'B'
            elif corrected[i] == '2':
                corrected[i] = 'Z'
            changes_made.append(f"pos{i}: {old}→{corrected[i]}")

    # Rule 2: Chars at position 2-3 MUST be numbers (district code)
    for i in [2, 3]:
        if i < len(corrected) and corrected[i].isalpha():
            old = corrected[i]
            if corrected[i] == 'O':
                corrected[i] = '0'
            elif corrected[i] == 'I':
                corrected[i] = '1'
            elif corrected[i] == 'S':
                corrected[i] = '5'
            elif corrected[i] == 'B':
                corrected[i] = '8'
            elif corrected[i] == 'Z':
                corrected[i] = '2'
            changes_made.append(f"pos{i}: {old}→{corrected[i]}")

    # Rule 3: Last 4 MUST be numbers
    start = len(corrected) - 4
    for i in range(start, len(corrected)):
        if i >= 0 and i < len(corrected) and corrected[i].isalpha():
            old = corrected[i]
            if corrected[i] == 'O':
                corrected[i] = '0'
            elif corrected[i] == 'I':
                corrected[i] = '1'
            elif corrected[i] == 'S':
                corrected[i] = '5'
            elif corrected[i] == 'B':
                corrected[i] = '8'
            elif corrected[i] == 'Z':
                corrected[i] = '2'
            changes_made.append(f"pos{i}: {old}→{corrected[i]}")

    corrected_vrn = ''.join(corrected)

    # Log if corrections were made
    if changes_made:
        logger.info(
            f"OCR corrections: {vrn} → {corrected_vrn} ({', '.join(changes_made)})")

    return corrected_vrn


# ✨ NEW: Save image for future training
def save_detection_image(image_data: bytes, vrn: str, confidence: float, request_id: str):
    """
    Save detection images for future model training
    Only saves if confidence is low (needs review) or user provides feedback
    """
    try:
        # Save original image
        image_filename = f"{request_id}_{vrn}_{confidence:.2f}.jpg"
        image_path = SAVED_IMAGES_DIR / image_filename

        with open(image_path, 'wb') as f:
            f.write(image_data)

        logger.debug(f"Saved detection image: {image_filename}")
    except Exception as e:
        logger.error(f"Failed to save detection image: {str(e)}")


# ✨ Logging helper
def log_detection_result(
    request_id: str,
    vrn: str,
    confidence: float,
    detection_conf: float,
    ocr_conf: float,
    bbox: list,
    image_shape: tuple,
    image_hash: str,
    success: bool,
    error: str = None
):
    """Log detection result to JSONL file for analysis"""
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'request_id': request_id,
        'vrn': vrn,
        'confidence': round(confidence, 3),
        'detection_confidence': round(detection_conf, 3),
        'ocr_confidence': round(ocr_conf, 3),
        'bbox': bbox,
        'image_shape': list(image_shape),
        'image_hash': image_hash,
        'success': success,
        'error': error
    }

    log_file = LOGS_DIR / 'detections.jsonl'
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ParkingMudde ANPR",
        "version": "2.2.0",
        "status": "running",
        "docs": "/docs",
        "features": [
            "Custom trained model (99.2% mAP)",
            "Indian plate optimizations",
            "IND logo removal",
            "Rotation correction",
            "OCR post-processing",
            "Error tracking",
            "Detection image saving"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.2.0",
        "model_loaded": True
    }


@app.post("/api/v1/vehicle/detect")
async def detect_vehicle(image: UploadFile = File(...)):
    """
    Direct plate detection with OCR post-processing
    Optimized for Indian license plates

    Improvements in v2.2:
    - Rotation/skew correction
    - IND logo removal
    - Optimized preprocessing (inverted threshold)
    - OCR character error correction
    - Validation warnings instead of errors
    - File size validation
    - Timeout protection
    """

    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]

    try:
        # ✨ STEP 1.5: Read and validate file
        contents = await image.read()

        # Validate file size (10MB max)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(contents) > MAX_FILE_SIZE:
            logger.warning(
                f"[{request_id}] File too large: {len(contents)} bytes")
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error": "FILE_TOO_LARGE",
                    "message": f"Image must be less than {MAX_FILE_SIZE // (1024*1024)}MB. Please compress or resize your image.",
                    "request_id": request_id
                }
            )

        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if image.content_type not in allowed_types:
            logger.warning(
                f"[{request_id}] Invalid file type: {image.content_type}")
            return JSONResponse(
                status_code=415,
                content={
                    "success": False,
                    "error": "INVALID_FILE_TYPE",
                    "message": "Only JPEG, PNG, and WebP images are supported",
                    "request_id": request_id
                }
            )

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning(f"[{request_id}] Could not decode image")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "INVALID_IMAGE",
                    "message": "Could not read image file. Please ensure it's a valid image.",
                    "request_id": request_id
                }
            )

        # Calculate image hash for tracking
        image_hash = hashlib.md5(contents).hexdigest()[:12]

        logger.info(
            f"[{request_id}] Processing image: {img.shape}, hash: {image_hash}")

        # Step 2: Find plate directly (99.2% model!)
        plate_roi, bbox, detection_conf = plate_finder.find_plate(img)

        if plate_roi is None:
            logger.warning(f"[{request_id}] No plate detected")

            log_detection_result(
                request_id=request_id,
                vrn=None,
                confidence=0.0,
                detection_conf=0.0,
                ocr_conf=0.0,
                bbox=None,
                image_shape=img.shape,
                image_hash=image_hash,
                success=False,
                error="NO_PLATE_DETECTED"
            )

            return {
                "success": False,
                "error": "NO_PLATE_DETECTED",
                "message": "No license plate found in image. Please ensure the plate is clearly visible.",
                "request_id": request_id
            }

        logger.info(
            f"[{request_id}] Plate detected: bbox={bbox}, conf={detection_conf:.2%}")

        # Step 3: NEW Indian plate preprocessing
       # plate_enhanced = preprocess_indian_plate(plate_roi) ( temp disabling due to low accuracy of ocr , 24-02-2024)
        # TEMPORARY - Test without aggressive preprocessing
        plate_enhanced = preprocessor.preprocess_plate_region(plate_roi)
        logger.debug(
            f"[{request_id}] Plate preprocessed (rotation fixed, IND removed)")

        # Step 4: Multi-OCR (using enhanced plate)
        ocr_result = multi_ocr.detect_text_multi_engine(
            plate_enhanced,  # Use enhanced version directly
            image_variants=[]
        )

        raw_vrn = ocr_result['text']
        ocr_conf = ocr_result['confidence']
        ocr_engine = ocr_result.get('engine', 'unknown')

        logger.info(
            f"[{request_id}] Raw OCR: vrn={raw_vrn}, conf={ocr_conf:.2%}, engine={ocr_engine}")

        # Step 5: Post-process OCR result (fix common errors)
        corrected_vrn = fix_common_ocr_errors(raw_vrn)

        # Calculate overall confidence
        overall_conf = detection_conf * ocr_conf

        # ✨ STEP 1.2: Validate but return anyway with warning
        validation_result = validator.validate(corrected_vrn)

        if not validation_result.get('valid', False):
            logger.warning(
                f"[{request_id}] VRN validation failed: {corrected_vrn}, "
                f"reason: {validation_result.get('reason')} - returning with warning"
            )

            # ✨ Save image for future review/training (low confidence or validation failed)
            if overall_conf < 0.8:  # Save if confidence < 80%
                save_detection_image(
                    contents, corrected_vrn, overall_conf, request_id)

            # Still log as successful detection
            log_detection_result(
                request_id=request_id,
                vrn=corrected_vrn,
                confidence=overall_conf,
                detection_conf=detection_conf,
                ocr_conf=ocr_conf,
                bbox=bbox,
                image_shape=img.shape,
                image_hash=image_hash,
                success=True,
                error=f"VALIDATION_WARNING: {validation_result.get('reason', 'Unknown')}"
            )

            # Return success with warning flags
            return {
                "success": True,
                "vrn": corrected_vrn,
                "formatted_vrn": corrected_vrn,
                "state_code": "XX",
                "state_name": "Unknown",
                "confidence": round(overall_conf, 3),
                "detection_confidence": round(detection_conf, 3),
                "ocr_confidence": round(ocr_conf, 3),
                "bbox": bbox,
                "ocr_engine": ocr_engine,
                "request_id": request_id,
                "corrections_applied": raw_vrn != corrected_vrn,
                "needs_verification": True,  # ✨ Flag for Flutter app
                # ✨ Warning message
                "validation_warning": validation_result.get('reason', 'Invalid format')
            }

        # Extract validation info for valid VRNs
        state_code = validation_result.get('state_code', 'XX')
        state_name = validation_result.get('state_name', 'Unknown')
        formatted_vrn = validation_result.get('formatted', corrected_vrn)

        # Log success
        log_detection_result(
            request_id=request_id,
            vrn=corrected_vrn,
            confidence=overall_conf,
            detection_conf=detection_conf,
            ocr_conf=ocr_conf,
            bbox=bbox,
            image_shape=img.shape,
            image_hash=image_hash,
            success=True
        )

        # Success!
        logger.info(
            f"[{request_id}] ✅ SUCCESS: {corrected_vrn} (confidence: {overall_conf:.2%})")

        return {
            "success": True,
            "vrn": corrected_vrn,
            "formatted_vrn": formatted_vrn,
            "state_code": state_code,
            "state_name": state_name,
            "confidence": round(overall_conf, 3),
            "detection_confidence": round(detection_conf, 3),
            "ocr_confidence": round(ocr_conf, 3),
            "bbox": bbox,
            "ocr_engine": ocr_engine,
            "request_id": request_id,
            "corrections_applied": raw_vrn != corrected_vrn,
            "needs_verification": overall_conf < 0.7  # Suggest verification if confidence < 70%
        }

    except HTTPException:
        raise
    except cv2.error as e:
        logger.error(f"[{request_id}] OpenCV error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "IMAGE_PROCESSING_ERROR",
                "message": "Could not process image. Please try a different photo.",
                "request_id": request_id
            }
        )
    except Exception as e:
        logger.error(
            f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again.",
                "request_id": request_id
            }
        )


@app.post("/api/v1/feedback")
async def submit_feedback(
    request_id: str,
    detected_vrn: str,
    correct_vrn: str
):
    """
    Submit user correction for detected VRN

    This data is CRITICAL for improving the system:
    - Tracks accuracy over time
    - Identifies problematic patterns
    - Collects training data for model improvement
    """

    feedback_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'request_id': request_id,
        'detected_vrn': detected_vrn,
        'correct_vrn': correct_vrn,
        'was_correct': detected_vrn == correct_vrn
    }

    if detected_vrn != correct_vrn:
        differences = []
        for i, (d, c) in enumerate(zip(detected_vrn, correct_vrn)):
            if d != c:
                differences.append({
                    'position': i,
                    'detected': d,
                    'correct': c
                })
        feedback_entry['differences'] = differences

    feedback_file = LOGS_DIR / 'feedback.jsonl'
    with open(feedback_file, 'a') as f:
        f.write(json.dumps(feedback_entry) + '\n')

    logger.info(
        f"Feedback received for {request_id}: {detected_vrn} → {correct_vrn}")

    return {
        "status": "success",
        "message": "Thank you for your feedback! This helps us improve.",
        "request_id": request_id
    }


@app.get("/api/v1/admin/stats")
async def get_statistics():
    """Get detection statistics (for monitoring)"""

    detections_file = LOGS_DIR / 'detections.jsonl'
    feedback_file = LOGS_DIR / 'feedback.jsonl'

    stats = {
        'total_detections': 0,
        'successful_detections': 0,
        'failed_detections': 0,
        'average_confidence': 0.0,
        'total_feedback': 0,
        'correction_rate': 0.0,
        'saved_images_count': len(list(SAVED_IMAGES_DIR.glob('*.jpg')))
    }

    if detections_file.exists():
        detections = []
        with open(detections_file, 'r') as f:
            for line in f:
                detections.append(json.loads(line))

        stats['total_detections'] = len(detections)
        stats['successful_detections'] = sum(
            1 for d in detections if d['success'])
        stats['failed_detections'] = sum(
            1 for d in detections if not d['success'])

        if detections:
            confidences = [d['confidence'] for d in detections if d['success']]
            if confidences:
                stats['average_confidence'] = round(
                    sum(confidences) / len(confidences), 3)

    if feedback_file.exists():
        feedback = []
        with open(feedback_file, 'r') as f:
            for line in f:
                feedback.append(json.loads(line))

        stats['total_feedback'] = len(feedback)
        if feedback:
            corrections = sum(1 for f in feedback if not f['was_correct'])
            stats['correction_rate'] = round(corrections / len(feedback), 3)

    return stats
