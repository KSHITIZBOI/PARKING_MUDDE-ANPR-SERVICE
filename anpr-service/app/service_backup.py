"""
Main ANPR Service
Orchestrates: Detection → Plate ROI → OCR → Validation

VERSION 1.1 - Now supports 1-4 images (flexible)
"""
import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path
import logging

from .detector import VehicleDetector
from .plate_finder import PlateFinder
from .ocr import OCREngine
from .validator import VRNValidator

logger = logging.getLogger(__name__)


class ANPRService:
    """
    Complete ANPR pipeline
    Thread-safe, production-ready

    NEW in v1.1:
    - Accepts 1-4 images (previously required 3-4)
    - Better error handling
    - Improved logging
    """

    def __init__(self):
        """Initialize all components ONCE"""
        logger.info("🔧 Initializing ANPR Service v1.1...")

        self.vehicle_detector = VehicleDetector()
        self.plate_finder = PlateFinder()
        self.ocr_engine = OCREngine()
        self.validator = VRNValidator()

        logger.info("✅ ANPR Service ready")

    def process_single_image(self, image_path: str, angle: str) -> Dict:
        """
        Process one image through complete ANPR pipeline

        Steps:
        1. Detect vehicle
        2. Crop vehicle region
        3. Find plate in vehicle
        4. OCR to read text
        5. Validate VRN format

        Args:
            image_path: Path to image file
            angle: Name/identifier for this image (e.g., "front", "image_1")

        Returns:
            Dict with success status and result/error
        """
        logger.info(f"   📸 Processing {angle}...")

        try:
            # ========================================
            # STEP 1: Detect Vehicle
            # ========================================
            vehicle_result = self.vehicle_detector.detect(image_path)

            if not vehicle_result["success"]:
                logger.warning(f"   ⚠️ {angle}: Vehicle detection failed")
                return {
                    "success": False,
                    "angle": angle,
                    "error": "NO_VEHICLE_DETECTED",
                    "details": vehicle_result.get("error", "Unknown error")
                }

            if not vehicle_result.get("vehicles"):
                logger.warning(f"   ⚠️ {angle}: No vehicles found in image")
                return {
                    "success": False,
                    "angle": angle,
                    "error": "NO_VEHICLE_FOUND"
                }

            # Get highest confidence vehicle
            vehicle = vehicle_result["vehicles"][0]
            logger.info(
                f"   ✓ {angle}: Vehicle detected ({vehicle.get('type', 'unknown')})")

            # ========================================
            # STEP 2: Crop Vehicle Region
            # ========================================
            try:
                # Read image
                image = cv2.imread(image_path)

                if image is None:
                    # Fallback to PIL if OpenCV fails
                    from PIL import Image
                    pil_img = Image.open(image_path)
                    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                if image is None:
                    logger.warning(f"   ⚠️ {angle}: Could not read image file")
                    return {
                        "success": False,
                        "angle": angle,
                        "error": "IMAGE_READ_FAILED"
                    }

                # Extract bounding box
                bbox = vehicle.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    # Format: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, bbox)
                elif isinstance(bbox, dict):
                    # Format: {x, y, width, height}
                    x = bbox.get('x', 0)
                    y = bbox.get('y', 0)
                    w = bbox.get('width', 0)
                    h = bbox.get('height', 0)
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    logger.warning(f"   ⚠️ {angle}: Invalid bbox format")
                    return {
                        "success": False,
                        "angle": angle,
                        "error": "INVALID_BBOX"
                    }

                # Ensure coordinates are within image bounds
                img_h, img_w = image.shape[:2]
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                # Crop vehicle
                vehicle_crop = image[y1:y2, x1:x2]

                if vehicle_crop.size == 0:
                    logger.warning(f"   ⚠️ {angle}: Empty vehicle crop")
                    return {
                        "success": False,
                        "angle": angle,
                        "error": "EMPTY_CROP"
                    }

                logger.info(f"   ✓ {angle}: Vehicle cropped")

            except Exception as crop_error:
                logger.error(f"   ❌ {angle}: Crop error - {str(crop_error)}")
                return {
                    "success": False,
                    "angle": angle,
                    "error": "CROP_FAILED",
                    "details": str(crop_error)
                }

            # ========================================
            # STEP 3: Find Plate in Vehicle
            # ========================================
            plate_crop = self.plate_finder.find_plate(vehicle_crop)

            if plate_crop is None or plate_crop.size == 0:
                logger.warning(f"   ⚠️ {angle}: Plate region not found")
                return {
                    "success": False,
                    "angle": angle,
                    "error": "PLATE_REGION_NOT_FOUND"
                }

            logger.info(f"   ✓ {angle}: Plate region found")

            # ========================================
            # STEP 4: OCR (Read Text from Plate)
            # ========================================
            ocr_result = self.ocr_engine.read(plate_crop)

            if not ocr_result.get('success'):
                error_msg = ocr_result.get('error', 'Unknown OCR error')
                logger.warning(f"   ⚠️ {angle}: OCR failed - {error_msg}")
                return {
                    "success": False,
                    "angle": angle,
                    "error": "OCR_FAILED",
                    "details": error_msg
                }

            # Log what OCR detected
            detected_text = ocr_result.get('text', '')
            ocr_confidence = ocr_result.get('confidence', 0)
            logger.info(
                f"   🔤 {angle}: OCR read '{detected_text}' (confidence: {ocr_confidence:.2%})")

            # ========================================
            # STEP 5: Validate VRN Format
            # ========================================
            validation = self.validator.validate(detected_text)

            if not validation.get('valid'):
                reason = validation.get('reason', 'Invalid format')
                logger.warning(
                    f"   ⚠️ {angle}: Validation failed for '{detected_text}' - {reason}")
                return {
                    "success": False,
                    "angle": angle,
                    "error": "INVALID_VRN_FORMAT",
                    "detected_text": detected_text,
                    "validation_reason": reason
                }

            # ========================================
            # STEP 6: SUCCESS!
            # ========================================
            formatted_vrn = validation.get('formatted', detected_text)
            logger.info(f"   ✅ {angle}: SUCCESS - {formatted_vrn}")

            return {
                "success": True,
                "angle": angle,
                "vrn": validation.get("raw", detected_text),
                "formatted_vrn": formatted_vrn,
                "confidence": ocr_confidence,
                "state_code": validation.get("state_code", "XX"),
                "state_name": validation.get("state_name", "Unknown"),
                "format": validation.get("format", "STANDARD"),
            }

        except Exception as e:
            logger.error(f"   ❌ {angle}: Unexpected error - {str(e)}")
            logger.exception(e)  # Log full stack trace
            return {
                "success": False,
                "angle": angle,
                "error": "PROCESSING_FAILED",
                "details": str(e),
            }

    def process(self, image_paths: List[str]) -> Dict:
        """
        Process 1-4 images and return best VRN

        NEW in v1.1: Works with any number of images from 1 to 4

        Strategy:
        - Try each image in order
        - Return first valid VRN found
        - If multiple valid VRNs, return highest confidence

        Args:
            image_paths: List of 1-4 image file paths

        Returns:
            Dict with success status and VRN details or error
        """
        logger.info(f"🔍 ANPR Processing: {len(image_paths)} image(s)")

        # Define angle names
        angles = ["front", "back", "left", "right"]
        detected_vrns = []

        # Process each image
        for idx, image_path in enumerate(image_paths):
            # Assign angle name
            angle = angles[idx] if idx < len(angles) else f"image_{idx+1}"

            # Process this image
            result = self.process_single_image(image_path, angle)

            if result["success"]:
                # Valid VRN found!
                detected_vrns.append(result)
                logger.info(
                    f"✅ VRN detected from {angle}: {result['formatted_vrn']}")

                # OPTIMIZATION: If we have high confidence, we can stop early
                if result['confidence'] > 0.9 and len(image_paths) == 1:
                    # Single image with high confidence - no need to continue
                    logger.info(
                        "⚡ High confidence on single image - stopping early")
                    break
            else:
                # Failed on this image
                error = result.get('error', 'UNKNOWN')
                logger.warning(f"⚠️ Failed on {angle}: {error}")

        # ========================================
        # Return Best Result
        # ========================================
        if detected_vrns:
            # Sort by confidence and take the best
            best = max(detected_vrns, key=lambda x: x.get("confidence", 0))

            logger.info(f"🎯 Best result: {best['formatted_vrn']} from {best['angle']} "
                        f"(confidence: {best['confidence']:.2%})")

            return {
                "success": True,
                "vrn": best["vrn"],
                "formatted_vrn": best["formatted_vrn"],
                "confidence": best["confidence"],
                "detected_in": best["angle"],
                # NEW: List all successful angles
                "all_detections": [v["angle"] for v in detected_vrns],
                "state_code": best["state_code"],
                "state_name": best["state_name"],
                "format": best["format"],
                "is_valid": True,
            }

        # ========================================
        # No Valid VRN Found
        # ========================================
        logger.warning(
            f"⚠️ No valid VRN found in any of the {len(image_paths)} image(s)")

        return {
            "success": False,
            "error": "PLATE_NOT_DETECTED",
            "message": "Could not read valid VRN from any image",
            "suggestion": "Please ensure number plate is clearly visible and try again",
            "images_attempted": len(image_paths)
        }
