"""
Debug ANPR - Save Detected Plate Images
========================================
This script saves the detected plate region so you can see what OCR is trying to read
"""

import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.plate_finder import PlateFinder
from app.anpr_v2_preprocessing import ImagePreprocessor
from app.anpr_v2_multi_ocr import MultiOCREngine
import cv2
from pathlib import Path

# Initialize
print("🔧 Initializing components...")
plate_finder = PlateFinder()
preprocessor = ImagePreprocessor()
multi_ocr = MultiOCREngine(use_easyocr=True, use_paddleocr=False)

# Create output directory
output_dir = Path("debug_output")
output_dir.mkdir(exist_ok=True)

def debug_image(image_path):
    """
    Debug ANPR pipeline step by step
    Saves intermediate results for inspection
    """
    
    print(f"\n🔍 DEBUGGING: {image_path}")
    print("=" * 70)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image")
        return
    
    print(f"✅ Image loaded: {img.shape}")
    
    # Step 1: Detect plate
    print("\n📍 Step 1: Detecting plate...")
    plate_roi, bbox, detection_conf = plate_finder.find_plate(img)
    
    if plate_roi is None:
        print("❌ No plate detected!")
        return
    
    print(f"✅ Plate detected: bbox={bbox}, confidence={detection_conf:.2%}")
    
    # Save detected plate
    plate_path = output_dir / "1_detected_plate.jpg"
    cv2.imwrite(str(plate_path), plate_roi)
    print(f"💾 Saved: {plate_path}")
    
    # Step 2: Preprocess
    print("\n🎨 Step 2: Preprocessing...")
    plate_enhanced = preprocessor.preprocess_plate_region(plate_roi)
    
    enhanced_path = output_dir / "2_enhanced_plate.jpg"
    cv2.imwrite(str(enhanced_path), plate_enhanced)
    print(f"💾 Saved: {enhanced_path}")
    
    # Step 3: OCR
    print("\n🔤 Step 3: Running OCR...")
    
    # Try on original
    print("   Testing on original...")
    ocr_result_original = multi_ocr.detect_text_multi_engine(
        plate_roi,
        image_variants=[]
    )
    print(f"   Original: '{ocr_result_original['text']}' (conf: {ocr_result_original['confidence']:.2%})")
    
    # Try on enhanced
    print("   Testing on enhanced...")
    ocr_result_enhanced = multi_ocr.detect_text_multi_engine(
        plate_roi,
        image_variants=[plate_enhanced]
    )
    print(f"   Enhanced: '{ocr_result_enhanced['text']}' (conf: {ocr_result_enhanced['confidence']:.2%})")
    
    # Generate more preprocessing variants
    print("\n🔬 Step 4: Trying different preprocessing variants...")
    
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    
    # Variant 1: OTSU threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_path = output_dir / "3_otsu_threshold.jpg"
    cv2.imwrite(str(otsu_path), otsu)
    print(f"💾 Saved: {otsu_path}")
    
    ocr_otsu = multi_ocr.detect_text_multi_engine(
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        image_variants=[]
    )
    print(f"   OTSU: '{ocr_otsu['text']}' (conf: {ocr_otsu['confidence']:.2%})")
    
    # Variant 2: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    adaptive_path = output_dir / "4_adaptive_threshold.jpg"
    cv2.imwrite(str(adaptive_path), adaptive)
    print(f"💾 Saved: {adaptive_path}")
    
    ocr_adaptive = multi_ocr.detect_text_multi_engine(
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
        image_variants=[]
    )
    print(f"   Adaptive: '{ocr_adaptive['text']}' (conf: {ocr_adaptive['confidence']:.2%})")
    
    # Variant 3: Inverted
    inverted = cv2.bitwise_not(otsu)
    inverted_path = output_dir / "5_inverted.jpg"
    cv2.imwrite(str(inverted_path), inverted)
    print(f"💾 Saved: {inverted_path}")
    
    ocr_inverted = multi_ocr.detect_text_multi_engine(
        cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR),
        image_variants=[]
    )
    print(f"   Inverted: '{ocr_inverted['text']}' (conf: {ocr_inverted['confidence']:.2%})")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY:")
    print("=" * 70)
    
    results = [
        ("Original", ocr_result_original),
        ("Enhanced", ocr_result_enhanced),
        ("OTSU", ocr_otsu),
        ("Adaptive", ocr_adaptive),
        ("Inverted", ocr_inverted)
    ]
    
    # Sort by confidence
    results.sort(key=lambda x: x[1]['confidence'], reverse=True)
    
    print("\nBest results (sorted by confidence):")
    for name, result in results:
        print(f"   {name:15s}: '{result['text']}' ({result['confidence']:.1%})")
    
    print(f"\n💾 All images saved to: {output_dir.absolute()}/")
    print("\n🔍 NEXT STEPS:")
    print("   1. Open debug_output/ folder")
    print("   2. Look at 1_detected_plate.jpg - is it the correct plate?")
    print("   3. Look at other images - which one is most readable?")
    print("   4. If plate crop is wrong → model needs retraining")
    print("   5. If plate is correct but unreadable → try commercial OCR")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_anpr_fixed.py <image_path>")
        print("\nExample:")
        print("  python debug_anpr_fixed.py test_images/1.jpeg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug_image(image_path)
