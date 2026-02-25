"""
Automated ANPR Testing & Error Logging Script
==============================================

Tests multiple images and automatically documents results in CSV + saves failed images

Usage:
    python test_and_log.py

Requirements:
    pip install requests pandas openpyxl
"""

import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import cv2
import time

# Configuration
API_URL = "http://localhost:8001/api/v1/vehicle/detect"
TEST_IMAGES_DIR = Path("test_images")  # Put your test images here
OUTPUT_DIR = Path("test_results")
RESULTS_FILE = OUTPUT_DIR / \
    f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
FAILED_IMAGES_DIR = OUTPUT_DIR / "failed_images"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
FAILED_IMAGES_DIR.mkdir(exist_ok=True)

# Ground truth data (you'll need to fill this)
GROUND_TRUTH = {
    # Format: "filename.jpg": "CORRECT_VRN"
    "1": "UP32LQ5525",
    "2": "UP32DD2019",
    "3": "UP32MF2267",
    "4": "UP32DX0474"
    # Add more as you test
}


def test_single_image(image_path: Path, ground_truth: str = None):
    """
    Test single image and return results

    Returns:
        dict with test results
    """

    print(f"\n📸 Testing: {image_path.name}")

    try:
        # Read image for metadata
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                'filename': image_path.name,
                'status': 'FAILED',
                'error': 'Could not read image',
                'ground_truth': ground_truth
            }

        height, width = img.shape[:2]
        file_size_kb = image_path.stat().st_size / 1024

        # Call API
        start_time = time.time()

        with open(image_path, 'rb') as f:
            response = requests.post(
                API_URL,
                files={'image': f},
                timeout=30
            )

        response_time = time.time() - start_time

        # Parse response
        result = response.json()

        # Extract data
        detected_vrn = result.get('vrn', 'N/A')
        confidence = result.get('confidence', 0.0)
        detection_conf = result.get('detection_confidence', 0.0)
        ocr_conf = result.get('ocr_confidence', 0.0)
        success = result.get('success', False)
        error = result.get('error', None)
        request_id = result.get('request_id', 'N/A')

        # Compare with ground truth
        is_correct = None
        if ground_truth:
            is_correct = (detected_vrn == ground_truth)

        # Save failed images
        if not success or (ground_truth and not is_correct):
            failed_img_path = FAILED_IMAGES_DIR / \
                f"{image_path.stem}_{datetime.now().strftime('%H%M%S')}.jpg"
            shutil.copy(image_path, failed_img_path)
            print(f"   💾 Saved failed image to: {failed_img_path.name}")

        # Print result
        if success:
            status_icon = "✅" if is_correct or ground_truth is None else "❌"
            print(
                f"   {status_icon} Detected: {detected_vrn} (confidence: {confidence:.1%})")
            if ground_truth:
                print(f"      Expected: {ground_truth}")
        else:
            print(f"   ❌ Failed: {error}")

        return {
            'timestamp': datetime.now().isoformat(),
            'filename': image_path.name,
            'image_width': width,
            'image_height': height,
            'file_size_kb': round(file_size_kb, 2),
            'status': 'SUCCESS' if success else 'FAILED',
            'detected_vrn': detected_vrn,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'confidence': round(confidence, 3),
            'detection_confidence': round(detection_conf, 3),
            'ocr_confidence': round(ocr_conf, 3),
            'response_time_sec': round(response_time, 3),
            'error': error,
            'request_id': request_id
        }

    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Error: Is the API running?")
        return {
            'filename': image_path.name,
            'status': 'CONNECTION_ERROR',
            'error': 'API not running',
            'ground_truth': ground_truth
        }

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {
            'filename': image_path.name,
            'status': 'ERROR',
            'error': str(e),
            'ground_truth': ground_truth
        }


def run_all_tests():
    """
    Run tests on all images in TEST_IMAGES_DIR
    """

    print("=" * 70)
    print("🚀 ANPR AUTOMATED TESTING")
    print("=" * 70)

    # Check if API is running
    try:
        health_check = requests.get("http://localhost:8001/health", timeout=5)
        if health_check.status_code != 200:
            print("❌ API is not healthy!")
            return
        print("✅ API is running")
    except:
        print("❌ API is not running! Start it with: python main.py")
        return

    # Find test images
    if not TEST_IMAGES_DIR.exists():
        print(f"\n❌ Test images directory not found: {TEST_IMAGES_DIR}")
        print(f"   Create it and add test images, then run again.")
        TEST_IMAGES_DIR.mkdir(exist_ok=True)
        return

    image_files = list(TEST_IMAGES_DIR.glob(
        "*.jpg")) + list(TEST_IMAGES_DIR.glob("*.jpeg")) + list(TEST_IMAGES_DIR.glob("*.png"))

    if not image_files:
        print(f"\n❌ No images found in {TEST_IMAGES_DIR}")
        print(f"   Add .jpg, .jpeg, or .png files and run again.")
        return

    print(f"\n📊 Found {len(image_files)} test images")

    # Test each image
    results = []

    for image_path in sorted(image_files):
        ground_truth = GROUND_TRUTH.get(image_path.name)
        result = test_single_image(image_path, ground_truth)
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate statistics
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)

    total = len(results)
    successful = df[df['status'] == 'SUCCESS'].shape[0]
    failed = total - successful

    print(f"\nTotal Tests: {total}")
    print(f"  ✅ Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"  ❌ Failed: {failed} ({failed/total*100:.1f}%)")

    if 'is_correct' in df.columns:
        with_ground_truth = df[df['is_correct'].notna()]
        if len(with_ground_truth) > 0:
            correct = with_ground_truth[with_ground_truth['is_correct']
                                        == True].shape[0]
            total_gt = len(with_ground_truth)
            print(
                f"\nAccuracy (with ground truth): {correct}/{total_gt} ({correct/total_gt*100:.1f}%)")

    if successful > 0:
        avg_confidence = df[df['status'] == 'SUCCESS']['confidence'].mean()
        avg_response_time = df[df['status'] ==
                               'SUCCESS']['response_time_sec'].mean()
        print(f"\nAverage Confidence: {avg_confidence:.1%}")
        print(f"Average Response Time: {avg_response_time:.2f}s")

    # Show errors
    errors = df[df['status'] != 'SUCCESS']
    if len(errors) > 0:
        print(f"\n❌ Failed Images:")
        for _, row in errors.iterrows():
            print(f"  - {row['filename']}: {row['error']}")

    # Show incorrect detections
    if 'is_correct' in df.columns:
        incorrect = df[df['is_correct'] == False]
        if len(incorrect) > 0:
            print(f"\n⚠️  Incorrect Detections:")
            for _, row in incorrect.iterrows():
                print(f"  - {row['filename']}")
                print(f"      Detected: {row['detected_vrn']}")
                print(f"      Expected: {row['ground_truth']}")
                print(f"      Confidence: {row['confidence']:.1%}")

    # Save to Excel
    with pd.ExcelWriter(RESULTS_FILE, engine='openpyxl') as writer:
        # Main results
        df.to_excel(writer, sheet_name='Results', index=False)

        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Tests',
                'Successful',
                'Failed',
                'Success Rate',
                'Average Confidence',
                'Average Response Time'
            ],
            'Value': [
                total,
                successful,
                failed,
                f"{successful/total*100:.1f}%",
                f"{avg_confidence:.1%}" if successful > 0 else 'N/A',
                f"{avg_response_time:.2f}s" if successful > 0 else 'N/A'
            ]
        }

        if 'is_correct' in df.columns:
            with_gt = df[df['is_correct'].notna()]
            if len(with_gt) > 0:
                correct = with_gt[with_gt['is_correct'] == True].shape[0]
                summary_data['Metric'].append('Accuracy (Ground Truth)')
                summary_data['Value'].append(
                    f"{correct/len(with_gt)*100:.1f}%")

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n💾 Results saved to: {RESULTS_FILE}")
    print(f"💾 Failed images saved to: {FAILED_IMAGES_DIR}")

    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Instructions
    print("\n📖 SETUP INSTRUCTIONS:")
    print("-" * 70)
    print("1. Put test images in 'test_images/' folder")
    print("2. Update GROUND_TRUTH dict in this script with correct VRNs")
    print("3. Make sure API is running (python main.py)")
    print("4. Run this script: python test_and_log.py")
    print("-" * 70)

    input("\nPress Enter to continue...")

    run_all_tests()
