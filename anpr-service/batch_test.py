"""
Batch ANPR Testing Script
==========================
Tests ANPR service on large dataset with CSV ground truth

Usage:
    python batch_test.py --csv path/to/labels.csv --images path/to/images --limit 100
"""

import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import argparse

# Configuration
API_URL = "http://localhost:8001/api/v1/vehicle/detect"
OUTPUT_DIR = Path("batch_test_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_vrn(raw_vrn):
    """
    Clean VRN from dataset format
    
    Examples:
        "KA19/TR01/2010-2011" → "KA19TR012010"
        "TN21AT0492" → "TN21AT0492"
        "MH20CS9817" → "MH20CS9817"
    """
    if pd.isna(raw_vrn):
        return None
    
    # Remove spaces, slashes, hyphens
    cleaned = str(raw_vrn).replace('/', '').replace('-', '').replace(' ', '').upper()
    
    # Remove any non-alphanumeric
    cleaned = ''.join(c for c in cleaned if c.isalnum())
    
    # Basic validation - should be 8-10 chars
    if len(cleaned) < 8 or len(cleaned) > 12:
        return None
    
    return cleaned


def test_single_image(image_path, ground_truth):
    """Test single image against API"""
    
    try:
        with open(image_path, 'rb') as f:
            # Determine content type from file extension
            ext = Path(image_path).suffix.lower()
            content_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.jpe': 'image/jpeg',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')
            
            response = requests.post(
                API_URL,
                files={'image': (Path(image_path).name, f, content_type)},
                timeout=30
            )
        
        if response.status_code != 200:
            return {
                'status': 'API_ERROR',
                'detected': None,
                'confidence': 0.0,
                'error': f"HTTP {response.status_code}"
            }
        
        result = response.json()
        
        return {
            'status': 'SUCCESS' if result.get('success') else 'FAILED',
            'detected': result.get('vrn'),
            'confidence': result.get('confidence', 0.0),
            'detection_conf': result.get('detection_confidence', 0.0),
            'ocr_conf': result.get('ocr_confidence', 0.0),
            'needs_verification': result.get('needs_verification', False),
            'error': result.get('error')
        }
        
    except requests.exceptions.Timeout:
        return {'status': 'TIMEOUT', 'detected': None, 'confidence': 0.0, 'error': 'Timeout'}
    except requests.exceptions.ConnectionError:
        return {'status': 'CONNECTION_ERROR', 'detected': None, 'confidence': 0.0, 'error': 'API not running'}
    except Exception as e:
        return {'status': 'ERROR', 'detected': None, 'confidence': 0.0, 'error': str(e)}


def calculate_accuracy(detected, ground_truth):
    """
    Calculate if detection is correct
    
    Handles variations:
    - Exact match
    - Case insensitive
    - Ignores spaces
    """
    if not detected or not ground_truth:
        return False
    
    d = detected.replace(' ', '').upper()
    g = ground_truth.replace(' ', '').upper()
    
    return d == g


def calculate_character_accuracy(detected, ground_truth):
    """Calculate character-level accuracy"""
    if not detected or not ground_truth:
        return 0.0
    
    d = detected.replace(' ', '').upper()
    g = ground_truth.replace(' ', '').upper()
    
    # Align to same length
    max_len = max(len(d), len(g))
    d = d.ljust(max_len, ' ')
    g = g.ljust(max_len, ' ')
    
    correct = sum(1 for i in range(max_len) if d[i] == g[i])
    return correct / max_len if max_len > 0 else 0.0


def run_batch_test(csv_path, images_dir, limit=None, sample_rate=1.0):
    """
    Run batch testing
    
    Args:
        csv_path: Path to CSV with labels
        images_dir: Directory containing images
        limit: Max number of images to test (None = all)
        sample_rate: Sample rate (0.1 = 10%, 1.0 = 100%)
    """
    
    print("=" * 70)
    print("🚀 BATCH ANPR TESTING")
    print("=" * 70)
    
    # Check API
    try:
        health = requests.get("http://localhost:8001/health", timeout=5)
        if health.status_code != 200:
            print("❌ API is not healthy!")
            return
        print("✅ API is running")
    except:
        print("❌ API is not running! Start it with: python main.py")
        return
    
    # Load CSV
    print(f"\n📊 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Clean VRNs
    df['cleaned_label'] = df['imageLabel'].apply(clean_vrn)
    
    # Remove invalid labels
    df = df[df['cleaned_label'].notna()]
    
    print(f"   Total images: {len(df)}")
    
    # Sample if needed
    if sample_rate < 1.0:
        df = df.sample(frac=sample_rate, random_state=42)
        print(f"   Sampled: {len(df)} images ({sample_rate*100:.0f}%)")
    
    # Limit if specified
    if limit:
        df = df.head(limit)
        print(f"   Limited to: {limit} images")
    
    # Test each image
    results = []
    start_time = time.time()
    
    print(f"\n🔬 Testing {len(df)} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
        image_filename = row['imageFile']
        ground_truth = row['cleaned_label']
        
        # Find image file
        image_path = Path(images_dir) / image_filename
        
        if not image_path.exists():
            # Try without extension variations
            base_name = image_path.stem
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.jpe']:
                test_path = image_path.parent / (base_name + ext)
                if test_path.exists():
                    image_path = test_path
                    found = True
                    break
            
            if not found:
                results.append({
                    'filename': image_filename,
                    'ground_truth': ground_truth,
                    'status': 'FILE_NOT_FOUND',
                    'detected': None,
                    'is_correct': False,
                    'char_accuracy': 0.0,
                    'confidence': 0.0
                })
                continue
        
        # Test image
        test_result = test_single_image(image_path, ground_truth)
        
        # Calculate accuracy
        is_correct = calculate_accuracy(test_result['detected'], ground_truth)
        char_acc = calculate_character_accuracy(test_result['detected'], ground_truth)
        
        results.append({
            'filename': image_filename,
            'ground_truth': ground_truth,
            'detected': test_result['detected'],
            'status': test_result['status'],
            'is_correct': is_correct,
            'char_accuracy': char_acc,
            'confidence': test_result['confidence'],
            'detection_conf': test_result.get('detection_conf', 0.0),
            'ocr_conf': test_result.get('ocr_conf', 0.0),
            'needs_verification': test_result.get('needs_verification', False),
            'error': test_result.get('error')
        })
    
    elapsed_time = time.time() - start_time
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    total = len(results_df)
    successful = len(results_df[results_df['status'] == 'SUCCESS'])
    correct = len(results_df[results_df['is_correct'] == True])
    
    avg_confidence = results_df[results_df['status'] == 'SUCCESS']['confidence'].mean()
    avg_char_acc = results_df[results_df['status'] == 'SUCCESS']['char_accuracy'].mean()
    avg_time = elapsed_time / total
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal Tested: {total}")
    print(f"  ✅ API Success: {successful} ({successful/total*100:.1f}%)")
    print(f"  ✅ Correct VRN: {correct} ({correct/total*100:.1f}%)")
    print(f"\nAccuracy Metrics:")
    print(f"  Full VRN Accuracy: {correct/successful*100:.1f}%" if successful > 0 else "  N/A")
    print(f"  Character Accuracy: {avg_char_acc*100:.1f}%" if successful > 0 else "  N/A")
    print(f"  Average Confidence: {avg_confidence:.1%}" if successful > 0 else "  N/A")
    print(f"\nPerformance:")
    print(f"  Total Time: {elapsed_time:.1f}s")
    print(f"  Avg Time/Image: {avg_time:.2f}s")
    print(f"  Throughput: {total/elapsed_time:.1f} images/sec")
    
    # Error breakdown
    print(f"\n⚠️  Errors:")
    error_counts = results_df['status'].value_counts()
    for status, count in error_counts.items():
        if status != 'SUCCESS':
            print(f"  {status}: {count}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Detailed results
    results_file = OUTPUT_DIR / f"batch_results_{timestamp}.xlsx"
    results_df.to_excel(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Summary report
    summary = {
        'timestamp': timestamp,
        'total_tested': total,
        'successful': successful,
        'correct': correct,
        'full_vrn_accuracy': correct/successful if successful > 0 else 0,
        'character_accuracy': avg_char_acc if successful > 0 else 0,
        'average_confidence': avg_confidence if successful > 0 else 0,
        'total_time': elapsed_time,
        'avg_time_per_image': avg_time,
        'throughput': total/elapsed_time
    }
    
    summary_file = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: {summary_file}")
    
    # Save failed cases for review
    failed = results_df[results_df['is_correct'] == False]
    if len(failed) > 0:
        failed_file = OUTPUT_DIR / f"failed_{timestamp}.xlsx"
        failed.to_excel(failed_file, index=False)
        print(f"💾 Failed cases saved to: {failed_file}")
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE")
    print("=" * 70)
    
    return results_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch test ANPR system')
    parser.add_argument('--csv', required=True, help='Path to CSV file with labels')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--limit', type=int, help='Limit number of images to test')
    parser.add_argument('--sample', type=float, default=1.0, help='Sample rate (0.1 = 10%%)')
    
    args = parser.parse_args()
    
    run_batch_test(
        csv_path=args.csv,
        images_dir=args.images,
        limit=args.limit,
        sample_rate=args.sample
    )
