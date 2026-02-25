"""Configuration settings"""
from pathlib import Path


class Settings:
    # Service
    SERVICE_NAME = "ParkingMudde ANPR"
    VERSION = "2.2.0"  # ✨ Updated version

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    # ✨ Updated to your trained model
    MODEL_PATH = BASE_DIR / "models" / "indian_plate_detector.pt"

    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8001

    # Image input flexibility
    MIN_IMAGES: int = 1  # Minimum images required
    MAX_IMAGES: int = 4  # Maximum images allowed

    # Processing
    MIN_CONFIDENCE = 0.25  # ✨ Lowered to match YOLO default
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT = 30  # ✨ NEW: seconds

    # OCR
    OCR_LANGUAGES = ['en']
    OCR_GPU = False
    USE_EASYOCR = True  # ✨ NEW
    USE_PADDLEOCR = False  # ✨ NEW

    # ✨ NEW: Image Saving
    SAVE_LOW_CONFIDENCE_IMAGES = True  # Enable/disable image saving
    SAVE_CONFIDENCE_THRESHOLD = 0.8  # Save if confidence < 80%
    SAVED_IMAGES_DIR = BASE_DIR / "data" / "saved_detections"

    # ✨ NEW: Logging
    LOG_DIR = BASE_DIR / "logs"
    LOG_LEVEL = "INFO"

    # Indian States (expanded list)
    VALID_STATE_CODES = [
        'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
        'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
        'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ',
        'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
    ]

    STATE_NAMES = {
        'AN': 'Andaman and Nicobar',
        'AP': 'Andhra Pradesh',
        'AR': 'Arunachal Pradesh',
        'AS': 'Assam',
        'BH': 'Bharat Series',
        'BR': 'Bihar',
        'CG': 'Chhattisgarh',
        'CH': 'Chandigarh',
        'DD': 'Daman and Diu',
        'DL': 'Delhi',
        'DN': 'Dadra and Nagar Haveli',
        'GA': 'Goa',
        'GJ': 'Gujarat',
        'HP': 'Himachal Pradesh',
        'HR': 'Haryana',
        'JH': 'Jharkhand',
        'JK': 'Jammu and Kashmir',
        'KA': 'Karnataka',
        'KL': 'Kerala',
        'LA': 'Ladakh',
        'LD': 'Lakshadweep',
        'MH': 'Maharashtra',
        'ML': 'Meghalaya',
        'MN': 'Manipur',
        'MP': 'Madhya Pradesh',
        'MZ': 'Mizoram',
        'NL': 'Nagaland',
        'OD': 'Odisha',
        'PB': 'Punjab',
        'PY': 'Puducherry',
        'RJ': 'Rajasthan',
        'SK': 'Sikkim',
        'TN': 'Tamil Nadu',
        'TR': 'Tripura',
        'TS': 'Telangana',
        'UK': 'Uttarakhand',
        'UP': 'Uttar Pradesh',
        'WB': 'West Bengal'
    }


settings = Settings()
