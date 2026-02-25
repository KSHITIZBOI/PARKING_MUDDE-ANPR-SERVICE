"""VRN (Vehicle Registration Number) validator for Indian formats"""
import re
from typing import Dict
from .config import settings


class VRNValidator:
    """
    Validates Indian number plate formats
    Supports: Standard & BH Series
    """

    # Standard format: XX00XX0000
    # Example: DL 01 AB 1234
    STANDARD_PATTERN = re.compile(
        r'^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$'
    )

    # BH series: 00BH0000XX
    # Example: 22 BH 1234 AB
    BH_PATTERN = re.compile(
        r'^(\d{2})(BH)(\d{4})([A-Z]{2})$'
    )

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean VRN text
        Remove spaces, special chars, make uppercase
        """
        text = text.upper()
        text = text.replace(' ', '')
        text = text.replace('-', '')
        # Remove any non-alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    @staticmethod
    def fix_ocr_errors(text: str) -> str:
        """
        Fix common OCR mistakes
        O → 0, S → 5, I → 1, etc.
        """
        # Common OCR confusions
        corrections = {
            'O': '0',  # Letter O → Number 0
            'I': '1',  # Letter I → Number 1
            'S': '5',  # Letter S → Number 5 (in number positions)
            'Z': '2',  # Letter Z → Number 2
            'B': '8',  # Letter B → Number 8 (sometimes)
        }

        result = text

        # Fix positions 2-3 (should be RTO code - numbers)
        if len(text) >= 4:
            # Position 2-3 should be numbers (RTO code)
            for i in range(2, 4):
                if i < len(text) and text[i] in corrections:
                    result = result[:i] + corrections[text[i]] + result[i+1:]

        # Fix last 4 positions (should be numbers)
        if len(text) >= 4:
            # Last 4 characters should be numbers
            for i in range(len(text) - 4, len(text)):
                if i >= 0 and i < len(text) and text[i] in corrections:
                    result = result[:i] + corrections[text[i]] + result[i+1:]

        return result

    @staticmethod
    def validate(text: str) -> Dict:
        """
        Validate VRN format

        Args:
            text: Raw VRN text from OCR

        Returns:
            {
                "valid": bool,
                "format": str,
                "formatted": str,
                "state_code": str,
                "state_name": str,
                "raw": str
            }
        """
        # Clean input
        clean = VRNValidator.clean_text(text)

        # 🔧 FIX OCR ERRORS
        clean = VRNValidator.fix_ocr_errors(clean)

        # Try standard format first
        match = VRNValidator.STANDARD_PATTERN.match(clean)
        if match:
            state, rto, series, number = match.groups()

            # Check if state code is valid
            if state in settings.VALID_STATE_CODES:
                return {
                    "valid": True,
                    "format": "STANDARD",
                    "formatted": f"{state} {rto} {series} {number}",
                    "state_code": state,
                    "state_name": settings.STATE_NAMES.get(state, "Unknown"),
                    "raw": clean
                }
            else:
                return {
                    "valid": False,
                    "format": "UNKNOWN",
                    "raw": clean,
                    "reason": f"Invalid state code: {state}"
                }

        # Try BH series format
        match = VRNValidator.BH_PATTERN.match(clean)
        if match:
            year, bh, number, series = match.groups()

            return {
                "valid": True,
                "format": "BH_SERIES",
                "formatted": f"{year} BH {number} {series}",
                "state_code": "BH",
                "state_name": "Bharat Series",
                "raw": clean
            }

        # No match
        return {
            "valid": False,
            "format": "UNKNOWN",
            "raw": clean,
            "reason": "Does not match Indian VRN format"
        }

    @staticmethod
    def is_valid_format(text: str) -> bool:
        """
        Quick check if text matches any valid format

        Returns:
            True if valid, False otherwise
        """
        clean = VRNValidator.clean_text(text)

        # Fix OCR errors before validation
        clean = VRNValidator.fix_ocr_errors(clean)

        # Check standard
        if VRNValidator.STANDARD_PATTERN.match(clean):
            return True

        # Check BH series
        if VRNValidator.BH_PATTERN.match(clean):
            return True

        return False
