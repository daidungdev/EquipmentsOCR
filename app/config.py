import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Bind configurations with backward compatibility
PADDLE_BASE_URL = os.getenv("PADDLE_BASE_URL", os.getenv("JOB_URL", "")).strip()
PADDLE_API_KEY = os.getenv("PADDLE_API_KEY", os.getenv("TOKEN", "")).strip()
PADDLE_API_SECRET = os.getenv("PADDLE_API_SECRET", "").strip()

# Gemini OCR configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

# Bind server port (Render passes PORT environment variable dynamically)
PORT = int(os.getenv("PORT", "8000"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "300"))
PADDLE_MODEL = os.getenv("MODEL", "PaddleOCR-VL-1.5").strip()

# Google Sheets Persistence Pipeline configuration
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "").strip()
GOOGLE_PRIVATE_KEY_ID = os.getenv("GOOGLE_PRIVATE_KEY_ID", "").strip()
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY", "").strip()
if GOOGLE_PRIVATE_KEY:
    GOOGLE_PRIVATE_KEY = GOOGLE_PRIVATE_KEY.replace("\\n", "\n")
GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL", "").strip()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "").strip()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr-api")

def validate_config():
    """Validates required configurations on startup."""
    logger.info("Validating configuration on startup...")
    errors = []
    
    if not PADDLE_BASE_URL:
        errors.append("PADDLE_BASE_URL (or JOB_URL)")
    if not PADDLE_API_KEY:
        errors.append("PADDLE_API_KEY (or TOKEN)")
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY")
        
    if not GOOGLE_PROJECT_ID:
        errors.append("GOOGLE_PROJECT_ID")
    if not GOOGLE_PRIVATE_KEY_ID:
        errors.append("GOOGLE_PRIVATE_KEY_ID")
    if not GOOGLE_PRIVATE_KEY:
        errors.append("GOOGLE_PRIVATE_KEY")
    if not GOOGLE_CLIENT_EMAIL:
        errors.append("GOOGLE_CLIENT_EMAIL")
    if not GOOGLE_CLIENT_ID:
        errors.append("GOOGLE_CLIENT_ID")
    if not GOOGLE_SHEET_ID:
        errors.append("GOOGLE_SHEET_ID")

    if errors:
        error_msg = f"Configuration validation failed. Missing: {', '.join(errors)}"
        logger.critical(error_msg)
        raise ValueError(error_msg)
        
    logger.info("Configuration successfully validated.")
    logger.info(f"Paddle Base URL: {PADDLE_BASE_URL}")
    logger.info(f"Paddle Model: {PADDLE_MODEL}")
    logger.info(f"Gemini Model: {GEMINI_MODEL}")
    logger.info(f"Google Sheet ID: {GOOGLE_SHEET_ID}")
