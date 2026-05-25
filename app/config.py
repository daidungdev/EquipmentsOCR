import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Bind configurations with backward compatibility
PADDLE_BASE_URL = os.getenv("PADDLE_BASE_URL", os.getenv("JOB_URL", "")).strip()
PADDLE_API_KEY = os.getenv("PADDLE_API_KEY", os.getenv("TOKEN", "")).strip()
PADDLE_API_SECRET = os.getenv("PADDLE_API_SECRET", "").strip()

# Bind server port (Render passes PORT environment variable dynamically)
PORT = int(os.getenv("PORT", "8000"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "300"))
PADDLE_MODEL = os.getenv("MODEL", "PaddleOCR-VL-1.5").strip()

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
        
    if errors:
        error_msg = f"Configuration validation failed. Missing: {', '.join(errors)}"
        logger.critical(error_msg)
        raise ValueError(error_msg)
        
    logger.info("Configuration successfully validated.")
    logger.info(f"Paddle Base URL: {PADDLE_BASE_URL}")
    logger.info(f"Paddle Model: {PADDLE_MODEL}")
