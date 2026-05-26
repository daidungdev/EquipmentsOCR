import threading
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, Any

from app.config import (
    GOOGLE_PROJECT_ID,
    GOOGLE_PRIVATE_KEY_ID,
    GOOGLE_PRIVATE_KEY,
    GOOGLE_CLIENT_EMAIL,
    GOOGLE_CLIENT_ID,
    GOOGLE_SHEET_ID,
    logger,
)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Thread safety lock and client/worksheet cache
_sheet_lock = threading.Lock()
_worksheet = None

def get_worksheet() -> gspread.Worksheet:
    """Retrieves and returns the cached worksheet client instance.
    If not initialized, credentials will be configured and verified.
    """
    global _worksheet
    if _worksheet is not None:
        return _worksheet

    with _sheet_lock:
        if _worksheet is not None:
            return _worksheet

        logger.info("Initializing Google Sheets client connection...")
        try:
            # Construct service account dictionary dynamically from env vars
            service_account_info = {
                "type": "service_account",
                "project_id": GOOGLE_PROJECT_ID,
                "private_key_id": GOOGLE_PRIVATE_KEY_ID,
                "private_key": GOOGLE_PRIVATE_KEY,
                "client_email": GOOGLE_CLIENT_EMAIL,
                "client_id": GOOGLE_CLIENT_ID,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{GOOGLE_CLIENT_EMAIL}"
            }
            
            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=SCOPES
            )
            client = gspread.authorize(credentials)
            
            # Open by spreadsheet key and grab the first worksheet
            _worksheet = client.open_by_key(GOOGLE_SHEET_ID).get_worksheet(0)
            logger.info("Google Sheets client initialized successfully.")
            return _worksheet
        except Exception as exc:
            logger.error(f"Failed to initialize Google Sheets client: {exc}", exc_info=True)
            raise exc


def append_ocr_result(data: Dict[str, Any]) -> None:
    """Appends OCR extracted key-value pairs to the target Google Sheet.
    - If the worksheet has no headers, inserts them first.
    - Maps data to columns: machine_name, Mã MMTB, Model, Xưởng, Vị trí.
    - Defensively handles exceptions so it never crashes the caller.
    """
    try:
        if not data:
            logger.warning("Empty data provided to append_ocr_result. Skipping write.")
            return

        logger.info("Appending OCR result to Google Sheets...")
        
        # Get sheet instance (will raise if authentication/loading fails)
        worksheet = get_worksheet()
        
        # Define exact column headers in order
        columns = ["machine_name", "Mã MMTB", "Model", "Xưởng", "Vị trí"]
        
        # Check if sheet is empty and write headers if needed
        try:
            first_row = worksheet.row_values(1)
        except Exception as e:
            logger.warning(f"Could not read first row from sheet (might be empty): {e}")
            first_row = []
            
        if not first_row:
            logger.info("Worksheet is empty. Appending headers first.")
            worksheet.append_row(columns)

        # Map dictionary values to their respective columns
        row = [
            str(data.get("machine_name", "") or "").strip(),
            str(data.get("Mã MMTB", "") or "").strip(),
            str(data.get("Model", "") or "").strip(),
            str(data.get("Xưởng", "") or "").strip(),
            str(data.get("Vị trí", "") or "").strip()
        ]

        # Log payload data before appending
        logger.info(f"Appending row to Google Sheets: {row}")
        
        worksheet.append_row(row)
        logger.info("Google Sheets row appended successfully.")
    except Exception as exc:
        logger.error(f"Error occurred while appending row to Google Sheets: {exc}", exc_info=True)
