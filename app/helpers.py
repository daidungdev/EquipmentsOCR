import re
import json
import time
import asyncio
import httpx
from typing import Dict, List
from fastapi import HTTPException, status

from app.config import (
    PADDLE_BASE_URL,
    PADDLE_API_KEY,
    PADDLE_MODEL,
    POLL_INTERVAL,
    MAX_WAIT_SECONDS,
    logger,
)
from app.schemas import OCRResult

# Validation limits
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB limit
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "application/pdf"}


def parse_markdown_to_key_value(markdown_text: str) -> Dict[str, str]:
    """Extracts structured key-value maps from PaddleOCR's page markdown output.
    - Captures titles/headers from the first line as 'machine_name' if no colon is present.
    - Identifies 'Key: Value' and 'Key：Value' structures on subsequent lines.
    - Preserves all Vietnamese accents and characters exactly.
    """
    key_value = {}
    if not markdown_text:
        return key_value

    # Parse lines that are not empty
    lines = [line.strip() for line in markdown_text.split("\n") if line.strip()]
    if not lines:
        return key_value

    # 1. Heading/Title identification on first line
    first_line = lines[0]
    first_line_clean = re.sub(r"^#+\s*", "", first_line).strip()
    
    # If first line contains no colons, assign it to machine_name
    if first_line_clean and ":" not in first_line_clean and "：" not in first_line_clean:
        key_value["machine_name"] = first_line_clean

    # 2. Key-Value pairs capture
    kv_pattern = re.compile(r"^\s*(?:\*\*)?\s*([^*：:]+?)\s*(?:\*\*)?\s*[:：]\s*(.*)$")

    for line in lines:
        match = kv_pattern.match(line)
        if match:
            k = match.group(1).strip()
            v = match.group(2).strip()

            # Clean residual markdown artifacts
            k = re.sub(r"^\*+\s*|\s*\*+$", "", k).strip()
            v = re.sub(r"^\*+\s*|\s*\*+$", "", v).strip()

            if k and v:
                key_value[k] = v

    return key_value


def validate_upload(filename: str, content_type: str, file_size: int):
    """Checks uploaded file extensions, MIME-types, and size limits."""
    import os

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has an empty filename."
        )

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{content_type}'. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size. Size: {file_size} bytes, Max: {MAX_FILE_SIZE} bytes (20MB)."
        )


async def submit_ocr_job(filename: str, file_bytes: bytes, content_type: str) -> str:
    """Uploads file bytes directly to the Paddle OCR API in memory."""
    headers = {
        "Authorization": f"bearer {PADDLE_API_KEY}"
    }
    files = {
        "file": (filename, file_bytes, content_type)
    }
    data = {
        "model": PADDLE_MODEL,
        "optionalPayload": json.dumps({
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        })
    }

    logger.info(f"Submitting in-memory OCR job for file '{filename}'...")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                PADDLE_BASE_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=120.0
            )
        except httpx.TimeoutException:
            logger.error("Timeout occurred while submitting file to Paddle API")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Paddle API request timed out during submission."
            )
        except httpx.RequestError as exc:
            logger.error(f"HTTP request error during submission: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to communicate with Paddle API: {exc}"
            )

    if r.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Paddle API authentication failure. Please check your token."
        )
    
    if r.status_code != 200:
        logger.error(f"Paddle API submission returned status {r.status_code}: {r.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream Paddle API submission failed with status {r.status_code}."
        )

    try:
        response_json = r.json()
        job_id = response_json["data"]["jobId"]
        logger.info(f"OCR Job submitted successfully. Job ID: {job_id}")
        return job_id
    except (KeyError, ValueError) as exc:
        logger.error(f"Malformed submission response from Paddle: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Malformed OCR submission response from Paddle API."
        )


async def poll_ocr_job(job_id: str) -> str:
    """Polls the Paddle API status until 'done' or 'failed'."""
    headers = {
        "Authorization": f"bearer {PADDLE_API_KEY}"
    }
    status_url = f"{PADDLE_BASE_URL}/{job_id}"
    start_time = time.time()

    logger.info(f"Starting polling loop for Job ID {job_id}...")
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < MAX_WAIT_SECONDS:
            try:
                r = await client.get(status_url, headers=headers, timeout=30.0)
            except httpx.TimeoutException:
                logger.warning(f"Timeout checking status for Job {job_id}, retrying...")
                await asyncio.sleep(POLL_INTERVAL)
                continue
            except httpx.RequestError as exc:
                logger.error(f"Error querying status for Job {job_id}: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to communicate with Paddle status API: {exc}"
                )

            if r.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Paddle API authentication failed during status polling."
                )

            if r.status_code != 200:
                logger.error(f"Downstream Paddle status query returned status {r.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Downstream Paddle API status check failed: Status {r.status_code}."
                )

            try:
                res_data = r.json()
                data = res_data["data"]
                state = data["state"]
            except (KeyError, ValueError) as exc:
                logger.error(f"Malformed status response for Job {job_id}: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Malformed status response from Paddle API."
                )

            if state == "done":
                return data["resultUrl"]["jsonUrl"]

            if state == "failed":
                error_msg = data.get("errorMsg", "OCR job failed")
                logger.error(f"OCR Job {job_id} failed on Paddle server: {error_msg}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Downstream Paddle OCR job failed: {error_msg}"
                )

            logger.info(f"Job {job_id} is '{state}'. Sleeping {POLL_INTERVAL}s...")
            await asyncio.sleep(POLL_INTERVAL)

        logger.error(f"Polling timed out for Job ID {job_id} after {MAX_WAIT_SECONDS} seconds.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Paddle OCR job polling timed out."
        )


async def download_and_parse_jsonl(jsonl_url: str) -> List[OCRResult]:
    """Downloads JSONL layout result from Paddle and parses structured pages."""
    logger.info(f"Downloading OCR result from: {jsonl_url}")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(jsonl_url, timeout=60.0)
            r.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timeout downloading results from Paddle storage."
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to download results from Paddle: Status {exc.response.status_code}."
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Network error downloading results from Paddle: {exc}"
            )

    pages = []
    try:
        lines = r.text.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            line_data = json.loads(line)
            result = line_data.get("result", {})

            for res in result.get("layoutParsingResults", []):
                markdown_text = res.get("markdown", {}).get("text", "")
                key_value = parse_markdown_to_key_value(markdown_text)

                pages.append(
                    OCRResult(
                        markdown=markdown_text,
                        key_value=key_value
                    )
                )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error(f"Error parsing final JSONL results: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse malformed OCR JSONL result from Paddle."
        )

    return pages


async def check_paddle_connectivity() -> bool:
    """Validates external endpoint availability."""
    headers = {"Authorization": f"bearer {PADDLE_API_KEY}"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(PADDLE_BASE_URL, headers=headers, timeout=2.0)
            return True
    except Exception as exc:
        logger.warning(f"Paddle connection check failed: {exc}")
        return False
