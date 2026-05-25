from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import validate_config, logger
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event manager for validating configurations on startup."""
    try:
        validate_config()
    except ValueError as exc:
        logger.critical(f"App configuration check failed on startup: {exc}")
        raise exc
    
    logger.info("OCR API Wrapper Service has started successfully.")
    yield
    logger.info("OCR API Wrapper Service is shutting down.")


app = FastAPI(
    title="PaddleOCR API Wrapper",
    description="A clean, production-ready FastAPI wrapper around PaddleOCR API.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# --- Standardized Error Handling Hook Procedures ---

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Intercepts standard HTTP exceptions to return normalized JSON error outputs."""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Intercepts Pydantic model and endpoint parameter validation failures."""
    errors = exc.errors()
    error_details = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "invalid value")
        error_details.append(f"[{loc}]: {msg}")
        
    error_message = f"Validation failed: {'; '.join(error_details)}"
    logger.error(error_message)
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": error_message}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catches unhandled server exceptions to format errors securely."""
    logger.exception(f"Unhandled system error encountered: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "OCR processing failed"}
    )


# Register endpoints router
app.include_router(router)
