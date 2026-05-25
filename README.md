# PaddleOCR FastAPI Wrapper Service

A lightweight, high-performance FastAPI wrapper API around PaddleOCR / Baidu AI Studio OCR. This wrapper processes files in memory, forwards requests to the downstream Paddle OCR API, and normalizes the parsed layout text into structured JSON page blocks with extracted key-value parameters.

---

## Folder Structure

```bash
project-root/
│
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application entrypoint
│   ├── routes.py        # Web route definitions (/health, /parse-text)
│   ├── config.py        # Safe environment validation loader
│   ├── schemas.py       # Pydantic response/request serializers
│   └── helpers.py       # Reusable parsing, upload, & connectivity helpers
│
├── requirements.txt     # Cleaned minimum package list for production
├── render.yaml          # Render Blueprint deployment configuration
├── .env.example
├── .gitignore
├── README.md
└── Procfile             # Command definition for Render runtime
```

---

## Local Development Setup

1. **Install Virtual Environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install Minimal Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Create a `.env` file from the example template and fill in your details:
   ```env
   PADDLE_API_KEY=your_paddle_token_here
   PADDLE_BASE_URL=https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
   PORT=8000
   ```

4. **Start local service:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

## Render.com Production Deployment

This project is configured for seamless deployment to **Render.com** using either a Blueprint spec or a direct Web Service config.

### Option A: Using Render Blueprints (Recommended)
1. Commit the files to your GitHub repository.
2. Go to the **Blueprints** dashboard on Render.
3. Link your repository. Render will automatically read `render.yaml` and provision your Web Service with the correct startup command, environment vars, and optimal runtime version.

### Option B: Manual Web Service Setup
If creating the service manually:
* **Runtime:** `Python`
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
* **Environment Variables:** Configure `PADDLE_API_KEY` and `PADDLE_BASE_URL` on the Environment dashboard tab.

---

## API Documentation & Verification

Once running, the interactive documentation is exposed at:
* **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Check Health
```bash
curl http://localhost:8000/health
```

### Parse Image File
```bash
curl -X POST -F "file=@/path/to/image.png" http://localhost:8000/parse-text
```
