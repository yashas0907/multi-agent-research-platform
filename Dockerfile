# Multi-stage Dockerfile — reproducible production build.
# Build:  docker build -t research-platform .
# Run:    docker run -p 8000:8000 --env-file .env research-platform

FROM python:3.12-slim AS backend

WORKDIR /app

# Install dependencies first (layer caching)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code
COPY backend ./backend

WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
