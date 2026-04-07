FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Health check hits the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# HF Spaces uses port 7860 by default
EXPOSE 7860

# Run FastAPI server — this is what the OpenEnv validator pings for /reset /step /state
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
