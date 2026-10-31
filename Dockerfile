FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/data/hf_cache \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

# curl is only used by the container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (much smaller than the default CUDA wheel), then the pinned requirements.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY app.py agent.py config.py database.py exceptions.py llm.py logging_config.py \
     models.py rag.py schemas.py security.py services.py tools.py LICENSE NOTICE ./
COPY templates ./templates
COPY static ./static
COPY scripts ./scripts

# Non-root runtime user; writable state directories are meant to be mounted as volumes.
RUN useradd --create-home --uid 10001 akgpt \
    && mkdir -p /app/data /app/uploads /app/faiss_db \
    && chown -R akgpt:akgpt /app
USER akgpt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
