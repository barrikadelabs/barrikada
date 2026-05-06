FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.runtime.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.runtime.txt


FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    HOME=/home/barrikada \
    BARRIKADA_ARTIFACTS_DIR=/app/core/models \
    BARRIKADA_LAYER_E_OUTPUT_DIR=/app/core/models/layer_e \
    HF_HOME=/home/barrikada/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/home/barrikada/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/home/barrikada/.cache/sentence_transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash barrikada
RUN mkdir -p /home/barrikada/.cache/huggingface /home/barrikada/.cache/sentence_transformers \
    && chown -R barrikada:barrikada /app /home/barrikada

COPY --from=builder /install /usr/local

COPY api /app/api
COPY core /app/core
COPY models /app/models
COPY config /app/config
COPY scripts /app/scripts
COPY docker_entrypoint.sh /app/docker_entrypoint.sh

RUN chmod +x /app/docker_entrypoint.sh && chown barrikada:barrikada /app/docker_entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=5)"

USER barrikada

ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
