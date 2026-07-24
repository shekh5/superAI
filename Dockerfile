# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install build deps only in this stage (keeps final image small)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

ARG BAKE_EMBEDDING_MODEL=true
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ENV SENTENCE_TRANSFORMERS_HOME=/build/models
RUN mkdir -p /build/models && if [ "$BAKE_EMBEDDING_MODEL" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"; \
    fi

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (never run containers as root in production)
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy only installed packages from builder stage, not build tools
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /build/models /opt/superai-models
COPY app ./app
COPY reasoning_chain ./reasoning_chain
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/opt/superai-models \
    PORT=8000

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
