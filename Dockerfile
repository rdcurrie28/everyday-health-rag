# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY healthrag/ healthrag/
COPY corpus.toml ./
COPY data/ data/
RUN uv sync --frozen --no-dev --no-editable

RUN DATA_DIR=/app/data CORPUS_FILE=/app/corpus.toml INDEX_DIR=/app/index MODEL_CACHE_DIR=/app/models \
    /app/.venv/bin/python -m healthrag.build_index


FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder --chown=app /app/.venv /app/.venv
COPY --from=builder --chown=app /app/index /app/index
COPY --from=builder --chown=app /app/models /app/models
COPY --chown=app app.py ./
COPY --chown=app .streamlit/ .streamlit/

ENV APP_MODE=deployed \
    INDEX_DIR=/app/index \
    MODEL_CACHE_DIR=/app/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

USER app
EXPOSE 8080
CMD ["sh", "-c", "exec python -m healthrag.serve --server.port=${PORT:-8080} --server.address=0.0.0.0"]
