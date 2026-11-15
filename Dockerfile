FROM ghcr.io/astral-sh/uv:trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_FIND_LINKS=/app/wheels

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
COPY wheels ./wheels

RUN uv python install 3.15t

RUN uv sync --frozen --no-dev --no-install-project --extra serve --find-links ./wheels

COPY src ./src

RUN uv sync --frozen --no-dev --extra serve --find-links ./wheels

FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/python /opt/python
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NASA_FARM_LAKE=/app/lake

EXPOSE 8000

CMD ["nasa-farm", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
