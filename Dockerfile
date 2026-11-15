FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    CRC32C_PURE_PYTHON=1 \
    HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial \
    HDF5_PKGCONFIG_NAME=hdf5-serial \
    CFLAGS="-I/usr/include/hdf5/serial" \
    LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/serial:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    pkg-config \
    libhdf5-dev \
    ninja-build \
    git \
    && ln -sf /usr/include/hdf5/serial/* /usr/include/ \
    && ln -sf /usr/lib/*/hdf5/serial/lib* /usr/lib/ \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version ./

RUN uv python install 3.15t

RUN uv sync --frozen --no-dev --no-install-project --extra serve > /tmp/sync.log 2>&1 || (tail -n 80 /tmp/sync.log >&2 && exit 1)

COPY src ./src

RUN uv sync --frozen --no-dev --extra serve > /tmp/sync2.log 2>&1 || (tail -n 80 /tmp/sync2.log >&2 && exit 1)

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libhdf5-103-1 \
    && ln -sf /usr/lib/*/hdf5/serial/lib* /usr/lib/ \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/python /opt/python
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NASA_FARM_LAKE=/app/lake \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/serial:${LD_LIBRARY_PATH}"

EXPOSE 8000

ENTRYPOINT ["nasa-farm", "serve", "--host", "0.0.0.0", "--port", "8000"]
