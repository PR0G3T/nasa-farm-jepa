FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_CONCURRENT_BUILDS=1 \
    MAKEFLAGS="-j1" \
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    MAX_JOBS=1 \
    CRC32C_PURE_PYTHON=1 \
    HDF5_INCLUDEDIR=/usr/include/hdf5/serial \
    HDF5_LIBDIR=/usr/lib/x86_64-linux-gnu/hdf5/serial \
    PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig" \
    CFLAGS="-I/usr/include/hdf5/serial" \
    LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/hdf5/serial:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libaec-dev \
    zlib1g-dev \
    libyaml-dev \
    ninja-build \
    git \
    && ln -sf /usr/include/hdf5/serial/* /usr/include/ \
    && ln -sf /usr/lib/*/hdf5/serial/lib* /usr/lib/ \
    && ldconfig \
    && printf '#!/bin/sh\nexec /usr/bin/ninja -j1 "$@"\n' > /usr/local/bin/ninja \
    && chmod +x /usr/local/bin/ninja \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version ./

RUN uv python install 3.15t

RUN uv sync --frozen --no-dev --no-install-project --extra serve --verbose --config-settings-package numcodecs:compile-args="-j1"

COPY src ./src

RUN uv sync --frozen --no-dev --extra serve --verbose --config-settings-package numcodecs:compile-args="-j1"

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
