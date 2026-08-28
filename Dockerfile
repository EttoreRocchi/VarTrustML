# =============================================================================
# Stage 1: Builder
# =============================================================================
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment accessible by any user
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python dependencies (filter nvidia-nccl which is an optional xgboost dep)
COPY requirements.lock.txt ./
RUN grep -v '^nvidia-' requirements.lock.txt > requirements.filtered.txt \
    && pip install -r requirements.filtered.txt

# Install vartrustml
COPY pyproject.toml .
COPY vartrustml/ vartrustml/
RUN pip install --no-deps .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM ${BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="VarTrustML" \
      org.opencontainers.image.description="Machine learning for structural variant trust assessment" \
      org.opencontainers.image.source="https://github.com/EttoreRocchi/VarTrustML" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PIP_DEFAULT_TIMEOUT=100 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder (accessible by any user)
COPY --from=builder /opt/venv /opt/venv

WORKDIR /data

# Verify installation
RUN vartrustml version

ENTRYPOINT ["vartrustml"]
CMD ["--help"]
