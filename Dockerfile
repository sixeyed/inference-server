# syntax=docker/dockerfile:1
# The engine as a container. No model runs in here - MTPLX and MLX are Apple
# Silicon only - so this is the router, recorder and dashboard, driving a Mac's
# model servers over SSH. See "Running in a container" in README.md.

FROM python:3.14-slim AS runtime-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home inference \
    && mkdir -p /data /etc/inference/ssh \
    && chown inference:inference /data
# Run from source, not a wheel: `python -m inference` from a bare checkout is
# the supported way in, and the package data (static/, prompts/, toolsets/)
# comes along without a packaging manifest to keep in step.
WORKDIR /app
COPY inference ./inference
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# ---- the suite, on Linux, against the code being shipped -------------------
FROM runtime-base AS test
RUN apt-get update \
    && apt-get install -y --no-install-recommends lsof procps git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
# No Mac to detect in here; the suite stands the m4 profile in.
ENV INFERENCE_MACHINE=m4
RUN python -m pytest -q

# ---- what runs ------------------------------------------------------------
FROM runtime-base
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source=https://github.com/sixeyed/inference-server \
      org.opencontainers.image.description="Local coding models on Apple Silicon, one at a time, on demand - the engine" \
      org.opencontainers.image.licenses=MIT
# Listen on every interface *inside* the container - loopback in here is not
# reachable from the host. What the host exposes is decided by the port
# mapping, which compose.yaml keeps on 127.0.0.1.
ENV INFERENCE_GIT_SHA=$GIT_SHA \
    INFERENCE_CONFIG=/etc/inference/config.json \
    INFERENCE_BIND=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER inference
EXPOSE 8084
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
