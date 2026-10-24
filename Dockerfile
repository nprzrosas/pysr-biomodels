# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 AS uv
FROM python:${PYTHON_VERSION}-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89

ARG TARGETARCH
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="PySR-only benchmark"
LABEL org.opencontainers.image.description="Reproducible ODE inference runtime"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/nprzrosas/pysr_only"

# libroadrunner 2.8.0 has no Linux arm64 wheel in the frozen dependency lock.
RUN if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "Unsupported Linux architecture: ${TARGETARCH}. libroadrunner==2.8.0 requires linux/amd64."; \
        exit 1; \
    fi

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libexpat1 \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/output/cache/uv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_JULIAPKG_PROJECT=/output/cache/julia_env \
    JULIA_DEPOT_PATH=/output/cache/julia:/opt/julia-depot \
    PYSR_MODEL_ROOT=/opt/pysr/inputs/expanded_100/models \
    PYSR_MODEL_TIME_METADATA_PATH=/opt/pysr/inputs/expanded_100/metadata/model_time.jsonl \
    PYSR_EXPANDED_COHORT_MANIFEST=/opt/pysr/inputs/expanded_100/metadata/benchmark_ready_100.json \
    PYSR_RESULTS_ROOT=/output/results \
    PYSR_RUNS_ROOT=/output/runs \
    PYSR_CACHE_ROOT=/output/cache/project \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    JULIA_NUM_THREADS=1 \
    PYSR_GIT_COMMIT=${VCS_REF}

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY main.py ./main.py
COPY docker/expanded_100_inputs /opt/pysr/inputs/expanded_100

RUN chmod -R a-w /opt/pysr/inputs/expanded_100 \
    && chmod -R a+rX /opt/pysr/inputs/expanded_100 \
    && .venv/bin/python -m src.container_smoke \
       --validate-cohort-manifest /opt/pysr/inputs/expanded_100/metadata/benchmark_ready_100.json

# Initialize Julia and the PySR Julia environment once in immutable image paths.
RUN mkdir -p /opt/julia-depot /opt/pysr/julia_env_template \
    && PYTHON_JULIAPKG_PROJECT=/opt/pysr/julia_env_template \
       JULIA_DEPOT_PATH=/opt/julia-depot \
       .venv/bin/python -c \
       "import pysr; from juliacall import Main as jl; print(jl.seval('VERSION'))" \
    && chmod -R a-w /opt/pysr/julia_env_template \
    && chmod -R a+rX /opt/pysr/julia_env_template

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /output \
    && chown -R appuser:appuser \
        /output \
        /home/appuser

USER appuser

VOLUME ["/output"]

ENTRYPOINT ["/app/.venv/bin/python", "-m", "src.container_entrypoint"]
CMD ["-m", "src.container_smoke", "--resources-only"]
