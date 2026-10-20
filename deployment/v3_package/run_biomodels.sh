#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="pysr-biomodels:latest"
WORKERS="${PYSR_WORKERS:-1}"

VOLUME_PREFIX="${PYSR_VOLUME_PREFIX:-pysr}"
RESULTS_VOLUME="${VOLUME_PREFIX}-results"
RUNS_VOLUME="${VOLUME_PREFIX}-runs"
CACHE_VOLUME="${VOLUME_PREFIX}-cache"

READY_MANIFEST="/deployment/config/biomodels_ready.json"
STANDARD_MANIFEST="/deployment/config/biomodels_standard.json"
EXTREME_MANIFEST="/deployment/config/biomodels_extreme.json"
SMOKE_MANIFEST="/deployment/config/biomodels_smoke_1_2_3.json"


usage() {
    cat <<USAGE
Usage:
  ./run_biomodels.sh preflight
  ./run_biomodels.sh smoke
  ./run_biomodels.sh infer
  ./run_biomodels.sh infer-extreme
  ./run_biomodels.sh export-results DESTINATION
  ./run_biomodels.sh export-runs DESTINATION

Commands:
  preflight
      Check all 43 protocol-compatible BioModels.

  smoke
      Run exact inference and validation on the
      1-, 2-, and 3-state smoke cohort.

  infer
      Run exact inference and validation on the 41-model
      standard cohort (1-59 states).

  infer-extreme
      Run exact inference and validation on the two
      402-state models. These are intentionally separate
      from the main inference cohort.

  export-results DESTINATION
      Export persistent scientific results from the Docker
      named volume to a host directory.

  export-runs DESTINATION
      Export manifests, logs, and telemetry from the Docker
      named volume to a host directory.

Environment:
  PYSR_WORKERS=N
      Number of models processed concurrently.
      Default: 1.

  PYSR_VOLUME_PREFIX=NAME
      Prefix for Docker named volumes.
      Default: pysr.

      Default volumes:
        pysr-results
        pysr-runs
        pysr-cache

Held-out test data are never opened by this launcher.
USAGE
}


require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: Docker is not installed or not in PATH." >&2
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Docker daemon is not accessible." >&2
        exit 1
    fi

    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo "ERROR: Docker image ${IMAGE_NAME} is not loaded." >&2
        echo "Run ./setup.sh first." >&2
        exit 1
    fi
}


require_volume() {
    local volume="$1"

    if ! docker volume inspect "${volume}" >/dev/null 2>&1; then
        echo "ERROR: Docker volume ${volume} does not exist." >&2
        echo "Run ./setup.sh first." >&2
        exit 1
    fi
}


export_volume() {
    local volume="$1"
    local destination="$2"

    if [[ -z "${destination}" ]]; then
        echo "ERROR: export destination cannot be empty." >&2
        exit 2
    fi

    mkdir -p "${destination}"
    destination="$(cd "${destination}" && pwd)"

    echo "=== Exporting Docker volume ==="
    echo "Volume:      ${volume}"
    echo "Destination: ${destination}"
    echo

    docker run --rm \
        --platform linux/amd64 \
        --entrypoint /bin/sh \
        -v "${volume}:/source:ro" \
        "${IMAGE_NAME}" \
        -c 'tar -C /source -cf - .' \
        | tar -C "${destination}" -xf -

    echo
    echo "EXPORT PASS"
}


if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

COMMAND="$1"

if [[ "${COMMAND}" == "-h" || "${COMMAND}" == "--help" || "${COMMAND}" == "help" ]]; then
    usage
    exit 0
fi

require_docker

case "${COMMAND}" in
    export-results)
        if [[ $# -ne 2 ]]; then
            usage >&2
            exit 2
        fi

        require_volume "${RESULTS_VOLUME}"
        export_volume "${RESULTS_VOLUME}" "$2"
        exit 0
        ;;

    export-runs)
        if [[ $# -ne 2 ]]; then
            usage >&2
            exit 2
        fi

        require_volume "${RUNS_VOLUME}"
        export_volume "${RUNS_VOLUME}" "$2"
        exit 0
        ;;

    preflight)
        [[ $# -eq 1 ]] || {
            usage >&2
            exit 2
        }
        MANIFEST="${READY_MANIFEST}"
        PHASE="preflight"
        RUN_ID="biomodels43_preflight"
        ;;

    smoke)
        [[ $# -eq 1 ]] || {
            usage >&2
            exit 2
        }
        MANIFEST="${SMOKE_MANIFEST}"
        PHASE="pretest"
        RUN_ID="biomodels_smoke_1_2_3_exact"
        ;;

    infer)
        [[ $# -eq 1 ]] || {
            usage >&2
            exit 2
        }
        MANIFEST="${STANDARD_MANIFEST}"
        PHASE="pretest"
        RUN_ID="biomodels41_standard_exact"
        ;;

    infer-extreme)
        [[ $# -eq 1 ]] || {
            usage >&2
            exit 2
        }
        MANIFEST="${EXTREME_MANIFEST}"
        PHASE="pretest"
        RUN_ID="biomodels2_extreme_exact"
        ;;

    *)
        echo "ERROR: unknown command: ${COMMAND}" >&2
        usage >&2
        exit 2
        ;;
esac

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PYSR_WORKERS must be a positive integer." >&2
    exit 2
fi

require_volume "${RESULTS_VOLUME}"
require_volume "${RUNS_VOLUME}"
require_volume "${CACHE_VOLUME}"

echo "=== PySR BioModels run ==="
echo "Command:           ${COMMAND}"
echo "Phase:             ${PHASE}"
echo "Workers:           ${WORKERS}"
echo "Run ID:            ${RUN_ID}"
echo "Derivative method: exact"
echo "Held-out test:     CLOSED"
echo "Results volume:    ${RESULTS_VOLUME}"
echo "Runs volume:       ${RUNS_VOLUME}"
echo "Cache volume:      ${CACHE_VOLUME}"
echo

exec docker run --rm \
    --platform linux/amd64 \
    -e PYSR_MODEL_ROOT=/inputs/models \
    -e PYSR_SIMULATION_METADATA_PATH=/inputs/metadata/time_curated_images.csv \
    -e PYSR_RESULTS_ROOT=/results \
    -e PYSR_RUNS_ROOT=/runs \
    -e PYSR_CACHE_ROOT=/cache/project \
    -v "${PACKAGE_DIR}/inputs/models:/inputs/models:ro" \
    -v "${PACKAGE_DIR}/inputs/metadata/time_curated_images.csv:/inputs/metadata/time_curated_images.csv:ro" \
    -v "${PACKAGE_DIR}/config:/deployment/config:ro" \
    -v "${RESULTS_VOLUME}:/results" \
    -v "${RUNS_VOLUME}:/runs" \
    -v "${CACHE_VOLUME}:/cache" \
    "${IMAGE_NAME}" \
    -m src.run_batch \
    --manifest "${MANIFEST}" \
    --derivative-method exact \
    --workers "${WORKERS}" \
    --phase "${PHASE}" \
    --run-id "${RUN_ID}"
