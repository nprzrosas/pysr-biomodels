#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PYSR_IMAGE:-pysr-biomodels:latest}"
VOLUME_PREFIX="${PYSR_VOLUME_PREFIX:-pysr}"
RESULTS_VOLUME="${VOLUME_PREFIX}-results"

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"


usage() {
    cat <<'EOF'
Usage:
  ./report_biomodels.sh all [OUTPUT_DIR]
  ./report_biomodels.sh model MODEL_ID [OUTPUT_DIR]

Examples:
  ./report_biomodels.sh all
  ./report_biomodels.sh all ./reports
  ./report_biomodels.sh model BIOMD0000000079
  ./report_biomodels.sh model BIOMD0000000079 ./reports

Environment:
  PYSR_VOLUME_PREFIX   Docker volume prefix used by setup/run scripts.
                       Default: pysr

  PYSR_IMAGE           Docker image containing the scientific runtime.
                       Default: pysr-biomodels:latest

Reporting reads the existing results volume read-only.

It does NOT:
  - run PySR;
  - perform model selection;
  - rerun ODE integration;
  - access held-out test data.
EOF
}


die() {
    echo "ERROR: $*" >&2
    exit 1
}


require_docker() {
    command -v docker >/dev/null 2>&1 \
        || die "docker was not found."

    docker image inspect "${IMAGE}" >/dev/null 2>&1 \
        || die "Docker image ${IMAGE} is not available."

    docker volume inspect "${RESULTS_VOLUME}" >/dev/null 2>&1 \
        || die "Results volume ${RESULTS_VOLUME} does not exist."
}


run_reporter() {
    local output_dir="$1"
    shift

    mkdir -p "${output_dir}"
    output_dir="$(cd "${output_dir}" && pwd)"

    echo "=== PySR BioModels reporting ==="
    echo "Image:          ${IMAGE}"
    echo "Results volume: ${RESULTS_VOLUME}"
    echo "Output:         ${output_dir}"
    echo "Held-out test:  CLOSED"
    echo

    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e MPLCONFIGDIR=/tmp/matplotlib \
        --entrypoint /app/.venv/bin/python \
        -v "${SCRIPT_DIR}/reporting:/reporting:ro" \
        -v "${RESULTS_VOLUME}:/results:ro" \
        -v "${output_dir}:/reports" \
        "${IMAGE}" \
        /reporting/report_biomodels.py \
        --results-root /results \
        --reports-root /reports \
        "$@"

    echo
    echo "REPORT PASS"
    echo "Reports written to:"
    echo "  ${output_dir}"
}


main() {
    [[ $# -ge 1 ]] || {
        usage
        exit 2
    }

    local command="$1"
    shift

    if [[ "${command}" == "-h" || \
          "${command}" == "--help" || \
          "${command}" == "help" ]]; then
        usage
        exit 0
    fi

    require_docker

    case "${command}" in
        all)
            [[ $# -le 1 ]] \
                || die "Too many arguments for 'all'."

            local output_dir="${1:-${SCRIPT_DIR}/reports}"

            run_reporter "${output_dir}"
            ;;

        model)
            [[ $# -ge 1 ]] \
                || die "'model' requires a BioModels model ID."

            [[ $# -le 2 ]] \
                || die "Too many arguments for 'model'."

            local model_id="$1"
            local output_dir="${2:-${SCRIPT_DIR}/reports}"

            [[ "${model_id}" =~ ^BIOMD[0-9]+$ ]] \
                || die "Invalid model ID: ${model_id}"

            run_reporter \
                "${output_dir}" \
                --model "${model_id}"
            ;;

        *)
            usage >&2
            echo >&2
            die "Unknown command: ${command}"
            ;;
    esac
}


main "$@"