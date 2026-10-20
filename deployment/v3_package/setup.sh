#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_TAR="${PACKAGE_DIR}/pysr-biomodels-amd64.tar"
IMAGE_NAME="pysr-biomodels:latest"

VOLUME_PREFIX="${PYSR_VOLUME_PREFIX:-pysr}"
RESULTS_VOLUME="${VOLUME_PREFIX}-results"
RUNS_VOLUME="${VOLUME_PREFIX}-runs"
CACHE_VOLUME="${VOLUME_PREFIX}-cache"

EXPECTED_IMAGE_SHA256="4712dc679f7688ee88bc3b3e72237ea373573fb4f9c904588650b04c4d5133d5"
EXPECTED_METADATA_SHA256="146aa3e65032400d582713e02594e2827da2edaaa0f1dd7775475e57d99cf0c9"
EXPECTED_MODEL_MANIFEST_SHA256="a4a42fc5c821c78fa8da66550f2e10a6913a679e669181bf5cc5d781902f13e2"


echo "=== PySR BioModels setup ==="


die() {
    echo "ERROR: $*" >&2
    exit 1
}


require_command() {
    local command_name="$1"

    command -v "${command_name}" >/dev/null 2>&1 \
        || die "Required command not found: ${command_name}"
}


sha256_of() {
    local path="$1"

    sha256sum "${path}" | awk '{print $1}'
}


ensure_volume() {
    local volume_name="$1"

    if docker volume inspect "${volume_name}" >/dev/null 2>&1; then
        echo "Docker volume already exists: ${volume_name}"
    else
        docker volume create "${volume_name}" >/dev/null
        echo "Created Docker volume: ${volume_name}"
    fi
}


require_command docker
require_command sha256sum
require_command awk

docker info >/dev/null 2>&1 || {
    echo "ERROR: Docker is not available to the current user." >&2
    echo "The current user must be able to run Docker without sudo." >&2
    exit 1
}


echo
echo "=== Verifying package files ==="

[[ -f "${IMAGE_TAR}" ]] \
    || die "Missing Docker image archive: ${IMAGE_TAR}"

[[ -d "${PACKAGE_DIR}/inputs/models" ]] \
    || die "Missing model input directory."

[[ -d "${PACKAGE_DIR}/inputs/metadata" ]] \
    || die "Missing metadata input directory."

[[ -f "${PACKAGE_DIR}/inputs/SHA256SUMS.models" ]] \
    || die "Missing model checksum manifest."

[[ -f "${PACKAGE_DIR}/inputs/metadata/time_curated_images.csv" ]] \
    || die "Missing simulation metadata."

ACTUAL_IMAGE_SHA256="$(sha256_of "${IMAGE_TAR}")"

[[ "${ACTUAL_IMAGE_SHA256}" == "${EXPECTED_IMAGE_SHA256}" ]] \
    || die "Docker image archive checksum mismatch."

ACTUAL_METADATA_SHA256="$(
    sha256_of "${PACKAGE_DIR}/inputs/metadata/time_curated_images.csv"
)"

[[ "${ACTUAL_METADATA_SHA256}" == "${EXPECTED_METADATA_SHA256}" ]] \
    || die "Simulation metadata checksum mismatch."

ACTUAL_MODEL_MANIFEST_SHA256="$(
    sha256_of "${PACKAGE_DIR}/inputs/SHA256SUMS.models"
)"

[[ "${ACTUAL_MODEL_MANIFEST_SHA256}" == "${EXPECTED_MODEL_MANIFEST_SHA256}" ]] \
    || die "Model checksum manifest checksum mismatch."

(
    cd "${PACKAGE_DIR}/inputs"
    sha256sum -c SHA256SUMS.models
)

echo "Package checksums: PASS"


echo
echo "=== Loading Docker image ==="

docker load -i "${IMAGE_TAR}"

docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 \
    || die "Expected image ${IMAGE_NAME} was not loaded from ${IMAGE_TAR}."

IMAGE_ARCH="$(
    docker image inspect "${IMAGE_NAME}" \
        --format '{{.Architecture}}'
)"

IMAGE_USER="$(
    docker image inspect "${IMAGE_NAME}" \
        --format '{{.Config.User}}'
)"

[[ "${IMAGE_ARCH}" == "amd64" ]] \
    || die "Expected an amd64 image, found: ${IMAGE_ARCH}"

[[ -n "${IMAGE_USER}" ]] \
    || die "Docker image does not define a non-root runtime user."

[[ "${IMAGE_USER}" != "root" && "${IMAGE_USER}" != "0" ]] \
    || die "Docker image runtime user must not be root."

echo "Image:        ${IMAGE_NAME}"
echo "Architecture: ${IMAGE_ARCH}"
echo "Runtime user: ${IMAGE_USER}"
echo "Docker image: PASS"


echo
echo "=== Creating persistent volumes ==="

ensure_volume "${RESULTS_VOLUME}"
ensure_volume "${RUNS_VOLUME}"
ensure_volume "${CACHE_VOLUME}"


echo
echo "=== Checking mounted inputs and writable volumes ==="

docker run --rm \
    --entrypoint /bin/bash \
    -v "${PACKAGE_DIR}/inputs/models:/inputs/models:ro" \
    -v "${PACKAGE_DIR}/inputs/metadata:/inputs/metadata:ro" \
    -v "${RESULTS_VOLUME}:/results" \
    -v "${RUNS_VOLUME}:/runs" \
    -v "${CACHE_VOLUME}:/cache" \
    "${IMAGE_NAME}" \
    -lc '
        set -euo pipefail

        test -d /inputs/models
        test -f /inputs/metadata/time_curated_images.csv

        test -w /results
        test -w /runs
        test -w /cache

        touch /results/.setup_write_test
        touch /runs/.setup_write_test
        touch /cache/.setup_write_test

        rm -f \
            /results/.setup_write_test \
            /runs/.setup_write_test \
            /cache/.setup_write_test
    '

echo "Mount and volume access: PASS"


echo
echo "=== Runtime smoke check ==="

docker run --rm \
    --entrypoint /bin/bash \
    "${IMAGE_NAME}" \
    -lc '
        set -euo pipefail

        /app/.venv/bin/python -c "
import platform
import pysr
import roadrunner
import libsbml

print(\"Python runtime: PASS\")
print(\"Architecture:\", platform.machine())
print(\"PySR:\", pysr.__version__)
print(\"RoadRunner:\", roadrunner.__version__)
print(\"libSBML:\", libsbml.getLibSBMLDottedVersion())
"
    '

echo "Runtime smoke check: PASS"


echo
echo "=== Setup complete ==="
echo "Image:          ${IMAGE_NAME}"
echo "Results volume: ${RESULTS_VOLUME}"
echo "Runs volume:    ${RUNS_VOLUME}"
echo "Cache volume:   ${CACHE_VOLUME}"
echo
echo "No sudo or host-directory ownership changes were required."
echo "SETUP PASS"