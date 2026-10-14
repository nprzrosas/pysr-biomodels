# Docker foundation

This document describes the Phase-6 container foundation. It validates the
runtime and deployment boundary; it does not establish scientific equivalence
between native and container executions. That comparison belongs to Phase 7.

## Prerequisites and supported platform

Use Docker Engine or Docker Desktop with Buildx. The currently validated image
platform is `linux/amd64`. The frozen `libroadrunner==2.8.0` dependency does not
publish a Linux arm64 wheel, so native `linux/arm64` builds are deliberately
rejected. Apple Silicon Macs can run the amd64 image through Docker Desktop
emulation by specifying `--platform linux/amd64`.

The Ubuntu compute machine's architecture must be checked with `uname -m`
before choosing an image platform.

## Build

```bash
docker build \
  --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --tag pysr-only:phase6-amd64 \
  .
```

The image uses Python 3.11 and installs the locked dependencies with
`uv sync --frozen`. Julia and the PySR Julia environment are initialized during
the build. Read-only Julia packages are baked under `/opt/julia-depot`; runtime
logs, compiled caches, and updates use the writable `/cache/julia` depot.

## Filesystem and mounts

The application uses environment-based paths with native-compatible defaults:

| Purpose | Container path | Environment variable | Access |
|---|---|---|---|
| BioModels tree | `/inputs/models` | `PYSR_MODEL_ROOT` | read-only |
| Simulation metadata | `/inputs/metadata/time_curated_images.csv` | `PYSR_SIMULATION_METADATA_PATH` | read-only |
| Scientific outputs | `/results` | `PYSR_RESULTS_ROOT` | writable |
| Manifests/logs/telemetry | `/runs` | `PYSR_RUNS_ROOT` | writable |
| Application cache | `/cache/project` | `PYSR_CACHE_ROOT` | writable |
| Julia depot | `/cache/julia` | `JULIA_DEPOT_PATH` | writable |
| uv cache | `/cache/uv` | `UV_CACHE_DIR` | writable |

Mount the input roots read-only and use isolated writable directories for
outputs. Do not point container smoke tests at frozen native results.

```bash
docker run --rm --platform linux/amd64 \
  --mount type=bind,src=/HOST/MODELS,dst=/inputs/models,readonly \
  --mount type=bind,src=/HOST/METADATA,dst=/inputs/metadata,readonly \
  --mount type=bind,src=/HOST/RESULTS,dst=/results \
  --mount type=bind,src=/HOST/RUNS,dst=/runs \
  --mount type=bind,src=/HOST/CACHE,dst=/cache \
  pysr-only:phase6-amd64 \
  -m src.container_smoke --model-id BIOMD0000000079 --simulate \
  --check-writable
```

Bind-mounted writable directories must grant the container user write access.
The default non-root user has UID 10001. Named volumes inherit container-side
ownership; bind mounts may require host-side ownership or ACL configuration.
Do not make scientific input mounts writable.

## CLI and worker controls

The image entrypoint is the environment's Python executable and preserves all
arguments. A future batch invocation therefore uses the same application CLI:

```bash
docker run --rm --platform linux/amd64 --cpus=2 --memory=8g \
  ...mounts... \
  pysr-only:phase6-amd64 \
  -m src.run_batch \
  --manifest /inputs/metadata/cohort.json \
  --derivative-method savgol \
  --workers 2 \
  --phase pretest \
  --run-id example-pretest \
  --min-available-memory-gib 6 \
  --emergency-available-memory-gib 2
```

`--workers` remains a runtime choice. The image does not select worker counts
automatically. Every scientific worker receives these fixed isolation values:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
JULIA_NUM_THREADS=1
```

Docker CPU quotas and cgroup-v2 memory limits are detected by the existing
resource monitor. The pre-launch memory floor and persistent runtime emergency
threshold are explicit CLI settings; they do not alter scientific algorithms.

## Lightweight checks

Resource discovery requires no scientific runtime initialization:

```bash
docker run --rm --cpus=2 --memory=4g pysr-only:phase6-amd64 \
  -m src.container_smoke --resources-only
```

The full smoke imports NumPy, SciPy, pandas, SymPy, psutil, PySR, juliacall,
RoadRunner, and libSBML. With mounted inputs, it may also perform preflight and
a three-point RoadRunner simulation. It never fits PySR or creates benchmark
results.

To verify the memory gate safely, run a preflight batch with a 4 GiB container
limit and `--min-available-memory-gib 6`. The expected nonzero exit occurs
before a run manifest, scientific namespace, or worker is created.

## Provenance

Inspect the embedded source revision and image platform with:

```bash
docker image inspect pysr-only:phase6-amd64 \
  --format '{{.Id}} {{.Os}}/{{.Architecture}} {{index .Config.Labels "org.opencontainers.image.revision"}}'
```

The container smoke report records Python, package, Julia, RoadRunner, and
libSBML versions. Run manifests use the embedded `PYSR_GIT_COMMIT` when the
excluded `.git` directory is unavailable.

## Apple Silicon and Ubuntu

On Apple Silicon, build or run with `--platform linux/amd64`; Docker Desktop
uses emulation. This is not native arm64 validation and may be slower. A native
Linux arm64 image remains blocked by the missing pinned libroadrunner wheel.

On Ubuntu Linux, first run `uname -m`. An `x86_64` host can use the validated
amd64 image natively. Any other architecture requires separate dependency and
runtime validation before scientific use.

## Signals and troubleshooting

The batch coordinator handles SIGINT and SIGTERM, stops new scheduling, and
terminates only its owned process groups. Completed checkpoints remain
available for verified resume.

- If Julia cannot write logs, verify that `/cache/julia` is writable.
- If uv cannot write, verify `/cache/uv` permissions.
- If mounted outputs are read-only, adjust the host directory ownership/ACL;
  do not run the scientific container as root by default.
- If the memory gate blocks, increase the container memory limit or reduce the
  explicitly requested workload in a separately reviewed run.
- Do not use an arm64 build failure as justification to loosen the frozen
  scientific dependency versions.
