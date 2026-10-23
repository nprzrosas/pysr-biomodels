# Expanded 100-model Docker deployment

This image packages the validated expanded benchmark inputs: the historical
43-model cohort plus the 57 models recovered through corrected filename
discovery and model-aware metadata. This is the same scientific protocol, not
a new methodological cohort. The historical 43-model manifest remains a
separate, unchanged artifact.

The supported runtime platform is `linux/amd64`. The frozen
`libroadrunner==2.8.0` dependency has no Linux arm64 wheel, so the Dockerfile
rejects other target architectures. The image remains non-root (UID 10001),
installs the frozen `uv.lock`, and initializes Julia/PySR during the build.

## Frozen input bundle

The deterministic bundle is stored at `docker/expanded_100_inputs`:

```text
docker/expanded_100_inputs/
├── models/
│   └── <100 model_id directories>/<original SBML filename>
└── metadata/
    ├── benchmark_ready_100.json
    ├── model_time.jsonl
    └── bundle_provenance.json
```

It contains exactly the 100 resolved SBML files, preserving their original
filenames, and exactly the corresponding 100 model-aware metadata rows. The
CSV inventory is validation evidence only and is not runtime configuration.
The full 497-model source tree and historical CSV metadata are not copied into
the image.

Rebuild the bundle only from the validated native sources. The destination
must not already exist; the builder deliberately refuses to overwrite it:

```bash
uv run python -m src.docker_input_bundle
uv run python -m src.docker_input_bundle \
  --validate-only \
  --output docker/expanded_100_inputs
```

The builder uses the production `load_cohort_manifest`, `resolve_model_path`,
and `load_simulation_time` functions. It rejects duplicate IDs, missing or
ambiguous models, missing or duplicate metadata, filename mismatches,
nonpositive/nonfinite simulation times, and any cohort size other than 100.
`bundle_provenance.json` records source hashes and per-model SBML/metadata
hashes; it never repairs source inputs.

## Build

From the repository root:

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --tag pysr-only:expanded-100-exact \
  --load \
  .
```

The build validates all bundled manifest entries against the bundled SBML and
model-aware JSONL without running PySR. The immutable image paths are:

| Purpose | Image path | Runtime setting |
|---|---|---|
| Expanded model root | `/opt/pysr/inputs/expanded_100/models` | `PYSR_MODEL_ROOT` |
| Model-aware times | `/opt/pysr/inputs/expanded_100/metadata/model_time.jsonl` | `PYSR_MODEL_TIME_METADATA_PATH` |
| 100-model manifest | `/opt/pysr/inputs/expanded_100/metadata/benchmark_ready_100.json` | `PYSR_EXPANDED_COHORT_MANIFEST` |

No scientific input mount is needed on the compute machine.

## Output boundary

One writable host root is mounted at `/output`:

| Purpose | Container path | Runtime setting |
|---|---|---|
| Scientific results | `/output/results` | `PYSR_RESULTS_ROOT` |
| Manifests, logs, telemetry, locks | `/output/runs` | `PYSR_RUNS_ROOT` |
| Project cache | `/output/cache/project` | `PYSR_CACHE_ROOT` |
| Julia runtime cache | `/output/cache/julia` | `JULIA_DEPOT_PATH` writable layer |
| uv runtime cache | `/output/cache/uv` | `UV_CACHE_DIR` |

The host output directory must exist and be writable by container UID 10001
(or have an equivalent ACL). Do not run the image as root to bypass output
permissions. Results, manifests, logs, telemetry, and mutable caches therefore
survive container removal while the image and its scientific inputs remain
immutable.

## Input and runtime smoke validation

This validates all 100 input mappings without initializing PySR or Julia:

```bash
docker run --rm --platform linux/amd64 \
  pysr-only:expanded-100-exact \
  -m src.container_smoke \
  --validate-cohort-manifest \
  /opt/pysr/inputs/expanded_100/metadata/benchmark_ready_100.json
```

The full dependency and RoadRunner smoke can inspect one bundled model without
fitting PySR:

```bash
docker run --rm --platform linux/amd64 \
  --mount type=bind,src=/HOST/OUTPUT,dst=/output \
  pysr-only:expanded-100-exact \
  -m src.container_smoke \
  --model-id BIOMD0000000079 \
  --simulate \
  --check-writable
```

## Memory controls

The three memory controls are deliberately separate:

1. Docker `--memory` is the hard cgroup ceiling for the entire container.
   Setting `--memory-swap` to the same value prevents additional swap-backed
   allowance. Crossing the hard ceiling can cause a kernel OOM kill.
2. `--emergency-available-memory-gib` is the existing global batch guard. After
   the configured consecutive low-available-memory samples, it stops new
   scheduling, terminates every owned worker, records the event, and produces
   `memory_safety_stopped`.
3. `--per-model-memory-gib` and `--per-model-timeout-minutes` are operational
   limits for one model worker. Persistent recursive RSS excess or timeout
   censors only that worker; remaining models continue and the manifest records
   the operational reason.

These controls do not alter PySR settings, simulation times, derivative
generation, candidate generation, selection, or held-out-test protections.
Choose machine-dependent values only after inspecting the Ubuntu host and a
representative run.

## One-command expanded pretest batch

Replace every angle-bracket placeholder with a reviewed value for the Ubuntu
machine. `--phase pretest` intentionally stops before held-out evaluation, and
exact derivatives reproduce the validated same-protocol expanded benchmark.

```bash
docker run --rm \
  --platform linux/amd64 \
  --cpus <HOST_CPU_LIMIT> \
  --memory <GLOBAL_MEMORY_LIMIT> \
  --memory-swap <GLOBAL_MEMORY_LIMIT> \
  --mount type=bind,src=/HOST/OUTPUT,dst=/output \
  pysr-only:expanded-100-exact \
  -m src.run_batch \
  --manifest /opt/pysr/inputs/expanded_100/metadata/benchmark_ready_100.json \
  --derivative-method exact \
  --phase pretest \
  --workers <SAFE_WORKER_COUNT> \
  --run-id <UNIQUE_RUN_ID> \
  --min-available-memory-gib <PRELAUNCH_AVAILABLE_GIB> \
  --emergency-available-memory-gib <GLOBAL_EMERGENCY_AVAILABLE_GIB> \
  --memory-emergency-samples 2 \
  --per-model-memory-gib <PER_MODEL_RSS_GIB> \
  --per-model-memory-samples 2 \
  --per-model-timeout-minutes <PER_MODEL_TIMEOUT_MINUTES> \
  --model-failure-exit-policy success
```

The explicit `success` exit policy returns zero only when orchestration reaches
`completed_with_failures`; interruption, orchestration failure, and global
memory-safety stop remain nonzero. The run manifest under `/output/runs` is
authoritative. Individual resource censoring never authorizes held-out testing.
A later `test` phase still requires the separate `--authorize-held-out-test`
flag and the verified pretest checkpoints.

## Provenance and troubleshooting

Inspect the image revision and platform with:

```bash
docker image inspect pysr-only:expanded-100-exact \
  --format '{{.Id}} {{.Os}}/{{.Architecture}} {{index .Config.Labels "org.opencontainers.image.revision"}}'
```

- If `/output` is not writable, fix host ownership or ACLs for UID 10001.
- If the global emergency guard fires, inspect telemetry before changing the
  reviewed worker count or thresholds.
- If one model is resource-censored, use its manifest `failure_info`; do not
  reinterpret it as a scientific inference failure.
- Do not loosen frozen dependencies or scientific settings to work around an
  unsupported architecture or insufficient host resources.
