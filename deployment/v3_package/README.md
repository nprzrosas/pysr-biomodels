# PySR BioModels Inference Package

This package provides a reproducible workflow for inferring systems of ordinary
differential equations (ODEs) from simulated BioModels trajectories using PySR
symbolic regression.

Starting from curated SBML models, the workflow generates training and
validation trajectories, constructs derivative targets, infers candidate
right-hand sides for each dynamic state, assembles them into coupled ODE
systems, evaluates their trajectory behavior, and selects systems using
validation data.

The package also provides batch execution, resumable runs, result reporting,
validation visualizations, and export of scientific results.

The supplied BioModels snapshot contains 43 models that satisfy the current
inference protocol. The main inference cohort contains 41 models with 1–59
dynamic states. Two additional 402-state models are provided separately because
their computational requirements are substantially larger.

## Target platform

The package has been validated on:

- Ubuntu 24.04 LTS
- x86_64 / AMD64
- Docker Engine
- CPU execution; no GPU is required

The included Docker image is `linux/amd64`.

The user running the package must be able to access the Docker daemon without
`sudo`.

## Quick start

From the package directory:

```bash
./setup.sh
./run_biomodels.sh preflight
./run_biomodels.sh smoke
./run_biomodels.sh infer
./report_biomodels.sh all
```

These commands:

1. prepare and verify the computational environment;
2. check the 43 compatible models;
3. perform a small end-to-end inference and validation run;
4. run the main 41-model inference cohort; and
5. generate tables of selected equations, validation metrics, and figures.

Scientific results and execution state are stored persistently. If an inference
run is interrupted, executing the same command again reuses completed work
rather than starting the entire workflow again.

## Setup

Run:

```bash
./setup.sh
```

`setup.sh`:

- verifies the packaged model and metadata checksums;
- loads the Docker image;
- creates persistent Docker named volumes;
- verifies that persistent storage is writable by the non-root container user;
  and
- checks the Python, Julia, PySR, RoadRunner, and libSBML runtime.

No `sudo`, `chown`, or host output-directory ownership changes are required.

## Check all compatible models

Run:

```bash
./run_biomodels.sh preflight
```

This checks all 43 models currently classified as compatible with the inference
protocol.

Preflight verifies model compatibility and required inputs without performing
full PySR inference.

## Small end-to-end inference test

Run:

```bash
./run_biomodels.sh smoke
```

This performs inference and coupled-system validation on:

- BIOMD0000000414 — 1 dynamic state
- BIOMD0000000036 — 2 dynamic states
- BIOMD0000000079 — 3 dynamic states

The workflow stops before held-out test evaluation.

This is the recommended end-to-end check before starting the larger inference
cohort.

## Run the main inference cohort

Run:

```bash
./run_biomodels.sh infer
```

This performs PySR inference and coupled-system validation on 41 compatible
BioModels containing between 1 and 59 inferred dynamic states.

The workflow stops before held-out test evaluation.

Execution is resumable. If execution is interrupted, run the same command
again. Completed work is detected and reused rather than recomputed.

## Generate reports and inspect the inferred equations

After inference results are available, run:

```bash
./report_biomodels.sh all
```

The reporting workflow automatically discovers available inference results and
generates summary tables and validation figures.

By default, reports are written under:

```text
reports/
├── benchmark_summary.csv
├── winning_equations.csv
├── failures.csv
└── models/
    └── BIOMD.../
        ├── model_summary.csv
        ├── winning_equations.csv
        └── figures/
            ├── train_04_validation_overview.png
            ├── train_08_validation_overview.png
            ├── train_16_validation_overview.png
            └── train_32_validation_overview.png
```

### Selected equations

The main file for inspecting the inferred ODEs is:

```text
reports/winning_equations.csv
```

It contains one row per selected state equation and training-set size,
including:

- BioModels model ID;
- training-set size;
- selected coupled-system ID;
- dynamic state;
- candidate rank;
- equation complexity;
- training loss;
- inferred equation;
- validation status;
- coupled-system validation NRMSE;
- state-level RMSE;
- state-level normalized RMSE; and
- state-level R².

For an individual model, the corresponding equations are also written to:

```text
reports/models/<MODEL_ID>/winning_equations.csv
```

### Model-level summary

The file:

```text
reports/benchmark_summary.csv
```

contains one row per model and training-set size. It summarizes the selected
coupled ODE system, including validation performance, integration success,
complexity, training loss, and selection information.

Each model also receives an individual summary:

```text
reports/models/<MODEL_ID>/model_summary.csv
```

### Validation figures

For each completed model and training-set size, the reporter generates a
validation overview comparing trajectories from the original BioModel with
trajectories predicted by the selected inferred ODE system.

The figures are written under:

```text
reports/models/<MODEL_ID>/figures/
```

### Failures and incomplete results

The file:

```text
reports/failures.csv
```

records missing, incomplete, or unsuccessful reporting cases so that they are
visible rather than silently omitted.

### Report a single model

To generate reports for a single model, for example:

```bash
./report_biomodels.sh model BIOMD0000000079
```

A custom output directory can also be specified:

```bash
./report_biomodels.sh all ./my-reports
```

Reporting reads existing inference results read-only. It does not run PySR,
repeat model selection, rerun ODE integration, or access held-out test data.

Reports can therefore be regenerated without modifying the scientific
inference results.

## Large models

Two additional models satisfy the current compatibility protocol but contain
402 inferred dynamic states each:

- BIOMD0000000469 — 402 states
- BIOMD0000000470 — 402 states

They are intentionally excluded from the main 41-model inference command
because their computational requirements have not been characterized to the
same extent as the main cohort.

To run these models explicitly:

```bash
./run_biomodels.sh infer-extreme
```

Passing the compatibility checks does not imply that these models have the same
computational requirements as the main cohort.

## Parallel execution

Different BioModels can be processed concurrently. The number of models allowed
to run at the same time is controlled with `PYSR_WORKERS`.

For example:

```bash
PYSR_WORKERS=8 ./run_biomodels.sh infer
```

allows up to eight BioModels to be processed concurrently.

`PYSR_WORKERS` controls model-level parallelism. It does not specify the number
of threads used by an individual PySR inference.

If `PYSR_WORKERS` is not specified, the workflow uses one worker:

```bash
./run_biomodels.sh infer
```

The one-worker default is intentionally conservative so that the package can
run on machines with limited computational resources. It is not a limit on the
parallelism supported by the workflow.

On larger compute servers, the worker count can be increased explicitly. For
example:

```bash
PYSR_WORKERS=16 ./run_biomodels.sh infer
```

The appropriate worker count depends on CPU capacity, available memory, disk
space, other workloads on the machine, and the sizes of the BioModels being
processed. Models in the main cohort range from 1 to 59 inferred dynamic states
and can have substantially different computational requirements.

Concurrent execution with two model workers has been tested successfully on a
two-CPU machine using small models. Larger worker counts have not yet been
systematically benchmarked.

On a new compute server, begin with a moderate worker count, monitor CPU,
memory, and disk utilization, and increase concurrency if sufficient resources
remain available.

For example, on a server with many CPU cores, reasonable initial configurations
could be:

```bash
PYSR_WORKERS=8 ./run_biomodels.sh infer
```

or:

```bash
PYSR_WORKERS=16 ./run_biomodels.sh infer
```

rather than assuming that the worker count should equal the total CPU count.

## Persistent storage

Runtime data are stored in Docker named volumes rather than host-owned output
directories.

By default:

```text
pysr-results
pysr-runs
pysr-cache
```

- `pysr-results` contains scientific inference artifacts.
- `pysr-runs` contains orchestration state, logs, telemetry, and manifests used
  for resume behavior.
- `pysr-cache` contains reusable runtime and cache data.

The volume prefix can be changed with `PYSR_VOLUME_PREFIX`.

For example:

```bash
PYSR_VOLUME_PREFIX=my-run ./setup.sh
PYSR_VOLUME_PREFIX=my-run ./run_biomodels.sh infer
PYSR_VOLUME_PREFIX=my-run ./report_biomodels.sh all
```

This creates and uses:

```text
my-run-results
my-run-runs
my-run-cache
```

Using a different prefix allows independent runs to use separate persistent
storage.

## Export scientific results

To export the persistent scientific results to a normal host directory:

```bash
./run_biomodels.sh export-results ./exported-results
```

To export run manifests, logs, telemetry, and execution records:

```bash
./run_biomodels.sh export-runs ./exported-runs
```

The exported files are extracted by the invoking user's host-side `tar`
process, so no `sudo` or ownership repair is required.

Generated reports are already ordinary host files and do not require the
volume export step.

## Scientific protocol

For each compatible SBML model, the inference workflow:

1. loads the original curated SBML model;
2. obtains the configured simulation horizon;
3. generates a deterministic experimental design;
4. simulates state trajectories from multiple initial conditions;
5. obtains exact numerical derivative targets from the SBML simulator;
6. normalizes each dynamic state using training data;
7. fits one PySR right-hand side per dynamic state using all inferred dynamic
   states as candidate input features;
8. retains candidate equations for each state;
9. assembles candidate equations into coupled ODE systems;
10. integrates the candidate systems on validation initial conditions;
11. evaluates coupled-system trajectory behavior;
12. selects the system using validation performance; and
13. writes inference and validation artifacts.

Four nested training-set sizes are evaluated:

```text
4, 8, 16, and 32 trajectories
```

The inference workflow uses exact numerical derivative targets obtained from
the SBML simulator.

Estimated-derivative and noisy-data experiments are separate methodological
experiments and are not enabled by the standard launcher.

## Training, validation, and held-out evaluation

Training trajectories are used to fit candidate equations.

Validation trajectories are used to evaluate coupled-system behavior and select
the inferred ODE system.

Held-out test trajectories are reserved for final scientific evaluation and are
not used for equation fitting, candidate selection, hyperparameter tuning, or
reporting.

The standard inference commands stop before held-out test evaluation, and the
launcher does not provide a command for opening the held-out test set.

This separation prevents held-out test information from influencing equation
inference or model selection.

## Computational environment

The included Docker environment contains:

- Python 3.11.17
- PySR 1.5.9
- Julia 1.11.9
- RoadRunner 2.8.0
- python-libSBML 5.21.2

The Docker image archive distributed with the package is:

```text
pysr-biomodels-amd64.tar
```

SHA256:

```text
4712dc679f7688ee88bc3b3e72237ea373573fb4f9c904588650b04c4d5133d5
```

The checksum can be used to verify that the Docker image archive has not
changed.

## Included BioModels snapshot

The package contains the complete curated input snapshot used for this
workflow:

- 974 files under `inputs/models/`
- 338 files matching `*_url.xml`
- 326 models discovered by the current canonical filename convention
- 43 models satisfying the current PySR-only compatibility protocol

The 43 compatible models are not intended to represent the complete public
BioModels database. They are the compatible subset of the supplied curated
snapshot under the current protocol.

The complete per-file SHA256 manifest is:

```text
inputs/SHA256SUMS.models
```

## Model manifests

The package contains several manifests defining the available model cohorts.

### All compatible models

```text
config/biomodels_ready.json
```

Contains all 43 models currently classified as compatible with the inference
protocol.

### Main inference cohort

```text
config/biomodels_standard.json
```

Contains the 41-model main inference cohort with 1–59 inferred dynamic states.

### Large-model cohort

```text
config/biomodels_extreme.json
```

Contains the two 402-state models requiring explicit execution.

### Small end-to-end cohort

```text
config/biomodels_smoke_1_2_3.json
```

Contains the 1-, 2-, and 3-state models used by the small end-to-end inference
test.

### Compatibility inventory

```text
config/biomodels_ready.csv
```

Contains the compatibility inventory and model metadata.

## Reproducibility checksums

The package includes checksums for the scientific inputs and model manifests.

### Simulation metadata

```text
146aa3e65032400d582713e02594e2827da2edaaa0f1dd7775475e57d99cf0c9
```

### Model-file checksum manifest

```text
a4a42fc5c821c78fa8da66550f2e10a6913a679e669181bf5cc5d781902f13e2
```

### 43-model compatible manifest

```text
50181533b565d8d3f6a004236dd72d8c9cf7a9cc898d58d09f292ab34c89c2cc
```

### 41-model main-cohort manifest

```text
504ac7efa3fc4071a29a84ad9fb683fcf0b26b2788ef42b2fd8f84c3c9f7537a
```

### Two-model large-model manifest

```text
1bb1ba54bc572f187a7f4e84525140fa239c869b607ee3dba3857dbbcc9277b2
```

### Three-model smoke manifest

```text
62ab719c402bbfba8aa185c6285f4a4100c9b0ce0c1ba711e41a925e9907ee66
```

These checksums allow the packaged scientific inputs and cohort definitions to
be verified independently of the inference results.

## Package validation

The package and scientific workflow have been validated on the intended Ubuntu
24.04 AMD64 target.

Validation included:

- Docker environment initialization without `sudo`;
- Python, PySR, Julia, RoadRunner, and libSBML runtime checks;
- compatibility checks for all 43 supported models;
- end-to-end inference and coupled-system validation on representative models;
- persistent non-root Docker storage;
- orchestration and interrupted-run/resume infrastructure;
- named-volume write and persistence checks;
- export of scientific results to user-owned host files;
- export of run records to user-owned host files; and
- concurrent execution with two model workers on small models.

The reporting workflow was tested using completed 1-, 2-, and 3-state inference
results. It successfully generated model summaries, selected-equation tables,
and validation figures without modifying the inference results.

No held-out test evaluation was performed as part of package validation.

## Resource considerations

Computational requirements vary substantially across models because the main
cohort contains systems with 1–59 inferred dynamic states.

CPU count alone should not be used to determine the maximum number of
concurrent workers. Memory consumption, disk usage, model size, and other
workloads on the machine should also be considered.

The two 402-state models are excluded from the main inference command because
their computational requirements have not been characterized. They must be
started explicitly with:

```bash
./run_biomodels.sh infer-extreme
```

Monitor CPU, memory, and available disk space during large inference runs,
particularly when increasing `PYSR_WORKERS`.

## Reviewing the scientific output

After inference is complete, the most direct files for reviewing the selected
ODE systems and their validation performance are:

```text
reports/winning_equations.csv
reports/benchmark_summary.csv
reports/failures.csv
```

Individual model equations and validation figures are available under:

```text
reports/models/<MODEL_ID>/
```

To preserve the complete inference artifacts and execution records outside the
Docker volumes, run:

```bash
./run_biomodels.sh export-results ./exported-results
./run_biomodels.sh export-runs ./exported-runs
```