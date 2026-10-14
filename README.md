# PySR-only Equation Inference

## Objective

This project independently reconstructs systems of ordinary differential
equations (ODEs) from simulated time-series data using PySR. It deliberately
does not use Snakemake or the LLM-based inference methods from the parent
project.

The current benchmark starts from curated BioModels SBML files, generates
trajectories and exact derivatives with RoadRunner, infers one right-hand side
(RHS) per dynamic state, selects the equations as a coupled ODE system, and
evaluates dynamic and mechanistic recovery.

## Current Status

The generic exact-derivative benchmark is complete and frozen. It contains:

- 10 BioModels models;
- 4 nested training sizes: 4, 8, 16, and 32 trajectories;
- 40 benchmark conditions (10 models × 4 training sizes), each with one
  frozen selected ODE system;
- 8 validation trajectories per condition;
- 8 held-out test trajectories per condition;
- exact RoadRunner derivative targets;
- post-hoc mechanistic-recovery evaluation;
- inferred SBML export;
- SBML portability evaluation; and
- original-versus-inferred SBML dynamic-fidelity evaluation.

The cohort-level tables and figures aggregate existing frozen artifacts. They
are reporting outputs, not additional benchmark experiments.

## Scientific Workflow

The implemented generic workflow is:

```text
original SBML
    -> deterministic initial-condition design
    -> RoadRunner state trajectories
    -> exact RoadRunner derivatives
    -> training-only state normalization
    -> independent PySR inference for each state RHS
    -> retained per-state candidate equations
    -> coupled candidate ODE systems
    -> validation-based Selector v1 coordinate search
    -> frozen selected system
    -> held-out test evaluation
    -> post-hoc mechanistic recovery
    -> inferred SBML rate-rule model
    -> SBML portability and dynamic-fidelity evaluation
    -> frozen reporting and aggregation
```

Model discovery, compatibility checks, and simulation metadata are implemented
in `src/model_registry.py`, `src/model_spec.py`,
`src/model_compatibility.py`, `src/preflight.py`, and
`src/simulation_metadata.py`. Source SBML files and curated simulation horizons
are expected in the sibling `../data2sbml/output/` tree.

`src/run_generic_benchmark.py` generates only training and validation
trajectories and runs PySR. Although the deterministic test initial conditions
already exist in the experimental design, their trajectories are not
materialized during inference or system selection. `src/run_generic_test.py`
materializes them only after the Selector v1 systems have been frozen.

## Initial Development: BIOMD0000000414

BIOMD0000000414 was the initial development and reference model. It is not a
member of the frozen 10-model benchmark.

The initial reconstruction began from:

- `BIOMD0000000414_url.xml`, the SBML model; and
- `BIOMD0000000414_url.sedml`, whose documented uniform time course runs from
  0 to 10 with 1,000 steps and identifies VENUS as the dynamic species.

This SED-ML configuration describes the original 0414 development input and
is not the simulation-time source used by the later generic benchmark.

The repository retains separate 0414 experiments involving exact derivatives,
estimated derivatives, measurement noise, and multiple noise seeds. These
historical development branches use the non-generic simulation, inference, and
evaluation scripts and must not be conflated with the generic exact-derivative
benchmark.

## Frozen 10-Model Benchmark Cohort

The completed cohort is defined directly by the frozen benchmark aggregation
and mechanistic-evaluation scripts. The repository does not assign or document
a scientific selection rationale for its composition.

| Model ID | Number of states | Simulation horizon |
|---|---:|---:|
| BIOMD0000000079 | 3 | 40 |
| BIOMD0000000156 | 3 | 30 |
| BIOMD0000000322 | 4 | 600 |
| BIOMD0000000174 | 4 | 1200 |
| BIOMD0000000039 | 5 | 50 |
| BIOMD0000000116 | 6 | 1600 |
| BIOMD0000000229 | 7 | 30 |
| BIOMD0000000010 | 8 | 150 |
| BIOMD0000000022 | 10 | 5 |
| BIOMD0000000476 | 16 | 50 |

BIOMD0000000036 and BIOMD0000000233 have generic artifacts in `results/`, but
they are not members of this frozen 10-model cohort. The repository does not
document why they were excluded.

## Experimental Design

`src/experimental_design_generic.py` creates reproducible multidimensional
initial conditions as multiplicative perturbations of each model's nominal
dynamic-state values. The current design therefore requires every nominal
state value to be finite and strictly positive.

- Candidate factors are Latin-hypercube samples drawn uniformly in log10 space
  over `[-1, +1]`, corresponding to `0.1x` through `10x` nominal.
- Independent pools of 256 candidates use seeds 1010, 2020, and 3030 for
  training, validation, and test, respectively.
- Training points are greedily maximin-ordered in log10 fold-change space.
- Eight validation points are selected against the complete 32-point training
  design and the growing validation set.
- Eight test points are selected against the complete training and validation
  designs and the growing test set.
- Training sets are nested: `train_04` is contained in `train_08`, which is
  contained in `train_16`, which is contained in `train_32`.
- Every trajectory contains 101 output points from time zero through the
  model-specific curated simulation horizon.

The resulting training tables contain 404, 808, 1,616, and 3,232 rows. The
validation set contains 808 rows. Test trajectories remain unmaterialized
during PySR inference and validation-based selection.

## Exact-Derivative Data Generation

`src/simulate_generic.py` simulates each initial condition with RoadRunner.
For every recorded time and state vector, it sets the RoadRunner model state
and calls `getRatesOfChange()` to obtain one exact target per dynamic state.

This is an exact, noiseless, or oracle-derivative symbolic-recovery benchmark.
It is not an experimental noisy derivative-estimation benchmark. The
Savitzky-Golay derivative estimation and measurement-noise utilities belong to
the separate BIOMD0000000414 development work.

## PySR Inference

`src/run_pysr_generic.py` fits an independent PySR model for every dynamic
state, using all dynamic states as input features. The frozen configuration is:

| Setting | Value |
|---|---|
| PySR version | 1.5.9 |
| `niterations` | 60 |
| Binary operators | `+`, `-`, `*`, `/` |
| Unary operators | None |
| `maxsize` | 20 |
| `parsimony` | 0.0 |
| `weight_optimize` | 0.0 |
| Constant optimization | Enabled, BFGS |
| `model_selection` | `best` |
| Random seed | 1010 |
| Execution | Deterministic, serial |
| Retained candidates | Up to 8 per state, ordered by training loss |

For each training condition, every state feature is divided by its maximum
absolute value in that training set. The derivative target for a state is
divided by the same state's scale. Each retained symbolic equation is then
transformed back to the original state coordinates and units before coupled
system evaluation.

## Coupled-System Selection

Each state has its own retained candidate set. A coupled candidate system is
defined by a tuple containing one candidate rank per state. Candidate counts
can be smaller than eight, so the Cartesian candidate space is the product of
the actual retained counts.

The frozen Selector v1 in `src/select_generic_system.py` uses deterministic,
single-start coordinate search:

1. Start from candidate rank 1 for every state.
2. Visit states in fixed `ModelSpec.state_ids` order.
3. For one state at a time, evaluate every available candidate while holding
   the remaining equations fixed.
4. Immediately adopt the best proposal and continue through the state order.
5. Repeat full sweeps until no candidate rank changes.
6. Cache rank tuples so that each encountered system is evaluated only once.

Candidate systems are ranked lexicographically by:

1. validation macro NRMSE;
2. total expression complexity;
3. total PySR training loss; and
4. canonical Cartesian system ID.

This is not guaranteed to find the global optimum over the complete Cartesian
candidate space. `src/run_generic_validation.py` and the exhaustive path in
`src/evaluate_generic_validation.py` implement an older generic route; the
completed benchmark is defined by the frozen `coordinate_search` selections.

## Validation and Held-Out Test

`src/evaluate_generic_validation.py` compiles the selected SymPy expressions
and integrates each coupled system with SciPy `solve_ivp` using:

- LSODA;
- relative tolerance `1e-7`;
- absolute tolerance `1e-9`; and
- a limit of 8,000 RHS evaluations per trajectory.

For each state and trajectory, the implementation records MSE, RMSE, MAE,
training-scale-normalized RMSE, and R2. A trajectory macro NRMSE is the mean of
its state NRMSE values, and a system macro NRMSE is the mean across its eight
trajectories. The normalization scale is derived only from the corresponding
training data.

If any trajectory fails to integrate, the condition is marked `failed` and
its system-level macro NRMSE is set to infinity. Successfully integrated
partial trajectories and their metrics remain recorded.

Validation selects the coupled system on eight fixed validation trajectories.
After selection is frozen, `src/run_generic_test.py` and
`src/evaluate_generic_test.py` materialize the eight held-out test trajectories
and evaluate that same system. Test evaluation performs no reselection,
refitting, or tuning.

## Mechanistic Recovery

Mechanistic ground-truth RHS equations are reconstructed directly from the
original SBML reaction network by `src/extract_mechanistic_ground_truth.py`.
Supported user-defined functions are expanded, numeric constants and local
parameters are substituted, reaction stoichiometry is applied, and each state
RHS is simplified. `src/validate_mechanistic_ground_truth.py` checks the
reconstructed symbolic rates against RoadRunner rates along a nominal
trajectory.

Three separate post-hoc recovery families are implemented. They are not
combined into a single composite mechanistic-recovery score.

1. **Dependency recovery** (`src/evaluate_dependency_recovery.py`) compares the
   sets of dynamic-state variables present in each true and inferred RHS. It
   reports per-RHS precision, recall, and F1, plus model-level macro and micro
   summaries.
2. **Functional-form Recovery v1**
   (`src/evaluate_functional_form_recovery.py`) reports numerator-monomial
   recovery, multiplicative interaction recovery, rational state-dependence
   agreement, denominator-dependency recovery, and numerator/denominator
   polynomial-degree differences. Constant-only numerator monomials are
   excluded from monomial precision, recall, and F1.
3. **Algebraic Recovery v1** (`src/evaluate_algebraic_recovery.py`) tests strict
   symbolic equivalence after canonicalization and simplification.

Algebraic Recovery v1 also reports a separate approximate numerical-equivalence
diagnostic. It evaluates eligible state-only RHS expressions at 32
deterministic positive state-space points using seed 4040, `rtol=1e-6`, and
`atol=1e-9`. Numerical equivalence is not counted as exact algebraic recovery.

## Mechanistic Comparability

The PySR feature space contains dynamic states only. Two frozen benchmark
models have ground-truth systems with explicit time dependence:

- BIOMD0000000174
- BIOMD0000000476

They are therefore labeled `feature_space_mismatch`: their full
non-autonomous ground-truth systems cannot be represented by the state-only
inference feature space.

All 10 models remain in the dynamic benchmark summaries. Primary
cohort-level mechanistic summaries use the eight feature-space-compatible
models; the two mismatch models are retained separately for transparency.

## Inferred SBML

`src/export_inferred_sbml.py` represents every frozen selected ODE system as
SBML Level 3 Version 2. It creates a unit-size compartment, one floating
species per dynamic state, and one `rateRule` per selected RHS. The generated
document is checked with libSBML before it is written as `inferred_model.xml`.

The exporter preserves the inferred ODE equations. It does not invent or claim
recovery of a biochemical reaction network.

## SBML Portability

`src/validate_sbml_portability.py` asks whether two implementations of the same
frozen inferred equations are numerically equivalent on the eight validation
initial conditions:

- the SymPy equations integrated by the SciPy/LSODA implementation; and
- the exported inferred SBML integrated by RoadRunner.

A trajectory passes the strict pointwise comparison using absolute tolerance
`1e-6` and relative tolerance `1e-5`. A condition passes only when both
implementations complete all eight trajectories and every trajectory is
equivalent within those tolerances.

A portability failure does not necessarily mean that the inferred SBML cannot
be simulated. Failure can mean that either implementation failed on at least
one trajectory or that both completed but did not satisfy the strict numerical
equivalence criterion. Portability is not a biological-fidelity or mechanistic
recovery measure.

## Dynamic Fidelity

`src/compare_original_inferred_sbml.py` compares the original biological SBML
with the inferred SBML. Both models are integrated with RoadRunner from the
same eight validation initial conditions, over the same interval and output
grid. The outputs include state RMSE, training-scale-normalized RMSE, R2, and
absolute-error summaries.

Dynamic fidelity measures trajectory agreement with the original model; it
does not establish recovery of the original mechanism. Because it uses the
same validation conditions involved in Selector v1, it is related to
selected-system validation and is not independent held-out evidence.

## Benchmark Results

The frozen benchmark contains 40 benchmark conditions (10 models × 4 training
sizes), each with one frozen selected ODE system. The existing frozen analysis
in `results/benchmark_analysis/benchmark_analysis_summary.txt` reports that
validation NRMSE improved from `train_04` to `train_32` in 9 of 10 models. This
is an endpoint comparison and does not imply monotonic improvement through 4,
8, 16, and 32 trajectories.

Frozen held-out full-integration success was:

| Training condition | Successful models |
|---|---:|
| `train_04` | 5/10 |
| `train_08` | 8/10 |
| `train_16` | 7/10 |
| `train_32` | 8/10 |

Finite held-out error summaries condition on successful full integration;
integration failures are reported separately and are not finite-error
observations.

## Benchmark Artifacts

The principal frozen cohort-level files are:

- `results/benchmark_summary.csv`: master 40-condition table;
- `results/benchmark_analysis/benchmark_analysis_summary.txt`: frozen textual
  benchmark summary;
- `results/selector_v1_aggregate_report.csv`: Selector v1 search and validation
  summary;
- `results/selector_v1_held_out_test.csv`: held-out test aggregate;
- `results/mechanistic_ground_truth_audit.csv`: ground-truth extraction audit;
- `results/dependency_recovery_summary.csv`;
- `results/functional_form_recovery_summary.csv`;
- `results/algebraic_recovery_summary.csv`;
- `results/integrated_recovery.csv`: all 10 models and 40 conditions;
- `results/integrated_recovery_compatible.csv`: eight comparable models and 32
  conditions; and
- `results/integrated_recovery_summary.csv`.

Important per-model paths follow this structure:

```text
results/<model_id>/generic_exact_experiment/
├── experiment_metadata.csv
├── train_<04|08|16|32>/
│   ├── run_metadata.csv
│   ├── state_<state_id>/
│   │   ├── all_equations.csv
│   │   └── top_8_by_loss.csv
│   └── coordinate_selection/
│       ├── coordinate_selected_system.csv
│       ├── coordinate_search_trace.csv
│       ├── coordinate_validation_*.csv
│       ├── test/test_*.csv
│       └── sbml_validation/
│           ├── inferred_model.xml
│           ├── portability_*.csv
│           └── dynamic_fidelity/dynamic_fidelity_*.csv
├── coordinate_selection/
│   ├── test/test_comparison.csv
│   └── sbml_validation/
│       ├── portability_comparison.csv
│       └── dynamic_fidelity/dynamic_fidelity_comparison.csv
├── mechanistic_recovery/
│   ├── dependency_recovery*.csv
│   ├── functional_form_recovery*.csv
│   └── algebraic_recovery*.csv
└── model_recovery_report/
    ├── report.md
    └── figures/
```

## Reproducibility and Script Roles

| Role | Scripts | Effect |
|---|---|---|
| Scientific computation | `src/run_generic_benchmark.py`, `src/run_pysr_generic.py`, `src/simulate_generic.py`, `src/experimental_design_generic.py` | Generate trajectories and derivatives and fit PySR equations. |
| Scientific computation | `src/select_generic_system.py`, `src/evaluate_generic_validation.py`, `src/run_generic_test.py`, `src/evaluate_generic_test.py` | Select and dynamically evaluate coupled systems. |
| Scientific computation | `src/extract_mechanistic_ground_truth.py`, `src/validate_mechanistic_ground_truth.py`, `src/evaluate_*_recovery.py`, `src/run_*_recovery.py` | Reconstruct and validate ground truth and compute frozen mechanistic metrics. |
| Scientific computation | `src/validate_sbml_portability.py`, `src/compare_original_inferred_sbml.py` | Perform portability and dynamic-fidelity evaluations. |
| Representation/export | `src/export_inferred_sbml.py` | Convert frozen selected ODEs to validated SBML rate-rule documents without changing the equations. |
| Reporting/aggregation only | `src/aggregate_*.py`, `src/summarize_*.py`, `src/integrate_recovery_results.py`, `src/make_benchmark_summary.py`, `src/analyze_benchmark_summary.py` | Read and summarize existing frozen artifacts; they do not rerun the benchmark. |
| Reporting/visualization only | `src/plot_*.py`, `src/make_model_recovery_report.py`, `src/reporting/*` | Generate tables, reports, and figures without changing frozen selections or scientific metrics. Some model-report figures reconstruct trajectories solely for visualization. |

`src/diagnose_exhaustive_system.py` performs scientific trajectory evaluation
for diagnostics and is not reporting-only. `src/migrate_coordinate_system_ids.py`
is a result-maintenance utility rather than a benchmark stage. `main.py` is a
placeholder and does not run the scientific workflow.

## Known Limitations

- The generic benchmark uses exact, noiseless derivatives rather than
  derivatives estimated from experimental observations.
- Trajectories come from synthetic multiplicative perturbations of the SBML
  nominal initial conditions.
- PySR receives dynamic states only; it does not receive time, parameters, or
  other independent variables as features.
- BIOMD0000000174 and BIOMD0000000476 consequently have a documented
  `feature_space_mismatch` because their ground-truth systems depend explicitly
  on time.
- Selector v1 is a single-start coordinate search and is not guaranteed to
  find the global optimum in the full Cartesian candidate space.
- Dynamic fidelity is evaluated on validation conditions used by system
  selection and is not independent held-out evidence.
- Held-out integration failure is distinct from finite trajectory error. A
  failed condition can retain partial successful-trajectory outputs while its
  overall macro NRMSE is infinite.
- Inferred SBML represents the selected ODEs as rate rules, not reconstructed
  biochemical reactions.
- Portability is a strict implementation-equivalence test. Its failure alone
  does not establish biological failure or inability to simulate the inferred
  SBML in every condition.

## Frozen Benchmark Definition

The completed benchmark is the 10 models listed above, evaluated at training
sizes 4, 8, 16, and 32: 40 benchmark conditions, each with one frozen Selector
v1 ODE system. Its scientific record consists of the frozen selections,
held-out evaluations, mechanistic evaluations, inferred-SBML exports,
portability evaluations, and dynamic-fidelity evaluations already stored under
`results/`.

Subsequent tables, summaries, reports, and figures that only aggregate or
visualize these artifacts are reporting products, not new benchmark
experiments.
