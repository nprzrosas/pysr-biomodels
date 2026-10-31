# PySR-BioModels

PySR-BioModels is a benchmark for evaluating symbolic regression on biological
ordinary differential equation (ODE) systems from curated BioModels SBML. The
current workflow uses RoadRunner simulations and exact simulator derivatives,
infers one right-hand side (RHS) per dynamic state with PySR, selects a coupled
ODE system on validation trajectories, and evaluates the frozen system on
held-out trajectories before any post-hoc mechanistic comparison.

[**GSoC 2026 Final Work Product**](docs/gsoc_2026_final_report.md) is the
authoritative scientific and methodological record for this repository.

This work was completed for Google Summer of Code 2026 with the National
Resource for Network Biology (NRNB) under the project title *Evaluating and
Improving Symbolic Regression Methods for Biological Network Dynamics*.

## Overview

The final GSoC benchmark contains 100 benchmark-ready BioModels. It tests one
fixed, deterministic PySR configuration at four nested training-set sizes and
keeps four outcomes distinct:

- workflow completion;
- predictive trajectory agreement;
- structural or mechanistic recovery under explicitly defined metrics; and
- approximate numerical versus exact algebraic RHS equivalence.

The benchmark is a standardized exact-derivative evaluation, not a replay of
all SED-ML experiments and not a noisy experimental-derivative benchmark.
Model-specific simulation horizons are project-curated, while the initial-
condition design, sampling density, and inference protocol are standardized.

## Benchmark at a Glance

| Item | Final benchmark |
|---|---:|
| Benchmark-ready BioModels launched | 100 |
| Models completing the workflow through held-out testing | 75 |
| Incomplete executions | 25 |
| Timeout-censored | 12 |
| Memory-censored | 2 |
| Unresolved candidate-stage worker exits | 11 |
| Training trajectories | 4, 8, 16, 32 |
| Validation trajectories per condition | 8 |
| Held-out trajectories per condition | 8 |
| Completed model/training-size conditions | 300 |
| Feature-space-compatible mechanistic cohort | 66 models, 264 conditions |

The 25 incomplete executions are operational outcomes, not a single class of
PySR or scientific-recovery failure. Likewise, the 75 completed models are not
75 “recovered models”: completion, prediction, and equation recovery answer
different questions.

### Corpus expansion

The expanded cohort combines 43 historically benchmark-ready models with 57
additional models admitted after robust SBML filename resolution. BioModels
identifiers were already standardized; the historical local resolver instead
assumed that every SBML file was named exactly `<BIOMD_ID>_url.xml`. Valid
downloads often retained other source filenames.

The current resolver prefers the canonical filename, otherwise accepts a
unique XML/SBML candidate, and rejects unresolved ambiguity. This is a local
benchmark-discovery fix, not a BioModels defect. Benchmark exclusions indicate
incompatibility with this benchmark configuration, not scientific invalidity
or intrinsic unsuitability for symbolic regression. An earlier frozen 10-model
cohort was used while developing and validating the generic workflow; the
final GSoC evaluation is the Expanded-100 benchmark.

## Workflow

```text
BioModels SBML
    -> filename-aware model resolution and structural preflight
    -> deterministic initial-condition design
    -> RoadRunner trajectories and exact derivatives
    -> nested training sets (4 / 8 / 16 / 32)
    -> independent PySR candidate inference for each state RHS
    -> retain up to 8 candidate equations per RHS
    -> coupled-system coordinate selection on validation trajectories
    -> freeze selected inferred ODE system
    -> held-out trajectory evaluation
    -> post-hoc mechanistic comparison with symbolic ground truth
```

Validation trajectories participate in coupled-system selection. Held-out
trajectories are materialized only after the selected system is frozen and do
not participate in inference or selection. Symbolic ground-truth equations are
used only for post-hoc recovery analysis.

## Experimental Design

- Independent 256-point Latin-hypercube candidate pools are generated for
  training, validation, and held-out conditions with seeds 1010, 2020, and
  3030, respectively.
- Nominal floating-state initial values are perturbed multiplicatively from
  approximately 0.1x to 10x in log space. Maximin selection constructs the
  final designs.
- Training sets are nested: `train_04` is contained in `train_08`, which is
  contained in `train_16`, which is contained in `train_32`.
- Each condition has 8 validation and 8 held-out trajectories. Every
  trajectory has 101 output points from time zero through the project-curated,
  model-specific simulation horizon.
- RoadRunner supplies exact, noiseless derivatives at the simulated state
  vectors. SBML parameters remain fixed at nominal values.

### Canonical PySR configuration

| Setting | Value |
|---|---|
| PySR | 1.5.9 |
| Iterations | 60 |
| Binary operators | `+`, `-`, `*`, `/` |
| Unary operators | None |
| Maximum expression size | 20 |
| Parsimony | 0 |
| Execution | Deterministic, serial |
| Random seed | 1010 |
| Retained candidates | Up to 8 per state RHS |

State features and derivative targets are scaled using training data only.
Retained equations are transformed back to original coordinates and units.
There is no post-selection coefficient refit. See the
[final report](docs/gsoc_2026_final_report.md) for the full configuration,
integration tolerances, selection tie-breaking, and metric definitions.

## Main Results

### Held-out predictive performance

Among the 75 workflow-complete models, a full held-out success means that all
eight held-out trajectories integrated successfully. Median macro-NRMSE is
reported only among systems with finite, full held-out results.

| Training trajectories | Full held-out integration | Proportion | Median held-out macro-NRMSE |
|---:|---:|---:|---:|
| 4 | 71/75 | 94.7% | 0.0428323 |
| 8 | 69/75 | 92.0% | 0.0282917 |
| 16 | 73/75 | 97.3% | 0.0243442 |
| 32 | 71/75 | 94.7% | 0.0262343 |

Overall, 284 of 300 selected systems completed all held-out integrations.
Sixteen training trajectories produced the strongest aggregate predictive
result among the tested sizes under this fixed setup; this is not evidence
that 16 is universally optimal. Performance did not improve monotonically for
every model or metric as training size increased.

### Mechanistic recovery

The primary mechanistic analysis contains 66 feature-compatible models at all
four training sizes. These are model-level macro summaries, so models with
more state equations do not automatically receive greater weight.

| Training trajectories | Dependency F1 | Numerator monomial F1 | Interaction F1 | Rational-structure accuracy |
|---:|---:|---:|---:|---:|
| 4 | 0.8015 | 0.5538 | 0.5666 | 0.6553 |
| 8 | 0.8191 | 0.5705 | 0.6298 | 0.6506 |
| 16 | 0.8439 | 0.6125 | 0.6539 | 0.6503 |
| 32 | 0.8277 | 0.5853 | 0.5894 | 0.6374 |

Across the final pooled equation comparison, 991 of 992 eligible RHSs
completed algebraic and numerical evaluation; one `train_08` RHS timed out.
Of the 991 evaluated RHSs, 140 (14.1%) passed the approximate numerical-
equivalence diagnostic and 2 (0.2%) were exact algebraic matches. Both exact
matches were trivial `0 = 0` RHSs in BIOMD0000000941. No nonzero RHS and no
complete coupled ODE system achieved exact algebraic recovery.

Predictive accuracy, dependency recovery, functional-form recovery, numerical
equivalence, and exact algebraic equality are complementary dimensions. They
are not combined into one recovery rate or a hierarchy.

## Example: BIOMD0000000156

BIOMD0000000156 illustrates why these distinctions matter. Under `train_32`,
all 8 held-out trajectories integrated and the held-out system macro-NRMSE was
`3.6540933364804454e-06`. Dependency, numerator-monomial, interaction, and
rational-structure scores were all 1.0. All 3 RHSs passed the approximate
numerical-equivalence diagnostic, while 0 of 3 were exact algebraic matches
because fitted coefficients differed slightly from the canonical equations.

This is an illustrative case, not an estimate of population frequency. The
report includes its held-out trajectories and canonical equation comparisons.

## Reproducibility with Docker

The container bundles the Expanded-100 SBML inputs, model-specific simulation
metadata, validated cohort manifest, project source, locked Python environment,
and a Julia/PySR environment initialized during the build. Runtime does not
download models from BioModels. Results are not baked into the image.

The supported target is `linux/amd64`; runtime uses non-root UID 10001 and one
writable `/output` host mount. Building requires network access, and the Julia
runtime is not completely version-pinned. The container improves environmental
reproducibility but does not establish bitwise-identical results.

Build from the repository root using the documented image tag:

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --tag pysr-only:expanded-100-exact \
  --load \
  .
```

Validate the bundled cohort without running PySR:

```bash
docker run --rm --platform linux/amd64 \
  pysr-only:expanded-100-exact \
  -m src.container_smoke \
  --validate-cohort-manifest \
  /opt/pysr/inputs/expanded_100/metadata/benchmark_ready_100.json
```

See [DOCKER.md](DOCKER.md) for the `/output` mount, resource controls, and the
phase-aware Expanded-100 batch command.

## Running the Code

Install the locked Python environment from the repository root:

```bash
uv sync --frozen
```

The current orchestrators expose explicit, resumable phases for one model or a
manifest-defined batch:

```bash
uv run python -m src.run_model --help
uv run python -m src.run_batch --help
```

The phases are `preflight`, `candidates`, `optional_exhaustive`, `selector`,
`pretest`, and `test`. Local scientific runs require configured model,
simulation-metadata, results, and run roots. Held-out testing additionally
requires explicit authorization and verified pretest checkpoints. For a
self-contained Expanded-100 execution, use the reviewed Docker command in
[DOCKER.md](DOCKER.md) rather than bypassing these boundaries.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | Simulation, inference, selection, evaluation, and orchestration code |
| `docker/expanded_100_inputs/` | Bundled SBML corpus and benchmark metadata |
| `analysis/` | Final aggregate tables and retained evaluation evidence |
| `results/` | Model-level results and corpus-selection audit evidence |
| `docs/` | Final GSoC report and figures |
| `tests/` | Unit and integration tests |

## Limitations

- The canonical benchmark covers deterministic SBML ODE models that pass its
  structural and numerical preflight requirements; it does not establish that
  excluded models are scientifically unsuitable.
- It uses exact, noiseless simulator derivatives, positive finite nominal
  floating-state initial values, fixed parameters, state-only features, one
  operator/search configuration, and one deterministic PySR seed.
- Explicit-time or otherwise unavailable dependencies can create a feature-
  space mismatch. Some SBML events, rules, and other constructs are outside
  the supported benchmark scope.
- One standardized initial-condition range, sampling density, and curated
  horizon is used per model rather than exhaustive SED-ML experiment replay.
- Coupled-system selection is a deterministic single-start coordinate search,
  not a globally optimal search over every candidate combination.
- Resource censoring, unresolved worker exits, and held-out integration
  failures limit the evaluated cohort. The Docker workflow currently targets
  CPU execution on `linux/amd64`.
- Exact algebraic comparison is intentionally strict, while numerical
  equivalence is a finite sampled-state diagnostic rather than a proof of
  global identity.

## Documentation and Results

- [GSoC 2026 Final Work Product](docs/gsoc_2026_final_report.md)
- [Final report claim audit](analysis/gsoc_2026_final_report_audit.md)
- [Final scientific summary](analysis/expanded100_final_scientific_summary.csv)
- [Expanded-100 master results](analysis/expanded100_master_results.csv)
- [Case-study candidates](analysis/expanded100_case_study_candidates.csv)
- [Mechanistic recovery summary](analysis/expanded100_mechanistic_recovery/mechanistic_recovery_summary.csv)
- [Historical corpus-selection funnel](results/model_selection_audit/biomodels_selection_funnel.csv)
- [Docker guide](DOCKER.md)

## GSoC 2026

- **Contributor:** Norma Pérez Rosas
- **Organization:** National Resource for Network Biology (NRNB)
- **Mentors:** Augustin Luna, Eran Agmon, and Mark Kittisopikul

The final report documents completed work, evidence classifications,
limitations, and future directions in greater detail.
