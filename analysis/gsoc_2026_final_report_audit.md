# GSoC 2026 final report claim audit

Status meanings: **VERIFIED** = read directly from implementation/configuration or a recorded artifact; **COMPUTED** = calculated or aggregated from final result files; **INTERPRETATION** = a qualified reading of verified results; **UNKNOWN** = retained evidence is insufficient.

| Report section | Claim | Evidence | Status | Caveat |
|---|---|---|---|---|
| Overview | Final cohort has 100 benchmark-ready models; 75 completed through held-out testing | `expanded100_manifest.json`; `expanded100_master_results.csv`; cohort validation | COMPUTED | Completion is not recovery |
| Scope | Final pipeline has separate design, simulation, fitting, validation selection, test, and mechanistic stages | Relevant `src/*.py` modules; manifest | VERIFIED | Describes current refactor only |
| Work completed | Scaling is training-only and expressions are back-transformed without coefficient refitting | `src/run_pysr_generic.py` | VERIFIED | No post-selection refit found |
| Corpus | Historical audit contains 1,075 local directories | `results/model_selection_audit/biomodels_selection_funnel.csv` | VERIFIED | Approximate larger attempted count was not reconstructed |
| Corpus | Historical counts are 1,075, 497, 326, 303, 66, and 43 | Funnel CSV; `scripts/make_model_selection_audit.py` | VERIFIED | Historical sequence, not final mutually exclusive funnel |
| Corpus | Current resolver prefers canonical name, accepts one alternative XML/SBML, and rejects ambiguity | `src/model_registry.py` | VERIFIED | Local discovery behavior |
| Corpus | Bundled cohort has 43 canonical and 57 noncanonical filenames | `benchmark_ready_100.csv`; packaged model directories; validation JSON | COMPUTED | All 100 passed stored readiness validation |
| Simulation | Start 0, model-specific horizon, 101 points, initial-state perturbations, fixed parameters, exact RoadRunner derivatives | `src/simulate_generic.py`; `src/simulation_metadata.py` | VERIFIED | Standardized protocol, not exhaustive SED-ML replay |
| Simulation | Duration candidates were automatically extracted from curated figures | Adjacent `data2sbml/extract_curated_image_times.py`; bundled metadata | VERIFIED | Extent of model-level manual review is UNKNOWN |
| Pipeline | PySR configuration is version 1.5.9, 60 iterations, `+ - * /`, no unary operators, maxsize 20, parsimony 0, deterministic serial seed 1010, top 8 | `src/run_pysr_generic.py`; `pyproject.toml`; `uv.lock` | VERIFIED | Manifest does not independently log imported runtime version |
| Design | Three 256-point pools, 0.1--10x factors, seeds 1010/2020/3030, nested 4/8/16/32 training, 8 validation, 8 test | `src/experimental_design_generic.py` | VERIFIED | Requires positive finite nominal states |
| Validation | LSODA, `rtol=1e-7`, `atol=1e-9`, maximum 8,000 RHS calls | `src/evaluate_generic_validation.py` | VERIFIED | Any trajectory failure makes aggregate score infinite |
| Selection | Deterministic single-start coordinate search; test opens after selections freeze | `src/select_generic_system.py`; `src/run_generic_test.py` | VERIFIED | Coordinate search is not globally optimal |
| Evaluation | Dependency metric compares dynamic-state symbol sets | `src/evaluate_dependency_recovery.py` | VERIFIED | Does not establish functional form |
| Evaluation | Functional metrics cover monomials, interactions, rational dependence, denominator dependencies, and degrees | `src/evaluate_functional_form_recovery.py`; `src/inspect_functional_structure.py` | VERIFIED | Separate components, not equation identity |
| Evaluation | Numerical diagnostic uses 32 log-uniform positive points, seed 4040, `rtol=1e-6`, `atol=1e-9` | `src/evaluate_algebraic_recovery.py` | VERIFIED | Finite sampling cannot prove identity |
| Evaluation | Exact criterion is SymPy identity after normalization/cancellation, without tolerance | `src/evaluate_algebraic_recovery.py` | VERIFIED | Subject to parsing/simplification limits |
| Execution | 75 complete and 25 incomplete; 12 timeout and 2 memory censored | `expanded100_manifest.json` | COMPUTED | Remaining 11 have generic candidate-stage worker exits |
| Execution | Remaining 11 incomplete runs have one common numerical cause | Retained manifest/log evidence | UNKNOWN | Report does not make this claim |
| Predictive | Full held-out success is 71/75, 69/75, 73/75, 71/75; total 284/300 | `expanded100_master_results.csv`; final summary | COMPUTED | Requires all eight test trajectories |
| Predictive | Finite/full medians are 0.042832, 0.028292, 0.024344, 0.026234 | `expanded100_final_scientific_summary.csv` | COMPUTED | Median used because distribution is heavy-tailed |
| Predictive | 16 trajectories is best among tested sizes under fixed setup | Final summary | INTERPRETATION | Not a universal optimum |
| Mechanistic | 276 evaluated, 264 compatible, 12 feature mismatches, 24 unavailable | `expanded100_master_results.csv` | COMPUTED | Unavailable = six ground-truth extraction failures |
| Mechanistic | Primary cohort is 66 models and 264 conditions | Master results; mechanistic summary | COMPUTED | Feature mismatches excluded |
| Mechanistic | Four rows of model-level mean F1/accuracy values in report | `expanded100_final_scientific_summary.csv` | COMPUTED | Models, not RHSs, receive equal weight |
| Mechanistic | Paired 4-to-32 counts are 24/27/15, 33/10/23, and 23/27/16 | Mechanistic summary joined by model | COMPUTED | Improved/unchanged/worsened respectively |
| Equivalence | Numerical counts are 33/248, 35/247, 35/248, 37/248; pooled 140/991 | Final summary; mechanistic recovery CSV | COMPUTED | One train-8 RHS timed out |
| Equivalence | Exact counts are 2/248, 0/247, 0/248, 0/248; pooled 2/991 | Final summary; mechanistic recovery CSV | COMPUTED | Exact and numerical criteria differ |
| Equivalence | Both exact matches are zero RHSs in BIOMD0000000941; no nonzero RHS or whole system is exact | `mechanistic_recovery.csv`; `selected_equations.csv` | VERIFIED | Complete-system check uses evaluable compatible systems |
| Case 0156 | All component scores 1; NRMSE about 3.65e-6--3.78e-6; train 32 numerical 3/3, exact 0/3 | `expanded100_case_study_candidates.csv` | COMPUTED | Illustrative, not prevalence estimate |
| Case 0319 | Dependency 1, monomial about 0.30--0.48, rational 0 or 1/3, NRMSE 0.0562--0.0737 | Case-study CSV | COMPUTED | Shows dependencies do not imply form |
| Case 0728 | Test integrates, NRMSE about 228--1,190, weak component scores, 0/2 equivalence | Case-study CSV | COMPUTED | Integration does not imply prediction |
| Engineering | Manifests, hashes, resume validation, locking, and telemetry are implemented | `src/run_manifest.py`; `src/run_batch.py`; manifest | VERIFIED | No claim of bitwise reproducibility |
| Docker | Inputs bundled; no live runtime download; non-root UID 10001; output mount; amd64 CPU workflow | `Dockerfile`; `DOCKER.md`; bundle | VERIFIED | Build needs network; no documented public image/GPU path |
| Docker | Julia is completely version-frozen | `Dockerfile` | UNKNOWN | Report explicitly says it is not completely pinned |
| Limitations | Six parser failures and three explicit-time feature mismatches affect mechanistic cohort | Master results; `mechanistic_errors.csv` | COMPUTED | Predictive results retained |
| Future work | Noise, estimated derivatives, multi-seed, alternatives, broader SBML are proposed | Report | INTERPRETATION | Explicitly not completed work |
