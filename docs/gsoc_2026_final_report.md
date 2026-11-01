---
layout: default
title: GSoC 2026 Final Report
description: Evaluating and Improving Symbolic Regression Methods for Biological Network Dynamics
---

# Evaluating and Improving Symbolic Regression Methods for Biological Network Dynamics

| GSoC metadata | Details |
|---|---|
| Program | Google Summer of Code 2026 |
| Organization | National Resource for Network Biology (NRNB) |
| Contributor | Norma C Pérez Rosas |
| Mentors | Augustin Luna, Eran Agmon, Mark Kittisopikul, and Matthias König |
| Work product | PySR-BioModels benchmark |

## 1. Abstract

Mechanistic models in systems biology are frequently expressed as systems of ordinary differential equations (ODEs), but discovering those equations from observations is a difficult inverse problem. Symbolic regression offers a data-driven approach in which the mathematical expressions themselves are inferred rather than specified in advance.

This project developed **PySR-BioModels**, a reproducible benchmark for evaluating symbolic regression on curated biological ODE models from BioModels [9]. Known SBML models are used as ground-truth dynamical systems. Standardized perturbation experiments generate state trajectories and exact derivatives, PySR independently proposes candidate right-hand sides for each dynamic state, and validation trajectories are used to assemble a coupled ODE system. The selected system is then frozen and evaluated on previously unseen initial conditions.

The benchmark was designed to distinguish predictive success from mechanistic recovery. In addition to held-out trajectory error, inferred equations are evaluated for state dependencies, numerator monomials, multiplicative interactions, rational structure, finite-point numerical equivalence, and exact algebraic equivalence.

The final benchmark contains 100 benchmark-ready BioModels models. Seventy-five completed the full workflow through held-out testing, producing 300 model/training-size conditions across training sets of 4, 8, 16, and 32 trajectories. Under the fixed experimental configuration, 16 training trajectories produced the lowest median held-out macro-NRMSE (0.0243) and the highest number of complete held-out successes (73/75). Mechanistic evaluation on 66 feature-space-compatible models showed substantially stronger recovery of state dependencies than exact equations: mean dependency F1 reached 0.844 at 16 trajectories, whereas only 2 of 991 evaluable right-hand sides were exact algebraic matches, both corresponding to trivial zero equations.

These results demonstrate an important distinction for biological equation discovery: an inferred ODE system can reproduce unseen dynamics and recover meaningful structural components without being algebraically identical to the generating equations. The project also provides an auditable execution pipeline, explicit training/validation/test boundaries, mechanistic evaluation tools, provenance records, and a containerized environment for future symbolic-regression studies.

## 2. Motivation and biological context

Mathematical models are widely used to describe how interacting biological components change over time. Examples include signaling networks, metabolic pathways, gene-regulatory systems, and other biochemical reaction networks. A deterministic model with $J$ dynamic states can be represented as

$$
\frac{d\mathbf{x}}{dt}
=
\mathbf{f}(\mathbf{x}, \boldsymbol{\theta}),
$$

where $\mathbf{x}$ contains the dynamic state variables, $\boldsymbol{\theta}$ contains model parameters, and $\mathbf{f}$ specifies the mechanisms governing their rates of change.

In a conventional mechanistic-modeling workflow, the mathematical form of $\mathbf{f}$ is constructed from biological knowledge and then its parameters are estimated from experimental data. Equation discovery asks a more difficult question: **can the mathematical form itself be inferred from observations?**

This distinction matters because different equations can generate similar trajectories over a limited region of state space. A model can therefore be an excellent predictor without reproducing the mechanism that generated the data. Conversely, an inferred equation may identify biologically relevant dependencies while differing in coefficients, interactions, or algebraic form.

The central scientific problem in this project is consequently not just whether symbolic regression can fit biological time-series data. It is whether an inferred system can generalize to unseen perturbations and how closely its mathematical structure resembles the known generating system.

BioModels provides a useful environment for studying this problem because it is a curated repository of published quantitative biological models [9]. Many of these models are represented using the Systems Biology Markup Language (SBML), a standardized machine-readable format for exchanging computational models of biological processes [10]. Together, BioModels and SBML provide known dynamical systems that can serve as ground truth for equation-discovery experiments. Instead of evaluating an inferred equation against an unknown experimental mechanism, the benchmark begins with an existing model, generates controlled observations from it, hides its equations from the symbolic-regression search, and then compares the inferred system with the original equations.

## 3. Symbolic regression for biological ODE discovery

### 3.1 From trajectories to equations

For an ODE system,

$$
\frac{d\mathbf{x}}{dt}
=
\mathbf{f}(\mathbf{x}),
$$

the equation-discovery problem can be expressed as learning an approximation

$$
\frac{d\mathbf{x}}{dt}
\approx
\hat{\mathbf{f}}(\mathbf{x}),
$$

where $\hat{\mathbf{f}}$ is inferred from observations rather than specified beforehand.

In this benchmark, each dynamic state is treated as a separate symbolic-regression target during candidate generation. For state $x_j$, PySR searches for expressions of the form

$$
\frac{dx_j}{dt}
\approx
\hat{f}_j(x_1,x_2,\ldots,x_J).
$$

The individual state equations cannot ultimately be judged in isolation because they interact when integrated as a coupled ODE system. The workflow therefore retains multiple candidate expressions for each right-hand side and subsequently evaluates combinations of those candidates on validation trajectories.

### 3.2 Why symbolic regression?

Conventional regression typically estimates coefficients within a predefined mathematical structure. Symbolic regression instead searches over both mathematical structure and numerical constants. Candidate expressions can therefore differ in which variables they use, how variables interact, and which algebraic operations connect them.

This property makes symbolic regression attractive for equation discovery: its output is an explicit mathematical expression rather than only a numerical predictor. That interpretability also creates a stricter evaluation problem. Two expressions can produce similar predictions without being mathematically identical.

PySR was used as the symbolic-regression engine for the final benchmark. The objective of the project is not to establish that PySR is universally optimal, but to use a controlled PySR configuration to establish a reproducible baseline for biological ODE discovery.

### 3.3 Related work

Data-driven discovery of dynamical systems has been studied using several methodological families. A prominent example is Sparse Identification of Nonlinear Dynamics (SINDy), which identifies a sparse set of active terms from a predefined library of candidate functions [1]. Symbolic regression addresses a related problem but searches more directly over mathematical expressions rather than selecting coefficients only within a fixed candidate library.

PySR is an open-source symbolic-regression system built on SymbolicRegression.jl. It combines multi-population evolutionary search, expression simplification, numerical optimization of constants, and explicit tradeoffs between prediction error and expression complexity [2]. Although many early scientific applications of PySR focused on static relationships or symbolic distillation, subsequent studies have increasingly used it as part of dynamical-system identification workflows.

Applications to biological dynamics already exist. AI-Aristotle combined physics-informed learning with symbolic regression for gray-box identification in systems biology and used PySR, alongside gplearn, to identify missing terms in pharmacokinetic and ultradian glucose-insulin ODE models [3]. de Rooij and colleagues subsequently used PySR to obtain an interpretable symbolic expression from the learned component of a conditional universal differential-equation model of C-peptide production [4]. These studies demonstrate that PySR can participate in biological equation-discovery workflows.

Related work has also examined the recovery of governing equations in more general dynamical systems. Grigorian and colleagues used hybrid neural ODE models followed by PySR to recover governing equations for unobserved states, including experiments with Lotka-Volterra and Lorenz systems, and evaluated the discovered equations through numerical reintegration and extrapolation [5]. Other recent studies have systematically examined how trajectory transformations, symbolic-regression methods, and hybrid sparse/symbolic approaches affect recovery of ODE systems [6,7].

A particularly relevant recent application is the work of de Pomereu and Fröhlich on intracellular signaling. Their 2026 preprint applies symbolic regression with PySR to discover coarse-grained kinetic laws for ERK phosphorylation dynamics across experimentally measured perturbation contexts, together with synthetic experiments in enzymatic kinetics [8]. This provides a direct example of PySR applied to molecular signaling dynamics, although the study remains a preprint and should therefore be distinguished from peer-reviewed applications such as AI-Aristotle.

These studies also motivate an important methodological distinction for the present benchmark. Low error when fitting a vector field does not necessarily imply that the resulting coupled ODE system will reproduce trajectories after numerical integration. Conversely, good trajectory prediction does not establish that the generating equation has been recovered. Evaluating equation discovery therefore benefits from separating local derivative agreement, trajectory-level generalization, structural similarity, and exact symbolic equivalence.

The goal of PySR-BioModels is not to claim the first use of PySR for biological ODEs. Instead, this project complements prior equation-discovery studies by evaluating a fixed PySR workflow across a heterogeneous corpus of curated SBML models with known generating equations. The benchmark uses explicit training, validation, and held-out perturbations and separately evaluates predictive generalization, structural recovery, numerical equivalence, and exact algebraic recovery.
### 3.4 Prediction is not equation recovery

The benchmark deliberately separates several increasingly different questions.

First, **does the workflow complete?** A computational run must finish before its scientific result can be evaluated.

Second, **does the inferred system integrate successfully?** Numerical integration is necessary for trajectory prediction, but successful integration alone says nothing about accuracy.

Third, **does the inferred system predict unseen trajectories?** This is measured on held-out initial conditions that are not used during fitting or system selection.

Fourth, **does the inferred equation recover important structural components of the generating equation?** The benchmark evaluates state dependencies, monomial structure, multiplicative interactions, and rational structure.

Finally, **is the inferred expression actually the same mathematical function as the generating expression?** Numerical and exact algebraic equivalence are evaluated separately.

These distinctions are central to interpreting the results. Workflow completion is not recovery, successful integration is not accurate prediction, and accurate prediction is not exact symbolic identification.

## 4. Research questions

The final benchmark addresses four primary questions:

1. **Can symbolic regression produce coupled ODE systems that generalize to unseen initial conditions across a heterogeneous collection of biological models?**

2. **How does the number of training trajectories affect predictive and mechanistic performance under a fixed symbolic-regression search budget?**

3. **To what extent do inferred equations recover structural components of the known generating equations, including state dependencies, monomial structure, interactions, and rational form?**

4. **How often does strong predictive or structural agreement correspond to numerical or exact algebraic recovery of the generating right-hand sides?**

A secondary engineering objective is to make these questions reproducible. The workflow therefore records model provenance, experimental designs, stage outputs, selections, held-out boundaries, resource failures, and mechanistic-evaluation artifacts rather than reporting only a final accuracy number.

## 5. Building the BioModels benchmark corpus

### Model retrieval

At the time of corpus acquisition, BioModels listed 1,096 manually curated models. The retained local corpus contains 1,075 model directories, which form the auditable starting point for the benchmark screening reported here. The disposition of the remaining 21 models could not be reconstructed from the retained acquisition records.

BioModels identifiers remained standardized `BIOMD...` identifiers. The discovery problem concerned the filenames of valid SBML documents inside local model directories, not identifier heterogeneity.

### Screening and compatibility

Preflight screening checks whether an SBML file can be resolved and loaded and whether the model fits the benchmark's executable scope. The current design requires deterministic ODE dynamics compatible with the state-only inference task, finite positive nominal values for dynamic floating species, and usable model-specific simulation metadata. Models with unsupported constructs are excluded from this benchmark configuration; exclusion is not evidence that the underlying biological model is invalid.

Compatibility also matters after execution. Three completed models contain an explicit time dependence that is absent from the inference feature space. Their 12 training-size conditions remain in the predictive results but are labeled as feature-space mismatches and excluded from primary mechanistic averages.

### Historical selection funnel

The repository retains the following historical screening stages:

| Historical stage | Models remaining |
|---|---:|
| Local curated model directories | 1,075 |
| Passed the no-variable-parameter criterion | 497 |
| Found by the earlier canonical-filename resolver | 326 |
| Passed the earlier load/preflight stage | 303 |
| Passed the earlier compatibility stage | 66 |
| Had required simulation-time metadata | 43 |

These figures document how the initial 43-model set was produced. They are not a fully reconstructed, mutually exclusive funnel for the final benchmark pipeline: the criteria and discovery logic subsequently changed, and some historical incompatibility categories overlap.

![Historical corpus screening and filename-aware expansion](figures/01_corpus_screening_and_expansion.svg)

*Figure 1. Historical screening counts and the filename-resolution expansion from 43 previously benchmark-ready models plus 57 additional models to the final 100-model cohort. The historical sequence is not presented as a reconstructed final-pipeline exclusion funnel.*

### SBML filename-resolution issue

The earlier local resolver expected the canonical filename `<BIOMD_ID>_url.xml`. Many valid downloaded SBML files instead used names such as an author/year filename, `Model.xml`, or `PVR.xml`. Directories containing those files were therefore missed even though their BioModels identifiers were standardized and their SBML could be valid.

The updated resolver first prefers the canonical filename. If it is absent, it accepts a unique XML/SBML candidate; if multiple candidates remain ambiguous, it rejects the directory rather than guessing. This is a robustness fix in local benchmark discovery, not a BioModels defect.

The change made 57 additional models benchmark-ready. Inspection of the bundled 100-model corpus confirms 43 canonical filenames and 57 noncanonical filenames. Archive identifiers can be standardized while local file layouts are not.

### Expanded 100-model benchmark

The final cohort is the disjoint union of the historical 43 models and 57 models admitted after filename-aware discovery, metadata completion, and preflight validation. Stored cohort validation reports that all 100 entries pass the final readiness checks. Each bundled model is associated with SBML content and provenance metadata.

Here, *benchmark-ready* means that a model passed structural, metadata, and preflight requirements. It does not mean that its computational run later finished, predicted accurately, or matched the reference mechanism.

## 6. Experimental design

Initial-condition factors span 0.1 to 10 times each positive nominal dynamic state value. To sample this multiplicative perturbation range systematically, independent 256-point Latin-hypercube candidate pools are generated in log10 space for training, validation, and held-out testing. Specifically, the sampling range extends from $\log_{10}(0.1)=-1$ to $\log_{10}(10)=1$ in each dimension. In this representation, a value of $-1$ corresponds to a factor of $10^{-1}=0.1$, a value of $0$ corresponds to $10^0=1$ (the nominal initial condition), and a value of $1$ corresponds to $10^1=10$. For example, if a state has a nominal initial value of 5, these three points correspond to initial values of 0.5, 5, and 50, respectively. Thus, perturbations below and above the nominal value are represented symmetrically on the logarithmic scale. Latin-hypercube sampling is a stratified sampling method that distributes samples across the range of every dimension, providing broad coverage of the multidimensional initial-condition space. A maximin criterion is then used to favor candidate points that are well separated from one another, further spreading the selected initial conditions across perturbation space.

| Split | Seed | Selected trajectories | Role |
|---|---:|---:|---|
| Training | 1010 | Nested prefixes of 4, 8, 16, and 32 | Fit state-wise candidates |
| Validation | 2020 | 8 | Select one coupled system |
| Held-out test | 3030 | 8 | Evaluate the frozen system |

A random seed fixes the initial state of a pseudorandom number generator, making a stochastic procedure reproducible. Here, the seeds 1010, 2020, and 3030 were used by the benchmark's initial-condition sampling procedure to generate the training, validation, and held-out test designs, respectively; these are not three PySR seeds. Their numerical values have no special scientific meaning and simply allow the same sampling designs to be regenerated. Separately, the PySR search itself used a fixed seed of 1010, as described below, to make the symbolic-regression search reproducible.

All training conditions use the same maximum 32-trajectory ordering, so smaller sets are nested within larger ones. Validation points are selected against the training design, and held-out points against the combined training and validation designs.

Before any test simulation is materialized, the runner validates that all selected systems and checkpoints are frozen. Held-out data are not used to choose candidates, refit coefficients, tune hyperparameters, or reselect a system.


## 7. Simulation and derivative generation

The benchmark uses a project-defined simulation protocol rather than executing every experiment in each model's SED-ML material. Every trajectory starts at time zero and contains 101 requested time points over a model-specific duration. Dynamic floating species are perturbed through their initial values; parameters remain fixed at nominal values.

For each simulated point, RoadRunner [11] returns the exact state derivatives used as PySR targets. Thus the experiment evaluates symbolic regression given noiseless simulator derivatives, not derivative estimation from sampled trajectories.

Simulation durations are read from bundled, model-aware metadata and checked for a matching model identifier and filename and for a positive finite value. During corpus preparation, an automated multimodal procedure extracted candidate simulation horizons from the largest plotted time axis in curated model figures. These model-specific durations were then retained as fixed benchmark metadata and used to define the standardized simulation horizon for each model. The benchmark therefore uses a consistent project-defined simulation protocol rather than attempting to reproduce every experimental or simulation configuration associated with each original BioModels entry.


## 8. PySR equation-discovery workflow

The current pipeline is:

1. resolve and preflight the SBML model, where *preflight* is a preliminary eligibility check that verifies that the model can be located and loaded and satisfies the structural, initial-condition, and simulation-metadata requirements of the benchmark before computationally expensive inference begins;
2. create deterministic training, validation, and test initial conditions;
3. simulate development data and extract exact derivatives;
4. compute state scales from training data only;
5. fit one PySR candidate set for each state equation's right-hand side (RHS)—the mathematical expression that defines that state's time derivative—using all dynamic states as input features;
6. transform retained expressions back to the original state coordinates and units;
7. select a coupled ODE system by integrating candidate combinations on the validation trajectories;
8. freeze and checkpoint the selected system so that its equations can no longer change;
9. load held-out initial conditions and evaluate the frozen system on previously unused trajectories; and
10. compare the selected RHSs with the known equations of the original SBML model, referred to here as the *mechanistic ground truth*, where the inferred and reference equations use compatible feature spaces.

![PySR benchmark pipeline and data boundaries](figures/02_refactored_pipeline.svg)

*Figure 2. Benchmark workflow. Validation trajectories participate in coupled-system selection; held-out trajectories are opened only after selection is frozen, and symbolic ground truth is used only for post-hoc mechanistic analysis.*

### PySR search and optimization

PySR performs symbolic regression by searching over mathematical expressions rather than assuming a predefined equation form. Its SymbolicRegression.jl backend uses an evolutionary search that maintains populations of candidate expressions and progressively modifies them through operations that alter their expression trees, such as changing operators, variables, constants, or subexpressions. The resulting expressions are evaluated according to their fit to the training data and continue to compete and evolve during the search. In this way, PySR explores alternative mathematical structures rather than optimizing only the coefficients of a fixed equation.

In this benchmark, each state derivative is inferred separately, with all dynamic states available as possible input variables. The search is restricted to the binary operators `+`, `-`, `*`, and `/`, meaning operators that combine two inputs, and uses no unary operators such as `exp`, `log`, or `sin`, which act on a single input. Expression size is limited to 20 nodes, constraining the structural complexity of candidate equations.

The configured PySR version is 1.5.9. Each RHS search uses 60 iterations, a fixed random seed of 1010, and deterministic serial execution. The explicit parsimony coefficient is set to 0. In symbolic regression, *parsimony* refers to penalizing more complex expressions so that simpler equations are preferred when appropriate. Setting this coefficient to zero disables this explicit complexity penalty in the objective; however, candidate expressions remain structurally constrained by the maximum expression size of 20.

PySR optimizes two related aspects of a candidate equation: its symbolic structure and the numerical constants appearing within that structure. The evolutionary search explores alternative mathematical structures, while PySR's internal constant optimizer adjusts numerical constants to reduce training loss for a given structure. For example, if the symbolic search proposes an expression of the form $a x + b y$, constant optimization determines numerical values for $a$ and $b$ that improve agreement with the exact training derivatives.

In this configuration, PySR uses the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for numerical constant optimization. BFGS is an iterative optimization method that uses information about how the loss changes with the constants, together with an approximation of the local curvature of the loss function, to update those constants in directions expected to reduce the loss. This optimization occurs during PySR candidate fitting and is distinct from a later coefficient-refitting step: once candidate equations leave the PySR search and the coupled system is selected, this workflow does not refit their coefficients.

Up to eight candidate expressions are retained for each RHS and ranked deterministically by training loss, which measures disagreement between a candidate derivative and the exact derivative values in the training data. These retained state-wise candidates provide the alternatives considered during subsequent coupled-system selection.

### Scaling and transformation back to model coordinates

Because state variables can differ substantially in numerical magnitude, the symbolic-regression problem is solved in scaled coordinates. For each state, the *state scale* is the maximum absolute value observed for that state in the training data. Input state values are divided by their corresponding scales, and each derivative target is divided by the scale of the state whose derivative is being modeled. These scales are calculated from training data only, so information from validation or held-out trajectories does not enter the fitting transformation.

Expressions retained by PySR are then symbolically back-transformed from these normalized coordinates to the original state coordinates and units. This produces equations that can be integrated and interpreted in the coordinate system of the original SBML model.

### Coupled-system selection

Although PySR fits candidate expressions separately for each state derivative, a biological ODE model is a *coupled system*: the equations interact because the derivative of one state can depend on other states. Consequently, choosing the individually best-fitting RHS for every state does not necessarily produce the best dynamical system when the equations are integrated together.

The selector therefore begins with the top-ranked PySR candidate for every RHS and performs a deterministic *coordinate search*. Here, coordinate search means changing the selected equation for one state at a time while keeping the equations for the other states fixed, then testing whether that substitution improves the performance of the complete coupled system on the validation trajectories. Candidate systems are evaluated by numerical integration—that is, by solving the inferred ODE system over time—rather than by derivative-fitting loss alone.

The primary validation criterion is macro-NRMSE (macro normalized root-mean-square error), a trajectory-error metric normalized to account for differences in state magnitude and aggregated across the system. When candidate systems are tied, the selector uses validation macro-NRMSE, total expression complexity, total PySR training loss, and finally a canonical system identifier to obtain a deterministic ordering.

This coordinate search starts from a single initial combination of RHS candidates. It is therefore reproducible for fixed inputs but is not an exhaustive search over every possible combination and is not guaranteed to identify the globally optimal coupled system. After selection, the chosen equations are frozen before held-out trajectories are loaded. The held-out data therefore measure generalization of an already selected system rather than contributing to model selection.


## 9. Evaluation framework

There is no single definition of successful ODE recovery. An inferred system may reproduce the trajectories of the original model while using different mathematical expressions, or it may recover important structural components without being exactly algebraically identical to the original equations. The benchmark therefore evaluates five complementary questions:

1. **Predictive dynamics:** Does the inferred ODE system reproduce the trajectories of the reference model?
2. **State-dependency recovery:** Does each inferred equation depend on the same dynamic state variables as the corresponding reference equation?
3. **Functional-form recovery:** Does the inferred equation recover important mathematical components of the reference equation, such as monomials, multiplicative interactions, or state-dependent denominators?
4. **Numerical equivalence:** Do the inferred and reference right-hand sides (RHSs) produce approximately the same derivative values at sampled points in state space?
5. **Exact algebraic equivalence:** After symbolic simplification, are the inferred and reference RHSs mathematically identical?

These questions measure different properties of an inferred model. In particular, good trajectory prediction does not by itself demonstrate recovery of the equation that generated those trajectories.

### Predictive dynamics

Predictive dynamics are evaluated by numerically integrating both the reference and inferred ODE systems from the same initial conditions. Integration uses SciPy's `solve_ivp` with the LSODA solver, a relative tolerance of `1e-7`, an absolute tolerance of `1e-9`, and at most 8,000 RHS evaluations per trajectory.

The trajectory error is calculated in several steps so that errors from states with very different numerical magnitudes can be compared fairly.

First, for state $j$ in trajectory $k$, sampled at $T$ time points, the root-mean-square error (RMSE) measures the typical difference between the inferred and reference trajectories:

$$
\mathrm{RMSE}_{k,j}
=
\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
\left(
\hat{x}_{k,j}(t)-x_{k,j}(t)
\right)^2
}.
$$

RMSE is expressed in the numerical scale of the corresponding state. Because different biological variables can have very different magnitudes, the benchmark normalizes this error using a state-specific scale calculated from training data only:

$$
s_j
=
\max_{r\in\mathcal{D}_{\mathrm{train}}}
|x_{r,j}|,
\qquad
\mathrm{NRMSE}_{k,j}
=
\frac{\mathrm{RMSE}_{k,j}}{s_j}.
$$

Here, $s_j$ is the largest absolute training value observed for state $j$. Using training data only prevents validation or held-out data from influencing the normalization.

For a model with $J$ dynamic states, the normalized errors are then averaged across states to obtain one error value for trajectory $k$:

$$
M_k
=
\frac{1}{J}
\sum_{j=1}^{J}
\mathrm{NRMSE}_{k,j}.
$$

This quantity is called the **trajectory macro-NRMSE**. The term *macro* indicates that each state contributes equally to the average rather than states being weighted according to their numerical magnitude or number of observations.

Finally, the benchmark evaluates $K=8$ trajectories in either the validation or held-out partition and averages their trajectory scores:

$$
M_{\mathrm{system}}
=
\frac{1}{K}
\sum_{k=1}^{K}
M_k,
\qquad K=8.
$$

This is the **system macro-NRMSE**, which provides one predictive-error score for the complete inferred ODE system. Lower values indicate closer agreement with the reference trajectories.

The same mathematical definition has two different roles in the workflow. **Validation macro-NRMSE** is used during the coupled-system selection described in Section 8. After the selected equations are frozen, **held-out macro-NRMSE** measures generalization on initial conditions that did not participate in fitting or system selection. The held-out score therefore does not influence which equations are selected.

A system macro-NRMSE is considered finite only when all eight trajectories in the corresponding partition integrate successfully. If any of the eight integrations fails, the implementation records a failed status and an infinite system macro-NRMSE. Integration success is therefore tracked separately from the magnitude of the trajectory error.

### State-dependency recovery

Predictive agreement does not reveal whether PySR recovered the same variables in the equations. State-dependency recovery therefore asks a simpler structural question: **does an inferred RHS depend on the same dynamic states as the corresponding reference RHS?** This evaluation builds on the broader idea of structural or support recovery used in sparse dynamical-system identification, where the active components of an inferred model are compared with those of the known generating system [1,12,13]. Because PySR produces free-form symbolic expressions rather than selecting terms from a fixed candidate-function library, this benchmark defines support at the level of dynamic-state dependencies. For each RHS, the dynamic state variables appearing in the reference and inferred expressions are treated as sets and compared using precision, recall, and F1.

For example, suppose a reference RHS depends on states $A$ and $B$, while the inferred RHS depends on $A$ and $C$. State $A$ is a true positive (TP) because it appears in both dependency sets. State $C$ is a false positive (FP) because it appears only in the inferred equation, and state $B$ is a false negative (FN) because the inferred equation failed to recover it.

From these counts, precision, recall, and F1 are calculated as

$$
P
=
\frac{TP}{TP+FP},
\qquad
R
=
\frac{TP}{TP+FN},
\qquad
F_1
=
\frac{2PR}{P+R}.
$$

The implementation evaluates F1 using the equivalent count expression

$$
F_1
=
\frac{2TP}{2TP+FP+FN}.
$$

Precision answers: *of the state dependencies inferred by PySR, how many were present in the reference equation?* Recall answers: *of the state dependencies present in the reference equation, how many were recovered?* F1 combines these two quantities into a single score.

Precision is undefined (`NaN`) when no dependencies are inferred, and recall is undefined when the reference dependency set is empty. If both dependency sets are empty, F1 is defined as 1 because the absence of state dependencies was correctly recovered. If only one set is empty, F1 is 0.

Model-level dependency F1 is calculated by macro-averaging the RHS-level F1 scores, giving each state equation equal weight. Importantly, this metric considers only **which state variables appear**. For example, $AB$, $A+B$, and $A/B$ all involve the dependency set $\{A,B\}$, even though their mathematical forms are very different. A high dependency F1 therefore does not establish recovery of powers, interactions, coefficients, or rational structure.

### Functional-form recovery

To examine mathematical structure in greater detail, the evaluator compares specific components of the inferred and reference RHSs after symbolic cancellation.

Consider, for example, an expression of the form

$$
\frac{A^2B+C}{1+D}.
$$

This expression contains several kinds of structural information. The numerator contains the monomials $A^2B$ and $C$; $A$ and $B$ participate in a multiplicative interaction in the first monomial; and state $D$ appears in the denominator, making the expression rational with respect to the dynamic states.

The evaluator therefore compares:

- **numerator monomial signatures:** state-exponent patterns describing the variable composition and powers of numerator terms; constant-only terms are recorded but excluded from monomial precision, recall, and F1;
- **multiplicative interactions:** pairs of distinct states that occur together within a numerator monomial;
- **rational structure:** whether the denominator contains dynamic-state dependence;
- **denominator dependencies:** which dynamic states appear in the denominator; and
- **polynomial degree information:** the numerator and denominator polynomial degrees and their corresponding errors.

For structural components represented as sets, the same precision, recall, and F1 definitions introduced above for state dependencies are reused. What changes is the type of object treated as a set element. Dependency F1 compares dynamic state variables, numerator monomial F1 compares numerator monomial signatures, and interaction F1 compares multiplicative state-state interactions. Thus, the meanings of TP, FP, and FN depend on the structural component being evaluated, but the F1 formula itself is unchanged. Other structural quantities, such as rational-structure agreement and polynomial-degree error, are evaluated with their corresponding non-F1 metrics.

These component-level metrics distinguish equations that contain similar biological or mathematical structure even when the complete expressions are not identical. The report uses model-level macro summaries so that models containing more state equations do not automatically receive greater weight. Functional-form recovery should therefore be interpreted as evidence of specific structural similarities, not as proof that the complete equation was recovered.

### Numerical equivalence

State dependencies and functional-form components describe structural similarity, but two expressions can also be compared directly by evaluating the derivatives they produce. Symbolic-regression benchmarks distinguish predictive or numerical agreement from recovery of the underlying symbolic expression [14,15]. Following this general distinction, the numerical-equivalence diagnostic used here tests whether the inferred and reference RHSs produce approximately the same derivative values at sampled points in state space.

For each evaluable RHS, the reference and inferred expressions are evaluated at 32 deterministic positive state vectors. Each state value is independently generated as

$$
x = 10^u,
\qquad
u \sim U(-1,1),
$$

using random seed 4040. Each state value is therefore sampled independently and uniformly in log10 space over the interval from $0.1$ to $10$. These values are synthetic state-space evaluation points used only for the numerical-equivalence diagnostic; unlike the initial-condition perturbations described in Section 6, they are not scaled relative to each model's nominal initial conditions. The number of evaluation points, sampling distribution, random seed, and numerical tolerances are benchmark-specific choices rather than standard thresholds adopted from these studies.

At every sampled state vector, the inferred and reference derivative values are compared using NumPy's `np.isclose`, with relative tolerance `1e-6` and absolute tolerance `1e-9`. An RHS is marked numerically equivalent only when all 32 evaluations are valid and all 32 comparisons satisfy these tolerances.

This test asks whether the two expressions behave approximately the same at the sampled points. It is deliberately stronger than agreement at a single point but remains a finite numerical test. Passing it does **not** prove that the expressions are identical everywhere in state space.

This numerical-equivalence test is a post-hoc diagnostic. It does not participate in PySR fitting or coupled-system selection.

### Exact algebraic equivalence

The strictest comparison asks whether the inferred and reference RHSs are mathematically the same expression, allowing for algebraically equivalent ways of writing that expression.

For each evaluable RHS, the exact evaluator uses SymPy to canonicalize the expressions, subtracts the inferred RHS from the reference RHS, and applies rational combination, cancellation, and symbolic simplification. If the simplified difference is exactly zero, either directly or through SymPy's symbolic equality test, the RHS is classified as an exact algebraic match.

For example, expressions that differ only because one has been algebraically expanded or factored can still be recognized as equivalent after symbolic simplification. In contrast, an inferred coefficient that is merely close to the reference coefficient does not qualify as an exact match. No numerical coefficient tolerance is applied.

Exact algebraic equivalence is also evaluated only after model selection and therefore does not influence which PySR equations are selected.

### How the evaluation dimensions relate

The five evaluation dimensions answer different questions and should not be interpreted as successive levels of a single recovery hierarchy. Predictive dynamics measure whether the complete inferred system reproduces trajectories. State-dependency and functional-form metrics examine progressively more detailed aspects of equation structure. Numerical equivalence tests whether individual RHSs produce approximately the same derivative values over a finite sample of state space, while exact algebraic equivalence tests whether the inferred and reference RHSs are symbolically identical.

This separation is important because an inferred system can achieve very low held-out trajectory error without exactly recovering the equations that generated those trajectories. Distinguishing predictive performance from mechanistic recovery is therefore central to the interpretation of the benchmark results.


## 10. Expanded-100 execution outcomes

The Expanded-100 benchmark contains 100 BioModels that were benchmark-ready. Being *benchmark-ready* means that a model had a resolved SBML input, valid simulation metadata, and passed the benchmark preflight checks. It does **not** mean that symbolic regression was guaranteed to finish successfully or recover the original equations.

Of these 100 models, 75 completed the full workflow through held-out testing, while 25 did not complete under the imposed computational constraints. Each completed model was evaluated using four nested training-set sizes—4, 8, 16, and 32 trajectories—so the 75 completed models produced

$$
75 \times 4 = 300
$$

final model-by-training-size conditions for predictive evaluation.

| Execution outcome | Models | Interpretation |
|---|---:|---|
| Workflow complete through held-out testing | 75 | Reached final held-out evaluation and was included in the main predictive analysis |
| Timeout-censored | 12 | Execution stopped because the configured time limit was reached |
| Memory-censored | 2 | Execution stopped because the configured memory limit was reached |
| Other incomplete candidate-stage worker exits | 11 | Candidate-generation worker exited unsuccessfully, but the retained evidence does not support a more specific cause |
| **Total benchmark-ready** | **100** | Passed the requirements needed to enter the benchmark |

Here, *resource-censored* means that execution was stopped by a predefined computational limit rather than producing a scientific recovery result. The 12 timeout-censored and 2 memory-censored models therefore provide information about the computational limits of the benchmark implementation, not about whether their underlying biological equations were recoverable.

The remaining 11 incomplete models require similar caution. Their retained manifests record nonzero worker exits during candidate generation, indicating that those processes did not finish successfully. However, the retained evidence is insufficient to determine a common underlying cause, so these runs are not classified more specifically.

Most importantly, **workflow completion is not the same as equation recovery**. A completed model is one for which the pipeline reached held-out evaluation. Whether the inferred equations accurately reproduce unseen trajectories, recover mechanistic structure, or match the original equations is evaluated separately in the following sections.

![Expanded-100 execution outcomes](figures/03_expanded100_execution_outcomes.svg)

*Figure 3. Execution outcomes for the 100 benchmark-ready models. Seventy-five models completed the workflow through held-out testing. The 25 incomplete runs comprise 12 timeout-censored runs, 2 memory-censored runs, and 11 candidate-stage worker exits whose causes cannot be classified more finely from the retained evidence.*

## 11. Predictive performance

The held-out trajectories provide the final test of predictive generalization because they are not used to fit the PySR equations or to select among candidate coupled systems. A full held-out success means that the selected system can be numerically integrated for all eight held-out trajectories. Among these successful systems, trajectory accuracy is summarized by macro-NRMSE: lower values indicate closer agreement between the inferred and reference trajectories. Because this metric averages normalized errors across states and trajectories, it should be interpreted as a comparative error score rather than as a direct percentage error.

The table summarizes the 75 workflow-complete models. Median macro-NRMSE is calculated only among systems with finite full held-out scores.

| Training trajectories | Full held-out successes | Median held-out macro-NRMSE |
|---:|---:|---:|
| 4 | 71/75 | 0.042832 |
| 8 | 69/75 | 0.028292 |
| 16 | 73/75 | 0.024344 |
| 32 | 71/75 | 0.026234 |
| **All conditions** | **284/300** | — |

The median is used as the primary summary because the distribution of finite errors is strongly right-skewed. In other words, most successful integrations have much smaller errors, while a small number produce extremely large errors; at least one condition that completed all eight held-out integrations has a macro-NRMSE above 1,000. Such a condition is numerically successful in the sense that integration completes, but it is not an accurate prediction. Under the fixed benchmark configuration, median held-out error decreases from 0.042832 with 4 training trajectories to 0.028292 with 8 and 0.024344 with 16, before increasing slightly to 0.026234 with 32. Among the four tested training-set sizes, the 16-trajectory condition had both the lowest conditional median held-out macro-NRMSE and the highest full held-out success rate, with 73/75 models (97.3%), under this fixed search configuration. This descriptive comparison does not establish 16 trajectories as a universally optimal training-set size.

![Predictive and mechanistic performance by training-set size](figures/04_predictive_and_mechanistic_metrics.svg)

*Figure 4. Predictive and mechanistic results across training-set sizes. Predictive panels use the 75 workflow-complete models; systems with full held-out integration define the NRMSE medians. Mechanistic curves are model-level means over the 66 compatible models.*

Figure 4 separates three aspects of performance. Panel A shows the conditional median held-out macro-NRMSE among systems that successfully completed all eight held-out trajectories. Panel B shows the complementary question of numerical robustness: the fraction of the 75 models that completed all eight trajectories. The two measures should be interpreted separately. For example, the 8-trajectory condition has a lower median error than the 4-trajectory condition, but fewer models achieve full held-out integration (69/75 versus 71/75). Thus, lower error among successful systems does not necessarily imply a higher probability of successful integration.

Panel C previews the structural-recovery results analyzed in Section 12. State-dependency F1 is consistently higher than the monomial- and interaction-level F1 scores, indicating that PySR more often identifies **which state variables participate in an RHS** than the precise mathematical way in which those variables are combined. Most of the structural metrics improve from 4 to 16 training trajectories and then decrease at 32, again showing that performance is not monotonic with training-set size under a fixed symbolic-regression search budget.

The aggregate trend also does not mean that every individual model improves when more training trajectories are supplied. Comparing the 4- and 32-trajectory conditions, validation macro-NRMSE improves for 43 models and worsens for 28 among the 71 models with finite scores at both endpoints. For held-out performance, 44 improve and 24 worsen among the 68 models with finite scores at both endpoints. In addition, 62 of the 75 workflow-complete models achieve full held-out integration at all four training-set sizes. These paired results show substantial model-to-model heterogeneity: increasing the amount of training data can improve predictive accuracy for many models, but it does not produce a uniform or monotonic response across the benchmark.

### Concrete example: predictive agreement versus exact symbolic recovery

Figure 5 illustrates what a highly successful reconstruction looks like for BIOMD0000000156 under `train_32`. The selected system completes all eight held-out trajectories with an overall held-out macro-NRMSE of `3.6540933364804454e-06`. The displayed `test_003` trajectory has macro-NRMSE `3.763991365656924e-06`; it was selected from the official trajectory metrics as the lower-error member of the two central trajectories among the eight held-out cases, rather than being chosen by visual inspection.

![Held-out trajectory reconstruction and canonical equations for BIOMD0000000156](figures/05_heldout_trajectory_example.svg)

*Figure 5. Observed SBML and selected inferred-system trajectories for the held-out `test_003` initial condition, together with the three canonical RHS comparisons. The near-overlapping curves measure predictive agreement for this example; they do not establish exact equation identity.*

The blue reference trajectories and orange inferred trajectories are nearly indistinguishable at the scale of the plot, demonstrating excellent predictive agreement on an initial condition that was not used for fitting or system selection. The structural metrics tell a similarly strong, but distinct, story. For each of the three RHSs, dependency F1, numerator monomial F1, and interaction F1 equal 1, and the rational structure matches. As defined in Section 9, these F1 scores use the same precision-recall framework but compare different structural objects; a value of 1 therefore indicates perfect set-level agreement for the corresponding structural component.

However, structural and predictive agreement do not imply exact symbolic identity. All 3/3 RHSs pass the finite-point numerical-equivalence diagnostic, but 0/3 are exact algebraic matches. Figure 5 shows why: the inferred equations reproduce the same terms and very similar coefficients, but small coefficient differences remain. Under the deliberately strict exact-recovery criterion, those expressions are therefore not algebraically identical to the SBML reference equations.

This example captures a central distinction in the benchmark. An inferred ODE system can generalize accurately to unseen trajectories, recover important mechanistic structure, and be numerically almost indistinguishable from the reference RHS while still failing exact symbolic recovery. BIOMD0000000156 is an illustrative example of that distinction, not an estimate of population-level recovery performance.


## 12. Mechanistic recovery

Predictive accuracy asks whether an inferred ODE system reproduces unseen trajectories. Mechanistic recovery asks a different and stricter question: does the inferred equation recover important mathematical components of the reference ODE? As defined in Section 9, the benchmark evaluates this question at several structural levels, including state dependencies, numerator monomials, multiplicative interactions, and rational structure.

The mechanistic analysis starts from the 75 models that completed the full workflow. Because each model was evaluated using four training-set sizes (4, 8, 16, and 32 trajectories), these 75 models produce **75 × 4 = 300 model-training conditions**. Here, one model-training condition means one BioModels model evaluated at one particular training-set size.

Not all 300 conditions could be used for mechanistic comparison. For six models, ground-truth equation extraction failed. Because each of these six models appears at all four training-set sizes, this removes **6 × 4 = 24 conditions**, leaving **276 conditions with mechanistic evaluation results**.

A second compatibility issue affects three explicit-time models. Their reference equations depend explicitly on time, whereas the PySR feature space used in this benchmark contains only dynamic state variables. These three models account for **3 × 4 = 12 conditions** for which a direct structural comparison would not be like-for-like. They are therefore excluded from the primary mechanistic comparison.

The resulting primary cohort contains **66 structurally compatible models**. Evaluating each at four training-set sizes gives **66 × 4 = 264 model-training conditions** used for the main mechanistic-recovery analysis. The filtering can therefore be summarized as:

**300 total conditions → 276 with ground truth available → 264 structurally compatible conditions (66 models × 4 training-set sizes).**

The following analysis asks how much of the reference equation structure PySR recovers within this 66-model cohort, and whether that recovery changes as more training trajectories are provided. Rather than treating mechanistic recovery as a single yes-or-no outcome, the analysis separates it into increasingly detailed structural components. **Dependency F1** measures whether the correct dynamic state variables appear in each inferred RHS; **numerator monomial F1** evaluates whether the corresponding mathematical terms are recovered; **interaction F1** focuses on multiplicative interactions between state variables; and **rational-structure accuracy** tests whether the inferred and reference equations agree on the presence of state-dependent denominator structure. The table reports the mean model-level score for each metric at each training-set size.

| Training trajectories | Mean dependency F1 | Mean numerator monomial F1 | Mean interaction F1 | Rational-structure accuracy |
|---:|---:|---:|---:|---:|
| 4 | 0.8015 | 0.5538 | 0.5666 | 0.6553 |
| 8 | 0.8191 | 0.5705 | 0.6298 | 0.6506 |
| 16 | 0.8439 | 0.6125 | 0.6539 | 0.6503 |
| 32 | 0.8277 | 0.5853 | 0.5894 | 0.6374 |

These values are means of model-level macro scores rather than pooled equation-level means, so each biological model contributes equally regardless of how many state equations it contains.

The clearest pattern is that state-dependency recovery is stronger than recovery of the more detailed mathematical structure. For example, with 16 training trajectories, mean dependency F1 reaches 0.8439, compared with 0.6125 for numerator monomials and 0.6539 for multiplicative interactions. In practical terms, PySR is more successful at recovering **which dynamic state variables participate in an equation** than at recovering **the precise mathematical way in which those variables are combined**. This distinction is important because identifying the correct variables is only one part of reconstructing the underlying biological mechanism.

Rational-structure accuracy measures a different structural property: whether the inferred and reference expressions agree on the presence of rational state dependence. It is therefore not an F1 score. Its mean remains relatively stable from 4 through 16 training trajectories (0.6553, 0.6506, and 0.6503) and decreases to 0.6374 at 32 trajectories.

At the population level, dependency, numerator-monomial, and interaction F1 all improve from 4 to 16 training trajectories and then decline at 32. This agrees with the broader pattern seen in predictive performance: under the fixed search configuration, increasing the training set from 4 to 16 trajectories is associated with better aggregate performance, but increasing it further to 32 does not produce an additional aggregate improvement. This should not be interpreted as evidence that 16 trajectories are intrinsically optimal, because the symbolic-regression search budget and all other benchmark settings remain fixed.

The aggregate means above summarize the overall behavior of the 66-model cohort, but they do not show whether individual models respond consistently to additional training data. To examine this, a paired model-level analysis compares each model's structural-recovery score using 4 training trajectories with its score using 32 training trajectories. For each metric, a model is classified as **improved** if its score is higher at 32 trajectories, **unchanged** if the score is the same, and **worsened** if its score is lower. The table therefore shows how many of the 66 models fall into each category and tests whether the aggregate trend is shared consistently across individual models:

| Metric | Improved | Unchanged | Worsened |
|---|---:|---:|---:|
| Dependency F1 | 24/66 | 27/66 | 15/66 |
| Numerator monomial F1 | 33/66 | 10/66 | 23/66 |
| Interaction F1 | 23/66 | 27/66 | 16/66 |

The model-level responses are heterogeneous. No structural metric improves for all 66 models when the training set increases from 4 to 32 trajectories. Numerator-monomial F1 shows the largest number of improvements (33/66), but 23 models still worsen. Dependency and interaction F1 each remain unchanged for 27 models, while 15 and 16 models, respectively, worsen. Thus, the aggregate trends do not describe a uniform response to additional training data: depending on the model and structural metric, additional trajectories may improve recovery, leave it unchanged, or coincide with worse recovery.

Together, these results indicate that mechanistic recovery is not a single all-or-none outcome. PySR can recover broad state dependencies more reliably than finer algebraic structure, and additional training trajectories affect different biological models differently. The stricter question of whether the complete inferred RHS is numerically or algebraically equivalent to the reference equation is examined next.


## 13. Exact versus approximate recovery

The previous section evaluates whether inferred equations recover particular structural components of the reference ODEs. A stricter test asks whether an inferred RHS behaves like, or is mathematically identical to, the complete reference RHS. The benchmark therefore distinguishes **numerical equivalence** from **exact algebraic equivalence**.

These RHS-level analyses use only the 66-model compatible cohort. Across those models, 248 individual RHS equations are eligible for comparison at training sizes 4, 16, and 32. At training size 8, one algebraic evaluation timed out, leaving 247 evaluated RHSs.

| Training trajectories | Numerically equivalent RHSs | Exactly equivalent RHSs |
|---:|---:|---:|
| 4 | 33/248 | 2/248 |
| 8 | 35/247 | 0/247 |
| 16 | 35/248 | 0/248 |
| 32 | 37/248 | 0/248 |
| **Pooled** | **140/991 (14.1%)** | **2/991 (0.2%)** |

Numerical equivalence is substantially more common than exact algebraic equivalence. Across all four training-set sizes, 140 of 991 evaluated RHSs (14.1%) satisfy the benchmark's numerical-equivalence criterion. As defined in Section 9, this means that the inferred and reference RHSs produce sufficiently similar derivative values at the fixed set of sampled state-space points under the specified numerical tolerances. It is a finite numerical diagnostic and does not imply that the two symbolic expressions are mathematically identical.

Exact algebraic recovery is much rarer. Only 2 of 991 evaluated RHSs (0.2%) satisfy the exact symbolic criterion. Both occur for BIOMD0000000941 in the four-trajectory training condition (`train_04`) and correspond to states whose reference RHS is identically zero. For `E2F_total`, the reference and inferred equations are

$$
\text{Reference:}\qquad
\frac{d\,E2F_{\mathrm{total}}}{dt}=0,
\qquad
\text{PySR:}\qquad
\frac{d\,E2F_{\mathrm{total}}}{dt}=0.
$$

The same occurs for `Cdc20_total`:

$$
\text{Reference:}\qquad
\frac{d\,Cdc20_{\mathrm{total}}}{dt}=0,
\qquad
\text{PySR:}\qquad
\frac{d\,Cdc20_{\mathrm{total}}}{dt}=0.
$$

In both cases, the symbolic difference is exactly zero after canonical simplification. These matches are therefore exact, but they are also trivial zero equations: PySR did not need to reconstruct a nonzero kinetic, regulatory, or interaction term. No nonzero RHS achieves exact algebraic equivalence, and no complete coupled ODE system is exactly equivalent to its reference system.

The contrast between the two criteria is important. The same model provides a simple example. For `E2F_total` with eight training trajectories (`train_08`), the reference RHS remains zero, whereas PySR infers

$$
\text{Reference:}\qquad
\frac{d\,E2F_{\mathrm{total}}}{dt}=0,
\qquad
\text{PySR:}\qquad
\frac{d\,E2F_{\mathrm{total}}}{dt}
=3.62222648906321\times10^{-23}.
$$

This RHS satisfies the numerical-equivalence criterion but not the exact algebraic criterion. The inferred value is extremely close to zero under the benchmark's numerical tolerances, but it is not symbolically identical to zero. More generally, numerical agreement without exact symbolic recovery can arise from small coefficient differences, alternative functional forms, or agreement restricted to the sampled state-space region. Conversely, the exact criterion deliberately requires symbolic identity after the canonical simplification procedure described in Section 9; numerical closeness alone cannot satisfy it.

These results also place the structural metrics from Section 12 in context. The substantially higher state-dependency F1 scores show that PySR often identifies which dynamic variables participate in an RHS, while the much lower RHS-level rates of numerical and exact equivalence show that recovering those dependencies does not imply recovery of the full governing equation. The benchmark therefore supports a graded interpretation of recovery: identifying relevant states is more common than reproducing the complete RHS numerically, and exact recovery of a nontrivial reference RHS is not observed in this experiment.


## 14. Case studies

The aggregate results in the previous sections summarize performance across many models, but they do not show how the different definitions of recovery relate within an individual biological system. The purpose of these case studies is therefore to examine several contrasting inference outcomes at the model level and make the distinction between successful integration, predictive accuracy, structural recovery, numerical equivalence, and exact algebraic recovery concrete.

Figure 6 compares three deliberately selected models that exhibit contrasting behaviors. They were not selected to estimate how common these behaviors are in the benchmark, nor should they be interpreted as the three best or worst models. Instead, each was chosen to illustrate a different relationship among the evaluation criteria. BIOMD0000000156 shows strong predictive and structural recovery without exact algebraic identity; BIOMD0000000319 shows that identifying the correct state dependencies does not guarantee recovery of the detailed functional form; and BIOMD0000000728 shows that successful numerical integration does not guarantee accurate prediction or mechanistic recovery.

To read the figure, begin with the upper row. Each point represents the held-out macro-NRMSE obtained with 4, 8, 16, or 32 training trajectories. Lower values indicate closer agreement between inferred and reference trajectories; because the errors span several orders of magnitude, the vertical axis is logarithmic. The annotation below each upper panel reports whether all eight held-out trajectories could be integrated successfully. Integration success is reported separately from error because an ODE system can integrate without reproducing the reference dynamics accurately.

The lower row evaluates equation structure rather than trajectories. Dependency F1 asks whether the inferred equations use the correct dynamic state variables. Monomial F1 asks whether the numerator contains the correct state-variable products and powers, while interaction F1 focuses on multiplicative interactions among states. Rational accuracy evaluates whether rational state dependence is represented correctly. For these structural scores, values closer to 1 indicate stronger agreement with the reference equations, whereas values closer to 0 indicate weaker agreement. The annotation inside each lower panel additionally reports, for the 32-trajectory condition, how many RHS equations pass the numerical- and exact-equivalence tests introduced in Section 9.

Reading the upper and lower panels together is the main purpose of the figure. If trajectory prediction, structural recovery, numerical equivalence, and exact algebraic recovery always agreed, the additional evaluation layers would provide little information. These three examples instead show that the layers can separate substantially, motivating the multi-level evaluation framework used throughout this benchmark.

![Predictive and structural comparison of three illustrative case studies](figures/06_case_studies.svg)

*Figure 6. Model-level comparison of three deliberately selected illustrative cases across all four training-set sizes. The upper panels show held-out macro-NRMSE on a logarithmic scale and report held-out integration success. The lower panels show dependency F1, numerator monomial F1, interaction F1, and rational-structure accuracy; annotations report numerical and exact equivalence of the right-hand-side (RHS) equations for the 32-trajectory condition. The cases were selected to illustrate contrasting relationships among evaluation criteria, not to estimate the frequency of these patterns in the benchmark.*

### BIOMD0000000156

BIOMD0000000156 provides the clearest example of strong predictive and mechanistic recovery without exact algebraic identity. This three-state model has dependency F1, numerator monomial F1, interaction F1, and rational-structure accuracy equal to 1 at every training-set size. Its held-out macro-NRMSE ranges from approximately $3.65 \times 10^{-6}$ to $3.78 \times 10^{-6}$, indicating extremely close agreement between the inferred and reference trajectories.

At 32 training trajectories, all three inferred RHSs also pass the numerical-equivalence diagnostic, but none passes the exact algebraic-equivalence test. The inferred equation for $x$ illustrates why. The reference RHS is

$$
\frac{dx}{dt} = -3.7xy + 2.0x,
$$

whereas PySR selects

$$
\frac{dx}{dt} = -3.70000024213422xy + 2.00000014281662x.
$$

The inferred equation therefore contains the same state variables, monomials, and multiplicative interaction as the reference equation, while its fitted coefficients differ only slightly from the reference values. Under the deliberately strict exact-equivalence criterion, however, these coefficient differences prevent the symbolic difference between the two expressions from simplifying to zero. BIOMD0000000156 therefore demonstrates that excellent held-out prediction, complete recovery of the structural components measured here, and numerical RHS agreement can all occur without exact symbolic equality.

### BIOMD0000000319

BIOMD0000000319 illustrates a different failure mode. Dependency F1 is 1 at all four training-set sizes, meaning that the inferred equations identify the correct dynamic state variables. However, numerator monomial F1 ranges only from approximately 0.30 to 0.48, interaction F1 ranges from approximately 0.67 to 1.00, and rational-structure accuracy is either 0 or 1/3. None of the three RHSs passes the numerical- or exact-equivalence test at any training-set size.

Despite these structural discrepancies, all eight held-out trajectories integrate successfully in every training condition, with held-out macro-NRMSE ranging from approximately 0.0562 to 0.0737. This case shows why dependency recovery alone should not be interpreted as equation recovery: PySR can identify which state variables participate while still reconstructing the mathematical terms and their organization incorrectly.

### BIOMD0000000728

BIOMD0000000728 demonstrates why successful numerical integration is not itself evidence of predictive or mechanistic accuracy. All eight held-out trajectories integrate successfully at every training-set size, yet held-out macro-NRMSE ranges from approximately 228 to 1,190. These errors indicate very poor agreement with the reference trajectories despite the absence of integration failures.

The structural metrics are also consistently weak across all four training-set sizes: dependency F1 is 1/3, numerator monomial F1 is 1/6, interaction F1 is 1/2, and rational-structure accuracy is 1/2. Neither of the model's two RHSs passes the numerical- or exact-equivalence diagnostic at any training-set size.

Together, these results make the distinction between numerical robustness and scientific accuracy explicit. An inferred ODE system may remain mathematically integrable over all held-out trajectories while reproducing neither the reference dynamics nor the underlying equation structure.

## 15. Discussion

The central result of this benchmark is that reproducing unseen dynamics and recovering the generating equations are related but distinct objectives. Many inferred systems integrate successfully and achieve useful held-out trajectory agreement, while exact algebraic recovery remains rare. Across the structurally compatible cohort, state-dependency recovery is also consistently stronger than recovery of numerator monomials and multiplicative interactions. Together with the case studies, these results show that symbolic-regression performance should not be summarized by a single notion of equation recovery.

Mechanistic recovery is better interpreted as a hierarchy of increasingly demanding comparisons. An inferred RHS may identify the correct participating state variables while combining them incorrectly, or it may recover the correct terms and interactions with coefficients that differ slightly from the reference equation. BIOMD0000000156 provides a concrete example of the latter: the inferred system closely reproduces held-out dynamics and recovers all evaluated structural components, and its three RHSs pass the numerical-equivalence diagnostic at 32 training trajectories, yet none is exactly algebraically equivalent. Conversely, BIOMD0000000319 and BIOMD0000000728 demonstrate that successful integration or correct state dependencies alone do not establish functional-form recovery.

Increasing the number of training trajectories does not produce a uniformly monotonic improvement under a fixed symbolic-regression search configuration. The 16-trajectory condition has the strongest aggregate predictive and several structural summaries among the four tested training-set sizes, but paired model-level comparisons show substantial heterogeneity, and several metrics decline again at 32 trajectories. These observations should therefore be interpreted as sensitivity results for this fixed benchmark configuration rather than evidence that 16 trajectories are generally optimal. More observations can provide broader dynamical coverage, but the benefit also depends on the search budget, candidate space, model structure, and selection procedure.

The benchmark also demonstrates why evaluation stages require distinct names and denominators. Benchmark readiness, workflow completion, numerical integration, held-out predictive accuracy, structural similarity, numerical RHS equivalence, and exact algebraic identity answer different questions. A model that fails because of a resource limit should not be classified as a scientific recovery failure, while a system that integrates successfully should not automatically be classified as an accurate prediction. Similarly, equation-level mechanistic metrics and model-level summaries must state their units of analysis so that models with more state equations do not receive unintended additional weight.

Several engineering decisions are therefore part of the scientific validity of the benchmark rather than merely implementation details. The improved SBML resolver showed that assuming a single canonical filename can impose an unintended selection effect even when BioModels identifiers themselves are standardized. Explicit file discovery, ambiguity rejection, and provenance records make cohort construction auditable. Likewise, materializing held-out trajectories only after the selected system is frozen creates an architectural barrier against test-data leakage rather than relying only on procedural convention.

Overall, the results support a cautious interpretation of symbolic regression for biological ODE reconstruction. PySR can recover highly predictive and, in some cases, strongly structure-preserving equations from simulated biological dynamics, but predictive agreement alone is insufficient evidence that the original mechanism has been identified. Reporting predictive, structural, numerical, and exact-algebraic outcomes separately provides a more informative picture of what has and has not been recovered.

## 16. Software engineering and reproducibility

A benchmark of symbolic-regression methods is not reproducible only because the inference code is available. It must also be possible to determine which biological models were analyzed, which simulation and search settings were used, which stages completed, and which artifacts belong to the same experimental run. The software developed for this project therefore treats provenance and execution state as part of the scientific output.

### Run manifests and provenance

Each benchmark run is represented by a manifest that records the requested model cohort, frozen experimental settings, SBML content hashes, simulation metadata, stage status, and the identities of selected equation systems. These records provide a link between an aggregate result in the final analysis and the model-specific computations that produced it.

Content hashes are particularly important because filenames alone do not establish that two runs used identical biological inputs. Recording the SBML hash makes changes to model content detectable even if the external model identifier or local filename remains unchanged.

### Safe resumption and concurrent execution

Symbolic-regression searches can be computationally expensive, and large benchmark runs may be interrupted by time limits, memory limits, machine restarts, or individual worker failures. The workflow therefore supports resumption from previously generated artifacts rather than requiring every model to restart from the beginning.

Before an existing artifact is reused, checkpoint information and artifact hashes are validated against the current run context. This reduces the risk of silently combining results produced with different inputs or experimental settings. Cross-run model/variant locks also prevent two concurrent processes from unintentionally performing and writing the same work.

Resource telemetry is recorded separately from scientific results. This distinction was important in the Expanded-100 benchmark: an incomplete model caused by a timeout, memory limit, or unresolved worker exit is an execution outcome, not evidence that symbolic regression scientifically failed to recover that model.

### Protecting the held-out evaluation

The held-out test set is intentionally separated from model selection. Candidate generation and system selection operate without access to the held-out trajectories. Test execution requires an already frozen selected-system artifact, validates that artifact, and only then constructs the held-out data used for final predictive evaluation.

This design turns the train/validation/test boundary into a software constraint rather than relying only on researcher convention. It also makes the boundary auditable: the selected equation system exists before the held-out trajectories used to evaluate it are materialized.

### Analysis artifacts

The final analysis joins workflow status, validation performance, held-out performance, and mechanistic-recovery results at the model and training-set-size level. Separate inventory and error records preserve why a result is missing, unavailable, or not mechanistically comparable. This is necessary because different denominators arise for different reasons: some models do not complete execution, while others complete prediction but cannot enter a particular structural comparison.

These machine-readable intermediate artifacts also make it possible to trace aggregate figures and tables back to individual models rather than treating the final report as the only record of the experiment.

### What reproducibility means here

In this project, reproducibility means that the benchmark inputs, configuration, software environment, provenance, execution status, and stage boundaries are explicitly recorded, and that deterministic components can be rerun from those records. The Docker workflow described in the next section further packages the software dependencies and benchmark inputs needed to reproduce the experiment on another machine.

This does not imply that every symbolic-search or floating-point result will be bitwise identical across all hardware platforms or future dependency stacks. Numerical libraries, processor architectures, and stochastic or implementation-dependent behavior can affect low-level results. The intended reproducibility target is therefore the documented scientific procedure and its numerical conclusions, rather than universal byte-for-byte identity of every generated artifact.

## 17. Docker and reproducible execution

The benchmark can also be executed through Docker so that a user does not need to reproduce the complete Python, Julia, PySR, and system-level environment manually. The purpose of the container is to package the software environment and benchmark inputs together while keeping generated results outside the image.

### What is included in the image

The Docker image contains the benchmark source code, the locked Python environment, an initialized PySR/Julia environment, and the packaged Expanded-100 SBML and simulation-metadata inputs. Because these biological inputs are already included, runtime execution does not require downloading models from BioModels or retrieving simulation metadata from an external service.

The image is therefore an execution environment rather than an archive of benchmark results. Generated results and writable runtime state are placed under `/output`, which is intended to be connected to a directory on the host computer.

### Host-mounted output

The Docker workflow uses a bind mount to connect a directory on the host computer to `/output` inside the container. Scientific results, run manifests, logs, telemetry, and mutable caches are stored below this boundary and therefore persist on the host when the container is removed. The host directory must already exist and be writable by container user ID 10001, or provide equivalent permissions through an ACL.

This separation means that the Docker image itself does not contain previously generated benchmark results. The same image can be reused while each user chooses where the resulting artifacts are stored.

### Portability and environment constraints

The container runs as non-root user ID 10001 and targets `linux/amd64`. This architecture is currently required because the frozen dependency lock uses `libroadrunner==2.8.0`, for which no Linux ARM64 wheel is available in the benchmark environment. A separate GPU execution path is not established by the current Docker workflow.

The Python base image and major build inputs are pinned by image digest or explicit version, and Python dependencies are installed from the project's lock file. Building the image still requires network access to obtain those dependencies. Runtime execution, however, does not require a live BioModels download because the benchmark inputs are already packaged in the image.

The Julia runtime is initialized as part of the image build, but its exact version is not completely frozen by the current Dockerfile. Docker therefore substantially reduces environmental variation without guaranteeing byte-for-byte identical outputs across every machine. The reproducibility target is the same packaged benchmark procedure, dependencies, inputs, and numerical workflow.

### Distribution

No public container-registry image is currently documented as part of this work product. The reproducible artifact is therefore the Dockerfile together with the bundled benchmark inputs and build instructions in the repository. The repository's [Docker deployment guide](https://github.com/nprzrosas/pysr-biomodels/blob/main/DOCKER.md) provides the verified build command, input validation, smoke tests, resource controls, and Expanded-100 execution procedure. A user builds the image locally and runs it with a writable host directory mounted at `/output`.

![Docker build inputs, built container image, runtime, and output boundary](figures/07_docker_reproducibility_architecture.svg)

*Figure 7. Docker reproducibility architecture. Source code, locked Python dependencies, the PySR/Julia environment, and the Expanded-100 benchmark inputs are packaged during image construction. At runtime, generated artifacts cross the container boundary through the host-mounted `/output` directory. No live BioModels download is required during benchmark execution. The container reduces environmental variation but does not claim bitwise reproducibility.*

## 18. Limitations

The benchmark was designed to isolate equation reconstruction under a controlled simulation protocol rather than to reproduce every difficulty encountered with experimental biological data. Its results should therefore be interpreted within the following methodological, model-scope, computational, and recovery-analysis limitations.

### Benchmark design

- The primary benchmark supplies exact, noiseless derivatives obtained directly from the SBML simulator. It therefore evaluates symbolic regression under an oracle-derivative setting and does not measure the additional difficulty introduced by estimating derivatives from sampled trajectories, measurement noise, or sparse observations.
- The reported Expanded-100 experiment uses one deterministic PySR seed and one fixed PySR configuration with a fixed search budget. Hyperparameter sensitivity and variability across independent symbolic-regression searches were not measured.
- Model parameters remain fixed at their nominal SBML values while initial states are perturbed. Candidate initial conditions are generated over a standardized 0.1--10-fold range relative to the nominal initial state.
- Each simulated trajectory contains 101 time points from time zero to one model-specific horizon. Different simulation horizons, temporal resolutions, or perturbation ranges could expose different dynamical regimes and change inference difficulty.
- The PySR feature space contains dynamic state variables only. Equations requiring explicit time, varying parameters, or other exogenous inputs cannot be represented by the current inference design.
- The benchmark uses a project-defined standardized simulation protocol rather than replaying all experimental conditions encoded in BioModels SED-ML files. The results therefore characterize recovery under this benchmark protocol, not recovery of every curated experiment associated with each BioModels entry.
- Model-specific simulation horizons were obtained from the project's curated-image time-extraction workflow. The retained acquisition records do not establish the extent of manual review applied to every automatically extracted horizon.

### Model scope

- The Expanded-100 cohort is a benchmark-compatible subset of the locally retained curated BioModels corpus, not a representative sample of all biological ODE models or all BioModels entries.
- The benchmark focuses on deterministic ODE systems that satisfy the current simulation and inference requirements. Events, unsupported assignment, rate, or algebraic rules, nonfinite or nonpositive nominal dynamic states, and other incompatible constructs can prevent a model from entering the benchmark.
- Earlier corpus filtering also excluded models with variable-parameter behavior that was outside the standardized inference protocol.
- Explicit-time models can still be evaluated for predictive behavior when execution is supported, but their reference equations are not directly comparable to a symbolic-regression feature space containing only dynamic states.
- These exclusions indicate incompatibility with the current benchmark protocol. They do not imply that the excluded models are scientifically invalid or intrinsically unsuitable for symbolic regression under a different formulation.

### Computational limitations

- Of the 100 benchmark-ready models launched in the Expanded-100 run, 75 completed the full workflow through held-out evaluation. Twelve incomplete runs were timeout-censored, two were memory-censored, and 11 exited during candidate generation without enough retained evidence for a more specific causal classification.
- Resource censoring is therefore part of the observed execution outcome. Aggregate scientific recovery results should not reinterpret these incomplete runs as demonstrated failures of equation reconstruction.
- The deterministic coordinate selector uses a fixed coordinate-search procedure and is not guaranteed to find the globally optimal combination of candidate RHS equations.
- The Docker workflow currently targets `linux/amd64` and a CPU-oriented execution path. The frozen `libroadrunner==2.8.0` dependency prevents the current image from supporting Linux ARM64 without changing the dependency environment.
- The expanded-run manifest does not separately record the imported PySR package version, although the project configuration used for the benchmark resolves PySR 1.5.9.
- Docker reduces variation in the execution environment but does not guarantee bitwise-identical symbolic-search or floating-point results across hardware and software environments.

### Recovery-analysis limitations

- Mechanistic ground-truth extraction failed for six workflow-complete models, making 24 model/training-size conditions unavailable for the structural recovery analysis.
- Three additional completed explicit-time models have a feature-space mismatch with the state-only inference representation. Their 12 conditions are classified separately rather than included in the primary mechanistic averages.
- The component-level metrics evaluate selected mathematical properties of each RHS: state dependencies, numerator monomials, multiplicative interactions, and rational state dependence. They do not constitute a complete test of biological mechanism identity.
- SymPy-based exact equivalence is a strict symbolic identity test and can be affected by expression parsing, canonicalization, or simplification limits. One otherwise eligible RHS evaluation timed out during this analysis.
- Numerical RHS equivalence is a diagnostic rather than a proof of identity. It evaluates the inferred and reference RHSs at 32 positive synthetic state-space points drawn directly over the fixed numerical range used by the analysis; these points are not multipliers of each model's nominal initial state. Agreement at those sampled points cannot establish global mathematical equivalence or guarantee accurate trajectories elsewhere in state space.
- Neither predictive accuracy nor the structural metrics establish biological parameter identifiability. Multiple equations can generate similar trajectories over the sampled domain, and the benchmark does not perform a formal structural- or practical-identifiability analysis.

These limitations narrow the claims that can be made from the benchmark but do not invalidate the comparisons within its controlled setting. The strongest conclusions concern the tested Expanded-100 protocol: under exact derivatives, a fixed PySR search configuration, and the specified train/validation/held-out design, predictive recovery is substantially easier than strict equation recovery. Generalization to noisy experimental measurements, different model classes, alternative symbolic-regression configurations, or broader biological mechanisms remains an empirical question.

## 19. Future work

The present benchmark establishes a controlled baseline for symbolic reconstruction from exact simulated derivatives. The following directions are proposed extensions rather than completed results.

### From oracle derivatives to realistic observations

A central next step is to replace exact simulator derivatives with derivatives estimated from sampled trajectories. This would make it possible to quantify how much recovery deteriorates when derivative estimation becomes part of the inference problem. Controlled measurement noise, temporal sparsity, alternative simulation horizons, and different initial-condition perturbation ranges could then be introduced systematically rather than changing several sources of difficulty at once.

This progression would help separate errors caused by symbolic search from errors introduced before symbolic regression begins. It would also move the benchmark closer to experimental biological settings, where exact derivatives are generally unavailable.

### Robustness of symbolic-regression recovery

The current results characterize one fixed PySR configuration and deterministic seed. Future experiments should repeat the benchmark across multiple seeds and systematically vary iteration budgets, operator sets, maximum expression size, parsimony settings, and candidate-pool size.

Such experiments could determine whether the observed gap between predictive and mechanistic recovery is robust to search configuration and whether additional computation improves equation recovery or mainly produces alternative predictive expressions. The same train/validation/held-out framework could also be used to compare PySR with additional symbolic-regression methods while preserving common data and evaluation controls.

### Broader biological model scope

The current state-only representation could be extended to include explicit time, parameters, or other exogenous variables when these quantities are required by the generating equations. Mechanistic ground-truth extraction could likewise be expanded to support a broader range of SBML rules and constructs.

Future corpus work should also improve simulation-duration curation and preserve explicit model-level review provenance. Broader use of available SED-ML information could complement the standardized benchmark by testing recovery under additional biologically curated simulation conditions, while keeping those experiments distinct from the controlled task reported here.

### Scaling and selection

The Expanded-100 experiment also exposes computational questions that become increasingly important as model size and symbolic-search complexity grow. Future work should characterize runtime and memory scaling as functions of the number of states, candidate complexity, and training-set size; analyze the causes of resource censoring in greater detail; and compare the current deterministic coordinate selector with alternative system-level selection strategies.

Together, these extensions would test whether the conclusions of the present benchmark persist when the inference problem becomes noisier, less constrained, computationally larger, and more representative of experimental biological data. They would also provide a path from the controlled oracle-derivative benchmark developed here toward a broader evaluation of symbolic regression for biological dynamical-system discovery.

## 20. Project evolution during GSoC

The original project proposal envisioned a broad evaluation of symbolic-regression approaches for biological network dynamics, including comparisons across methodological families and questions related to identifiability. During GSoC, the project was narrowed to establish a rigorous and reproducible PySR baseline before extending the comparison to additional methods.

This change in scope reflected an important methodological requirement that became clearer during development: meaningful comparisons between equation-discovery methods require a stable benchmark definition, reproducible biological inputs, controlled data splits, explicit recovery criteria, and auditable execution outcomes. Establishing those foundations therefore became the primary objective of the completed work.

The resulting benchmark separates training, validation, and held-out trajectories; evaluates nested training-set sizes; performs candidate selection at the level of the coupled dynamical system; distinguishes predictive performance from several levels of mechanistic recovery; and records computational censoring separately from scientific evaluation. Corpus provenance, simulation metadata, run manifests, model-specific results, and a containerized execution path were also incorporated so that the benchmark can be inspected and reproduced beyond the original development environment.

The final GSoC work product should therefore be viewed as a completed PySR baseline and an extensible benchmark foundation, rather than as a completed comparison of all symbolic-regression approaches proposed at the beginning of the project. This distinction is important: the broader methodological questions remain relevant, but they can now be investigated against a defined experimental framework rather than requiring each future method to establish its own data, evaluation, and execution conventions.

Future extensions can build on this foundation to evaluate estimated and noisy derivatives, alternative symbolic-regression algorithms, search variability, identifiability-aware analyses, and additional biological model classes while preserving the same separation between predictive, mechanistic, and computational outcomes.

## 21. Work product and repository navigation

Key work-product paths are:

- [Project overview and usage](https://github.com/nprzrosas/pysr-biomodels/blob/main/README.md)
- [Docker build and execution guide](https://github.com/nprzrosas/pysr-biomodels/blob/main/DOCKER.md)
- [Docker definition](https://github.com/nprzrosas/pysr-biomodels/blob/main/Dockerfile)
- [Bundled expanded-100 inputs](https://github.com/nprzrosas/pysr-biomodels/tree/main/docker/expanded_100_inputs)
- [Final combined results](https://github.com/nprzrosas/pysr-biomodels/blob/main/analysis/expanded100_master_results.csv)
- [Final scientific summary](https://github.com/nprzrosas/pysr-biomodels/blob/main/analysis/expanded100_final_scientific_summary.csv)
- [Case-study table](https://github.com/nprzrosas/pysr-biomodels/blob/main/analysis/expanded100_case_study_candidates.csv)
- [Expanded-run manifest](https://github.com/nprzrosas/pysr-biomodels/blob/main/analysis/expanded100_manifest.json)
- [Mechanistic-analysis outputs](https://github.com/nprzrosas/pysr-biomodels/tree/main/analysis/expanded100_mechanistic_recovery)
- [Held-out comparison outputs](https://github.com/nprzrosas/pysr-biomodels/tree/main/analysis/expanded100_test_comparisons)
- [Experimental design](https://github.com/nprzrosas/pysr-biomodels/blob/main/src/experimental_design_generic.py)
- [Simulation](https://github.com/nprzrosas/pysr-biomodels/blob/main/src/simulate_generic.py)
- [PySR candidate fitting](https://github.com/nprzrosas/pysr-biomodels/blob/main/src/run_pysr_generic.py)
- [Validation system selection](https://github.com/nprzrosas/pysr-biomodels/blob/main/src/select_generic_system.py)
- [Held-out runner](https://github.com/nprzrosas/pysr-biomodels/blob/main/src/run_generic_test.py)
- [Ground-truth extraction](https://github.com/nprzrosas/pysr-biomodels/blob/main/src/extract_mechanistic_ground_truth.py)
- [Algebraic evaluation](https://github.com/nprzrosas/pysr-biomodels/blob/main/src/evaluate_algebraic_recovery.py)
- [Corpus-selection audit](https://github.com/nprzrosas/pysr-biomodels/tree/main/results/model_selection_audit)
- [Expanded model inventory](https://github.com/nprzrosas/pysr-biomodels/tree/main/results/expanded_model_inventory)
- [Report claim audit](https://github.com/nprzrosas/pysr-biomodels/blThe analysis CSVs are the authoritative quantitative sources for the reported benchmark results. The expanded-run manifest records execution status and run-level provenance, while the source modules document the implemented simulation, inference, selection, and evaluation procedures. The report claim audit provides an additional trace between major quantitative statements in this work product and their supporting analysis artifacts.

les provide execution and methodology provenance.

## 22. Conclusions

PySR-BioModels establishes a controlled benchmark for asking whether symbolic regression can reconstruct biological ODE systems from simulated observations. Known SBML models provide the reference dynamics, while the benchmark separates equation fitting and system selection from held-out evaluation and measures recovery at predictive, structural, numerical, and exact algebraic levels.

Of the 100 benchmark-ready models launched in the Expanded-100 experiment, 75 completed the full workflow through held-out evaluation. Across these completed models, 284 of the 300 model/training-size conditions produced complete predictions for all eight held-out trajectories. Increasing the number of training trajectories improved aggregate predictive performance through 16 trajectories under the tested configuration, but the trend was not monotonic at 32. The lowest median held-out macro-NRMSE was 0.0243 at 16 training trajectories, where 73 of 75 completed models produced predictions for all held-out trajectories.

The mechanistic analyses reveal a sharper distinction between predictive success and recovery of the generating equations. At 16 training trajectories, mean state-dependency F1 reached 0.844, while more detailed monomial and interaction recovery remained lower. Across all eligible RHS comparisons, 140 of 991 inferred equations (14.1%) satisfied the numerical-equivalence criterion, whereas only 2 of 991 (0.2%) were exact algebraic matches. Both exact matches corresponded to trivial zero RHS equations; no nonzero RHS and no complete inferred system was recovered as an exact algebraic identity.

The model-level results illustrate why these distinctions matter. For BIOMD0000000156, PySR recovered the evaluated dependency, monomial, interaction, and rational structure across all training sizes and achieved held-out macro-NRMSE on the order of $10^{-6}$. At 32 training trajectories, all three inferred RHS equations also passed the numerical-equivalence test, but none was an exact algebraic match because of small coefficient differences. Conversely, BIOMD0000000319 retained correct state dependencies while missing important details of functional form, and BIOMD0000000728 demonstrated that successful numerical integration alone does not imply useful predictive or mechanistic recovery.

The central conclusion is therefore that, under the controlled exact-derivative protocol studied here, reproducing unseen dynamics is substantially easier than reconstructing the exact generating equations. Trajectory accuracy, structural recovery, numerical RHS equivalence, and exact symbolic identity represent different levels of evidence and should not be treated as interchangeable measures of successful equation discovery.

The principal contribution of this GSoC project is consequently both scientific and infrastructural: a quantitative PySR baseline on known biological dynamical systems and a reproducible framework that connects curated model inputs, controlled simulation and data splits, system-level equation selection, held-out testing, mechanistic evaluation, execution provenance, and containerized deployment. This foundation makes it possible for future symbolic-regression methods and more realistic data conditions to be compared against the same explicitly defined recovery criteria.

## 23. Acknowledgements

I thank mentors Augustin Luna, Eran Agmon, Mark Kittisopikul, and Matthias König for their guidance, and the National Resource for Network Biology for supporting this work. I also acknowledge the BioModels maintainers and model contributors whose curated SBML resources make systematic benchmarking possible, and the developers of PySR, Julia, RoadRunner, libSBML, SciPy, SymPy, and the broader open-source scientific-computing ecosystem used by this project.

## 24. References

1. Brunton SL, Proctor JL, Kutz JN. **Discovering governing equations from data by sparse identification of nonlinear dynamical systems.** *Proceedings of the National Academy of Sciences*. 2016;113(15):3932-3937. doi:10.1073/pnas.1517384113.

2. Cranmer M. **Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.** *arXiv*. 2023. arXiv:2305.01582.

3. Ahmadi Daryakenari N, De Florio M, Shukla K, Karniadakis GE. **AI-Aristotle: A physics-informed framework for systems biology gray-box identification.** *PLOS Computational Biology*. 2024;20(3):e1011916. doi:10.1371/journal.pcbi.1011916.

4. de Rooij M, van Riel NAW, O'Donovan SD. **Conditional universal differential equations capture population dynamics and interindividual variation in C-peptide production.** *npj Systems Biology and Applications*. 2025;11:84. doi:10.1038/s41540-025-00570-6.

5. Grigorian G, George SV, Arridge S. **Learning governing equations of unobserved states in dynamical systems.** *Physica D: Nonlinear Phenomena*. 2025;472:134499. doi:10.1016/j.physd.2024.134499.

6. Brum BR, Lober L, Previdelli I, Rodrigues FA. **Discovering equations from data: symbolic regression in dynamical systems.** *arXiv*. 2025. arXiv:2508.20257.

7. Basiri MA, Nicholson C. **Discovery of Nonlinear Dynamics with Automated Basis Function Generation.** *arXiv*. 2026. arXiv:2605.09696.

8. de Pomereu T, Fröhlich F. **Symbolic regression enables coarse-grained model discovery of intracellular signalling dynamics.** *bioRxiv*. 2026. doi:10.64898/2026.08.20.745973.

9. Malik-Sheriff RS, Glont M, Nguyen TVN, et al. **BioModels—15 years of sharing computational models in life science.** *Nucleic Acids Research*. 2020;48(D1):D407-D415. doi:10.1093/nar/gkz1055.

10. Hucka M, Bergmann FT, Dräger A, et al. **The Systems Biology Markup Language (SBML): Language Specification for Level 3 Version 2 Core.** *Journal of Integrative Bioinformatics*. 2018;15(1):20170081. doi:10.1515/jib-2017-0081.

11. Somogyi ET, Bouteiller JM, Glazier JA, et al. **libRoadRunner: a high performance SBML simulation and analysis library.** *Bioinformatics*. 2015;31(20):3315-3321. doi:10.1093/bioinformatics/btv363.

12. Mangan NM, Brunton SL, Proctor JL, Kutz JN. **Inferring Biological Networks by Sparse Identification of Nonlinear Dynamics.** *IEEE Transactions on Molecular, Biological and Multi-Scale Communications*. 2016;2(1):52-63. doi:10.1109/TMBMC.2016.2633265.

13. Pantazis Y, Tsamardinos I. **A unified approach for sparse dynamical system inference from temporal measurements.** *Bioinformatics*. 2019;35(18):3387-3396. doi:10.1093/bioinformatics/btz065.

14. La Cava W, Burlacu B, Virgolin M, et al. **Contemporary Symbolic Regression Methods and their Relative Performance.** *Advances in Neural Information Processing Systems*. 2021;2021(DB1):1-16.

15. de Franca FO, Virgolin M, Kommenda M, et al. **SRBench++: Principled Benchmarking of Symbolic Regression With Domain-Expert Interpretation.** *IEEE Transactions on Evolutionary Computation*. 2025;29(4):1127-1134. doi:10.1109/TEVC.2024.3423681.
