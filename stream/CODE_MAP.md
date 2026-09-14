# CODE_MAP.md — Stream Pipeline (MSc Thesis: Learning in Stream)

Generated 2026-04-26 from `src/main/java/`.

## E) Build configuration (`pom.xml`)

- **groupId/artifactId:** `thesis:stream-pipeline:1.0-SNAPSHOT`
- **Java target:** 17 (source + target)
- **Dependencies**
  - `nz.ac.waikato.cms.moa:moa:2024.07.0` — MOA framework (streams, ADWIN, HDDM, HoeffdingTree, ARF, SRP, generators)
  - `org.apache.commons:commons-math3:3.6.1` — KS test, χ² / F distributions, Wilcoxon
  - `com.fasterxml.jackson.core:jackson-databind:2.17.2` — JSON config loading
  - `org.projectlombok:lombok:1.18.30` (provided) — `@Getter` annotations
- **No JUnit / TestNG dependency** — the project ships its own ad-hoc smoke-test harness as classes with `main()`.

## F) Configuration files

- `src/main/java/thesis/experiments/master_experiments.json` — the only JSON config in the
  source tree (lives next to `UnifiedStreamExperimentRunner.java`, not in `resources/`). It
  defines all five blocks `E1`…`E5` in one file: global settings (`warmup`, `window_size`,
  `ram_sample_every`, `num_threads`, `seeds`, `default_max_instances`) plus, per block, the
  variant list and the dataset list. This replaced the five per-block configs
  (`E1_baselines.json`, `e3_da_srp.json`, …), none of which exist any more.
- No `src/main/resources/`, no YAML, no `.properties`.

## D) Tests

- **No `src/test/`.** Tests are co-located runnable classes (`*SmokeTest.java`) inside each
  package, each with its own `main()` and counters `passed`/`failed`. Listed under each package
  below.
- **Run them all with `bash stream/run_smoke_tests.sh`** — it compiles the project and executes
  every `*SmokeTest` main, printing a combined total and exiting non-zero on any failure.
  Current state: **19 classes, 276 assertions, all passing.** Three of them
  (`SyntheticStreamSmokeTest`, `EvaluationSmokeTest`, `SelectorStrategiesSmokeTest`) print
  diagnostics rather than a `RESULT:` line and so contribute 0 to the count.

---

## A) Architecture interfaces

| Interface | Package | Methods |
|---|---|---|
| `DriftDetector` | `thesis.detection` | `update(double)`, `isChangeDetected()`, `isWarningDetected()`, `getEstimation()`, `reset()`, default `name()` |
| `FeatureSelector` | `thesis.selection` | `initialize(double[][], int[])`, `update(double[], int, boolean, Set<Integer>)`, `getSelectedFeatures()`, `getCurrentSelection()`, `filterInstance(double[])`, `getNumFeatures()`, `getK()`, `isInitialized()`, default `name()` |
| `FilterRanker` | `thesis.selection` | `update(int[], int)`, `getFeatureScores()`, `selectTopK(int)`, `reset()`, `getNumFeatures()`, default `name()` |
| `ModelWrapper` | `thesis.models` | `predictProba(Instance)`, `predict(Instance)`, `train(Instance, int)`, `train(Instance, int, boolean, Set<Integer>)`, `getSelector()`, `getCurrentSelection()`, `reset()`, `name()` |

---

## B) Inheritance / implementation tree

```
DriftDetector  (interface)
├── ADWINChangeDetector              [wraps moa ADWIN]
└── HDDMChangeDetector               [wraps moa HDDM_A_Test / HDDM_W_Test]

(no interface) PerFeatureKSWIN  → composes KSWINSingleFeature[]
(no interface) TwoLevelDriftDetector → composes DriftDetector + PerFeatureKSWIN

FilterRanker  (interface)
└── AbstractFrequencyRanker (abstract)
    └── InformationGainRanker

FeatureSelector  (interface)
├── StaticFeatureSelector
├── PeriodicSelector
├── AlarmTriggeredSelector
└── DriftAwareSelector

ModelWrapper  (interface)
├── MajorityClassWrapper
├── NoChangeWrapper
├── HoeffdingTreeWrapper             [wraps moa HoeffdingTree]
├── ARFWrapper                       [wraps moa AdaptiveRandomForest]
├── SRPWrapper                       [wraps moa StreamingRandomPatches]
├── NativeDriftAwareSRP              [own ensemble of ARFHoeffdingTree, explicit subspaces]
└── DAARFWrapper                     [own ensemble, per-tree ADWIN + background learners]

moa.options.AbstractOptionHandler  +  moa.streams.InstanceStream
└── SyntheticStreamFactory.NoiseAugmentedStream  (private inner)
```

---

## C) Entry points (classes with `main()`)

| Class | Purpose |
|---|---|
| `thesis.experiments.UnifiedStreamExperimentRunner` | **The production runner.** Loads `master_experiments.json`, expands `blocks × datasets × variants × seeds` into a flat WorkItem list, runs it on a fixed thread pool, and writes per-block CSVs + `master_summary.csv` + `runs_raw.csv` + per-block `stat_tests/`. Replaces the five per-block runners (`E1Baselines`, `E2AdaptiveFS`, `E3DASRP`, `E4DriftAnalysis`, `E5Detectors`) that this document used to list; none of them exist any more, and neither do the per-block JSON configs. |
| `thesis.models.DAARFDiversityProbe` | Diagnostic — measures ensemble diversity inside DA-ARF. |
| `thesis.pipeline.ArffSanityCheck` | Diagnostic — prints attributes + class distribution for the three real ARFFs. |
| `thesis.pipeline.SyntheticStreamSmokeTest` | Prints per-stream summary stats for the 5 synthetic generators. |
| `thesis.detection.DetectionSmokeTest` | ADWIN / KSWINSingle / PerFeatureKSWIN unit-style tests. |
| `thesis.selection.SelectionSmokeTest` | Rankers + StaticFeatureSelector. |
| `thesis.selection.PeriodicSelectorSmokeTest` | PeriodicSelector. |
| `thesis.selection.AlarmTriggeredSelectorSmokeTest` | AlarmTriggeredSelector. |
| `thesis.selection.DriftAwareSelectorSmokeTest` | DriftAwareSelector. |
| `thesis.selection.SelectorStrategiesSmokeTest` | Side-by-side comparison of the four selectors. |
| `thesis.discretization.DiscretizationSmokeTest` | PiD layers. |
| `thesis.models.WrapperSelectionSmokeTest` | FeatureSpace, FilteredHeaderBuilder, HT/ARF/SRP wrappers and how they follow selection changes. |
| `thesis.models.DAARFRepairSmokeTest` | DA-ARF subspace repair / background-learner promotion. |
| `thesis.models.ImportanceSamplerSmokeTest` | FeatureImportance + WeightedSubspaceSampler. |
| `thesis.evaluation.EvaluationSmokeTest` | Per-instance metrics (κ, κ_per, recovery, RAM-h, stability). |
| `thesis.evaluation.StatisticalTestsSmokeTest` | Friedman / Nemenyi / Wilcoxon. |
| `thesis.evaluation.MetricsSmokeTest` | MetricsCollector wiring, RAM-Hours model-size sampling, window vs cumulative accuracy. |
| `thesis.experiments.RecoveryMetricSmokeTest` | Two-phase recovery-time metric and its four outcome categories. |
| `thesis.detection.TwoLevelDriftSmokeTest` | Level-1 global + Level-2 per-feature detection with BH-FDR. |

---

# Class catalog (per package)

## Package `thesis.detection`

### `DriftDetector` — interface
- File: `src/main/java/thesis/detection/DriftDetector.java`
- Methods: see table above.
- State: complete.

### `ADWINChangeDetector implements DriftDetector`
- File: `src/main/java/thesis/detection/ADWINChangeDetector.java`
- Wraps `moa.classifiers.core.driftdetection.ADWIN`. No warnings (always returns false).
- Fields: `final double delta`, `ADWIN adwin`, `boolean changeDetected`.
- Public methods: `ADWINChangeDetector()`, `ADWINChangeDetector(double delta)`, `update(double)`, `isChangeDetected()`, `isWarningDetected()`, `getEstimation()`, `reset()`, `getDelta()`, `getWindowLength()`, `name()`.
- Deps: MOA only.
- State: complete.

### `HDDMChangeDetector implements DriftDetector`
- File: `src/main/java/thesis/detection/HDDMChangeDetector.java`
- Wraps `HDDM_A_Test` or `HDDM_W_Test` per `Variant` enum. Reports both warning + drift.
- Inner enum: `Variant { A, W }`.
- Fields: `final Variant variant`, `final double alphaD`, `final double alphaW`, `final double lambda`, `AbstractChangeDetector detector`.
- Public methods: ctor `(Variant, double, double, double)`, factories `ofA`, `ofW`, `update`, `isChangeDetected`, `isWarningDetected`, `getEstimation`, `reset`, `name`. `@Getter` exposes all fields.
- State: complete.

### `KSWINSingleFeature` (no interface)
- File: `src/main/java/thesis/detection/KSWINSingleFeature.java`
- Two-window KS detector for a single feature (commons-math3 `KolmogorovSmirnovTest`); first window becomes the reference, sliding current window is compared.
- Fields: `final int windowSize`, `final double alpha`, `final KolmogorovSmirnovTest ks`, `double[] reference`, `Deque<Double> current`, `lastPValue/lastKsStatistic/lastDrift`.
- Public methods: ctor `(int, double)`, `update`, `testDrift`, `getPValue`, `getKSStatistic`, `isDrift`, `isReady`, `setReferenceWindow`, `promoteCurrentToReference`, `reset`. `@Getter` covers fields.
- State: complete.

### `PerFeatureKSWIN`
- File: `src/main/java/thesis/detection/PerFeatureKSWIN.java`
- Per-feature parallel KSWIN with Benjamini-Hochberg FDR control.
- Fields: `numFeatures`, `alpha`, `fdrQ`, `windowSize`, `KSWINSingleFeature[] detectors`, `double[] lastPValues`.
- Public methods: ctor `(int, double, int)`, ctor `(int, double, int, double)`, `update(double[])`, `getDriftingFeatures()` (BH-corrected), `getRawDriftingFeatures()`, `getLastPValues()`, `getKSStatistic(int)`, `isReady()`, `resetAll()`, `resetFeature(int)`. `@Getter` exposes fields.
- State: complete.

### `TwoLevelDriftDetector`
- File: `src/main/java/thesis/detection/TwoLevelDriftDetector.java`
- Hierarchical drift: Level-1 = global error stream (ADWIN/HDDM_A/HDDM_W); on alarm, Level-2 = `PerFeatureKSWIN` returns drifting feature indices and (optionally) is reset for them.
- Inner: `Level1Type` enum, `Config` static class (numFeatures, level1Type, level1Delta, level1AlphaW, level1Lambda, kswinAlpha, kswinWindowSize, bhQ, promoteReferenceOnDrift).
- Fields: `cfg`, `level1`, `level2`, `lastGlobalDrift/Warning`, `lastDriftingFeatures`, `updates`, `globalAlarms`.
- Public methods: ctor `(Config)`, `update(double, double[])`, `isGlobalDriftDetected/Warning`, `getDriftingFeatureIndices`, `getLastPValues`, `getLevel1Estimation`, `isLevel2Ready`, `getUpdateCount`, `getConfig`, `level1Name`, `reset`.
- State: complete.

### `DetectionSmokeTest` — runnable tests
14 tests covering ADWIN, KSWINSingle and PerFeatureKSWIN.

---

## Package `thesis.discretization` (PiD = Partition Incremental Discretization)

### `Layer1Histogram`
- File: `src/main/java/thesis/discretization/Layer1Histogram.java`
- Equal-width fine histogram (`b1` bins) with per-bin class counts. `fromWarmup` derives min/max with 5% margin.
- Fields: `final int b1`, `final int numClasses`, `min`, `max`, `width`, `final int[] binCounts`, `final int[][] classCounts`.
- Public methods: ctor `(int, int, double, double)`, static `fromWarmup`, `update(double, int)`, `bin(double)`. `@Getter` exposes all.
- State: complete (bin bounds fixed at warmup — does not adapt to range drift).

### `Layer2Merger` — utility (final, private ctor)
- File: `src/main/java/thesis/discretization/Layer2Merger.java`
- Greedy adjacent-bin merging: collapses `b1` Layer-1 bins into `b2` Layer-2 groups by minimising total-variation distance between their class-conditional Laplace-smoothed distributions, weighted by `min(|a|,|b|)+1`.
- Public methods: `static int[] merge(int[] binCounts, int[][] classCounts, int b2, int numClasses)`.
- State: complete.

### `FeatureDiscretizer`
- File: `src/main/java/thesis/discretization/FeatureDiscretizer.java`
- One feature: warmup buffer → build `Layer1Histogram` → maintain `l1ToL2` mapping refreshed via `Layer2Merger`.
- Fields: `b1`, `b2`, `numClasses`, `warmupN`, `ready`, `warmupBuffer`, `warmupClasses`, `warmupCount`, `Layer1Histogram l1`, `int[] l1ToL2`, `updatesSinceRecompute`, `totalUpdates`.
- Public methods: ctor `(int b1, int b2, int numClasses, int warmupN)`, `reset`, `update(double, int)`, `discretize(double)`, `recomputeLayer2()`, `l2Counts()`, `l2ClassCounts()`, `getLayer1`, `getL1ToL2Mapping`. `@Getter` covers fields.
- State: complete.

### `PiDDiscretizer`
- File: `src/main/java/thesis/discretization/PiDDiscretizer.java`
- One `FeatureDiscretizer` per feature; auto-recomputes Layer-2 every `recomputeEvery` instances.
- Defaults: `b1=100, b2=10, warmupN=500, recomputeEvery=1000`.
- Fields: `numFeatures`, `numClasses`, `b1`, `b2`, `warmupN`, `recomputeEvery`, `FeatureDiscretizer[] features`.
- Public methods: ctor `(int, int)`, ctor `(int, int, int b1, int b2, int warmupN, int recomputeEvery)`, `update(double[], int)`, `update(int, double, int)`, `discretize(int, double)`, `discretizeAll(double[])`, `recomputeLayer2()`, `recomputeLayer2(int)`, `reset()`, `resetFeature(int)`, `isReady()`, `isReady(int)`, `getL2Counts(int)`, `getL2ClassCounts(int)`, `getFeature(int)`. `@Getter` for fields.
- State: complete.

### `DiscretizationSmokeTest` — runnable tests.

---

## Package `thesis.selection`

### `FilterRanker` / `FeatureSelector` — interfaces (see Architecture).

### `AbstractFrequencyRanker implements FilterRanker` (abstract)
- File: `src/main/java/thesis/selection/AbstractFrequencyRanker.java`
- Maintains `int[F][B][C] joint` counts and `int[F][B] featureBinTotals`. Subclass supplies `score(featureIdx)`. `selectTopK` sorts by score desc.
- Fields: `numFeatures`, `numBins`, `numClasses`, `joint`, `featureBinTotals`.
- Public methods: ctor `(int, int, int)`, `update`, `getFeatureScores`, `selectTopK`, `reset`, `resetFeature(int)`, `getNumFeatures`. Protected: `score`, `featureTotal`, `classMarginal`. `@Getter` on fields.
- State: complete.

### `InformationGainRanker extends AbstractFrequencyRanker`
- IG = H(Y) − H(Y|X) in base 2. State: complete.

> `MutualInformationRanker` and `ChiSquaredRanker` used to sit here as sibling implementations.
> Both were deleted: no variant in `master_experiments.json` ever selected them, so they were
> dead code that still had to be kept compiling and tested. `InformationGainRanker` is now the
> only implementation, and the ranker is fixed for every selector (S1-S4) and every DA-* variant.

### `StaticFeatureSelector implements FeatureSelector`
- File: `src/main/java/thesis/selection/StaticFeatureSelector.java`
- "S1": one-shot selection from the warmup window; `update(...)` is a no-op.
- Fields: `numFeatures`, `numClasses`, `k`, `PiDDiscretizer discretizer`, `BiFunction<Integer,Integer,FilterRanker> rankerFactory`, `int[] selection`, `boolean initialized`.
- Public methods: ctor `(int, int)`, ctor `(int, int, int, PiDDiscretizer, BiFunction<…>)`, static `defaultK = ceil(sqrt(F))`, `initialize`, `update` (no-op), `getSelectedFeatures`, `getCurrentSelection`, `filterInstance`, `getNumFeatures`, `getK`, `isInitialized`, `name`.
- Deps: `thesis.discretization.PiDDiscretizer`.
- State: complete.

### `PeriodicSelector implements FeatureSelector`
- File: `src/main/java/thesis/selection/PeriodicSelector.java`
- "S2": every `periodN` instances, re-rank using a ring buffer of the last `periodN` discretised rows; swap up to `maxSwapsPerCycle = ceil(0.3·k)` features (subject to `minTenure`).
- Fields: + `periodN`, `minTenure`, `maxSwapsPerCycle`, `tenure[]`, `ringBins[][]`, `ringLabels[]`, `ringPos`, `ringCount`, `swapEvents`, `swappedFeatures`. `@Getter`.
- Deps: `PiDDiscretizer`, `StaticFeatureSelector` (default-K + default ranker).
- State: complete.

### `AlarmTriggeredSelector implements FeatureSelector`
- File: `src/main/java/thesis/selection/AlarmTriggeredSelector.java`
- "S3": only re-ranks after a drift alarm — collects `wPostDrift` post-drift instances into a fresh ranker, then commits a new top-k.
- Fields: `wPostDrift`, `ranker`, `selection`, `initialized`, `collecting`, `collected`, `reSelections`. `@Getter`.
- Deps: `PiDDiscretizer`, `StaticFeatureSelector.defaultK`, `InformationGainRanker` (default factory).
- State: complete.

### `DriftAwareSelector implements FeatureSelector`
- File: `src/main/java/thesis/selection/DriftAwareSelector.java`
- "S4": combines periodic re-selection with alarm-triggered swaps that target only the alarm's drifting features. Maintains both a long-window ring buffer and a per-alarm post-drift ranker.
- Adds counters: `periodicSwapEvents`, `alarmSwapEvents`, `swappedByAlarm`, `swappedByPeriodic`. Helpers: `startCollecting`, `whereSwap`, `periodicReSelect`, `commitSelection`, `pushRing`.
- State: complete.

### Smoke-test classes (5)
`SelectionSmokeTest`, `PeriodicSelectorSmokeTest`, `AlarmTriggeredSelectorSmokeTest`, `DriftAwareSelectorSmokeTest`, `SelectorStrategiesSmokeTest`.

---

## Package `thesis.models`

### `ModelWrapper` — interface (see Architecture).

### `FeatureSpace` (final)
- File: `src/main/java/thesis/models/FeatureSpace.java`
- Maps feature index (0..d−1, class excluded) → attribute index in MOA `InstancesHeader`. Provides `extractFeatures(Instance) → double[d]`.
- Methods: ctor `(InstancesHeader)`, `numFeatures`, `classIndex`, `attrIndexOf(int)`, `extractFeatures(Instance)`. `@Getter` on header.
- State: complete.

### `FilteredHeaderBuilder` (final, utility)
- File: `src/main/java/thesis/models/FilteredHeaderBuilder.java`
- Builds a reduced `InstancesHeader` (selected attrs + class) and projects an `Instance` onto it via `DenseInstance`.
- Methods: `static InstancesHeader build(FeatureSpace, int[], String)`, `static Instance filteredInstance(Instance, FeatureSpace, int[], InstancesHeader)`.
- State: complete.

### `MajorityClassWrapper implements ModelWrapper`
- File: `src/main/java/thesis/models/MajorityClassWrapper.java`
- Predicts argmax of running class counts. Selector held but not used for prediction.
- Fields: `selector`, `numClasses`, `counts[]`, `total`.
- State: complete (Level-1 baseline).

### `NoChangeWrapper implements ModelWrapper`
- File: `src/main/java/thesis/models/NoChangeWrapper.java`
- "Predict last seen label." Level-1 baseline.
- Fields: `selector`, `numClasses`, `lastLabel`.
- State: complete.

### `HoeffdingTreeWrapper implements ModelWrapper`
- File: `src/main/java/thesis/models/HoeffdingTreeWrapper.java`
- Wraps `moa.classifiers.trees.HoeffdingTree` with a filtered header and a cached selection. On selection change, optionally reset the tree.
- Defaults: `gracePeriod=200, splitConfidence=0.01, resetOnSelectionChange=false`.
- Fields: selector, space, gracePeriod, splitConfidence, resetOnSelectionChange, `HoeffdingTree tree`, `InstancesHeader reducedHeader`, `int[] cachedSelection`. `@Getter`.
- Methods: predict/predictProba/train + `rebuild`, `syncSelection`, `newTree`. `train(...)` also calls `selector.update(...)`.
- State: complete.

### `ARFWrapper implements ModelWrapper`
- File: `src/main/java/thesis/models/ARFWrapper.java`
- Wraps `moa.classifiers.meta.AdaptiveRandomForest`. Lambda is set via CLI option `'a'` (silently ignored if MOA's option layout differs).
- Defaults: `ensembleSize=10, lambda=6.0, resetOnSelectionChange=false`.
- State: complete.

### `SRPWrapper implements ModelWrapper`
- File: `src/main/java/thesis/models/SRPWrapper.java`
- Wraps `moa.classifiers.meta.StreamingRandomPatches`. Uses reflection to read the ensemble array and per-learner subspace fields (tries `ensemble/learners/baseLearners/classifiers` and `subSpaceIndexes/subspaceIndexes/…`).
- Defaults: `ensembleSize=10, lambda=6.0, resetOnSelectionChange=false`.
- Adds: `getSubspaceIndices(int)`, `getAllSubspaceIndices()`, `getActualEnsembleSize()`, `requireSubspaceField()`, `buildFilteredInstance(Instance)`, `getReducedHeader()`, `getFeatureSpace()`, `getSRP()`. `@Getter`.
- State: complete (depends on MOA's internal field naming — guards via `*FIELD_CANDIDATES`).

### `FeatureImportance`
- File: `src/main/java/thesis/models/FeatureImportance.java`
- Combines max-normalised MI scores (relevance) and `1/(KS+ε)` (stability) into normalised importance vector: `imp = w1·MI + w2·stab`.
- Defaults: `w1=0.7, w2=0.3, epsilon=1e-6, normalizeInputs=true`.
- Methods: ctor `(int)`, ctor `(int, double, double, double, boolean)`, `update(double[], double[])`, `updateMIOnly(double[])`, `updateKSOnly(double[])`, `getImportance/MIScores/KSStatistics`, `getNumFeatures/W1/W2/Updates`, `projectToReduced(int[])`.
- State: complete.

### `WeightedSubspaceSampler` (final, utility)
- File: `src/main/java/thesis/models/WeightedSubspaceSampler.java`
- Weighted reservoir sampling à la Efraimidis–Spirakis (key = `ln(u)/w`). Excludes a forbidden index set; falls back to uniform if too few weighted candidates.
- Method: `static int[] sample(double[] weights, int size, Random rng, Set<Integer> exclude)`.
- State: complete.

### `DriftActionSummary` (final)
- File: `src/main/java/thesis/models/DriftActionSummary.java`
- Records per-learner outcome of `NativeDriftAwareSRP.handleDrift` / `DAARFWrapper`: `Action { KEEP, SURGICAL, FULL, NO_REPLACEMENT }` + overlap counts + subspace sizes + aggregates.
- State: complete.

### `NativeDriftAwareSRP implements ModelWrapper`
- File: `src/main/java/thesis/models/NativeDriftAwareSRP.java`
- **The DA-SRP implementation every variant runs.** Owns its ensemble explicitly: MOA
  `ARFHoeffdingTree` base learners, each with a fixed feature subspace (a "patch"), online
  bagging, and a per-learner ADWIN drift channel with a background learner. No reflection.
- Replaced `DriftAwareSRP`, which wrapped MOA `StreamingRandomPatches` and reached into its
  private per-learner subspace arrays via reflection. That class sat behind a `da_srp_native`
  flag no config ever set, and has been **deleted** together with the flag.
- On drift (`handleDrift`), per learner: overlap 0 → **KEEP**; overlap below `tau` → **SURGICAL**
  swap of the drifting picks; above → **FULL** rebuild (`WeightedSubspaceSampler` when
  `FeatureImportance` is present, uniform otherwise). When no acceptable replacement exists the
  learner is recorded as **NO_REPLACEMENT**.
- Component B: subspaces drawn importance-weighted (`importancePower`, `samplingBeta`).
  Component C: `predictProba` blends the plain vote toward a top-K importance-weighted
  correction (`correctionAlpha`, capped by `maxBlendAlpha`).
- The supplied `FeatureSelector` is kept only for interface compatibility — this model routes the
  FULL feature space. `getCurrentSelection()` returns the union of the ensemble's subspaces,
  which is what `RunDetailedRecorder` logs.
- Emits a `DriftEvent` (top-level class in `thesis.models`, formerly nested in `DriftAwareSRP`)
  to the runner's listener, carrying the `DriftActionSummary` with per-learner actions.
- State: complete.

### Smoke-tests
`WrapperSelectionSmokeTest` (FeatureSpace, FilteredHeaderBuilder, HT/ARF/SRP and how they follow selection changes), `ImportanceSamplerSmokeTest` (FeatureImportance + WeightedSubspaceSampler — rescued from the deleted `DriftAwareSRPSmokeTest`), `DAARFRepairSmokeTest` (DA-ARF subspace repair).

---

## Package `thesis.evaluation`

### `PrequentialAccuracy`
- File: `src/main/java/thesis/evaluation/PrequentialAccuracy.java`
- Sliding-window accuracy of size `windowSize`.
- Methods: ctor `(int)`, `update(int, int)`, `getAccuracy`, `getCount`, `reset`.
- State: complete.

### `CohenKappa`
- File: `src/main/java/thesis/evaluation/CohenKappa.java`
- Sliding-window confusion matrix → κ.
- Fields: `numClasses`, `windowSize`, `cm[][]`, `rowTotals[]`, `colTotals[]`, `Deque<int[]> window`, `total`, `correct`.
- Methods: ctor `(int, int)`, `update`, `getKappa`, `getAccuracy`, `getWindowCount`, `getWindowSize`, `reset`.
- State: complete.

### `TemporalKappa`
- File: `src/main/java/thesis/evaluation/TemporalKappa.java`
- κ vs. NoChange baseline (`yNc = prevTrue`) inside a sliding window.
- Methods: ctor `(int)`, `update`, `getKappaTemporal`, `getNoChangeAccuracy`, `reset`.
- State: complete.

### `RAMHours`
- File: `src/main/java/thesis/evaluation/RAMHours.java`
- Trapezoid integration of `usedBytes·hours`. `sampleFromRuntime()` reads JVM heap usage.
- Methods: `start`, `sample(long)`, `sampleFromRuntime`, `getRamHours`, `getPeakBytes`, `getPeakMB`, `getElapsedHours`, `getPeakRamHours`, `reset`.
- State: complete.

### `RecoveryTime`
- File: `src/main/java/thesis/evaluation/RecoveryTime.java`
- After a drift alarm, counts instances until current κ ≥ pre-drift κ − tolerance, capped by `maxRecoveryWindow`. Defaults: `tolerance=0.05, maxRecoveryWindow=10000`.
- Methods: ctor `()`, ctor `(double, int)`, `tick`, `onDriftAlarm(double)`, `update(double)`, getters, `reset`.
- State: complete.

### `FeatureStabilityRatio`
- File: `src/main/java/thesis/evaluation/FeatureStabilityRatio.java`
- Average Jaccard-like ratio `|prev ∩ curr| / |prev|` over consecutive selections.
- Methods: `update(int[])`, `getLastRatio`, `getAverageRatio`, `getUpdateCount`, `reset`.
- State: complete.

### `MetricsCollector`
- File: `src/main/java/thesis/evaluation/MetricsCollector.java`
- Aggregates: `CohenKappa`, `TemporalKappa`, `PrequentialAccuracy`, `RecoveryTime`, `RAMHours`, `FeatureStabilityRatio`. Defaults: `windowSize=1000, logEvery=1000`.
- Methods: ctor `(int)`, ctor `(int, int, int)`, `update(int, int, long)`, `onDriftAlarm()`, `onSelectionChanged(int[])`, `shouldLog`, `formatLogLine`, `snapshot()`, getters, plus public `Snapshot` POJO.
- State: complete.

### `FriedmanTest`
- File: `src/main/java/thesis/evaluation/FriedmanTest.java`
- Per-row ranking with tie averaging → χ² statistic + Iman-Davenport F. Uses commons-math3 `ChiSquaredDistribution`/`FDistribution` for p-values.
- Inner: `Result` with `numMethods, numDatasets, averageRanks, ranks, chiSquared, pValueChi, imanDavenport, pValueF, dfChi, dfF1, dfF2`, helper `rejectsNull(double)`.
- State: complete.

### `NemenyiPostHoc`
- File: `src/main/java/thesis/evaluation/NemenyiPostHoc.java`
- Built-in q-tables for α∈{0.05, 0.10}, k∈[2..20]. `CD = q · sqrt(k(k+1)/(6N))`.
- Inner: `Result` with `numMethods, numDatasets, alpha, qAlpha, criticalDifference, averageRanks, rankDifferences, significant, significantPairs`.
- State: complete (limited to those two α values and k≤20).

### `WilcoxonSignedRank`
- File: `src/main/java/thesis/evaluation/WilcoxonSignedRank.java`
- Paired test via commons-math3. Exact mode for n≤30.
- Inner: `Result { n, statistic, pValue, exact, wins, losses, ties; rejectsNull }`.
- State: complete.

### `StatisticalTests`
- File: `src/main/java/thesis/evaluation/StatisticalTests.java`
- Façade: `friedman`, `nemenyi`, `wilcoxon`, `runFull(matrix, datasetNames, methodNames)` → `Report`.
- Inner: `Report` with `exportCD(Path)` (writes 3 CSVs via `CDDiagramExporter`) and `summary()`.
- State: complete.

### `CDDiagramExporter`
- File: `src/main/java/thesis/evaluation/CDDiagramExporter.java`
- Writes `avg_ranks.csv`, `rank_matrix.csv`, `pairwise_significance.csv`. Note: it writes raw CSV — there is no actual diagram (PNG/SVG) generation, only the data tables for downstream plotting.
- State: data-export complete; visual rendering not implemented (likely intentional — done in Python).

### Smoke-tests
`EvaluationSmokeTest`, `StatisticalTestsSmokeTest`, `MetricsSmokeTest`.

---

## Package `thesis.pipeline`

> Shrank to three classes. `StreamPipeline`, `StreamMetrics`, `RecordingMetrics`,
> `ExperimentRunner` and `Shims` were **deleted**: the production path is
> `UnifiedStreamExperimentRunner`, which drives its own prequential loop and builds streams,
> models, selectors and detectors itself. The pipeline classes duplicated that orchestration
> without being called by it, and `Shims`' four factories were
> `UnsupportedOperationException` stubs.

### `SyntheticStreamFactory`
- Builds every synthetic stream used by the blocks: `createSEA`, `createMultiDriftSEA`,
  `createSTAGGER`, `createHyperplane`, `createRandomRBF`, `createCustomFeatureDrift`,
  `createLEDDrift`, plus `addNoiseFeatures` (wraps a stream in the private inner
  `NoiseAugmentedStream`) and `buildCyclicAbruptStream` (the `numDrifts+1`-segment cyclic
  concept switcher used for the Low/HiDyn pairs).

### `ArffSanityCheck` — runnable diagnostic
- Prints attributes and class distribution for the three real ARFFs.

### `SyntheticStreamSmokeTest` — runnable diagnostic
- Prints per-stream summary stats for the synthetic generators (no `RESULT:` line).

---

# Component coverage vs. plan

| Area | Implementation | State |
|---|---|---|
| **Detection** — Level-1 global + Level-2 per-feature | `ADWINChangeDetector`, `HDDMChangeDetector`, `KSWINSingleFeature`, `PerFeatureKSWIN`, `TwoLevelDriftDetector` (+ BH-FDR) | ✔ complete |
| **Discretization** — PiD two-layer | `Layer1Histogram` + `Layer2Merger` + `FeatureDiscretizer` + `PiDDiscretizer` | ✔ complete |
| **Selection** — S1–S4 | `StaticFeatureSelector`, `AlarmTriggeredSelector`, `PeriodicSelector`, `DriftAwareSelector`, `NoFeatureSelection`; ranker: `InformationGainRanker` | ✔ complete and wired in `buildSelector` |
| **Models** — baselines + ensembles + drift-aware | `MajorityClassWrapper`, `NoChangeWrapper`, `HoeffdingTreeWrapper`, `ARFWrapper`, `SRPWrapper`, `DriftAwareSRP` / `NativeDriftAwareSRP`, `DAARFWrapper` (+ `FeatureImportance`, `WeightedSubspaceSampler`, `DriftActionSummary`, `ModelSize`) | ✔ complete and wired in `buildModel` |
| **Evaluation** — metrics + statistical tests | Prequential Acc, Cohen κ, Temporal κ, `RecoveryTime`, `RAMHours` (model-size based), `FeatureStabilityRatio`, `MetricsCollector` + Friedman / Nemenyi / Wilcoxon / CD-export | ✔ complete |
| **Pipeline** — orchestration | — | **removed.** `StreamPipeline`/`StreamMetrics`/`RecordingMetrics` were bypassed by the runner's own prequential loop |
| **Experiments** | `UnifiedStreamExperimentRunner` + `RunDetailedRecorder` + `BlockStatisticalAnalysis`, driven by `master_experiments.json` | ✔ all five blocks, all selectors, both drift-aware families |
| **Generic JSON runner** | — | **removed.** `ExperimentRunner` + `Shims` were `UnsupportedOperationException` stubs the production path never touched |
| **Tests** | 17 runnable `*SmokeTest` classes with `main()`, **237 assertions, all passing**; no `src/test/`, no JUnit | functional but not a real test framework |
| **Configs** | One JSON next to `UnifiedStreamExperimentRunner.java`; no `resources/`, no YAML | OK |
| **Analysis / reporting** | Python package `analysis/` (`python -m analysis`) → 63 LaTeX tables + 87 figures | ✔ complete |

**What's missing or partial**

- `CDDiagramExporter` writes CSVs only — the diagrams themselves are drawn in `analysis/`.
- No real JUnit suite (only `main`-driven smoke tests with a custom `report(...)` helper), so
  nothing runs the tests automatically; they have to be invoked class by class. That is how two
  of them silently rotted: `WrapperSelectionSmokeTest` still asserted that a model wrapper
  forwards drift alarms to the selector (the runner took that job over), and
  `AlarmTriggeredSelectorSmokeTest` asserted that tie-breaking preserves a feature whose score
  had collapsed by a full `tieEpsilon` bucket. Both are fixed; a one-line script that runs every
  `*SmokeTest` main would have caught them at the time.
**Recently fixed (was: what's wrong)**

- **DA-* selection artefacts were degenerate.** `RunDetailedRecorder` logged
  `selector.getCurrentSelection()`, but the DA models ignore the selector and draw their own
  per-learner subspaces, so every DA run looked like a frozen S1: one `initial` row,
  `feature_stability_mean` 1.000, `selection_changes_mean` 0, `mean_selected_feature_count`
  = `ceil(sqrt(d))`. The runner now passes `model.getCurrentSelection()` (which the DA wrappers
  override to return the union of the ensemble's subspaces) and tags model-driven changes
  `subspace_change`.
- **The `initial` trigger was emitted twice per run.** `onInitialSelection` left
  `lastSelectionChangeInstance` at -1, so the first genuine re-selection was also labelled
  `initial` — 75 such rows across 40 `ARF+S2` runs in E3. Fixed for every block.

---

# Data flow

**Entry point** (production): `thesis.experiments.UnifiedStreamExperimentRunner.main` reads
`master_experiments.json`. Launch with `bash stream/run_experiments.sh` — it compiles with
JDK 17 and adds `-javaagent:sizeofag`, without which every RAM-Hours measurement is `NaN`.

The runner expands `blocks × datasets × variants × seeds` into a flat `WorkItem` list and
submits it to a fixed thread pool. Each `RunWorker` is self-contained:

1. **Stream construction** — `buildStream(...)` → `SyntheticStreamFactory.create*` (optionally
   wrapped in `addNoiseFeatures`) or `ArffFileStream`; `prepareForUse()` is called.
2. **Warm-up** — `warmupSize` (1500) instances are pulled into a window, fed to the ranking PiD,
   then `selector.initialize(window, labels)`. `FeatureImportance` is seeded from the full ranker.
3. **Selector / model / detector** built by `buildSelector`, `buildModel`, `buildDetector` from
   the variant spec; `MetricsCollector` gets `setModelSizeSupplier(model::modelByteSize)` so
   RAM-Hours measures the model, not the heap. `RunDetailedRecorder` is attached, and for
   DA-SRP a drift listener forwards `DriftActionSummary` per adaptation event.
4. **Per-instance loop** (`runPrequentialLoop`), for each `Instance x`:
   - `feats = space.extractFeatures(x)`, `yTrue = (int) x.classValue()`.
   - `updateFullFeatureRanker(feats, yTrue)` — ranking PiD + full-space ranker.
     **Outside the timed region**, so baselines are not charged for work only DA models consume.
   - *timed region starts*: `yHat = model.predict(x)` (own `predictNanos`).
   - `detector.update(err, feats)` — Level-1 sees the 0/1 error, Level-2 sees the raw feature
     vector. On alarm Level-2 returns BH-FDR-corrected drifting feature indices.
   - on alarm: `updateFeatureImportanceFromDetector()` — importance is refreshed from the
     ranker scores plus `1 − pValue` per feature.
   - `selector.update(feats, yTrue, alarm, drifting)` — called by the runner, **not** by the
     model wrapper.
   - `model.train(x, yTrue, alarm, drifting)`. `NativeDriftAwareSRP`/`DAARFWrapper` decide
     KEEP / SURGICAL / FULL per learner here.
   - *timed region ends*: `metrics.update(yTrue, yHat, predictNanos, stepNanos)`, so
     `throughput` covers the whole prequential step while `predict_latency_us` isolates
     inference. CSV buffering stays outside the timed region.
   - `recorder.onInstance(...)` buffers window metrics, alarms, selection changes; DA-ARF delta
     counters are polled and turned into adaptation events when they move.
5. **Per-run output** — the recorder is drained single-threaded after the pool finishes and
   written into the block's `windows.csv`, `drift_alarms.csv`, `feature_selections.csv`,
   `feature_importance.csv`, `recovery_time.csv`, `adaptation_events.csv`.
6. **Per-block aggregation** — one summary row per run in `E{n}_*.csv`, plus
   `BlockStatisticalAnalysis` writing Friedman / Nemenyi / Wilcoxon-Holm / CD CSVs into
   `stat_tests/`. Globally: `master_summary.csv` (means over seeds) and `runs_raw.csv`.
7. **Reporting** — `python -m analysis` turns those CSVs into `stream/results/tables/*.tex`
   and `stream/results/figures/**`.

In short: `Instance → FeatureSpace.extractFeatures → ranking PiD + full ranker → model.predict →
TwoLevelDriftDetector → (on alarm) FeatureImportance refresh → selector.update →
model.train (DA: per-learner KEEP/SURGICAL/FULL) → MetricsCollector + RunDetailedRecorder`.

> **Note.** Earlier versions of this document described `StreamPipeline` as the production
> path. It never was — the runner has always driven its own loop — and those classes have now
> been deleted along with `ExperimentRunner`/`Shims`.
