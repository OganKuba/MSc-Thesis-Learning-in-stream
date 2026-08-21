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

- `src/main/java/thesis/experiments/E1_baselines.json` — only JSON config in the source tree (lives next to `E1Baselines.java`, not in `resources/`).
- No `src/main/resources/`, no YAML, no `.properties`.

## D) Tests

- **No `src/test/`.** Tests are co-located runnable classes (`*SmokeTest.java`) inside each package, each with its own `main()` and counters `passed`/`failed`. Listed under each package below.

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
    ├── InformationGainRanker
    ├── MutualInformationRanker
    └── ChiSquaredRanker

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
└── DriftAwareSRP                    [composes SRPWrapper + FeatureImportance]

StreamMetrics
└── RecordingMetrics                 [adds CSV row writing + κ + κ_per + drift/recovery]

moa.options.AbstractOptionHandler  +  moa.streams.InstanceStream
└── SyntheticStreamFactory.NoiseAugmentedStream  (private inner)
```

---

## C) Entry points (classes with `main()`)

| Class | Purpose |
|---|---|
| `thesis.experiments.E1Baselines` | Production runner — loads `E1_baselines.json`, sweeps datasets × detectors × variants × seeds, writes `summary.csv` + `validation_level1.txt`. |
| `thesis.pipeline.ExperimentRunner` | Generic JSON-driven runner (writes per-run CSV). **Stub-bound** — depends on `DatasetFactory/ModelFactory/SelectorFactory/DetectorFactory` in `Shims.java` which all `throw UnsupportedOperationException`. |
| `thesis.pipeline.ArffSanityCheck` | Diagnostic — prints attributes + class distribution for the three real ARFFs. |
| `thesis.pipeline.SyntheticStreamSmokeTest` | Prints per-stream summary stats for the 5 synthetic generators. |
| `thesis.pipeline.PipelineSmokeTest` | Asserts pipeline + RecordingMetrics behaviour. |
| `thesis.detection.DetectionSmokeTest` | ADWIN / KSWINSingle / PerFeatureKSWIN unit-style tests. |
| `thesis.selection.SelectionSmokeTest` | Rankers + StaticFeatureSelector. |
| `thesis.selection.PeriodicSelectorSmokeTest` | PeriodicSelector. |
| `thesis.selection.AlarmTriggeredSelectorSmokeTest` | AlarmTriggeredSelector. |
| `thesis.selection.DriftAwareSelectorSmokeTest` | DriftAwareSelector. |
| `thesis.selection.SelectorStrategiesSmokeTest` | Side-by-side comparison of the four selectors. |
| `thesis.discretization.DiscretizationSmokeTest` | PiD layers. |
| `thesis.models.ModelsSmokeTest` | FeatureSpace, FilteredHeaderBuilder, HT/ARF/SRP wrappers. |
| `thesis.models.DriftAwareSRPSmokeTest` | DriftAwareSRP + FeatureImportance + WeightedSubspaceSampler. |
| `thesis.evaluation.EvaluationSmokeTest` | Per-instance metrics (κ, κ_per, recovery, RAM-h, stability). |
| `thesis.evaluation.StatsSmokeTest` | Friedman / Nemenyi / Wilcoxon. |

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

### `MutualInformationRanker extends AbstractFrequencyRanker`
- MI in nats from joint/marginal frequencies. State: complete.

### `ChiSquaredRanker extends AbstractFrequencyRanker`
- Pearson χ² over the contingency table. State: complete.

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
- Records per-learner outcome of `DriftAwareSRP.handleDrift`: `Action { KEEP, SURGICAL, FULL, NO_REPLACEMENT }` + overlap counts + subspace sizes + aggregates.
- State: complete.

### `DriftAwareSRP implements ModelWrapper`
- File: `src/main/java/thesis/models/DriftAwareSRP.java`
- Composes a `SRPWrapper`. On drift, for each learner:
  - compute overlap of its subspace with the drifting features (in reduced index space),
  - if overlap = 0 → **KEEP**,
  - else if overlap-fraction < `tau` → **SURGICAL** swap (replace drifting picks with best non-drifting candidates by score),
  - else → **FULL** rebuild (`generateSubspace` via `WeightedSubspaceSampler` if `FeatureImportance` is set, otherwise uniform) and `resetLearning()`.
- Predictions: importance-weighted vote (`predictProbaWeighted`) or fallback to plain `srp.predictProba` if no importance / no ensemble yet.
- Constants: `ENSEMBLE_FIELD_CANDIDATES`, `SUBSPACE_FIELD_CANDIDATES`, `CLASSIFIER_FIELD_CANDIDATES`.
- Fields: `srpWrapper`, `tau`, `rng`, `importance`, plus counters: `handleDriftCalls`, `totalKept/Surgical/Full/NoReplacement`, `refreshCalls`, `totalRefreshed`, `weightedPredictions`, `unweightedFallbacks`, `lastSummary`, `lastLearnerWeights`.
- Inner: `RefreshSummary { ensembleSize, refreshedCount }`.
- Public methods: ctor `(SRPWrapper)`, `(SRPWrapper, double, long)`, `(SRPWrapper, double, long, FeatureImportance)`; `setFeatureImportance`, `getFeatureImportance`, `train(...)` (delegates), `predict`, `predictProba`, `predictProbaWeighted`, `handleDrift(Set<Integer>, double[])`, `refreshAllSubspaces()`, getters/counters, `getSRPWrapper`, `getTau`, `name`.
- Deps: `SRPWrapper`, `FeatureImportance`, `WeightedSubspaceSampler`, `DriftActionSummary`, MOA `Classifier`.
- State: complete.

### Smoke-tests
`ModelsSmokeTest` (FeatureSpace, FilteredHeaderBuilder, HT/ARF/SRP), `DriftAwareSRPSmokeTest` (importance/sampler/drift actions).

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
`EvaluationSmokeTest`, `StatsSmokeTest`.

---

## Package `thesis.pipeline`

### `StreamMetrics`
- File: `src/main/java/thesis/pipeline/StreamMetrics.java`
- Lightweight throughput/accuracy counters used by `StreamPipeline`.
- Fields: `count`, `correct`, `totalTimeNanos`, `peakMemoryBytes`, `lastUpdateNanos`. `@Getter`.
- Methods: `update`, `recordMemory`, `reset`, `getAccuracy`, `getAvgTimeMicros`.
- State: complete.

### `RecordingMetrics extends StreamMetrics`
- File: `src/main/java/thesis/pipeline/RecordingMetrics.java`
- Streaming CSV writer attached as the pipeline's metrics. Emits a row every `sampleEvery` instances with `instance_num,kappa,kappa_per,accuracy,ram_hours,feature_stability_ratio,drift_count,recovery_time`.
- Maintains an inline κ (per-class true/pred totals) and κ_per (vs. no-change baseline), plus its own RAM-hour integration and drift/recovery tracking (`RECOVERY_THRESHOLD = 0.01`).
- Methods: ctor `(BufferedWriter, int, FeatureSelector)`, package-private `bindPipeline(StreamPipeline, TwoLevelDriftDetector)`, override `update`, package-private `flushFinalRow`. Private helpers: `writeRow`, `accuracy`, `cohenKappa`, `kappaTemporal`.
- Deps: `StreamPipeline`, `TwoLevelDriftDetector`, `FeatureSelector`.
- State: complete.

### `SyntheticStreamFactory` (final, utility)
- File: `src/main/java/thesis/pipeline/SyntheticStreamFactory.java`
- Factories: `createSEA`, `createHyperplane`, `createRandomRBF`, `createSTAGGER`, `createCustomFeatureDrift`, `addNoiseFeatures`. SEA & STAGGER are built as 4- or 3-segment `ConceptDriftStream` with abrupt changepoints (positions 25k/50k/75k for SEA; 20k/40k/60k for STAGGER).
- Inner private: `NoiseAugmentedStream extends AbstractOptionHandler implements InstanceStream` — wraps a base stream and appends `nNoise` U(0,1) attributes; rebuilds the header with the class moved to the end.
- State: complete.

### `StreamPipeline`
- File: `src/main/java/thesis/pipeline/StreamPipeline.java`
- Core orchestration:
  1. `warmupIfNeeded` — drains `warmupSize` instances, builds `FeatureSpace`, fills the optional ranking PiD, calls `selector.initialize(...)`, seeds `FeatureImportance` with MI scores + zero KS.
  2. `processInstance` — predict → compute 0/1 error → `detector.update(error, x)` → on alarm: reset drifting features in ranking PiD, refresh `FeatureImportance` with current MI + `1−p` from KSWIN → `model.train(...)`. If model is `DriftAwareSRP`, also call `handleDrift(drifting, fullRanker.scores)` and (every `refreshEvery` instances) `refreshAllSubspaces`.
  3. `metrics.update` records prediction error + elapsed nanos.
- Inner: `Builder` with all options; defaults `warmupSize=1500, logEvery=1000, refreshEvery=0, maxInstances=Long.MAX_VALUE, verbose=true`.
- Fields: `source`, `selector`, `model`, `detector`, `metrics`, `space`, optional `rankingPid`/`fullRanker`/`importance`, configuration ints, `warmedUp`, `processed`, `globalAlarmsSeen`. `@Getter`.
- Public methods: `builder`, `run`, `processInstance`, `warmupIfNeeded`. Private: `log`, `finalLog`, static `invertPValues`.
- Deps: every `thesis.*` package except `evaluation` and `experiments`.
- State: complete.

### `ExperimentRunner`
- File: `src/main/java/thesis/pipeline/ExperimentRunner.java`
- Loads JSON config (`experiment_group, datasets[], variants[], seeds[], output_dir, warmup, log_every, refresh_every, max_instances, verbose`); writes `{outputDir}/{group}/{dataset}/{variant}/seed_{s}.csv` using `RecordingMetrics`.
- Inner DTOs: `Variant {name, model, selector, detector="TWO_LEVEL"}`, `Config`.
- Public: `main(String[])`, `runAll(Config)`. Private: `runOne`, `loadConfig`.
- State: **stub-bound.** Calls `DatasetFactory/ModelFactory/SelectorFactory/DetectorFactory.create(...)` from `Shims.java`, all of which throw `UnsupportedOperationException`. The actual production runner is `E1Baselines`.

### `Shims.java`
- File: `src/main/java/thesis/pipeline/Shims.java`
- Four package-private final classes (`DatasetFactory`, `ModelFactory`, `SelectorFactory`, `DetectorFactory`), each with a single static `create(name, seed)` that throws — placeholders for `ExperimentRunner`. **State: TODO.**

### `ArffSanityCheck`
- Diagnostic with hard-coded ARFF paths under `…/preproccessing/…`. State: complete (works only on the user's machine).

### Smoke-tests
`PipelineSmokeTest`, `SyntheticStreamSmokeTest`.

---

## Package `thesis.experiments`

### `E1Baselines`
- File: `src/main/java/thesis/experiments/E1Baselines.java`
- The functional E1 driver. Loads `E1_baselines.json`, sweeps `datasets × detectors × variants × seeds`, builds the matching `InstanceStream` (synthetic generator or `ArffFileStream`) + `StaticFeatureSelector` + chosen `ModelWrapper` + chosen `TwoLevelDriftDetector`, runs the pipeline manually (calls `pipe.processInstance` per instance and feeds `MetricsCollector`), writes `summary.csv` and `validation_level1.txt`.
- The validation step requires every non-baseline variant to beat both `MajorityClass` and `NoChange` on the chosen metric (default `kappa`) by `min_margin` (default 0.0).
- Public: `main(String[])`. Private: `runOne`, `buildStream`, `buildModel`, `buildDetector`, `pickMetric`, `writeSummary`, `validateLevel1`, three list helpers.
- Deps: `pipeline.*` (StreamPipeline, SyntheticStreamFactory), `models.*` (ARF, HT, SRP, MajorityClass, NoChange), `selection.StaticFeatureSelector`, `discretization.PiDDiscretizer`, `detection.TwoLevelDriftDetector`, `evaluation.MetricsCollector`.
- State: complete for the variants listed (`HT+S1, ARF+S1, SRP+S1, MajorityClass, NoChange`); does NOT yet wire other selectors (S2/S3/S4) or the drift-aware SRP — those would need new `case` arms in `buildModel` / a configurable selector.
  Bug-watch: in `runOne` the loop calls `model.predict` + `mc.update` AND then `pipe.processInstance(raw)` — the pipeline itself also predicts/trains, so each instance is being predicted twice (once for `mc`, once inside the pipeline) and trained once via the pipeline's path. Worth reviewing.
  In the JSON, `YahooFinance` is mistakenly pointed at `data/arff/nhts.arff` (same as `NHTS`).

### `E1_baselines.json`
- 5 variants, 8 datasets (5 synthetic + 3 ARFF), 5 seeds, validation against MC + NoChange.

---

# Component coverage vs. plan

| Plan component | Implementation | Status |
|---|---|---|
| **Detection** — global + per-feature | ADWIN, HDDM_A, HDDM_W (Level-1) ; KSWINSingleFeature → PerFeatureKSWIN with BH-FDR (Level-2) ; combined as `TwoLevelDriftDetector` | ✔ complete |
| **Selection** — S1/S2/S3/S4 | `StaticFeatureSelector` (S1), `PeriodicSelector` (S2), `AlarmTriggeredSelector` (S3), `DriftAwareSelector` (S4) — all over an `AbstractFrequencyRanker` (IG/MI/χ²) on PiD output | ✔ complete |
| **Discretization** — PiD two-layer | `Layer1Histogram` + `Layer2Merger` + `FeatureDiscretizer` + `PiDDiscretizer` | ✔ complete |
| **Models** — baselines + ensembles | `MajorityClassWrapper`, `NoChangeWrapper`, `HoeffdingTreeWrapper`, `ARFWrapper`, `SRPWrapper`, `DriftAwareSRP` (+ `FeatureImportance` + `WeightedSubspaceSampler`) | ✔ complete |
| **Evaluation** — metrics + statistical tests | Prequential Acc, Cohen κ, Temporal κ, Recovery, RAM-h, Stability, MetricsCollector + Friedman / Nemenyi / Wilcoxon / CD-export | ✔ complete (no PNG diagrams generated in Java; CSVs only) |
| **Pipeline** — orchestration | `StreamPipeline` (Builder), `StreamMetrics`, `RecordingMetrics` | ✔ complete |
| **Experiments** | `E1Baselines` (E1 baselines vs. MajorityClass/NoChange) | ✔ for E1's 5 variants. **Other experiments (E2/E3/…) — selectors S2/S3/S4 and DriftAwareSRP — NOT YET wired into a runner.** |
| **Generic JSON runner** | `ExperimentRunner` + `Shims.java` factories | ✗ stubs — `DatasetFactory/ModelFactory/SelectorFactory/DetectorFactory` all throw |
| **Tests** | 14 ad-hoc `*SmokeTest.java` `main` classes; no `src/test/`, no JUnit | functional but not a real test framework |
| **Configs** | One JSON next to `E1Baselines.java`; no `resources/`, no YAML | OK |

**What's missing or partial**
- The "official" generic runner `ExperimentRunner` is not connected — its four factories are `UnsupportedOperationException` stubs (`Shims.java`). `E1Baselines` works around this with hard-coded `switch` arms.
- No experiment configs for S2/S3/S4 selectors, no config for `DriftAwareSRP` — the code exists, the JSON sweeps don't reach it.
- `E1_baselines.json` has the YahooFinance entry pointing at the NHTS ARFF (typo).
- `E1Baselines.runOne` evaluates each instance twice (own `model.predict` + `pipe.processInstance` which also predicts and trains) — likely an unintended overlap to verify before reporting numbers.
- `CDDiagramExporter` writes CSVs only — actual CD-diagram drawing must be done externally.
- No real JUnit test suite (only `main`-driven smoke tests with a custom `report(...)` helper).

---

# Data flow

**Entry point** (production): `thesis.experiments.E1Baselines.main` reads `E1_baselines.json`.

For each `(dataset, detector, variant, seed)`:

1. **Stream construction** — `buildStream(...)` → either `SyntheticStreamFactory.create*` (optionally wrapped in `addNoiseFeatures`) or `ArffFileStream`. `prepareForUse()` is called.
2. **Selector + model + detector + PiD** built from the JSON keys; `MetricsCollector` instantiated.
3. **Pipeline build** — `StreamPipeline.builder()...build()` discovers the header via `FeatureSpace`, marks itself warmed up if the selector is already initialised.
4. **Warm-up** — `warmupIfNeeded()` pulls `warmupSize` instances, feeds them into the optional ranking PiD, then `selector.initialize(window, labels)`. Optional `FeatureImportance` is seeded.
5. **Per-instance loop** (per instance: `Instance raw`):
   - `x = space.extractFeatures(raw)`, `y = (int) raw.classValue()`.
   - Optional ranking PiD update; optional `fullRanker.update(pid.discretizeAll(x), y)`.
   - `yhat = model.predict(raw)`; `error = yhat==y ? 0 : 1`.
   - `detector.update(error, x)` — Level-1 sees `error`, Level-2 sees `x[]`. On alarm Level-2 returns BH-corrected drifting feature indices.
   - On alarm: drifting features reset in ranking PiD; `FeatureImportance` updated with current MI scores + `1 − pValue` per feature.
   - `model.train(raw, y, alarm, drifting)` — wrapper calls `selector.update(...)` internally so the selection responds to drift signals. `DriftAwareSRP` additionally calls `handleDrift(drifting, fullRanker.scores)` — per-learner KEEP / SURGICAL / FULL action.
   - `metrics.update(y, yhat, elapsedNanos)` (or `RecordingMetrics.update` to also flush a CSV row every `sampleEvery`).
6. **Per-run output** — `RecordingMetrics` writes one CSV; `E1Baselines` instead writes one row per run into `summary.csv` and aggregates means for level-1 validation.
7. **Aggregation across datasets/seeds** — `validateLevel1(...)` writes `validation_level1.txt`. Cross-method statistical comparison (Friedman / Nemenyi / Wilcoxon / CD CSVs) is available in `thesis.evaluation.StatisticalTests` but is *not* yet invoked by `E1Baselines` — so far that wiring stops at `summary.csv`.

In short: `Instance → FeatureSpace.extractFeatures → (ranking PiD + fullRanker) + (model.predict, error, TwoLevelDriftDetector) → on alarm: feature-importance refresh + DriftAwareSRP.handleDrift → model.train (which updates the selector) → metrics`.
