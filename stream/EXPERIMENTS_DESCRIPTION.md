# EXPERIMENTS_DESCRIPTION.md — szczegółowy opis eksperymentów E1–E5

> Dokument opisuje pięć runnerów eksperymentalnych zlokalizowanych w
> `src/main/java/thesis/experiments/`. Każdy z nich zaczytuje swoją konfigurację
> z pliku JSON sąsiadującego w tym samym folderze i zapisuje wyniki w `results/<EX>/`.
>
> Eksperymenty (klasa → config → output dir):
> - `E1Baselines`      → `E1_baselines.json` → `results/E1_baselines/`
> - `E2AdaptiveFS`     → `E2_adaptive_fs.json` → `results/E2/`
> - `E3DASRP`          → `e3_da_srp.json` → `results/E3/`
> - `E4DriftAnalysis`  → `e4_synthetic_drift.json` → `results/E4/`
> - `E5Detectors`      → `e5_detectors.json` → `results/E5/`

---

## E1 — `E1Baselines`

### 1. Cel eksperymentu

E1 to **production-grade benchmark baseline'ów modeli** (HT, ARF, SRP) w wariancie
*Static Feature Selection (S1) bez wykrywania dryftu*. Cel: ustalić **referencyjne
poziomy κ / accuracy / RAM-h / throughput** dla każdej pary `(model, dataset)`,
do których kolejne eksperymenty (E2/E3/E4/E5) będą się porównywać.

Hipoteza nie jest statystyczna — E1 dostarcza **tabelarycznych baseline'ów** z
porównaniem do `MajorityClass` (predykcja klasy najczęstszej) i `NoChange` (predykcja
ostatniej etykiety) — oba liczone równolegle w tym samym strumieniu, co służy jako
sanity check (model poważny powinien bić oba).

### 2. Konfiguracja eksperymentu

Z pliku `E1_baselines.json`:

| Parametr | Wartość |
|---|---|
| `experiment_group` | `E1_baselines` |
| `output_dir` | `results/E1_baselines` |
| `warmup` | `1500` |
| `window_size` | `1000` |
| `log_every` | `1000` |
| `ram_sample_every` | `200` |
| `max_instances` | `100000` |
| `skip_missing_arff` | `true` |
| `real_datasets_read_all` | `true` (ARFF czytane do końca, nie do `max_instances`) |
| `seeds` | `[1]` |
| `models` | `["HT", "ARF", "SRP"]` |

**Modele** (z `E1Baselines.buildModel`):
- `HT` → `new HoeffdingTreeWrapper(sel, header)` (default: `gracePeriod=200, splitConfidence=0.01, resetOnSelectionChange=false`).
- `ARF` → `new ARFWrapper(sel, header)` (default: `ensembleSize=10, lambda=6.0, resetOnSelectionChange=false`).
- `SRP` → `new SRPWrapper(sel, header)` (default: `ensembleSize=10, lambda=6.0, resetOnSelectionChange=false`).
- `MAJORITY`/`MAJORITYCLASS` → `new MajorityClassWrapper(sel, numClasses)` (zawsze tworzony jako kanał baseline, niezależnie od listy `models`).
- `NOCHANGE` → `new NoChangeWrapper(sel, numClasses)` (analogicznie).

> Wszystkie wrappery dostają **ten sam selektor** S1 — model widzi `K = ⌈√d⌉`
> wybranych cech, nigdy pełnego `d`.

**Feature selection:**
- Tylko **S1** — `new StaticFeatureSelector(numFeatures, numClasses)` z domyślnym
  `k = ⌈√d⌉` (z `StaticFeatureSelector.defaultK`).
- Selektor inicjalizowany jednorazowo na warmup-oknie (`selector.initialize(window, labels)`),
  `update()` jest no-op (selekcja zamrożona na cały run).
- **Brak osobnych parametrów selekcji** w E1 — domyślne PiD + IG ranker zaszyte w klasie.

**Drift detector:**
- **BRAK** — E1 nie tworzy żadnego detektora dryftu. `error` nie jest karmiony do
  ADWIN/HDDM/KSWIN. Pipeline jest "płaski": predict → metrics.update → train.
- Konsekwencja: kolumna `drift_count` w plikach wynikowych będzie **zawsze równa 0**
  (`MetricsCollector.driftCount` rośnie tylko gdy ktoś wywoła `mc.onDriftAlarm()`,
  a w E1 nikt tego nie robi).

**Datasety (8)** — wszystkie z configa:
- 5 syntetycznych (każdy `n=100000`):
  - `SEA` (`generator=SEA`, `noise_features=5`),
  - `Hyperplane` (`sigma=0.01`, `noise_features=5`),
  - `RandomRBF` (`speed=0.001`, `noise_features=5`),
  - `STAGGER` (`noise_features=0`),
  - `FeatureDrift` = `CustomFeatureDrift` (`drift_features=5`, `sigma=0.01`, `noise_features=5`).
- 3 ARFF (czytane do końca dzięki `real_datasets_read_all=true`):
  - `YahooFinance` → `/home/kubog/MSc-Thesis-Learning-in-stream/preproccessing/yahoo_finance/data/arff/yahoo_finance.arff`,
  - `NYCTaxi` → `/home/kubog/MSc-Thesis-Learning-in-stream/preproccessing/nyc_taxi/data/arff/nyc_taxi.arff`,
  - `NHTS` → `/home/kubog/MSc-Thesis-Learning-in-stream/preproccessing/data/arff/nhts.arff`.

**Typy dryftu**: implicite zaszyte w generatorach syntetycznych (np. SEA jest
4-segmentowym `ConceptDriftStream` z punktami zmiany w `n/4, n/2, 3n/4` —
patrz `SyntheticStreamFactory.createSEA`). E1 NIE sweepuje magnitudes (to robi E4).

**Inne ustawienia**:
- `effectiveMaxInstances`: dla ARFF i `realDatasetsReadAll=true` zwraca `Long.MAX_VALUE`
  (czytamy do EOF), inaczej `cfg.maxInstances=100000`.
- `MetricsCollector × 3` (model, majority, no-change), wszystkie z identycznymi
  `windowSize=1000`, `logEvery=1000`, `ramSampleEvery=200`.
- `skip_missing_arff=true` → run pomijany cicho jeśli ARFF nie istnieje.

### 3. Przebieg eksperymentu

Główna pętla w `run(E1Config)`:

```
for each dataset ds in cfg.datasets:
    skip if arff missing && skipMissingArff
    for each seed in cfg.seeds:
        for each modelName in cfg.models:
            runOne(cfg, ds, modelName, seed, csv, ...)
            // result -> wpis do E1_summary.csv
```

Krok po kroku w `runOne`:

1. **Build stream** (`buildStream`):
   - Jeśli `type=arff` → `new ArffFileStream(ds.path, -1)` + `prepareForUse()`.
   - Jeśli `type=synthetic` → `SyntheticStreamFactory.create*(seed, ...)` (SEA/Hyperplane/RandomRBF/STAGGER/CustomFeatureDrift),
     opcjonalnie wrappowane w `addNoiseFeatures(base, ds.noiseFeatures, seed)`.
2. **Header → wymiary**: `numFeatures = header.numAttributes() - 1`, `numClasses = header.numClasses()`. Walidacja: `numFeatures ≥ 1`, `numClasses ≥ 2`.
3. **Warmup** (`cfg.warmup=1500` instancji):
   - Czytamy do bufora `window[cfg.warmup][]` + `labels[]`,
   - cechy ekstrahowane przez `FeatureSpace.extractFeatures(x)`.
4. **Selector init**: `selector.initialize(window, labels)` (S1 raz, na zawsze).
5. **Modele**: `model = buildModel(modelName, selector, header, numClasses)`,
   `majority = new MajorityClassWrapper(selector, numClasses)`,
   `noChange = new NoChangeWrapper(selector, numClasses)`.
6. **Karmienie baseline'ów warmupem**: pętla `for i<collected: majority.train(null, labels[i]); noChange.train(null, labels[i]);` — żeby na pierwszej "prawdziwej" instancji baseline'y już znały rozkład klas / ostatnią etykietę.
7. **`MetricsCollector × 3`** (`metrics`, `mMaj`, `mNC`).
8. **Pętla strumieniowa** (do `n < effMax`):
   - `yHat = model.predict(x)` z pomiarem `elapsed` (`System.nanoTime`),
   - `yMaj = majority.predict(x)`, `yNC = noChange.predict(x)`,
   - `metrics.update(yTrue, yHat, elapsed)`, `mMaj.update(yTrue, yMaj, 0)`, `mNC.update(yTrue, yNC, 0)`,
   - `model.train(x, yTrue)`, `majority.train(x, yTrue)`, `noChange.train(x, yTrue)`,
   - co `logEvery=1000` snapshot do `E1_results.csv` (window-level wpis) +
     liczony lokalny throughput `(n - lastLogN) / dtSec`.
9. **Po pętli**: finalny `metrics.snapshot()` zwracany do `run`, gdzie zostaje
   wpisany do `E1_summary.csv` z `status=OK` (lub `FAIL` przy wyjątku).

**Używane klasy/funkcje:**
- `thesis.pipeline.SyntheticStreamFactory` (createSEA, createHyperplane, createRandomRBF, createSTAGGER, createCustomFeatureDrift, addNoiseFeatures),
- `thesis.models.{FeatureSpace, MajorityClassWrapper, NoChangeWrapper, HoeffdingTreeWrapper, ARFWrapper, SRPWrapper}`,
- `thesis.selection.StaticFeatureSelector`,
- `thesis.evaluation.MetricsCollector` (3 instancje: model, majority, no-change),
- MOA: `ArffFileStream`, `InstancesHeader`, `Instance`.

### 4. Zapisywane wyniki

E1 produkuje **3 pliki** w `results/E1_baselines/`:

#### Plik `E1_results.csv`
- **Lokalizacja**: `results/E1_baselines/E1_results.csv`
- **Kiedy tworzony**: na początku `run(...)`, w try-with-resources. Wpisy co
  `logEvery=1000` instancji dla każdego runu.
- **Co zawiera**: window-level snapshot per run.
- **Kolumny** (z `csv.println(...)` na liniach 140–144):

| Kolumna | Znaczenie |
|---|---|
| `dataset` | Nazwa datasetu (np. `SEA`, `YahooFinance`). |
| `model` | Nazwa modelu (`HT`/`ARF`/`SRP`). |
| `selector` | Stała `S1`. |
| `seed` | Ziarno losowe dla tego runu. |
| `instance_num` | Numer aktualnej instancji w strumieniu. |
| `selected_count` | `selector.getCurrentSelection().length` (oczekiwane = `K = ⌈√d⌉`). |
| `model_num_attributes` | `model.getCurrentSelection().length` — tyle atrybutów widzi model. |
| `accuracy_window` | Sliding-window accuracy z `MetricsCollector.snapshot().accuracyWindow`. |
| `kappa_window` | Cohen's κ z okna. |
| `kappa_per_window` | Temporal κ (vs. NoChange) z okna. |
| `majority_baseline_window` | Window accuracy dla `MajorityClassWrapper`. |
| `nochange_baseline_window` | Window accuracy dla `NoChangeWrapper`. |
| `ram_hours` | `s.ramHoursGB` — RAM-hours (GB·h). |
| `drift_count` | `s.driftCount` — w E1 zawsze 0 (brak detektora). |
| `feature_stability` | `s.lastFeatureStabilityRatio` (NaN→0.0). Dla S1 = 1.0 lub NaN, bo selekcja się nie zmienia. |
| `throughput_inst_per_sec` | Lokalny throughput w oknie międzyloggowania. |
| `peak_ram_mb` | `s.peakMB`. |

#### Plik `E1_summary.csv`
- **Lokalizacja**: `results/E1_baselines/E1_summary.csv`
- **Kiedy tworzony**: na początku `run(...)`. Wpis (jeden wiersz) dodawany **po
  zakończeniu każdego runu** (sukces lub FAIL).
- **Co zawiera**: jeden wiersz per `(dataset × seed × model)` run z finalnymi metrykami.
- **Kolumny** (z `sum.println(...)` na liniach 146–149):

| Kolumna | Znaczenie |
|---|---|
| `dataset, model, selector, seed` | Identyfikatory runu. `selector` zawsze `S1`. |
| `instance_num` | Łączna liczba przetworzonych instancji (`nOut[0]`). |
| `accuracy_window` | Final window accuracy. |
| `kappa_window` | Final Cohen κ. |
| `kappa_per_window` | Final temporal κ. |
| `majority_baseline_window` | Final accuracy MajorityClass. |
| `nochange_baseline_window` | Final accuracy NoChange. |
| `ram_hours` | RAM-hours (GB·h). |
| `drift_count` | Z `s.driftCount`; w E1 zawsze 0. |
| `feature_stability` | Final stability ratio (NaN→0.0). |
| `throughput_inst_per_sec` | Średnia globalna `n / elapsedSec` (z `s.elapsedHours * 3600`). |
| `peak_ram_mb` | Peak RAM. |
| `status` | `OK` / `FAIL`. Wiersze `FAIL` mają NaN w polach numerycznych. |

#### Plik `E1_baselines_drifts.csv`
- **Lokalizacja**: `results/E1_baselines/E1_baselines_drifts.csv` (nazwa zbudowana
  z `cfg.experimentGroup + "_drifts.csv"`).
- **Kiedy tworzony**: na początku `run(...)`, **z samym headerem zapisanym natychmiast**.
- **Kolumny**: `dataset, variant, seed, alarm_at, kappa_before_500, kappa_after_500, recovery_instances, drift_type`.
- **Co zawiera**: **TYLKO header — żadnych wpisów**. E1 nie używa `DriftLogger`
  ani nie wywołuje detektora, więc plik pozostaje pusty (poza pierwszą linią).
  Header jest tworzony dla **konsystencji formatu** z E2/E3/E4/E5.

---

## E2 — `E2AdaptiveFS`

### 1. Cel eksperymentu

E2 testuje **hipotezę, że adaptacyjne strategie selekcji cech (S2/S3/S4) z detekcją
dryftu zachowują się lepiej niż statyczna baseline S1** dla tego samego modelu (ARF lub SRP).

Walidacja w `writeValidation` (`validation_level1.txt`) jest sformułowana wprost:
- dla każdego datasetu liczona jest średnia κ po seedach per wariant,
- każdy wariant adaptacyjny porównywany z odpowiednim S1 (ARF+S1 lub SRP+S1):
  - `mean(adaptive) >= mean(S1)` → **PASS**,
  - `|mean(adaptive) - mean(S1)| < 1e-6` → **WARN** (selektor nieefektywny),
  - inaczej → **FAIL**.

Dodatkowo eksperyment sweepuje:
- 3 detektory (ADWIN/HDDM_A/HDDM_W) dla S2,
- 3 interwały okresowe (500/1000/2000) dla S3,
- 1 wariant S4 (`SRP+S4`).

### 2. Konfiguracja eksperymentu

Z pliku `E2_adaptive_fs.json`:

| Parametr | Wartość |
|---|---|
| `experiment_group` | `E2` |
| `output_dir` | `results/E2` |
| `warmup` | `1500` |
| `window_size` | `1000` |
| `log_every` | `1000` |
| `ram_sample_every` | `200` |
| `max_instances` | `100000` |
| `detector_delta` | `0.002` |
| `skip_missing_arff` | `true` |
| `real_datasets_read_all` | `true` (ARFF czytane do końca, nie do `max_instances`) |
| `seeds` | `[1]` |

**Modele** (`buildModel` w E2 obsługuje TYLKO ARF/SRP — HT należy do E1, DASRP do E3):
- `ARF` → `new ARFWrapper(selector, header, ensembleSize=10, lambda=6.0, resetOnSelectionChange=false, exposeOptions=true)`
- `SRP` → `new SRPWrapper(selector, header, 10, 6.0, false, true)`

**Feature selection** (`buildSelector`, `K = ⌈√d⌉`):
- **S1** → `new StaticFeatureSelector(d, C)` (jednorazowa selekcja na warmup-ie).
- **S2** → `new AlarmTriggeredSelector(d, C, K, max(50, wPostDrift), new PiDDiscretizer(d, C), (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc))`.
- **S3** → `new PeriodicSelector(d, C, K, max(100, periodicInterval), 100, new PiDDiscretizer(d, C), (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc))` (`minTenure=100`).
- **S4** → `new DriftAwareSelector(d, C, K, max(100, periodicInterval), 100, max(50, wPostDrift), new PiDDiscretizer(d, C), (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc))` (łączy okresowość z reakcją na alarm).

**Parametry FS sterowane z JSON**:
- `periodic_interval` (S3, S4): `500 / 1000 / 2000` w configu.
- `w_post_drift` (S2, S4): `1000`.

**Detektory** (`buildDetector` → zawsze opakowane w `TwoLevelDriftDetector` z `Config`):
- Level-1 typu: `ADWIN` / `HDDM_A` / `HDDM_W` (case sensitive na uppercase'owanej nazwie).
- `level1Delta = detector_delta = 0.002` (pozostałe parametry — `level1AlphaW`, `level1Lambda`,
  `kswinAlpha`, `kswinWindowSize`, `bhQ`, `promoteReferenceOnDrift` — przyjmują default'y
  z `TwoLevelDriftDetector.Config`, NIE są ustawiane z JSON).

**Lista wariantów (14)** — pełen iloczyn z configa:
- `ARF+S1`, `SRP+S1`,
- `ARF+S2-ADWIN/HDDM_A/HDDM_W`, `SRP+S2-ADWIN/HDDM_A/HDDM_W` (`w_post_drift=1000`),
- `ARF+S3-N500/N1000/N2000`, `SRP+S3-N500/N1000/N2000`,
- `SRP+S4` (`detector=ADWIN`, `periodic_interval=1000`, `w_post_drift=1000`).

**Datasety (8)**:
- 5 syntetycznych (każdy `n=100000`):
  - `SEA` (`+ noise_features=5`),
  - `Hyperplane` (`sigma=0.01`, `+5 noise`),
  - `RandomRBF` (`speed=0.001`, `+5 noise`),
  - `STAGGER` (`noise_features=0`),
  - `FeatureDrift` = `CustomFeatureDrift` (`drift_features=5`, `sigma=0.01`, `+5 noise`).
- 3 ARFF: `YahooFinance`, `NYCTaxi`, `NHTS` — czytane do końca (`real_datasets_read_all=true`).

**Typy dryftu** są implicite zaszyte w generatorach (E2 nie ma osobnego sweepu po
"magnitude"); to E4 sweepuje magnitudes.

**Inne ustawienia**:
- `attachListeners(...)` podpina `EventListener`-y do każdego adaptacyjnego selektora,
  rozróżniając trigger ALARM (kod=1) vs PERIODIC (kod=2) → wpisywany w `trigger_type`
  CSV i licznik `periodic_triggers` / `re_selections`.
- `effectiveMaxInstances`: dla ARFF i `realDatasetsReadAll=true` zwraca
  `Long.MAX_VALUE`, inaczej `cfg.maxInstances`.

### 3. Przebieg eksperymentu

```
for each dataset ds:
    skip if arff missing && skipMissingArff
    for each seed s:
        for each variant v:
            runOne(cfg, ds, v, s, csv, drifts, selections, ...)
```

Krok po kroku w `runOne`:

1. `stream = buildStream(ds, seed)` (analogicznie do E1, ale z większą liczbą generatorów).
2. Header, wymiary, `K = defaultK(d)`, `FeatureSpace`.
3. Warmup `cfg.warmup=1500` instancji, init selektora (`buildSelector(v, d, C, K)` + `selector.initialize(window, labels)`).
4. **`attachListeners`** — subskrybuje selektor pod ALARM/PERIODIC/REASEL events,
   ustawiając `triggerCode` (0 NONE / 1 ALARM / 2 PERIODIC) co iterację.
5. `model = buildModel(v.model, selector, header)`, baseline `majority` + `noChange`.
6. `detector = buildDetector(v.detector, d, cfg.detectorDelta)` — `TwoLevelDriftDetector`.
7. **Karmienie baseline'ów warmupem** (pętla `for i<collected`).
8. `MetricsCollector` dla głównego modelu + 2 baseline'ów. `DriftLogger dl` z oknami
   `before=500, after=500`.
9. **Pętla strumieniowa**:
   - `triggerCode.set(0)` na początku iteracji.
   - `yhat = model.predict(raw)`, `yMaj`/`yNC` z baseline'ów.
   - `err = (yhat==y) ? 0 : 1`; `detector.update(err, x)`.
   - `alarm = detector.isGlobalDriftDetected()`,
     `drifting = alarm ? detector.getDriftingFeatureIndices() : Set.of()`.
   - Update `mc`, `mM`, `mN`. `dl.tick(n, kappa)`.
   - **Jeśli alarm**: `mc.onDriftAlarm()`, `dl.onAlarm(...)`, `driftCount++`.
   - `selector.update(x, y, alarm, drifting)` — adaptacyjna reakcja.
   - `model.train(raw, y, alarm, drifting)` — wrapper sam wywoła `selector.update` wewnątrz
     (uwaga: w E2 selektor jest aktualizowany dwa razy — raz jawnie tutaj,
     raz przez wrapper — to świadome lub bug, kod tak działa).
   - `majority.train`, `noChange.train`.
   - `mc.onSelectionChanged(sel)`, jeśli zmiana → wpis do `selections` CSV.
10. Co `logEvery=1000` snapshot do `E2_results.csv` (nazwa wewn. `csvPath`).
11. `dl.flushPending(n, kappa)` na końcu — domyka pending alarms.
12. Zwraca `RunSummary`, agregowane do `writeSummary` + `writeValidation`.

**Używane klasy** (poza tymi z E1):
- `thesis.detection.TwoLevelDriftDetector` (+ `Config`),
- `thesis.discretization.PiDDiscretizer`,
- `thesis.selection.{StaticFeatureSelector, AlarmTriggeredSelector, PeriodicSelector, DriftAwareSelector, InformationGainRanker}`,
- `thesis.experiments.DriftLogger`.

### 4. Zapisywane wyniki

E2 produkuje **5 plików** w `results/E2/`:

#### Plik `E2_results.csv`
- **Lokalizacja**: `results/E2/E2_results.csv`
- **Kiedy tworzony**: na początku `run` (try-with-resources w `try`).
- **Co zawiera**: window-level log co `logEvery=1000` instancji per run.
- **Kolumny** (z `windowHeader()`):

| Kolumna | Znaczenie |
|---|---|
| `instance_num` | Numer instancji. |
| `dataset` | Nazwa datasetu. |
| `variant` | Nazwa wariantu z JSON (np. `ARF+S2-ADWIN`). |
| `model` | `ARF` lub `SRP`. |
| `selector` | `S1`/`S2`/`S3`/`S4`. |
| `detector` | `ADWIN`/`HDDM_A`/`HDDM_W`. |
| `periodic_interval` | Interwał okresowego re-rankingu (S3/S4); 1000 default'em dla S1/S2. |
| `seed` | Ziarno. |
| `selected_features` | Aktualna selekcja, pipe-separated `1\|3\|7\|9` w cudzysłowach. |
| `selected_count` | Liczba wybranych cech. |
| `selection_changed` | `true` jeśli selekcja zmieniła się od ostatniego logu. |
| `trigger_type` | `ALARM` / `PERIODIC` / `NONE` — kto wywołał ostatnią zmianę. |
| `drift_alarm` | `true`/`false` — czy w tej iteracji `detector.isGlobalDriftDetected()`. |
| `drift_count` | Skumulowana liczba alarmów od początku runu. |
| `feature_stability` | `s.lastFeatureStabilityRatio` (NaN→1.0). |
| `model_num_attributes` | `model.getCurrentSelection().length + 1` (cechy + klasa). |
| `accuracy_window` | Window accuracy. |
| `kappa_window` | Cohen κ. |
| `kappa_per_window` | Temporal κ. |
| `majority_baseline_window` | Window accuracy dla MajorityClass. |
| `nochange_baseline_window` | Window accuracy dla NoChange. |
| `recovery_time` | Ostatni recovery time (instancje od alarmu do κ ≥ pre-drift κ − 0.05). |
| `ram_hours` | RAM-hours (GB·h). |
| `throughput_inst_per_sec` | Aktualny throughput w oknie międzyloggowania. |
| `peak_ram_mb` | Peak RAM w MB. |

#### Plik `E2_summary.csv`
- **Lokalizacja**: `results/E2/E2_summary.csv`
- **Kiedy tworzony**: po pętli wszystkich runów, w `writeSummary(all, summaryPath)`.
- **Co zawiera**: jeden wiersz per `(dataset × seed × variant)` run.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `dataset, variant, model, selector, detector, periodic_interval, seed` | Identyfikatory runu. |
| `instances` | Łączna liczba przetworzonych instancji. |
| `d` | Liczba cech. |
| `k` | `K = ⌈√d⌉`. |
| `accuracy` | Final accuracy z `s.accuracyWindow`. |
| `kappa` | Final κ. |
| `kappa_per` | Final temporal κ. |
| `drift_count` | Łączna liczba alarmów dryftu. |
| `periodic_triggers` | Liczba wyzwoleń okresowego re-rankingu (z listenera). |
| `re_selections` | Liczba wywołanych re-selekcji (z listenera). |
| `selection_change_count` | Ile razy `lastSel` zmieniło się w stosunku do poprzedniej iteracji. |
| `avg_feature_stability` | Średni Jaccard-like ratio kolejnych selekcji. |
| `last_feature_stability` | Ostatnia wartość ratio. |
| `acc_majority` | Final accuracy MajorityClass. |
| `acc_nochange` | Final accuracy NoChange. |
| `ram_hours` | RAM·h. |
| `throughput_inst_per_sec` | Średni throughput. |
| `peak_ram_mb` | Peak RAM. |
| `status` | `OK` / `FAIL`. Wiersze `FAIL` mają NaN we wszystkich numerycznych. |

#### Plik `E2_drifts.csv`
- **Lokalizacja**: `results/E2/E2_drifts.csv`
- **Kiedy tworzony**: na początku `run`. Wpisy dodawane przez `DriftLogger` przy każdym
  alarmie (i przy `flushPending` na końcu runu).
- **Co zawiera**: jeden wiersz per zarejestrowany alarm (Pending → po `afterWindow=500`
  od alarmu).
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `dataset` | Nazwa datasetu. |
| `variant` | Nazwa wariantu. |
| `seed` | Ziarno. |
| `alarm_at` | Numer instancji, w której odpalił alarm. |
| `kappa_before_500` | Średnia κ z okna `before=500` instancji *przed* alarmem. |
| `kappa_after_500` | κ na chwilę zamknięcia wpisu (after=500 instancji po alarmie LUB końcu strumienia). |
| `recovery_instances` | Liczba instancji do odzyskania κ ≥ `startKappa − 0.02`; `-1` jeśli nie odzyskano w oknie. |
| `drift_type` | `FEATURE` jeśli `drifting` był niepusty, inaczej `GLOBAL`. |

#### Plik `E2_selections.csv`
- **Lokalizacja**: `results/E2/E2_selections.csv`
- **Kiedy tworzony**: na początku `run`. Wpisy dodawane przy każdej zmianie selekcji
  (`writeSelectionChange` w pętli, po `selector.update`).
- **Co zawiera**: jeden wiersz per zmiana selekcji.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `dataset` | Nazwa datasetu. |
| `variant` | Nazwa wariantu. |
| `seed` | Ziarno. |
| `changed_at` | Numer instancji, w której zaszła zmiana. |
| `old_selection` | Stara selekcja w nawiasach `[1,3,7,9]`. |
| `new_selection` | Nowa selekcja `[2,3,7,12]`. |
| `added_features` | Cechy nowo dodane: `[2,12]`. |
| `removed_features` | Cechy usunięte: `[1,9]`. |
| `trigger` | `ALARM` / `PERIODIC` (na podstawie `triggerCode` z listenera). Default `PERIODIC`, gdy kod = 0 (patrz linia `: "PERIODIC"` w runOne). |

#### Plik `validation_level1.txt`
- **Lokalizacja**: `results/E2/validation_level1.txt`
- **Kiedy tworzony**: na końcu `run` przez `writeValidation(...)`.
- **Co zawiera**: tekstowy raport per dataset z mean κ wszystkich wariantów,
  każdy adaptacyjny porównany z S1 (PASS/FAIL/WARN) + zbiorczy `TOTAL`.

---

## E3 — `E3DASRP`

### 1. Cel eksperymentu

E3 testuje **`DriftAwareSRP` w trzech wersjach progresywnego włączania komponentów**
(ablation study) i porównuje je z mocnymi baseline'ami (`SRP+S1`, `ARF+S2`):

- **A** — `DASRP_A` — sam mechanizm KEEP/SURGICAL/FULL bez `FeatureImportance` i bez
  ważonego głosowania.
- **AB** — `DASRP_AB` — A + `FeatureImportance` (MI + 1−p), ale bez ważenia predykcji
  (`DASRPNoWeighting` nadpisuje `predictProba` zwykłym wywołaniem `srp.predictProba`).
- **ABC** — `DASRP_ABC` — pełne A + B + C (ważone głosowanie ensembla przez
  `predictProbaWeighted`).

Hipoteza: każdy kolejny komponent (B, C) poprawia metryki względem poprzedniego
i pełna wersja ABC dominuje baseline'y. Walidacja przez:
- per-dataset κ matrix (`E3_kappa_matrix.csv`),
- testy statystyczne Friedman + Nemenyi + Wilcoxon (`StatisticalTests.runFull`),
- targeted Wilcoxon DA-SRP* vs `SRP+S1` i `ARF+S2` z tagiem WIN/LOSE/ns/DEGEN.

### 2. Konfiguracja eksperymentu

Z pliku `e3_da_srp.json`:

| Parametr | Wartość |
|---|---|
| `experiment_group` | `E3` |
| `output_dir` | `results/E3` |
| `warmup` | `1500` |
| `window_size` | `1000` |
| `log_every` | `1000` |
| `ram_sample_every` | `200` |
| `importance_update_every` | `1000` |
| `max_instances` | `100000` |
| `detector_delta` | `0.002` |
| `skip_missing_arff` | `true` |
| `real_datasets_read_all` | `true` |
| `seeds` | `[1]` |

**Modele / Mode-y** (`buildModel` w E3):
- `BASELINE_SRP_S1`: `SRPWrapper(sBase, header, ensembleSize=10, lambda=6.0, false, true)` z S1.
- `BASELINE_ARF_S2`: `ARFWrapper(s2, header, 10, 6.0, false, true)` z `AlarmTriggeredSelector`
  (S2, `K=⌈√d⌉`, `wPostDrift=max(50, v.wPostDrift)=1000`, `PiDDiscretizer + InformationGainRanker`).
- `DASRP_A` / `DASRP_AB` / `DASRP_ABC`:
  - **`wideSel`** = `StaticFeatureSelector(d, C, k=d, PiDDiscretizer, InformationGainRanker)`
    z `initializeIdentity()` — model SRP widzi WSZYSTKIE cechy (subspaces są zarządzane
    przez DriftAwareSRP, nie przez selektor).
  - `srp = SRPWrapper(wideSel, header, ensembleSize=10, lambda=6.0, false, false)`.
  - `scorePid = new PiDDiscretizer(d, C)` + `scoreRanker = InformationGainRanker(d, b2, C)`,
    karmione warmupem żeby od startu mieć MI scores.
  - `imp = new FeatureImportance(d, w1, 1-w1, 1e-6, true)` dla mode'ów AB i ABC
    (`w1=0.7` z configa, `w2=0.3`).
  - Dla AB: `da = new DASRPNoWeighting(srp, tau=0.5, seed, imp)` (override predict).
  - Dla ABC: `da = new DriftAwareSRP(srp, tau=0.5, seed, imp)`.
  - Dla A: `da = new DriftAwareSRP(srp, tau=0.5, seed, null)` (importance=null → uniform sampling).
  - `da.setScoreProvider(...)` — dostarczana funkcja zwracająca aktualne MI scores.

**Feature selection**:
- Dla DA-SRP* selekcja jest "pełna" (`d` cech) — **per-learner subspaces** zarządzane przez
  `DriftAwareSRP.handleDrift` z progiem `tau=0.5`.
- Baseline'y używają S1 (SRP) lub S2 (ARF).
- `wPostDrift = 1000` dla S2; `K = ⌈√d⌉` dla S2.

**Detektor** (`buildDetectorWithKswin`):
- Zawsze `TwoLevelDriftDetector` z Level-1 = ADWIN/HDDM_A/HDDM_W (na podstawie `v.detector`).
- `c.level1Delta = 0.002` (hardcoded w buildDetectorWithKswin, NIE czytane z `cfg.detectorDelta`).
- `c.kswinAlpha = v.kswinAlpha` (`0.005` w configu dla DA-SRP-ABC; default 0.005 dla pozostałych).
- `c.kswinWindowSize = max(10, v.kswinWindow)` (`200` w configu).

**Lista wariantów (5)**:
- `SRP+S1` (`mode=BASELINE_SRP_S1`),
- `ARF+S2` (`mode=BASELINE_ARF_S2`, `w_post_drift=1000`),
- `DA-SRP-A` (`mode=DASRP_A`, `tau=0.5`),
- `DA-SRP-AB` (`mode=DASRP_AB`, `tau=0.5`, `w1=0.7`),
- `DA-SRP-ABC` (`mode=DASRP_ABC`, `tau=0.5`, `w1=0.7`, `kswin_alpha=0.005`, `kswin_window=200`).

**Sensitivity grid** (opcjonalnie, w configu nie podany; obsługiwany w
`Cfg.expandGrid` — generuje warianty `DA-SRP-ABC_w1=...,a=...,W=...`).

**Datasety**: identyczne 8 jak w E2 (5 syntetycznych + 3 ARFF z tymi samymi parametrami).

**Inne ustawienia**:
- `importance_update_every=1000` — co tyle instancji `mh.importance.update(mi, ks)`
  z `ks[i] = 1 - clamp01(pv[i])`.
- `tau=0.5`, `w1=0.7` — globalne default'y wariantów.
- `ensemble_size=10`, `lambda=6.0` — default'y SRP.

### 3. Przebieg eksperymentu

```
for each dataset ds (skip if arff missing):
    for each seed s:
        for each variant v:
            runOne(cfg, ds, v, s, csv, drifts, actions, ...)
```

Krok po kroku w `runOne`:

1. `stream = E2AdaptiveFS.buildStream(ds, seed)` (ten sam helper co E2).
2. Header → `d, C`, `FeatureSpace`.
3. Warmup `cfg.warmup=1500`.
4. **`mh = buildModel(v, header, d, C, seed, window, labels)`** — buduje:
   - `mh.main` (model główny — wrapper SRP/ARF lub DriftAwareSRP),
   - `mh.da` (referencja do DriftAwareSRP, NULL dla baseline'ów),
   - `mh.importance` (FeatureImportance lub NULL dla DASRP_A i baseline'ów),
   - `mh.scorePid` + `mh.scoreRanker` (PiD + IG do liczenia MI),
   - `mh.selectorForBaselines` (StaticFeatureSelector używany przez Majority/NoChange).
5. `detector = buildDetectorWithKswin(v, d)`.
6. `majority`/`noChange` z `mh.selectorForBaselines`, karmione warmupem.
7. `MetricsCollector` × 3 + `DriftLogger`.
8. **Pętla strumieniowa** (do `n < effMax`):
   - Jeśli `mh.scorePid != null` i wartości skończone: `mh.scorePid.update(x, y)`,
     a jeśli `isReady` i jest ranker — `mh.scoreRanker.update(disc(x), y)`.
   - `yhat = mainModel.predict(raw)` + measurement.
   - Predykcje baseline'ów.
   - `err = (yhat==y) ? 0 : 1`; `detector.update(err, x)`.
   - `alarm`, `drifting`.
   - Update metrics, `dl.tick`.
   - Jeśli alarm: `mc.onDriftAlarm()`, `driftCount++`, `dl.onAlarm`. Wpis do
     `actionsCsv` z aktualnymi licznikami DriftAwareSRP (`getTotalKept/Surgical/Full/NoReplacement`,
     `getWeightedPredictions`, `getUnweightedFallbacks`, `|drifting|`).
   - `mainModel.train(raw, y, alarm, drifting)` — wewnątrz DriftAwareSRP wywołuje się
     `handleDrift(drifting, scoreProvider.get())` z per-learner KEEP/SURGICAL/FULL.
   - `majority.train`, `noChange.train`.
   - Co `importance_update_every=1000`: `mh.importance.update(mi, ks)` gdzie
     `ks[i] = 1 - clamp01(detector.getLastPValues()[i])`.
   - `mc.onSelectionChanged(sel)`.
   - Co `logEvery=1000`: snapshot + delta `surgDelta`/`fullDelta` od poprzedniego logu,
     `overlapStr` = `da.lastSummary.overlapCounts`, `impStr` = top-5 importance, `lwStr` = learner weights → wpis do `E3_window.csv`.
9. `dl.flushPending(n, kappa)`.
10. `RunResult` + agregacja.
11. Po wszystkich runach: `writeSummary` + `runStatistics(cfg, all)` (kappa matrix + Friedman/Nemenyi/Wilcoxon + CD export).

**Używane klasy** (poza E2):
- `thesis.models.{DriftAwareSRP, FeatureImportance, SRPWrapper, ARFWrapper}`,
- `thesis.selection.{InformationGainRanker, FilterRanker, StaticFeatureSelector, AlarmTriggeredSelector}`,
- `thesis.evaluation.StatisticalTests` (Friedman/Nemenyi/Wilcoxon).

### 4. Zapisywane wyniki

E3 produkuje **6 plików** w `results/E3/` (+ 3 pliki CD-diagramu z `exportCD`).

#### Plik `E3_window.csv`
- **Lokalizacja**: `results/E3/E3_window.csv`
- **Kiedy tworzony**: na początku `run`. Wpisy co `logEvery=1000` instancji.
- **Kolumny** (z `windowHeader()`):

| Kolumna | Znaczenie |
|---|---|
| `instance_num` | Numer instancji. |
| `dataset` | Nazwa datasetu. |
| `variant` | Nazwa wariantu (np. `DA-SRP-ABC`). |
| `seed` | Ziarno. |
| `selected_count` | Długość selekcji `mainModel.getCurrentSelection()`. |
| `selected_features` | Selekcja `1\|3\|7`. |
| `drifting_features` | Ostatnie wykryte cechy dryfujące (od ostatniego alarmu). |
| `overlap_per_learner` | Z `da.getLastSummary().getOverlapCounts()` — overlap |learner.subspace ∩ drifting| per learner. |
| `num_surgical_updates` | Delta `totalSurgical` od ostatniego logu. |
| `num_full_replacements` | Delta `totalFull` od ostatniego logu. |
| `num_kept` | `da.getTotalKept()` (cumulative). |
| `num_no_replacement` | `da.getTotalNoReplacement()` (cumulative). |
| `importance_top5` | Top-5 cech wg importance: `f:value\|f:value\|...`. |
| `learner_weights` | Wagi learnerów z `da.getLastLearnerWeights()`, `0.1234\|0.0567\|...`. |
| `accuracy_window`, `kappa_window`, `kappa_per_window` | Metryki okna. |
| `majority_baseline_window`, `nochange_baseline_window` | Baseline acc. |
| `recovery_time` | `s.lastRecoveryTime`. |
| `drift_count` | Skumulowana liczba alarmów. |
| `feature_stability` | Last feature stability ratio. |
| `ram_hours`, `throughput_inst_per_sec`, `peak_ram_mb` | Resource metrics. |

#### Plik `E3_summary.csv`
- **Lokalizacja**: `results/E3/E3_summary.csv`
- **Kiedy tworzony**: po pętli runów (`writeSummary`).
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `dataset, variant, seed` | Identyfikatory. |
| `instances` | Łącznie. |
| `accuracy, kappa, kappa_per` | Final metrics. |
| `recovery_time` | `r.recoveryTime` (-1.0 jeśli NaN). |
| `drift_count` | Liczba alarmów. |
| `avg_feature_stability`, `last_feature_stability` | Średnia + ostatnia. |
| `total_kept, total_surgical, total_full, total_no_replacement` | Cumulative liczniki DriftAwareSRP. |
| `refresh_calls` | `da.getRefreshCalls()`. |
| `weighted_predictions` | Liczba predykcji ważonych (DASRP_ABC). |
| `unweighted_fallbacks` | Liczba fallbacków do plain `srp.predictProba`. |
| `ram_hours, throughput_inst_per_sec, peak_ram_mb` | Resource. |
| `status` | `OK`/`FAIL`. |

#### Plik `E3_drifts.csv`
- **Lokalizacja**: `results/E3/E3_drifts.csv` (nazwa zbudowana z `experimentGroup + "_drifts.csv"`).
- **Kiedy tworzony**: na początku `run`, wpisy z `DriftLogger` (analogicznie do E2).
- **Kolumny**: identyczne jak w E2 — `dataset, variant, seed, alarm_at, kappa_before_500, kappa_after_500, recovery_instances, drift_type`.

#### Plik `E3_dasrp_actions.csv`
- **Lokalizacja**: `results/E3/E3_dasrp_actions.csv`
- **Kiedy tworzony**: na początku `run`, wpis na każdy alarm.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `dataset, variant, seed` | Identyfikatory. |
| `alarm_at` | Numer instancji alarmu. |
| `kept` | `da.getTotalKept()` w momencie alarmu (cumulative). |
| `surgical` | `da.getTotalSurgical()`. |
| `full` | `da.getTotalFull()`. |
| `no_replacement` | `da.getTotalNoReplacement()`. |
| `weighted_preds_total` | `da.getWeightedPredictions()`. |
| `fallbacks_total` | `da.getUnweightedFallbacks()`. |
| `drifting_features_detected` | `|drifting|` w tym alarmie. |

> Dla baseline'ów (gdzie `da == null`) wpis ma zera w polach `kept..fallbacks_total`.

#### Plik `E3_kappa_matrix.csv`
- **Lokalizacja**: `results/E3/E3_kappa_matrix.csv`
- **Kiedy tworzony**: w `runStatistics` po pętli runów (jeśli ≥2 metody i ≥2 datasety).
- **Co zawiera**: macierz `[dataset × variant]` ze średnimi κ po seedach.
- **Kolumny**: pierwsza `dataset`, kolejne — po jednej na każdy `variant`. Pusta cell = NaN.

#### Plik `E3_stats_report.txt`
- **Lokalizacja**: `results/E3/E3_stats_report.txt`
- **Kiedy tworzony**: w `runStatistics`.
- **Co zawiera** (tekst):
  - `rep.summary()` — wynik Friedmana + Nemenyi (avg ranks, χ², Iman-Davenport F, p-value, CD).
  - Macierz pairwise Wilcoxon p-values (NIE CSV, tylko ASCII tabela z przecinkami).
  - Sekcja "Targeted comparisons vs baselines" — DA-SRP* vs `SRP+S1` i DA-SRP* vs `ARF+S2`
    z tagami `WIN`/`LOSE`/`ns`/`DEGEN`.

#### Pliki CD-diagramu (`rep.exportCD(Paths.get(cfg.outputDir))`)
- **Lokalizacja**: `results/E3/avg_ranks.csv`, `results/E3/rank_matrix.csv`, `results/E3/pairwise_significance.csv`.
- **Co zawierają**: dane do narysowania CD-diagramu (NIE PNG/SVG — generowanie wykresu poza Javą).
  - `avg_ranks.csv` — średnie rankingi metod.
  - `rank_matrix.csv` — macierz rankingów per dataset.
  - `pairwise_significance.csv` — pary istotnie różniące się.

---

## E4 — `E4DriftAnalysis`

### 1. Cel eksperymentu

E4 mierzy **jakość detekcji dryftu** (Precision/Recall/F1, mean detection delay,
false alarm rate) dla każdej metody, na trzech poziomach intensywności dryftu
(`LOW`/`MEDIUM`/`HIGH`):
- czy detektor potrafi w ogóle wykryć dryft (TP),
- ile fałszywych alarmów generuje (FP),
- jak szybko reaguje (delay),
- minimalna intensywność (`min_detectable_magnitude`), przy której F1 ≥ próg
  i κ ≥ próg.

Ground truth:
- **Abrupt** (SEA, STAGGER) — punkty zmiany w `n/4, n/2, 3n/4` (SEA) lub
  `n/5, 2n/5, 3n/5` (STAGGER); alarm w oknie `±tolerance_window=500` od GT = TP.
- **Gradual/incremental** (Hyperplane, RandomRBF) — przyjmowane jako jeden onset
  na `warmup`; pierwszy alarm po warmup = TP, pozostałe = FP.
- **CustomFeatureDrift** — dodatkowo ground-truth feature-level (`featureDetection`):
  cechy `0..driftFeatures-1` powinny być w `drifting` zwracanym przez Level-2.

### 2. Konfiguracja eksperymentu

Z pliku `e4_synthetic_drift.json`:

| Parametr | Wartość |
|---|---|
| `experiment_group` | `E4` |
| `output_dir` | `results/E4` |
| `warmup` | `1500` |
| `window_size` | `1000` |
| `log_every` | `1000` |
| `ram_sample_every` | `200` |
| `importance_update_every` | `500` |
| `tolerance_window` | `500` (matching alarm ↔ GT) |
| `max_instances` | `50000` |
| `detector_delta` | `0.002` |
| `f1_threshold` | `0.5` (do `min_detectable_magnitude`) |
| `kappa_threshold` | `0.5` |
| `drift_before_window` / `drift_after_window` | `500` / `500` (pola w configu, ale `DriftLogger` w E4 dostaje `500/500` na sztywno) |
| `seeds` | `[1]` |
| `magnitudes` | `["LOW", "MEDIUM", "HIGH"]` |

**Methods (10)** — z configa, parsowane przez `buildMethodModel`:
- `HT+S1` → `E1Baselines.buildModel("HT", sBase, header, C)`,
- `ARF+S1`, `ARF+S2`, `ARF+S3` → `E2AdaptiveFS.buildSelector` + `E2AdaptiveFS.buildModel`
  (`detector="ADWIN"`, `periodicInterval=cfg.methodPeriodicInterval=500`, `wPostDrift=cfg.methodWPostDrift=300`),
- `SRP+S1`, `SRP+S2`, `SRP+S3` — analogicznie,
- `DA-SRP-A`, `DA-SRP-AB`, `DA-SRP-ABC` — przez `E3DASRP.buildModel` z hardcoded:
  `tau=0.5, w1=0.7, kswinAlpha=0.005, kswinWindow=200, detector="ADWIN", ensembleSize=10, lambda=6.0, wPostDrift=1000`.

**Detector** (`E2AdaptiveFS.buildDetector(methodOptDetector(method), d, cfg.detectorDelta)`):
- `methodOptDetector(method)` zwraca **zawsze `"ADWIN"`** — więc w E4 wszystkie
  metody używają TwoLevelDriftDetector z Level-1=ADWIN, niezależnie od ich nazwy.
- `level1Delta = 0.002`.
- Pozostałe parametry domyślne z `TwoLevelDriftDetector.Config`.

**Generators (5)** z configu (każdy `n=50000`):
- `SEA` — abrupt, 4-segment ConceptDriftStream, GT pos: `[n/4, n/2, 3n/4]`.
  - `abruptWidth(LOW)=500`, `abruptWidth(MEDIUM)=100`, `abruptWidth(HIGH)=1` — szerokość
    przejścia (mniejsza = ostrzejsze).
  - `noisePercentageOption=10`.
- `STAGGER` — abrupt, 4-segment, GT pos: `[n/5, 2n/5, 3n/5]`. Także `abruptWidth`.
- `Hyperplane` — gradual, parametr `sigma`:
  - `LOW=0.001`, `MEDIUM=0.01`, `HIGH=0.1`.
  - GT pos: `[warmup]` (jeden onset).
- `RandomRBF` — gradual, parametr `speed`:
  - `LOW=0.0001`, `MEDIUM=0.001`, `HIGH=0.01`.
- `CustomFeatureDrift` — `drift_features=5`, parametr `sigma`:
  - `LOW=0.02`, `MEDIUM=0.05`, `HIGH=0.10`.
  - `noise_features=5`, `gtFeatures = {0,1,2,3,4}`.

**Inne ustawienia**:
- `recovery_cap_fraction=0.5` — dla generatorów gradualnych jeśli `recoveryTime > 0.5 ·
  monitoredInstances`, zerowane do `-1.0` (uznajemy, że nie odzyskał).
- `methodPeriodicInterval=500`, `methodWPostDrift=300` — używane przy budowie ARF/SRP+S2/S3.
- DA-SRP zawsze ma `wPostDrift=1000` (z hardcoded fragmentu w `buildMethodModel`).

### 3. Przebieg eksperymentu

```
for each generator g:
    for each magnitude m in cfg.magnitudes:
        for each seed s:
            for each method:
                runOne(cfg, g, m, method, s, ...)
```

Krok po kroku w `runOne`:

1. `stream = buildStreamWithMagnitude(g, mag, seed)` — generator zbudowany z parametrami
   zależnymi od magnitude (np. SEA z `width = abruptWidth(mag)`); opcjonalne
   `addNoiseFeatures`.
2. Header → `d, C, FeatureSpace`.
3. Warmup `cfg.warmup=1500`.
4. **`mh = buildMethodModel(cfg, methodName, header, d, C, seed, window, labels)`** — buduje
   `mh.main`, `mh.da` (dla DASRP), `mh.selector`, `mh.baselineSelector`, opcjonalne
   `scoreUpdater` i `importanceUpdater` (lambdy aktualizujące PiD/ranker/importance dla DASRP).
5. `detector = E2AdaptiveFS.buildDetector("ADWIN", d, 0.002)`.
6. Baseline'y MajorityClass, NoChange — karmione warmupem.
7. `MetricsCollector × 3`, `DriftLogger`.
8. `gtPositions = groundTruthPositions(g, warmup)`, `gtFeatures = groundTruthFeatures(g)`.
9. **Pętla strumieniowa**:
   - Jeśli `mh.scoreUpdater != null` → wywołanie (DASRP only).
   - `yhat = main.predict(raw)`, predykcje baseline'ów.
   - `detector.update(err, x)`, `alarm`, `drifting`.
   - Update metrics + `dl.tick`.
   - Jeśli alarm: `mc.onDriftAlarm`, dodanie `AlarmEvent` (z `n` i `detectedFeatures=drifting`),
     wpis do `actionsCsv` (analogicznie do E3).
   - Jawne `mh.selector.update(x, y, alarm, drifting)` jeśli `selector != null`.
   - `main.train(raw, y, alarm, drifting)` (wrapper sam może wywołać selector.update wewnątrz).
   - Co `importance_update_every=500`: `mh.importanceUpdater.accept(detector.getLastPValues())`
     (dla DASRP).
   - `mc.onSelectionChanged(sel)`, `selChgCount` jeśli zmiana.
   - Co `logEvery=1000`: wpis do `E4_window.csv`.
10. `dl.flushPending`.
11. `r.detection = computeDetection(gtPositions, alarms, tolerance_window=500, abrupt, n)`:
    - **Abrupt**: każdy alarm matchowany do najbliższego niewykorzystanego GT w oknie
      ±tolerance; matched → TP + delay = |alarm − GT|; nie matched → FP. Niewykorzystane GT → FN.
    - **Gradual**: pierwszy alarm po `gt[0]` → TP (delay = alarm − onset); pozostałe → FP;
      brak alarmu → FN.
12. `r.featureDetection = computeFeatureDetection(alarms, gtFeatures, isCustomFeatureDrift)`:
    - Tylko dla `CustomFeatureDrift` z `gtFeatures` niepustym.
    - Dla każdego TP-alarmu: cechy z `detectedFeatures` ∈ gtFeatures → tp; ∉ → fp;
      cechy z gtFeatures ∉ detectedFeatures → fn. Stąd P/R/F1.
13. Recovery cap dla gradualnych (`r.recoveryTime > 0.5 * monitored` → -1.0).
14. Po pętli runów: `writeSummary` + `writeAggregated` + `writeMinDetectableMagnitude` + `writeRanking`.

**Używane klasy** (poza E1/E2/E3):
- `moa.streams.{ConceptDriftStream, generators.SEAGenerator, generators.STAGGERGenerator}` —
  bezpośrednio do budowania abrupt-drift streamów z parametrem `width`.

### 4. Zapisywane wyniki

E4 produkuje **8 plików CSV/TXT** w `results/E4/`:

#### Plik `E4_window.csv`
- **Lokalizacja**: `results/E4/E4_window.csv`
- **Kiedy tworzony**: na początku `run`. Wpisy co `logEvery=1000`.
- **Kolumny** (z `windowHeader()`):

| Kolumna | Znaczenie |
|---|---|
| `instance_num` | Numer instancji. |
| `generator` | Nazwa generatora (np. `SEA`). |
| `method` | Nazwa metody (np. `DA-SRP-ABC`). |
| `magnitude` | `LOW`/`MEDIUM`/`HIGH`. |
| `seed` | Ziarno. |
| `true_drift` | `1` jeśli w oknie `(n - logEvery, n]` mieści się jakieś `gtPositions[i]`, inaczej `0`. |
| `detected_drift` | `1` jeśli w tej iteracji `alarm`, inaczej `0`. |
| `drifting_features_detected` | Aktualnie wykryte cechy dryfujące, `0\|3\|7`. |
| `drifting_features_true` | Ground-truth feature set (np. `0\|1\|2\|3\|4` dla CustomFeatureDrift, puste dla pozostałych). |
| `accuracy_window`, `kappa_window`, `kappa_per_window` | Window metrics. |
| `majority_baseline_window`, `nochange_baseline_window` | Baseline acc. |
| `recovery_time` | `s.lastRecoveryTime`. |
| `drift_count` | Skumulowana liczba alarmów. |
| `selection_change_count` | Skumulowana liczba zmian selekcji. |
| `feature_stability` | Last stability ratio. |
| `ram_hours`, `throughput_inst_per_sec`, `peak_ram_mb` | Resource. |

#### Plik `E4_alarms.csv`
- **Lokalizacja**: `results/E4/E4_alarms.csv`
- **Kiedy tworzony**: na początku `run`. Wpisy w `writeAlarmsCsv(a, r)` po każdym
  zakończonym runie (gdy mamy już `computeDetection`).
- **Co zawiera**: jeden wiersz per alarm, z informacją czy to TP/FP i delay.
- **Kolumny** (z `alarmHeader()`):

| Kolumna | Znaczenie |
|---|---|
| `generator` | Nazwa generatora. |
| `method` | Nazwa metody. |
| `magnitude` | LOW/MEDIUM/HIGH. |
| `seed` | Ziarno. |
| `instance_num` | Numer instancji alarmu. |
| `is_tp` | `TP` lub `FP`. |
| `delay` | Dla TP: liczba instancji od najbliższego GT. Dla FP: -1. |
| `matched_gt` | Indeks GT, do którego dopasowano alarm (lub -1). |
| `drifting_features_detected` | `f1\|f2\|...` cechy z Level-2 (puste jeśli brak). |

#### Plik `E4_summary.csv`
- **Lokalizacja**: `results/E4/E4_summary.csv`
- **Kiedy tworzony**: po pętli runów (`writeSummary`).
- **Co zawiera**: jeden wiersz per `(generator × method × magnitude × seed)` run.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `generator, method, magnitude, seed` | Identyfikatory runu. |
| `instances` | Łącznie. |
| `d` | Liczba cech. |
| `accuracy, kappa, kappa_per` | Final metrics. |
| `recovery_time` | Z capowaniem dla gradual. |
| `avg_feature_stability, last_feature_stability` | Stability. |
| `ram_hours, throughput_inst_per_sec, peak_ram_mb` | Resource. |
| `drift_count` | Liczba alarmów. |
| `selection_change_count` | Liczba zmian selekcji. |
| `tp, fp, fn` | Stats detekcji global. |
| `precision, recall, f1, false_alarm_rate, mean_detection_delay` | Metryki global. |
| `feat_tp, feat_fp, feat_fn` | Stats detekcji feature-level (tylko CustomFeatureDrift, inaczej 0). |
| `feat_precision, feat_recall, feat_f1` | Metryki feature-level. |
| `total_kept, total_surgical, total_full, total_no_replacement` | Cumulative liczniki DASRP (0 dla nie-DASRP). |
| `weighted_predictions, unweighted_fallbacks` | Liczniki ważonych/fallback predykcji DASRP. |
| `gt_positions` | Pozycje GT, `1250\|2500\|3750`. |
| `status` | `OK`/`FAIL`. |

#### Plik `E4_drifts.csv`
- **Lokalizacja**: `results/E4/E4_drifts.csv` (`experimentGroup + "_drifts.csv"`).
- **Kiedy tworzony**: na początku `run`, wpisy z `DriftLogger`.
- **Kolumny**: identyczne jak w E2/E3 — `dataset, variant, seed, alarm_at, kappa_before_500, kappa_after_500, recovery_instances, drift_type`.
  W E4 `dataset` zawiera nazwę generatora, `variant` zawiera nazwę metody.

#### Plik `E4_dasrp_actions.csv`
- **Lokalizacja**: `results/E4/E4_dasrp_actions.csv`
- **Kiedy tworzony**: na początku `run`, wpis na każdy alarm.
- **Kolumny**: identyczne jak w `E3_dasrp_actions.csv` — `dataset, variant, seed, alarm_at, kept, surgical, full, no_replacement, weighted_preds_total, fallbacks_total, drifting_features_detected`. Dla nie-DASRP zera.

#### Plik `E4_aggregated.csv`
- **Lokalizacja**: `results/E4/E4_aggregated.csv`
- **Kiedy tworzony**: po pętli runów (`writeAggregated`).
- **Co zawiera**: agregacje per `(generator × method × magnitude)` po seedach.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `generator, method, magnitude` | Klucz agregacji. |
| `n_seeds` | Liczba seedów w tej grupie. |
| `mean_accuracy, std_accuracy` | Średnia i odchylenie z `accuracy`. |
| `mean_kappa, std_kappa` | Z `kappa`. |
| `mean_recovery, std_recovery` | Z `recoveryTime` (tylko ≥0). |
| `mean_f1, std_f1` | Z `detection.f1`. |
| `mean_precision, mean_recall, mean_false_alarm_rate, mean_detection_delay` | Z `detection.*`. |
| `mean_feat_f1, mean_feat_precision, mean_feat_recall` | Z `featureDetection.*`. |
| `mean_throughput, mean_peak_mb` | Średnia po seedach. |

#### Plik `E4_min_detectable_magnitude.csv`
- **Lokalizacja**: `results/E4/E4_min_detectable_magnitude.csv`
- **Kiedy tworzony**: po pętli runów (`writeMinDetectableMagnitude`).
- **Co zawiera**: dla każdej pary `(method, generator)` — najmniejszy magnitude
  (LOW < MEDIUM < HIGH), przy którym mean_F1 ≥ `f1_threshold=0.5` ORAZ mean_kappa ≥ `kappa_threshold=0.5`.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `method` | Nazwa metody. |
| `generator` | Nazwa generatora. |
| `minimum_detectable_magnitude` | `LOW`/`MEDIUM`/`HIGH`/`none` (gdy żaden próg nie spełniony). |
| `mean_F1_at_min` | Średnie F1 na tej minimalnej magnitudzie. |
| `mean_kappa_at_min` | Średnie κ na tej minimalnej magnitudzie. |
| `mean_recovery_at_min` | Średnie recovery na tej minimalnej magnitudzie. |

#### Plik `E4_ranking.csv`
- **Lokalizacja**: `results/E4/E4_ranking.csv`
- **Kiedy tworzony**: po pętli runów (`writeRanking`).
- **Co zawiera**: ranking metod sortowany malejąco po `mean_kappa`.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `rank` | Miejsce w rankingu (1 = najlepsze). |
| `method` | Nazwa metody. |
| `n_runs` | Liczba runów po wszystkich generators × magnitudes × seeds. |
| `mean_kappa` | Średnia κ. |
| `mean_f1` | Średnia F1 detekcji. |
| `mean_recovery` | Średnia recovery (z 0 dla -1). |
| `mean_feature_stability` | Średnia stability. |

---

## E5 — `E5Detectors`

### 1. Cel eksperymentu

E5 testuje **detektory dryftu w izolacji** — porównuje 4 detektory operujące tylko na
strumieniu błędów (bez per-feature Level-2):

- `ADWIN` — wrapper `ADWINChangeDetector`, reset po każdym alarmie.
- `HDDM_A` — `HDDMChangeDetector.ofA(driftConfidence, warningConfidence)`.
- `HDDM_W` — `HDDMChangeDetector.ofW(driftConfidence, warningConfidence, lambda)`.
- `KSWIN_GLOBAL` — `KSWINSingleFeature` na strumieniu errorów (NIE na cechach!),
  testowany co `kswin_test_every` instancji.

Hipoteza: który detektor ma najlepsze TP/FP/F1, najmniejszy delay, najmniejszy
false alarm rate? E5 dodatkowo robi **sanity checks** (`E5_sanity_checks.txt`):
- detektor nigdy nie odpalił (`WARN_NEVER_FIRED`),
- detektor odpalił "ciągle" (>1% instancji = `WARN_ALWAYS_FIRED`),
- delay = 0 mimo TP > 0 (`WARN_ZERO_DELAY` — podejrzane),
- wszystkie detektory dały identyczną liczbę alarmów (`WARN_IDENTICAL` — wszystkie się
  rozjeżdżają tak samo).

### 2. Konfiguracja eksperymentu

Z pliku `e5_detectors.json`:

| Parametr | Wartość |
|---|---|
| `experiment_group` | `E5` |
| `output_dir` | `results/E5` |
| `warmup` | `1500` |
| `window_size` | `1000` |
| `log_every` | `1000` |
| `ram_sample_every` | `200` |
| `tolerance_window` | `500` |
| `max_instances` | `50000` |
| `f1_threshold` | `0.5` |
| `recovery_cap_fraction` | `0.5` |
| `method_periodic_interval` | `500` |
| `method_w_post_drift` | `300` |
| `drift_before_window` / `drift_after_window` | `500` / `500` |
| `seeds` | `[1]` |
| `magnitudes` | `["LOW", "MEDIUM", "HIGH"]` |

**Detektory** i ich parametry (z configa, `buildAdapter`):
- `ADWIN`: `delta = adwin_delta = 0.002`.
- `HDDM_A`: `driftConfidence = 0.001`, `warningConfidence = 0.005`.
- `HDDM_W`: `driftConfidence = 0.001`, `warningConfidence = 0.005`, `lambda = 0.05`.
- `KSWIN_GLOBAL`: `windowSize = 200`, `alpha = 0.005`, `testEvery = 50` (test KS co 50 instancji,
  na strumieniu errorów; przy alarmie `promoteCurrentToReference`).

**Modele** (tylko 2): `ARF+S2`, `SRP+S2`. Parsowane przez `model.split("+")` — w obu
przypadkach selektor = `S2` = `AlarmTriggeredSelector`. Build przez
`E2AdaptiveFS.buildSelector` + `E2AdaptiveFS.buildModel`.
- `S2 wPostDrift = method_w_post_drift = 300` (uwaga: różne od E3/E4 gdzie 1000).
- `S2 periodic_interval` nie używany (S2 nie jest okresowy).

**Generators (5)** — identyczne jak w E4 (każdy `n=50000`):
- `SEA`, `STAGGER` (abrupt + `noise_features=0`),
- `Hyperplane`, `RandomRBF` (gradual + `noise_features=0`),
- `CustomFeatureDrift` (`noise_features=5`, `drift_features=5`).

**Magnitudes**: LOW/MEDIUM/HIGH (te same parametry co w E4 — używana ta sama metoda
`E4DriftAnalysis.buildStreamWithMagnitude`).

**Ważne różnice względem innych eksperymentów**:
- Detektor jest **niezależnym `DetectorAdapter`-em** (interfejs zdefiniowany
  w E5), NIE `TwoLevelDriftDetector`. To znaczy:
  - tylko global error stream, **bez per-feature KSWIN**,
  - `drifting = Set.of()` zawsze,
  - `selector.update(x, y, alarm, drifting)` zawsze dostaje pusty zbiór (S2 reaguje
    tylko na fakt alarmu, nie na konkretne cechy).

### 3. Przebieg eksperymentu

```
for each generator g:
    for each magnitude m:
        for each seed s:
            for each model model:           // ARF+S2 lub SRP+S2
                for each detector det:      // ADWIN/HDDM_A/HDDM_W/KSWIN_GLOBAL
                    runOne(...)
```

Krok po kroku w `runOne`:

1. `stream = E4DriftAnalysis.buildStreamWithMagnitude(g, mag, seed)` (reuse z E4).
2. Header → `d, C, FeatureSpace`.
3. Warmup `cfg.warmup=1500`.
4. **`baselineSel = StaticFeatureSelector(d, C)` + `initialize`** (dla MajorityClass + NoChange).
5. **`fs = E2AdaptiveFS.buildSelector(v, d, C, K)`** z `v.detector=detName`,
   `v.periodicInterval=500`, `v.wPostDrift=300`. Dla `S2` wPostDrift=max(50, 300)=300.
6. **`main = E2AdaptiveFS.buildModel(mPart, fs, header)`** (`ARFWrapper` lub `SRPWrapper`).
7. **`detector = buildAdapter(detName, cfg)`** — natywny adapter (NIE TwoLevelDriftDetector).
8. Baseline'y MajorityClass + NoChange karmione warmupem.
9. `MetricsCollector × 3`, `DriftLogger`.
10. **Pętla strumieniowa**:
    - `yhat = main.predict(raw)` + measurement.
    - Predykcje baseline'ów.
    - `err = (yhat==y) ? 0 : 1`; `detector.update(err)` (TYLKO error, NIE x).
    - `alarm = detector.isAlarm()`. `drifting = Set.of()` (puste).
    - Update metrics + `dl.tick`.
    - Jeśli alarm: `mc.onDriftAlarm`, dodanie `AlarmEvent`. (`dl.onAlarm` z `featureLevel=false`).
    - `fs.update(xv, y, alarm, drifting)`, `main.train(raw, y, alarm, drifting)`.
    - `mc.onSelectionChanged(sel)`.
    - Co `logEvery=1000`: wpis do `E5_window.csv`.
11. `dl.flushPending`.
12. `r.detection = E4DriftAnalysis.computeDetection(gtPositions, alarms, tolerance_window, abrupt, n)`
    (reuse z E4; UWAGA: brak feature-level detection — tu zawsze 0, nie jest liczone).
13. Recovery cap dla gradual.
14. Po pętli: `writeSummary`, `writeAggregated`, `writeRanking`, `writeSanityChecks`.

**Używane klasy**:
- `thesis.detection.{ADWINChangeDetector, HDDMChangeDetector, KSWINSingleFeature, DriftDetector}`,
- `thesis.experiments.E2AdaptiveFS.{buildSelector, buildModel}` (re-use builderów),
- `thesis.experiments.E4DriftAnalysis.{buildStreamWithMagnitude, computeDetection, AlarmEvent, DetectionStats, GeneratorSpec, Magnitude}` (reuse).

### 4. Zapisywane wyniki

E5 produkuje **6 plików** w `results/E5/`:

#### Plik `E5_window.csv`
- **Lokalizacja**: `results/E5/E5_window.csv`
- **Kiedy tworzony**: na początku `run`. Wpisy co `logEvery=1000`.
- **Kolumny** (z `windowHeader()`):

| Kolumna | Znaczenie |
|---|---|
| `instance_num` | Numer instancji. |
| `generator` | Nazwa generatora. |
| `detector` | Nazwa detektora (ADWIN/HDDM_A/HDDM_W/KSWIN_GLOBAL). |
| `model` | `ARF+S2` lub `SRP+S2`. |
| `magnitude` | LOW/MEDIUM/HIGH. |
| `seed` | Ziarno. |
| `true_drift` | `1` jeśli w oknie `(n - logEvery, n]` mieści się `gtPositions[i]`. |
| `detected_drift` | `1` jeśli `alarm` w tej iteracji. |
| `accuracy_window`, `kappa_window`, `kappa_per_window` | Window metrics. |
| `majority_baseline_window`, `nochange_baseline_window` | Baseline acc. |
| `recovery_time` | `s.lastRecoveryTime`. |
| `drift_count` | Skumulowana liczba alarmów. |
| `selection_change_count` | Skumulowana liczba zmian selekcji. |
| `feature_stability` | `s.featureStabilityRatio` (NaN→1.0). |
| `ram_hours`, `throughput_inst_per_sec`, `peak_ram_mb` | Resource. |

#### Plik `E5_alarms.csv`
- **Lokalizacja**: `results/E5/E5_alarms.csv`
- **Kiedy tworzony**: na początku `run`. Wpisy w `writeAlarmsCsv` po runie (gdy
  `computeDetection` ustawi `is_tp` i `delay`).
- **Kolumny** (z `alarmHeader()`):

| Kolumna | Znaczenie |
|---|---|
| `generator` | Nazwa generatora. |
| `detector` | Nazwa detektora. |
| `model` | `ARF+S2`/`SRP+S2`. |
| `magnitude` | LOW/MEDIUM/HIGH. |
| `seed` | Ziarno. |
| `instance_num` | Numer instancji alarmu. |
| `is_tp` | `TP`/`FP`. |
| `delay` | Dla TP: liczba instancji od najbliższego GT; dla FP: -1. |
| `matched_gt` | Indeks dopasowanego GT. |

> Brak kolumny `drifting_features_detected` (E5 nie ma feature-level info).

#### Plik `E5_summary.csv`
- **Lokalizacja**: `results/E5/E5_summary.csv`
- **Kiedy tworzony**: po pętli runów (`writeSummary`).
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `generator, detector, model, magnitude, seed` | Identyfikatory. |
| `instances` | Łącznie. |
| `d` | Liczba cech. |
| `accuracy, kappa, kappa_per` | Final metrics. |
| `recovery_time` | Z capowaniem dla gradual. |
| `feature_stability, last_feature_stability` | Stability. |
| `ram_hours, throughput_inst_per_sec, peak_ram_mb` | Resource. |
| `drift_count, selection_change_count` | Skumulowane liczniki. |
| `tp, fp, fn, precision, recall, f1, false_alarm_rate, mean_detection_delay` | Detection stats global. |
| `status` | `OK`/`FAIL`. |
| `gt_positions` | `1250\|2500\|3750`. |

#### Plik `E5_drifts.csv`
- **Lokalizacja**: `results/E5/E5_drifts.csv` (na sztywno, nie z `experimentGroup`).
- **Kiedy tworzony**: na początku `run`, wpisy z `DriftLogger`.
- **Kolumny**: identyczne jak w E2/E3/E4 — `dataset, variant, seed, alarm_at, kappa_before_500, kappa_after_500, recovery_instances, drift_type`.
  - **Uwaga**: w E5 `dataset` to *złożony tag* `{generator}|{magnitude}|{detector}|{model}` (z `dlTag`),
    a `variant` to nazwa modelu (np. `ARF+S2`). `drift_type` zawsze `GLOBAL` (E5 nie ma feature-level).

#### Plik `E5_aggregated.csv`
- **Lokalizacja**: `results/E5/E5_aggregated.csv`
- **Kiedy tworzony**: po pętli runów (`writeAggregated`).
- **Co zawiera**: agregacje per `(generator × detector × model × magnitude)` po seedach.
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `generator, detector, model, magnitude` | Klucz. |
| `n_seeds` | Liczba seedów. |
| `mean_kappa, std_kappa` | Z kappa. |
| `mean_recovery, std_recovery` | Z recoveryTime (≥0). |
| `mean_drift_count` | Z driftCount. |
| `mean_f1, std_f1` | Z f1. |
| `mean_precision, mean_recall, mean_false_alarm_rate, mean_detection_delay` | Detection. |
| `mean_throughput, mean_peak_mb` | Resource. |

#### Plik `E5_ranking.csv`
- **Lokalizacja**: `results/E5/E5_ranking.csv`
- **Kiedy tworzony**: po pętli runów (`writeRanking`).
- **Co zawiera**: ranking detektorów (sortowanie malejąco po `mean_f1`).
- **Kolumny**:

| Kolumna | Znaczenie |
|---|---|
| `rank` | Miejsce. |
| `detector` | Nazwa detektora. |
| `n_runs` | Liczba wszystkich runów dla tego detektora (po generators × magnitudes × seeds × models). |
| `mean_f1` | Średnie F1. |
| `mean_kappa` | Średnia κ. |
| `mean_recovery` | Średnie recovery (z 0 dla -1). |
| `mean_false_alarm_rate` | Średni FAR. |
| `mean_detection_delay` | Średni delay (z 0 dla -1). |

#### Plik `E5_sanity_checks.txt`
- **Lokalizacja**: `results/E5/E5_sanity_checks.txt`
- **Kiedy tworzony**: po pętli runów (`writeSanityChecks`).
- **Co zawiera** (tekst):
  - `WARN_IDENTICAL <key>` — gdy wszystkie detektory dla tego samego (generator|magnitude|seed|model)
    odpaliły identyczną liczbę alarmów.
  - `WARN_NEVER_FIRED <detector> on <key>` — `driftCount==0`.
  - `WARN_ALWAYS_FIRED <detector> on <key>` — alarms > 1% monitored.
  - `WARN_ZERO_DELAY <detector> on <key>` — TP>0 ale `meanDelay <= 0.01`.
  - `--- TOTALS: identical=X never_fired=Y always_fired=Z zero_delay=W ---`.
  - `--- FAILED RUNS: N ---`.
  - "Per-detector totals" — `drift_total/TP_total/FP_total` per detektor (sumy po wszystkich runach).

---

## 5. Podsumowanie

### Czym różnią się eksperymenty E1–E5

| Eks. | Główne pytanie badawcze | Modele | Selectory | Detektory | Magnitudes | Sweep |
|---|---|---|---|---|---|---|
| **E1** | Jakie są referencyjne baseline'y (κ/acc/RAM) modeli z S1? | **HT, ARF, SRP** | tylko **S1** | **brak** | — | seed × model × dataset (8 datasetów) |
| **E2** | Czy adaptacyjne FS bije S1? | ARF, SRP | **S1, S2, S3, S4** | ADWIN, HDDM_A, HDDM_W (Two-Level) | — (jeden poziom) | seed × variant × dataset |
| **E3** | Czy DA-SRP (A→AB→ABC) bije baseline'y? | SRP+S1, ARF+S2, **DA-SRP-{A,AB,ABC}** | S1, S2, identity (DA-SRP zarządza subspaces) | ADWIN (+KSWIN dla L2) | — | seed × variant × dataset |
| **E4** | Jak dobre są metody w detekcji dryftu na różnych intensywnościach? | HT+S1, ARF+{S1,S2,S3}, SRP+{S1,S2,S3}, DA-SRP-{A,AB,ABC} | wszystkie | **wszystkie** mają hardcoded ADWIN (Two-Level) | **LOW/MEDIUM/HIGH** | generator × magnitude × seed × method |
| **E5** | Który detektor (w izolacji) jest najlepszy? | ARF+S2, SRP+S2 | tylko S2 | **ADWIN, HDDM_A, HDDM_W, KSWIN_GLOBAL** (single-stream) | LOW/MEDIUM/HIGH | generator × magnitude × seed × model × detector |

### Wspólne elementy

- **Wszystkie 5** zaczytują JSON config z `src/main/java/thesis/experiments/`
  (Jackson + `Cfg.load(Path)`), z fallbackiem do `findDefaultConfig()`
  (poza E1, gdzie ścieżka domyślna jest na sztywno
  `src/main/java/thesis/experiments/E1_baselines.json` w `main(args)`).
- **Wszystkie** używają tego samego pipeline'u: warmup → init selektora →
  `MetricsCollector` × 3 (model, MajorityClass, NoChange) → pętla per-instance
  z `predict / detect / update / train`.
- **Wszystkie** używają `FeatureSpace` + `MetricsCollector` + `MajorityClassWrapper` + `NoChangeWrapper`.
- **E2-E5** używają `DriftLogger` z oknami `before=500, after=500`.
- **E2-E5** generują plik `*_drifts.csv` z **wpisami** alarmów w identycznym formacie.
  E1 też tworzy `E1_baselines_drifts.csv`, ale **tylko z headerem** (brak detektora).
- **E3, E4** generują `*_dasrp_actions.csv` w identycznym formacie.
- **E4, E5** używają tych samych `GeneratorSpec`, `Magnitude`, `AlarmEvent`,
  `DetectionStats`, `computeDetection` (E5 importuje z E4).
- **Detektor**: w E2/E3/E4 zawsze opakowany w `TwoLevelDriftDetector`; w E5
  detektor jest w izolacji (single-stream `DetectorAdapter`).

### Specyficzne dla każdego

- **E1** — jako jedyny **NIE używa detektora dryftu** w ogóle (`drift_count` zawsze = 0,
  `E1_baselines_drifts.csv` zawiera tylko header). Pipeline: predict → metrics.update → train.
  Pełni rolę **referencji** (κ/acc/RAM-h dla HT/ARF/SRP+S1) dla pozostałych eksperymentów.
  Skala identyczna z E2/E3 (`warmup=1500`, `max_instances=100000`, ARFF do EOF).
- **E2** — jako jedyny ma **plik `E2_selections.csv`** ze szczegółową historią zmian selekcji + `validation_level1.txt` z porównaniem PASS/FAIL adaptacyjnych vs S1.
- **E3** — jako jedyny robi **testy statystyczne** (Friedman/Nemenyi/Wilcoxon) w `runStatistics`, eksportuje **CD-diagram CSVs** (`avg_ranks.csv`, `rank_matrix.csv`, `pairwise_significance.csv`) i `E3_kappa_matrix.csv` + `E3_stats_report.txt`.
- **E4** — jako jedyny ma **ground truth alarmy** (pozycje zmian + drifting features dla CustomFeatureDrift), liczy **TP/FP/FN/precision/recall/F1/delay/false_alarm_rate** + dodatkowe agregacje (`E4_aggregated.csv`, `E4_min_detectable_magnitude.csv`, `E4_ranking.csv`). Sweepuje **magnitudes** dla pełnej macierzy detekcyjnej.
- **E5** — jako jedyny używa **single-stream `DetectorAdapter`** (nie Two-Level), porównuje **4 typy detektorów** (w tym KSWIN_GLOBAL na strumieniu errorów), generuje **`E5_sanity_checks.txt`** z heurystycznymi alertami (NEVER_FIRED, ALWAYS_FIRED, ZERO_DELAY, IDENTICAL).

### Notatka o brakach informacji w kodzie

- **E2**: domyślne parametry `TwoLevelDriftDetector.Config` (poza `level1Delta` i `level1Type`)
  nie są ustawiane z JSON-a — pochodzą z konstruktora `Config(numFeatures)` (wartości
  patrz `TwoLevelDriftDetector.java`, **NIE są dokumentowane w samym configu E2**).
- **E3**: `level1Delta` w `buildDetectorWithKswin` jest **hardcoded `0.002`**, ignoruje `cfg.detectorDelta` (zachowanie różne od E2).
- **E4**: `methodOptDetector(method)` zawsze zwraca `"ADWIN"` — pomimo że w E2 i E3 `detector` jest sweepowalne, **w E4 dla każdej metody używany jest ADWIN** (kod TODO zostawia wyjaśnienie wkomentowane).
- **E4 / E5**: pola `drift_before_window` i `drift_after_window` z configu są czytane do `Cfg`, ale w E4 do `DriftLogger` przekazywane jest na sztywno `500/500`; w E5 używane są wartości z configu.
- **E5 `E5_drifts.csv`**: kolumna `dataset` zawiera złożony tag `generator|magnitude|detector|model`,
  co jest **niespójne** z E2/E3/E4 (gdzie to czysta nazwa datasetu).

