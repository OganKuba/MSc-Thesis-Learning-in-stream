# METHODOLOGY.md — analiza metodologiczna projektu

> Kompanion do `CODE_MAP.md`. Tam masz „co jest w kodzie", tu masz „dlaczego to tak działa".
> Pisane w tonie mentora — z naciskiem na intuicję, a nie na opis klas.

---

## 1. High-level idea systemu

To jest **system uczenia się ze strumienia danych z adaptacyjną selekcją cech i wykrywaniem dryftu konceptu** — czyli klasyfikator online, który nie tylko uczy się instancja po instancji, ale potrafi też rozpoznać:

- **kiedy świat się zmienił** (concept drift na globalnym strumieniu błędów),
- **co konkretnie się zmieniło** (które cechy zaczęły zachowywać się inaczej),
- **jak na to zareagować** (przebudować selekcję cech, zresetować część modelu, podmienić podprzestrzenie ensembla).

Jeden zdaniem: to *pipeline*, w którym **detekcja dryftu napędza selekcję cech**, a **selekcja cech napędza model** — wszystko w trybie strumieniowym, z minimalnym zużyciem pamięci.

### Jaki problem rozwiązuje

W klasycznym ML zakładamy, że dane są i.i.d. — pochodzą z jednego, stabilnego rozkładu. W realnych strumieniach (ceny akcji, ruch w taxi, telemetria) to **zawsze fałsz**: rozkład zmienia się sezonowo, gwałtownie albo dryfuje powoli. Klasyczny model trenowany na pełnym datasetcie:

1. nie nadąża pamięciowo (dane są nieskończone),
2. „pamięta" stare wzorce, które już nie obowiązują,
3. nie wie, że pewne cechy stały się nieinformatywne (a inne nagle stały się ważne).

Ten projekt celuje dokładnie w te trzy bóle.

### Główna intuicja

> **Selekcja cech to nie jest jednorazowy krok preprocessingu — to powinien być proces ciągły, sterowany sygnałem dryftu.**

Cała architektura jest zbudowana wokół tej obserwacji. Jeśli rozkład cechy `f7` nagle się zmienia (np. nowa polityka cenowa Ubera), to:

1. test KS na sliding window dla `f7` da niskie p-value,
2. system to zauważy (poziom 2 detekcji),
3. obniży zaufanie do `f7` w wektorze importance,
4. selektor cech (S3/S4) wyrzuci `f7` lub podmieni go w top-k,
5. model (np. `DriftAwareSRP`) usunie drzewa, których podprzestrzeń mocno polegała na `f7`.

Cała reszta to inżynieria tej intuicji.

---

## 2. Metodologia — fundament teoretyczny

### Charakter podejścia

System jest **hybrydą siedmiu paradygmatów**, każdy odpowiada za inny kawałek:

| Komponent | Paradygmat | Po co |
|---|---|---|
| **ADWIN / HDDM** | probabilistyczny (granice typu Hoeffdinga) | Wykrycie globalnej zmiany w strumieniu błędów ze statystyczną gwarancją |
| **KSWIN + BH-FDR** | **statystyczny nieparametryczny** (test Kołmogorowa-Smirnowa) | Wykrycie zmian w rozkładzie pojedynczych cech, bez założeń o rodzinie rozkładu |
| **IG / MI / χ²** | teoria informacji + statystyka | Ocena, *ile* dana cecha mówi o klasie — w trybie online, na podstawie tabel kontyngencji |
| **PiD (dyskretyzacja)** | heurystyka (greedy merge) | Sprowadzenie cech ciągłych do dyskretnych binów, żeby w ogóle dało się liczyć IG/MI/χ² |
| **DriftAwareSRP / DA-ARF** | meta-uczenie ensemblowe | Per-learner KEEP/SURGICAL/FULL (DA-SRP) i KEEP/FULL z background-learner (DA-ARF) — heurystyka decyzji, kogo „resetować" po dryfcie |
| **Reservoir sampling (Efraimidis-Spirakis)** | losowy próbkowanie ważone | Generowanie nowych podprzestrzeni cech proporcjonalnie do ich ważności (z opcjonalnym `importancePower` ostrzącym i `samplingBeta` mieszającym w stronę uniformu) |
| **Top-K rank-weighted voting** | rangowe agregowanie predykcji | W DA-SRP/DA-ARF głosują tylko najlepsze `⌈topKFraction·N⌉` learnery z descending wagami `(K−r)`; w DA-SRP-ABC dodatkowo `correctionAlpha · importance` modyfikuje rozkład probabilistyczny |
| **Friedman + Nemenyi + Wilcoxon (Holm)** | testowanie wielokrotne | Porównanie wariantów across `datasets × seeds`; CD-diagram + paired Wilcoxon z Holm-korektą p-wartości |

### Założenia bazowe

1. **Strumień jest *prequential*** — najpierw przewidujemy, potem widzimy etykietę, potem aktualizujemy. Nigdy nie patrzymy w przyszłość.
2. **Pamięć ograniczona** — każda struktura ma stałą lub logarytmiczną pamięć (windows, ADWIN-owe okno adaptacyjne, fixed-size histogramy).
3. **Cechy są warunkowo niezależne dla potrzeb rankingu** — IG/MI liczone *per cecha* (filter approach), bez modelowania interakcji. To świadome uproszczenie: szybkość kontra dokładność.
4. **Etykieta jest dostępna natychmiast** — to klasyczne założenie *prequential evaluation*. W realu często fałszywe (delayed labels), ale tu pomijane.
5. **Dryft może być abrupt LUB gradual** — dlatego są dwa typy detekcji: ADWIN (zmiana wartości średniej) i KSWIN (zmiana całego rozkładu).

### Co zbierane, dlaczego

| Co zbierane | Po co |
|---|---|
| `joint[f][b][c]` — tensor 3D częstości (cecha × bin × klasa) | Z tego wyliczamy *każdy* score: IG, MI, χ². Jedna struktura, trzy różne rankery |
| Sliding window wartości cechy (KSWIN) | Test KS porównuje rozkład „stary vs nowy" → bez założenia rodziny rozkładu |
| Strumień błędów 0/1 (ADWIN/HDDM) | Globalny sygnał: czy mój model traci celność? |
| `FeatureImportance` (MI + KS) | Wektor wagowy do ważonego głosowania w ensemblu i do próbkowania nowych podprzestrzeni |
| Histogramy Layer-1 (PiD) | Surowy rozkład wartości cechy — punkt wyjścia do dyskretyzacji adaptacyjnej |

---

## 3. Model danych — jak system reprezentuje wiedzę

### Centralna struktura: tensor kontyngencji

```
joint[F][B][C]  ←  liczby (zmiennoprzecinkowe, bo decay)
   F = liczba cech
   B = liczba binów (po dyskretyzacji)
   C = liczba klas
```

`joint[f][b][c]` = ile razy widzieliśmy: cechę `f` w binie `b` z etykietą `c`.

Wszystko inne (marginalne, totale) to projekcje tego tensora:

- `featureBinTotals[f][b] = Σ_c joint[f][b][c]` — ile razy `f` wpadło w bin `b`
- `featureClassMarginals[f][c] = Σ_b joint[f][b][c]` — rozkład klasy widziany przez cechę `f`
- `featureTotals[f] = Σ_b Σ_c joint[f][b][c]` — ile próbek ma cecha `f` w sumie

**Interpretacja matematyczna:** `joint[f][b][c] / featureTotals[f]` to *empiryczny estymator* wspólnego rozkładu `P(X_f = b, Y = c)`. Cały ranking opiera się o ten estymator.

### Dyskretyzacja PiD (Partition Incremental Discretization)

Po co dwie warstwy?

- **Layer 1** — bardzo drobny histogram (`b1=100` binów). To „surówka" rozkładu cechy.
- **Layer 2** — gruboziarnista mapa (`b2=10`), zbudowana **przez chciwe sklejanie sąsiednich binów Layer-1** tak, żeby sąsiedzi mieli *podobne rozkłady warunkowe klasy* (TV-distance + Laplace smoothing).

**Intuicja:** Layer 1 mówi „gdzie są wartości", Layer 2 mówi „gdzie są *ważne* progi klasyfikacyjne". Co `recomputeEvery=1000` instancji Layer 2 jest przebudowywane — to jest sposób, w jaki PiD adaptuje się do dryftu *bez* odrzucania danych historycznych.

> Limitacja: **min/max Layer-1 są ustawiane w warmup-ie i nie zmieniają się**. Jeśli rozkład cechy „ucieknie" poza ten zakres, PiD nie zauważy. To bug-watch.

### Wektor importance

```
importance[f] = w1 · normalizedMI[f]  +  w2 · (1 / (KS[f] + ε))
                ↑ relevance              ↑ stability
                (default w1=0.7)         (default w2=0.3)
```

Dwa składniki:

1. **Relevance** — „ile cecha mówi o klasie" (z MI, znormalizowane do max).
2. **Stability** — „jak bardzo cecha jest spokojna" (odwrotność statystyki KS — im mniejszy dryft, tym wyższa waga).

To jest **klucz do zrozumienia DriftAwareSRP/DA-ARF**: nie chcemy cech, które są tylko informatywne (te się zaraz zmienią pod nogami), ani tylko stabilnych (te mogą być nieinformatywne). Chcemy iloczynu cech *informatywnych i stabilnych*.

### Ostrzenie i regularizacja importance

Przed podaniem `importance` do reservoir samplera dwa knoby modulują rozkład:

```
weights[f] = (1 − samplingBeta) · importance[f]^importancePower  +  samplingBeta · uniform[f]
```

- **`importancePower` (default 2.0)** — wykładnik *sharpening*. Wartości > 1 wzmacniają różnice między cechami (większa koncentracja na top-cechach), wartości < 1 wypłaszczają rozkład. To jest soft-temperature; dla `power=1` mamy proporcjonalność.
- **`samplingBeta` ∈ [0,1] (default 0.7)** — blend w stronę uniformu. `beta=0` → tylko importance; `beta=1` → uniformowe próbkowanie (tracimy informację o ważności). `0.7` to świadomy kompromis: 30% wagi z importance, 70% z uniform → eksploracja nie ginie, ale prior działa.

> Intuicja: bez `samplingBeta` ensemble degeneruje do replikatów top-`subspaceSize` cech we wszystkich learnerach. Bez `importancePower` różnice w MI rzędu 0.01 vs 0.05 są praktycznie niewidoczne dla samplera. Razem dają „kontrolowaną zachłanność".

---

## 4. Flow algorytmu — co się dzieje, gdy przyjdzie próbka

### Faza warm-up (pierwsze ~1500 instancji)

1. Drainujemy `warmupSize` instancji do bufora.
2. Budujemy `FeatureSpace` (mapa indeksów cech ↔ indeksy atrybutów MOA).
3. Karmimy nimi PiD — Layer 1 dostaje `min/max ± 5% margines`.
4. Rankery (IG/MI/χ²) dostają zdyskretyzowane warmup-okno → liczą pierwsze score'y.
5. `selector.initialize(window, labels)` → wybierany jest **pierwszy top-k** (`k = ⌈√F⌉` domyślnie).
6. `FeatureImportance` jest seedowane: MI z rankera, KS = 0 (bo nie ma jeszcze sygnału dryftu).

System jest teraz „gotowy".

### Pętla per-instancja

```
Instance raw  ──►  x = extractFeatures(raw),  y = raw.classValue()
                       │
                       ├── (opcjonalnie) PiD.update(x, y); fullRanker.update(disc(x), y)
                       │
                       ├── ŷ = model.predict(raw)
                       ├── error = (ŷ == y) ? 0 : 1
                       │
                       ├── detector.update(error, x)
                       │     │
                       │     ├── Level 1 widzi `error` → ADWIN/HDDM
                       │     └── Level 2 widzi `x[]`   → per-feature KSWIN
                       │
                       ├── if (alarm): {
                       │       drifting = level2.getDriftingFeatures()  // BH-FDR-corrected
                       │       PiD.resetFeature(d) for d in drifting
                       │       importance.update(MI_scores, 1 - p_values)
                       │       if (model is DriftAwareSRP):
                       │           model.handleDrift(drifting, fullRanker.scores)
                       │   }
                       │
                       ├── model.train(raw, y, alarm, drifting)
                       │     └── wrapper wywołuje selector.update(...) wewnątrz
                       │
                       └── metrics.update(y, ŷ, elapsedNanos)
```

### Kiedy system jest „gotowy"

- **PiD jest ready**, gdy *każda* cecha zebrała ≥ `warmupN=500` próbek.
- **Ranker jest ready**, gdy `featureTotals[f] >= minSamplesReady=50` dla każdej `f` (świeży konstrukt — patrz `AbstractFrequencyRanker.isReady()`).
- **Detektor poziomu 2 jest ready**, gdy każdy KSWINSingle ma wypełnione okno referencyjne i bieżące.

Dopóki nie są ready, ich wyjścia są ignorowane / rollback do uniform.

### Jak powstaje wynik

Końcowy „wynik" to:

- **per-instancja** — predykcja `ŷ` i metryki (κ, accuracy, RAM-h, recovery time, stability ratio),
- **per-run** — wiersz w `summary.csv` lub `RecordingMetrics`-owy CSV co `sampleEvery` instancji,
- **per-eksperyment** — `validation_level1.txt` (czy nasz wariant bije baseline-y MajorityClass + NoChange).

---

## 5. Scoring i ranking — serce „filter" approach

### Trzy score'y, jedna struktura

Wszystkie trzy rankery operują na *tym samym* tensorze `joint[F][B][C]`:

| Score | Wzór (intuicyjnie) | Co mierzy |
|---|---|---|
| **Information Gain** | `H(Y) − H(Y\|X_f)` | Ile niepewności o klasie *redukuje* znajomość cechy |
| **Mutual Information** | `Σ p(x,y) · log(p(x,y) / (p(x)p(y)))` | Wzajemna informacja między cechą a klasą (w natach) |
| **Chi-Squared** | `Σ (obs − exp)² / exp` | Jak bardzo zaobserwowana tabela odbiega od niezależności |

Matematycznie IG i MI są **niemal tożsame** (różnica głównie w bazie logarytmu). Różnica jest praktyczna: IG jest zwyczajowo używane w drzewach decyzyjnych, MI jest bardziej „information-theoretyczne". χ² jest stricte statystyczny — ma rozkład pod H₀ i można z niego liczyć p-value, ale tu używany jako *score*, nie jako test.

### Wysoki vs niski score

| Score | Wysoki znaczy | Niski znaczy |
|---|---|---|
| IG | Cecha mocno tnie niepewność klasy | Cecha mówi tyle, co losowanie |
| MI | Silna zależność cecha-klasa | Brak zależności (≈ 0 dla niezależnych) |
| χ² | Tabela mocno odbiega od niezależności | Tabela ≈ taka jak przy niezależności |

### Świadome konsekwencje wyboru filter approach

1. **Plus:** Score liczony niezależnie od modelu → szybko, ranker można podmieniać bez retreningu.
2. **Plus:** Działa online i z decay'em → adaptuje się do dryftu naturalnie.
3. **Minus:** **Ignoruje interakcje cech.** Cecha może mieć MI ≈ 0 sama, ale być kluczowa razem z drugą (XOR). Filter tego nie zobaczy.
4. **Minus:** **χ² ma bias proporcjonalny do liczby binów** — cechy bardziej rozdrobnione w PiD wyjdą sztucznie wyżej. (Dlatego dyskretyzacja Layer 2 robi `b2=10` dla wszystkich.)
5. **Minus:** **MI/IG mają bias proporcjonalny do *empirycznej* entropii** — przy małych próbkach są przeszacowywane.

**Decyzja architekta:** używaj IG jako default (najbardziej znana intuicja, min wpływ skali); χ² jest dostępny do experymentów typu „a co, jeśli...".

### Selekcja top-k z preferencjami

`selectTopK(k, preferredOrder, tieEpsilon)` — to jest piękna heurystyka:

- Sortujemy descending po score.
- **W przypadku remisu** (różnica < `tieEpsilon`) — wygrywa cecha wcześniej w `preferredOrder`.
- `preferredOrder` to zwykle *poprzednia selekcja* — to jest mechanizm **stabilizacji selekcji**, żeby cechy nie skakały co iterację dla σ-ε różnic w score.

Bez tego mielibyśmy „flickering" selektora i destabilizację modelu (`HoeffdingTreeWrapper` resetuje drzewo na zmianę selekcji).

---

## 6. Decyzje, heurystyki i ich uzasadnienie

### UNKNOWN_BIN = -1

Co jeśli wartość cechy wpadnie poza zakres Layer-1? PiD zwraca `UNKNOWN_BIN`. Ranker **pomija** taką próbkę dla tej cechy (ale liczy dla innych). To jest *graceful degradation* — nie crashujemy, tylko ignorujemy informację, której nie umiemy zaklasyfikować.

```java
if (b == UNKNOWN_BIN) continue;  // skip update for this feature
```

`rejectedSamples` liczy, ile takich pominięć. To jest *health metric* — jeśli rośnie, znaczy że PiD nie nadąża za rozkładem.

### Mała liczba próbek — `minSamplesReady`

Ranker uznaje się za gotowy dopiero, gdy każda cecha ma ≥ 50 próbek. To jest **regulamin proti przedwczesnym decyzjom** — ranking po 5 próbkach jest losowy.

### Szum — Benjamini-Hochberg FDR correction

W Level-2 mamy F testów KS równolegle. Naiwne „odrzuć jeśli p < α" prowadzi do `αF` fałszywych alarmów na 100 cech. BH koryguje to na **False Discovery Rate** = oczekiwana frakcja błędnych odrzuceń wśród *odrzuconych*. Domyślnie `bhQ = 0.10`.

To jest klasyczna higiena multi-test, ale w kontekście strumienia — zaskakująco rzadko spotykana.

### Decay — exponential forgetting

`decay(factor)` mnoży wszystkie liczby w `joint` przez `factor ∈ (0,1]`. Po `n` zastosowaniach próbka sprzed `n` ma efektywną wagę `factor^n`.

> Praktycznie: `factor = 0.999` co update → próbka po 1000 instancji ma wagę ~0.37 (1/e).

Po co? **Stare obserwacje powinny się wycofywać same**, bez ostrego resetu. Reset cech (`resetFeature(f)`) jest „cięciem mieczem" — robimy go tylko, gdy KSWIN krzyknie, że ta konkretna cecha dryfowała. Decay jest „cięciem nożycami" — łagodne, ciągłe.

### τ (tau) — próg surgical vs full w DriftAwareSRP

```
overlap = |learner.subspace ∩ drifting_features|
overlap_fraction = overlap / |learner.subspace|

if overlap == 0           → KEEP   (ten learner nie dotyka dryfujących cech)
if overlap_fraction < τ   → SURGICAL (podmień TYLKO te cechy)
else                      → FULL    (resetuj cały learner, nowa podprzestrzeń)
```

Domyślnie `τ = 0.5`. Intuicja: **jeśli dryft dotknął więcej niż połowy cech learnera, to tego learnera już nie naprawimy — tańszy jest reset niż łatanie**.

### DA-ARF — dwa kanały adaptacji

`DAARFWrapper` używa **innego** mechanizmu niż DA-SRP, świadomie. Mamy dwa równoległe kanały:

1. **Kanał wewnętrzny — per-learner ADWIN-y na błędzie treningowym.**
   Każde drzewo ma własną parę ADWIN-ów: jeden z `warningDelta=1e-4` (próg ostrzeżenia), drugi z `driftDelta=1e-5` (próg dryftu). Na *warning* spawnujemy **background learner** z nową, ostrzejszą próbką importance-weighted; na *drift* background podmienia foreground (lub pełny reset jeśli nie było tła).
2. **Kanał zewnętrzny — globalny alarm z `TwoLevelDriftDetector`.**
   Gdy strumień błędów odpala globalny alarm, `train(..., driftAlarm=true, drifting={...})` wykonuje **KEEP/FULL pass**: learnery z `subspace ∩ drifting = ∅` zostają (KEEP); pozostałe są pełnym reset-em z nową podprzestrzenią, która *unika* cech dryfujących (`sampleSubspace(avoid=drifting)`).

> Różnica vs DA-SRP: domyślnie **brak SURGICAL** — DA-ARF albo nie rusza (KEEP), albo resetuje całkowicie (FULL).
>
> **AKTUALIZACJA (diagnoza):** tryb `SURGICAL` dla DA-ARF *zaimplementowano* (przebudowa `reducedHeader` przy zachowaniu drzewa, swap tylko typo-zgodny) — więc nie jest „fikcją". Ale runy diagnostyczne pokazały, że **empirycznie szkodzi** (NYCTaxi κ 0.81→0.71): głębsze drzewo trenowane na starej podprzestrzeni gubi się po podmianie cech. Dlatego domyślny tryb to `RESET`. Osobno okazało się, że external layer prawie się nie odpala na syntetykach (całą pracę robi intrinsic ADWIN przez promocję background). Prawdziwa słabość DA-ARF leżała gdzie indziej — w konfiguracji drzew i szerokości podprzestrzeni (patrz 13.2).

### top-K rank-weighted voting (Component C)

W DA-SRP-ABC / DA-ARF predykcja **nie** jest średnią po całym ensemblu:

```
1. ranking[i] = sliding-window accuracy of learner i  (last accWindow=1000 instances)
2. K = ⌈topKFraction · N⌉                              (np. 5 z 10)
3. top-K learners głosują wagami w_r = (K − r), r ∈ [0..K-1]
4. probs[c] = Σ_{i∈topK} w_r(i) · learner[i].predictProba()[c]
5. (opcjonalnie, DA-SRP-ABC) probs[c] += correctionAlpha · importance_blend[c]
   gdzie correctionAlpha jest auto-clip-owane do maxBlendAlpha
```

Intuicja: w ensemblu po dryfcie część learnerów jest „dezorientowana" (jeszcze nie zdążyła się przebudować). Plain average je włącza i degraduje wynik. Top-K rank-weighted głosowanie **dyskwalifikuje** najsłabszych i **wzmacnia** najlepszych — bez konieczności ich resetowania.

`correctionAlpha` (DA-SRP-ABC only) wpisuje dodatkowy bias proporcjonalny do feature-importance — to jest *prior* mówiący „nawet jeśli wszystkie learnery są zgubione, pamiętaj, które cechy są ważne". Wartości > `maxBlendAlpha=0.5` są przycinane, żeby prior nie zdominował głosowania.

### Min tenure i max swap rate

`PeriodicSelector` ma `maxSwapsPerCycle = ⌈0.3·k⌉` i `minTenure`. To znaczy:

- W jednym cyklu **max 30% selekcji wymieniamy**,
- każda cecha musi „zostać" minimum `minTenure` instancji.

To jest jawne ograniczenie destabilizacji — wymiana całej selekcji raz na 1000 instancji = chaos w modelu.

---

## 7. Dynamika systemu — uczenie online + zapominanie

### Czy uczy się online?

Tak, **w pełni online**:
- żadna struktura nie wymaga pełnego datasetu w pamięci,
- każda aktualizacja jest O(F·log B) lub lepiej,
- nie ma „epok" — jest jeden przebieg.

### Jak zapomina?

Trzy mechanizmy, **na różnych skalach czasowych**:

| Mechanizm | Skala czasowa | Co zapomina |
|---|---|---|
| **Sliding window** (KSWIN, CohenKappa, RecoveryTime) | krótka (rzędu `windowSize` ≈ 1000) | Stare wartości cech / etykiet wypadają poza okno |
| **Decay multiplikatywny** (`AbstractFrequencyRanker.decay`) | średnia, kontrolowana parametrem | Stare obserwacje wykładniczo tracą wagę |
| **Hard reset na alarm** (`resetFeature`, `resetLearning`, `handleDrift FULL`) | natychmiastowa | Pełne zerowanie struktur dla cech z dryftu |
| **ADWIN adaptive window** (wewnątrz MOA) | adaptacyjna | Okno samo się obcina, gdy wykryje zmianę średniej |

### Jak system zmienia się w czasie

Patrząc na *wektor selekcji* w czasie:

```
t=0       t=1500   t=warmup_end           t=alarm_1            t=periodic_recompute
[ - ]  →  [3,7,2,9,5]  ────────stable────►  [3,7,2,9,5]  ──drift──►  [3,7,15,9,12]
                                            (drifting=[2,5])         (S3/S4 podmienia 2 i 5)
```

W S1 wektor jest zamrożony po warmup-ie. W S2 zmienia się co `periodN`. W S3 — tylko po alarmie. W S4 — i regularnie, i po alarmie (z preferencją do wymiany dryfujących).

---

## 8. Architektura runnera i analiza statystyczna

### `UnifiedStreamExperimentRunner` — jeden runner dla całego E1–E5

Cała macierz eksperymentów (E1 baselines, E2 adaptive FS sweep, E3 DA-SRP/DA-ARF ablation, E4 high-dynamics drift, E5 detectors comparison) jest zdefiniowana w **jednym pliku** `master_experiments.json`. Runner czyta go i rozwija pętlę:

```
for each block ∈ {E1..E5}:
  for each dataset ∈ block.datasets:
    for each variant ∈ block.variants:
      for each seed ∈ cfg.seeds:           # default [1,2,3,4,5]
         spawn WorkItem(block, dataset, variant, seed)
```

`WorkItem`-y są wrzucane do `ExecutorService` z `numThreads=12` workerami. Każdy worker buduje *własny* stream + selector + model + detector + `MetricsCollector` + `RunDetailedRecorder` i puszcza prequential loop w izolacji. Wszystkie wyniki (`RunArtifacts`) lecą do `ConcurrentLinkedQueue`, którą po zamknięciu poola sortujemy deterministycznie i piszemy single-threaded.

> **Decyzja architekta:** żadnego współdzielonego mutable state w hot-pathu, żadnych lock-ów. Worker albo wraca z `status=OK` i kompletem artefaktów, albo z `status=FAIL` i stringiem błędu — runner się **nie wywala na pojedynczym FAIL-u**, tylko zapisuje go do `runs_raw.csv` jako wiersz statusu. To pozwala dokończyć 919 OK runów nawet gdy 1 z 920 ma problem.

### Co dokładnie jest mierzone — `RunDetailedRecorder`

Oprócz finalnego wiersza w `runs_raw.csv` każdy run zapisuje sześć typów detalicznych zdarzeń, agregowanych potem do per-blockowych CSV w `results/<block>/`:

| Plik | Granularność | Co tam jest |
|---|---|---|
| `windows.csv` | co `windowSize=1000` instancji | accuracy, κ, κ_per, temporal_κ, RAM-h, peak_MB, throughput, drift_count_in_window |
| `drift_alarms.csv` | każdy alarm globalny | instance_index, drifting_features list, num_drifting, error_at_alarm, window_acc before/after |
| `feature_selections.csv` | każda zmiana selekcji | trigger_type (warmup/periodic/alarm), selected_features, changed_features, jaccard_to_previous, stability_ratio |
| `feature_importance.csv` | co alarm + warmup | snapshot wektora importance per feature, rank, is_selected, is_drifting |
| `recovery_time.csv` | każdy alarm | drift_id, recovery_length, baseline_accuracy_before, max_drop, area_under_recovery_curve |
| `adaptation_events.csv` | każdy DA-SRP/DA-ARF event | kept/surgical/full/no_replacement counts (DA-SRP), ext_keep/ext_full (DA-ARF) |

To są **surowe dane do tezy** — z nich robi się wykresy i tabele. Plus jeden zagregowany `master_summary.csv` (mean ± std po seedach).

### Per-block statistical analysis (`BlockStatisticalAnalysis`)

Dla każdego bloku eksperymentu osobno, dla każdej metryki (kappa, accuracy, ram_hours_gb, throughput…) odpalany jest pełen pipeline testowy:

1. **Budowa macierzy** `datasets × variants` z wartościami uśrednionymi po seedach.
2. **Friedman test** — czy *którakolwiek* para wariantów różni się istotnie. Statystyka χ² + Iman-Davenport F.
3. **Nemenyi post-hoc** — pairwise comparison z critical-difference threshold `CD = q_α · √(k(k+1)/(6N))`. Eksport do `cd_diagram_*.csv` (wizualizacja w Pythonie).
4. **Wilcoxon signed-rank** — pary wariantów over wszystkie `(dataset, seed)` próbki, czyli `N_datasets · N_seeds` punktów per para.
5. **Holm step-down** — korekta p-values *w obrębie metryki*, kontrola family-wise error rate.

> Friedman jest **omnibus**: mówi „coś jest różne, gdzieś". Nemenyi pokazuje gdzie *średnie rangi* się różnią (uśrednione po datasetach). Wilcoxon-Holm daje per-parę porównanie z większą mocą (bo działa na surowych wynikach, nie na rangach). Trzy testy to nie redundancja — to **trzy poziomy ostrości**: omnibus → ranking → pairwise.

### Defensywność pomiarów

Pomiary RAM-h opierają się o `Runtime.totalMemory() − Runtime.freeMemory()`, co przy 12 wątkach w jednym JVM-ie i aktywnym GC może momentalnie dawać wartość ujemną (dwa nie-atomiczne odczyty). `RAMHours.sample(long)` **clampuje** takie próbki do 0 (zamiast rzucać wyjątkiem) i loguje pojedynczy warning per instance. To jest **świadoma rezygnacja z kawałka precyzji** w zamian za stabilność: pojedyncza patologiczna próbka 1× na 1000 i tak ginie w trapezoidalnej integracji.

---

## 9. Ograniczenia i biasy metodologii

### Kiedy działa dobrze

- **Strumień ma realny dryft** — system jest dla niego zaprojektowany, statystyczne testy wykryją go z gwarancją.
- **Cechy są niezależne lub słabo skorelowane** — filter ranker się tu nie myli.
- **Klas niewiele (≤ 10)** — tensor `joint` rośnie liniowo z `C`.
- **Cechy ciągłe lub porządkowe** — PiD je sensownie zdyskretyzuje.

### Kiedy może zawodzić

1. **Interakcje cech (XOR-like)** — filter ranker da im zerowy MI, system je odrzuci. Lekarstwem byłby wrapper approach (kosztowny).
2. **Range drift cech** — Layer 1 w PiD ma sztywne min/max ustawione w warmup. Po dryfcie wartości mogą uciec poza zakres → wszystko ląduje w `UNKNOWN_BIN`.
3. **Cechy kategoryczne o wysokiej kardinalności** — PiD je „roztopi" w ciągłej skali, χ² się rozjedzie.
4. **Bardzo szybki dryft (co ~100 instancji)** — okno KSWIN nie zdąży się napełnić; ADWIN się resetuje cały czas.
5. **Imbalanced classes** — `featureClassMarginals` zdominowane przez majority class → MI/IG zaniżone.
6. **Delayed labels** — system zakłada natychmiastową dostępność `y`. Jeśli etykieta przychodzi z opóźnieniem 100 instancji, *prequential* się rozjeżdża.

### Potencjalne biasy

- **χ² faworyzuje cechy o dużej liczbie binów** — w tym kodzie wszystkie mają ten sam `b2=10`, więc bias zniwelowany, ale gdyby ktoś podmienił PiD — wraca.
- **MI faworyzuje cechy o wysokiej entropii brzegowej** — cecha „bogata" wygląda lepiej niż cecha „uboga", nawet jeśli obie mówią tyle samo o klasie.
- **`FeatureImportance` z domyślnym `w1=0.7, w2=0.3`** — relevance dominuje stability. Dla strumienia z dużym dryftem warto odwrócić.
- **`tau=0.5` w DriftAwareSRP** — agresywny próg, wiele learnerów leci do FULL. Konserwatywne `tau=0.8` zachowałoby więcej wiedzy, ale wolniej adaptowało.
- **Filter approach jest *model-blind*** — nie wie, że Twoje SRP woli pewne cechy. Ranking może być teoretycznie poprawny, a praktycznie suboptymalny dla konkretnego klasyfikatora.

---

## 10. TL;DR

> Strumieniowy klasyfikator z dwupoziomową detekcją dryftu (ADWIN/HDDM globalnie + per-cechowy KSWIN z poprawką BH lokalnie). Cechy ciągłe są dyskretyzowane przez dwuwarstwowe PiD do tensora kontyngencji `[F][B][C]`, z którego liczone są score'y filter (IG/MI/χ²). Selektor (S1/S2/S3/S4) na różne sposoby reaguje na sygnał dryftu — od „nigdy" (S1) do „okresowo + na alarm z chirurgiczną podmianą" (S4). Dwa modele drift-aware: `DriftAwareSRP` per-learner decyduje KEEP / SURGICAL-swap / FULL-reset zależnie od overlapu z cechami dryfującymi (+ top-K rank-weighted voting + opcjonalny `correctionAlpha · importance` prior); `DAARFWrapper` używa per-tree ADWIN-ów z background-learnerami (kanał wewnętrzny) i KEEP/FULL pass-u na zewnętrzny alarm. Nowe podprzestrzenie próbkowane Efraimidis-Spirakis-em z wagami `importance^power · (1−β) + uniform · β`. Cała pamięć stała (sliding windows + multiplikatywny decay), nigdy nie wymagamy całego datasetu. Całość pakuje `UnifiedStreamExperimentRunner` — 12-wątkowy executor który mieli `blocks × datasets × variants × seeds`, zapisuje 6 typów per-run detalicznych CSV i puszcza per-block Friedman/Nemenyi/CD + Wilcoxon-Holm.

**Sercem metody jest sprzężenie zwrotne:** *dryft → reset cech w ranker → nowy ranking → nowa selekcja → reset/podmiana w modelu → nowe predykcje → nowy strumień błędów do detektora.*

I to wszystko w jednym przebiegu, instancja po instancji.

---
---

# CZĘŚĆ II — Szczegółowy flow implementacyjny i ścieżki eksperymentów

> Ta część jest „przewodnikiem po kodzie w ruchu". Część I mówiła *dlaczego*; tutaj
> prześledzisz *co dokładnie* dzieje się od kliknięcia `main()` aż do zapisania
> ostatniego CSV — krok po kroku, w kolejności, w jakiej wykonuje się kod. Wszystkie
> nazwy klas/pól odnoszą się do rzeczywistej implementacji w
> `src/main/java/thesis/…` (a nie do starszych, osobnych runnerów E1–E5).
>
> **Uwaga architektoniczna:** cała macierz E1–E5 jest dziś obsługiwana przez
> **jeden** plik — `thesis.experiments.UnifiedStreamExperimentRunner` — czytający
> **jeden** config `master_experiments.json`. Starszy dokument
> `EXPERIMENTS_DESCRIPTION.md` opisuje historyczne, osobne klasy `E1Baselines`…`E5Detectors`;
> tu opisujemy stan aktualny (unified).

---

## 11. Poziom 0 — orkiestracja: od `main()` do CSV

### 11.1. Wejście i parsowanie configu

```
main(args)
  └── configPath = args[0] ?? "src/main/java/thesis/experiments/master_experiments.json"
  └── Cfg cfg = Cfg.load(configPath)              // Jackson: JSON → obiekty Cfg/Block/DatasetSpec/VariantSpec
  └── new UnifiedStreamExperimentRunner().run(cfg)
```

`Cfg.load` czyta **globalne** ustawienia (`warmup=1500`, `window_size=1000`,
`ram_sample_every=200`, `num_threads=12`, `seeds=[1,2,3,4,5]`, `skip_missing_arff`,
`real_datasets_read_all`) oraz listę **bloków** (`E1..E5`), a w każdym bloku listę
`datasets` i `variants`. Walidacje twarde już tu:
- `num_threads ∈ [1,64]` inaczej wyjątek,
- jeśli `seeds` puste → domyślnie `[1,2,3,4,5]`,
- jeśli `< 5` seedów → `WARN` (teza wymaga 5),
- każdy blok musi mieć niepuste `datasets` i `variants`.

Każdy `VariantSpec` ma **komplet** hiperparametrów z default'ami (patrz sekcja 12) —
JSON nadpisuje tylko to, co jawnie ustawi. Dzięki temu np. `DA-SRP-A` dostaje
`correctionAlpha` wyzerowane w fabryce mimo że w JSON go nie ma.

### 11.2. Rozwinięcie macierzy pracy (`expandWork`)

```
for each block b in cfg.blocks:
  for each dataset ds in b.datasets:
     if ds.type == "arff" && plik nie istnieje:
         if skipMissingArff → log + continue      // ciche pominięcie brakującego ARFF
         else → RuntimeException
     for each variant v in b.variants:
        for each seed in cfg.seeds:
           work.add(new WorkItem(b, ds, v, seed))
```

Wynikiem jest **płaska lista `WorkItem`** — kartezjański iloczyn
`bloki × datasety × warianty × seedy`. Dla obecnego `master_experiments.json`:

| Blok | variants × datasets × seeds | = runów |
|---|---|---|
| E1 | 8 × 8 × 5 | 320 |
| E2 | 10 × 5 × 5 | 250 |
| E3 | 10 × 8 × 5 | 400 |
| E4 | 6 × 4 × 5 | 120 |
| E5 | 8 × 4 × 5 | 160 |
| **Σ** | | **~1250** WorkItem-ów (minus brakujące ARFF) |

### 11.3. Wykonanie równoległe (`executeAll`)

```
pool = Executors.newFixedThreadPool(numThreads=12, "exp-worker")
for each WorkItem wi:
    futures.add(pool.submit(() -> {
        RunArtifacts ra = new RunWorker(cfg, wi).call();   // CAŁY run w izolacji
        log "(k/N) block|dataset|variant|seed → n, kappa, acc, thr [status]"
        sink.add(ra)                                        // ConcurrentLinkedQueue
        return ra;
    }))
for each future f: f.get()   // czekamy; ExecutionException tylko printStackTrace (nie wywala poola)
pool.shutdown(); awaitTermination(120s) else shutdownNow()
```

**Kluczowa decyzja:** każdy `RunWorker` jest **całkowicie samowystarczalny** — buduje
własny stream, selektor, importance, ranker, model, detektor, `MetricsCollector` i
`RunDetailedRecorder`. Zero współdzielonego mutable state w hot-pathie → zero locków.
Worker zwraca `RunArtifacts{ RunResult, RunDetailedRecorder }`; przy wyjątku łapie
`Throwable` i zwraca `status=FAIL` z `recorder=null` (run nie wywala całej macierzy).

### 11.4. Deterministyczne domknięcie i zapisy

Po opróżnieniu poola kolejka jest sortowana `ARTIFACT_ORDER`
(`blockId → dataset → variant → seed`) — dzięki temu CSV są **bit-w-bit
deterministyczne** niezależnie od kolejności ukończenia wątków. Potem
**single-threaded** w ustalonej kolejności:

```
writeRunsRaw               → results/runs_raw.csv            (1 wiersz / run, także FAIL)
writeMasterAndBlockSummaries → results/master_summary.csv    (mean±std po seedach)
                             + results/<Ex>/<output_file>.csv (per-blok summary)
writeDetailedPerBlockCsvs  → results/<Ex>/{windows,drift_alarms,feature_selections,
                                           feature_importance,recovery_time,
                                           adaptation_events}.csv
writeStatisticalTests      → results/<Ex>/stat_tests/…       (Friedman+Nemenyi+Wilcoxon+CD)
```

`blockFolder` mapuje `E1_baselines`→`results/E1/` (bierze prefiks przed `_`).

---

## 12. Poziom 1 — cykl życia jednego runu (`RunWorker.call`)

To jest **serce** — dokładnie to, co dzieje się dla pojedynczego `(block, dataset,
variant, seed)`. Pięć faz:

```
call():
  openStream()            // 12.1
  collectWarmup()         // 12.2
  buildComponents()       // 12.3
  attachRecorder()        // 12.4
  n = runPrequentialLoop()// 12.5  ← 99% czasu CPU
  recorder.finalizeAtEnd(n, metrics)
  return RunArtifacts(buildSuccessResult(n), recorder)
```

### 12.1. `openStream` — budowa strumienia

`buildStream(ds, seed)`:
- `type=arff` → `new ArffFileStream(path, -1)` + `prepareForUse()` (klasa = ostatni atr.).
- `type=synthetic` → `switch(generator)`:
  - `SEA` → `createSEA` (jeśli `num_drifts=0`) **lub** `createMultiDriftSEA(seed,n,numDrifts)`,
  - `STAGGER` → `createSTAGGER` / `createMultiDriftSTAGGER`,
  - `HYPERPLANE` → `createHyperplane(seed, sigma, n)`,
  - `RANDOMRBF` → `createRandomRBF(seed, speed, n)`,
  - `CUSTOMFEATUREDRIFT`/`FEATUREDRIFT` → `createCustomFeatureDrift(seed, driftFeatures, sigma, n)`.
- Jeśli `noise_features>0` → owijamy w `NoiseAugmentedStream` (dokłada `nNoise` kolumn
  `~U(0,1)` **przed** kolumną klasy → cechy nieinformatywne, test odporności selekcji).

Następnie: `header`, `numFeatures = numAttributes()-1`, `numClasses`, walidacja
(`numFeatures ≥ 1`, `numClasses ≥ 2`), `space = new FeatureSpace(header)`.

> **Jak powstaje dryft w danych syntetycznych** (bardzo ważne dla E4/E5):
> - **SEA (single)** — `ConceptDriftStream` sklejający 4 funkcje SEA z punktami zmiany
>   w `n/4, n/2, 3n/4` (`width=1` → *abrupt*), `noise=10%`.
> - **SEA/STAGGER multi-drift** — `buildCyclicAbruptStream`: `numDrifts+1` segmentów
>   równomiernie po `n/(numDrifts+1)`, funkcje cyklują `1→2→3→4→1…` (SEA) / `1→2→3→1…`
>   (STAGGER). Dla `numDrifts=10, n=100k` → **dryft co 10k instancji**. To jest „high dynamics".
> - **Hyperplane** — 15 atr., 5 dryfujących, `magChange=sigma` → *gradual* dryft ciągły.
> - **RandomRBF** — 15 atr., 20 centroidów, 5 dryfujących, `speedChange=speed` → gradual.
> - **CustomFeatureDrift** — Hyperplane 20 atr. z `numDriftFeatures` dryfującymi — celowo
>   testuje *lokalizację cech* (czy KSWIN wskaże właściwe kolumny).

### 12.2. `collectWarmup` — bufor rozgrzewki

Drenuje **`warmup=1500`** pierwszych instancji do `warmupWindow[1500][d]` +
`warmupLabels[1500]`, ekstrahując cechy przez `space.extractFeatures`. Jeśli strumień
skończy się wcześniej — bufor przycinany do `collected`. `warmupCollected` zapamiętane.

### 12.3. `buildComponents` — budowa wszystkich obiektów runu

W tej **dokładnej** kolejności (są zależności!):

1. **`selector = buildSelector(v, d, C)`** → `selector.initialize(warmupWindow, warmupLabels)`
   (patrz sekcja 14 — S1..S4/NONE). Selektor **od razu** ma pierwszą selekcję top-k.
2. **`importance = new FeatureImportance(d)`** — na razie pusty.
3. **`buildFullFeatureRankerFromWarmup()`** — buduje **osobny, pełnowymiarowy** ranker
   (nie ten w selektorze!): własny `PiDDiscretizer(d,C)` + `InformationGainRanker`
   nad **wszystkimi** `d` cechami. Karmi go całym warmupem; jeśli `isReady()` →
   **seeduje `importance`** wartościami `MI/IG` (KS = wektor zer). Ten ranker służy potem
   do dwóch rzeczy: (a) zasilania `importance` po alarmie, (b) dostarczania score'ów do
   `DriftAwareSRP.handleDrift` przy chirurgicznej podmianie cech.
4. **`model = buildModel(v, selector, header, C, seed, importance)`** (sekcja 13).
5. **`detector = buildDetector(v, d)`** → `TwoLevelDriftDetector` (sekcja 15).
6. **`metrics = new MetricsCollector(C, windowSize=1000, logEvery=0, ramSampleEvery=200)`**.

### 12.4. `attachRecorder`

`RunDetailedRecorder` dostaje tożsamość runu i:
- `onInitialSelection(warmupCollected, selector.getCurrentSelection())`,
- `onImportanceSnapshot(...)` — snapshot importance po warmupie,
- jeśli `model instanceof DriftAwareSRP` → podpina `driftListener` przekazujący
  `DriftActionSummary` do `recorder.onDASRPEvent`. (Dla DA-ARF liczniki `extKeep/extFull`
  są odczytywane różnicowo w pętli — patrz 12.5.)

### 12.5. `runPrequentialLoop` — pętla per-instancja (dokładna kolejność)

To jest **prequential**: predykcja → błąd → detekcja → adaptacja selekcji → trening.
Kolejność operacji ma znaczenie (np. `detector.update` **przed** `selector.update`):

```
n = warmupCollected
while stream.hasMoreInstances() && n < effMax:
    x     = stream.nextInstance().getData()
    yTrue = (int) x.classValue()
    feats = space.extractFeatures(x)

    updateFullFeatureRanker(feats, yTrue)     // (1) pełny ranker uczy się w tle
    t0 = nanoTime()
    yHat = model.predict(x)                    // (2) PREDYKCJA (mierzymy czas)
    elapsed = nanoTime() - t0
    err = (yHat == yTrue) ? 0.0 : 1.0

    detector.update(err, feats)                // (3) detekcja: L1(err) + L2(feats)
    alarm    = detector.isGlobalDriftDetected()
    drifting = alarm ? detector.getDriftingFeatureIndices() : {}
    if alarm: updateFeatureImportanceFromDetector()   // (4) odśwież importance (MI + 1-p)

    metrics.update(yTrue, yHat, elapsed)       // (5) κ, acc, RAM-h, throughput
    if alarm: metrics.onDriftAlarm()           // (6) start licznika recovery

    selector.update(feats, yTrue, alarm, drifting)    // (7) adaptacja selekcji
    model.train(x, yTrue, alarm, drifting)            // (8) TRENING (+ wewn. reakcja modelu)
    n++

    recorder.onInstance(n, yTrue, yHat, selection, alarm, drifting, metrics)  // (9)
    if alarm: recorder.onImportanceSnapshot(...)
    if daArf: różnicowy odczyt extKeep/extFull → recorder.onDAARFEvent(...)    // (10)
```

**Dlaczego ta kolejność:**
- `(1)` pełny ranker uczy się **z każdej** instancji (niezależnie od selekcji modelu),
  żeby po alarmie mieć aktualny globalny ranking do odbudowy importance i podprzestrzeni.
- `(3) przed (7)`: detektor musi najpierw powiedzieć *czy* i *co* dryfuje, zanim
  selektor zareaguje.
- `(4)`: `updateFeatureImportanceFromDetector` bierze `fullRanker.getFeatureScores()`
  (relevance) i `invertPValues(detector.getLastPValues())` jako proxy KS (stability =
  `1 − p`, im niższe p, tym silniejszy dryft, tym niższa stabilność).
- `(7) i (8)`: selektor jest wołany **jawnie**, ale wrapper modelu w `train(...)` też
  woła `selector.update` wewnętrznie — dla selektorów adaptacyjnych oznacza to podwójną
  aktualizację licznika/okna w jednej iteracji (świadomy kompromis: hot-path bez
  dodatkowej koordynacji; efekt netto to nieco szybsze narastanie okien).

`effMax = effectiveMaxInstances`: ARFF przy `realDatasetsReadAll` → do EOF; inaczej
`ds.maxInstances` lub `defaultMaxInstances=100000`.

Na końcu `buildSuccessResult` robi `metrics.snapshot()`, liczy throughput
`n / wallSeconds`, i wyciąga liczniki adaptacji: dla `DAARFWrapper` →
`extKeepCount/extFullCount`; dla `DriftAwareSRP` → `totalKept/Surgical/Full/NoReplacement`.

---

## 13. Poziom 2 — fabryka modeli (`buildModel`) i warianty drift-aware

`key = model.toUpperCase().replace('_','-')`; `noExternalFS = selector ∈ {NONE,NO_FS,ALL}`:

| `model` | Konstrukcja | Uwagi |
|---|---|---|
| `HT` | `HoeffdingTreeWrapper(selector, header)` | pojedyncze drzewo MOA |
| `ARF` | `ARFWrapper(sel, header, ensemble, lambda, false, exposeOptions=!noExternalFS)` | MOA AdaptiveRandomForest |
| `SRP` | `SRPWrapper(sel, header, ensemble, lambda, false, useHardFilter=!noExternalFS)` | MOA StreamingRandomPatches |
| `MAJORITY` | `MajorityClassWrapper(sel, C)` | baseline poziom-0 |
| `NOCHANGE` | `NoChangeWrapper(sel, C)` | baseline poziom-0 |
| `DA-SRP-A` | `newDASRP(..., imp=null, topK=false)` | tylko Komponent A (KEEP/SURGICAL/FULL) |
| `DA-SRP-AB` | `newDASRP(..., imp=importance, topK=false)` | + importance-weighted sampling |
| `DA-SRP-ABC` | `newDASRP(..., imp=importance, topK=true)` | + top-K rank-weighted voting + `correctionAlpha` |
| `DA-ARF` | `newDAARF(...)` | custom ensemble (patrz 13.2) |

### 13.1. `NativeDriftAwareSRP` — ablacja A / AB / ABC (bez refleksji)

> **AKTUALIZACJA (Option B):** DA-SRP to teraz `NativeDriftAwareSRP` — **własny ensemble bez
> refleksji**. Stara wersja (`DriftAwareSRP`) owijała MOA `StreamingRandomPatches` i refleksją
> nadpisywała jego prywatne podprzestrzenie (krucha, nieobronna). Nowa klasa jest własnym
> ensemblem: rzutowane patche `ARFHoeffdingTree` (strojone jak MOA: grace=50, δ=0.01) + online
> bagging + per-learner ADWIN, a podprzestrzenie są **jawne** w kodzie. Domyślny patch = **60%
> cech** (jak MOA SRP; wąskie ⌈√d⌉ kolapsuje na realnych). Domyślne `daSrpNative=true` — stara
> refleksyjna klasa nie jest już używana w żadnym eksperymencie. Walidacja head-to-head: natywny
> ≈ refleksyjny (lepszy na syntetyce, minimalnie słabszy na NYCTaxi/NHTS, bez kolapsów).

`newDASRP` (przy `daSrpNative=true`) buduje `NativeDriftAwareSRP(selector, header, numClasses,
ensemble, subspace=0.6·d, lambda, accWindow, tau, useBkg, warnDelta, driftDelta, seed, importance)`
i ustawia knoby: `importancePower`, `samplingBeta`, `unlocalizedFallbackFraction`,
`unstableImportanceQuantile`, `surgicalReplacementTolerance`. Dla ABC dodatkowo
`topKFraction`, `correctionAlpha`, `maxBlendAlpha`; dla A/AB → `correctionAlpha=0`.

**`handleDrift(drifting, fullRankerScores)`** — per-learner, w reduced-index space:
```
overlap = |learner.subspace ∩ drifting|
if overlap == 0            → KEEP
else if overlap/|sub| < tau→ SURGICAL: podmień TYLKO dryfujące cechy na najlepsze
                                        niedryfujące kandydatki wg score (do tolerancji)
else                       → FULL: nowa podprzestrzeń (WeightedSubspaceSampler jeśli jest
                                    importance, inaczej uniform) + resetLearning()
```
Każdy learner raportuje `Action{KEEP,SURGICAL,FULL,NO_REPLACEMENT}` w
`DriftActionSummary` → listener → `recorder.onDASRPEvent`.

**Predykcja** (`predictProba`): baza = zwykły głos własnego ensembla (średnia po learnerach).
Przy importance i `correctionAlpha>0` (ABC): wynik blendowany ku korekcji top-K —
learnery rankowane po średniej importance podprzestrzeni, top-`⌈topKFraction·N⌉` głosuje wagami
`(K−r)`, a `final = (1−α)·baseProba + α·weightedTopK`, gdzie `α=correctionAlpha` przycinane do
`maxBlendAlpha`. Bez importance lub `α=0` → sam głos ensembla (A/AB).

### 13.2. `DAARFWrapper` — DA-ARF (trzy komponenty, dwa kanały)

`newDAARF`: `subspaceSize = daArfSubspaceSize>0 ? … : ⌈daArfSubspaceFraction·d⌉` (domyślnie
**0.5·d**), ensemble `ARFHoeffdingTree` **strojony jak MOA ARF** (grace=50, δ=0.01, maxByte=2e6).

> **AKTUALIZACJA (naprawa DA-ARF):** dwa handicapy bazowego lasu, które fałszowały porównanie z ARF:
> (1) `newTree()` ustawiał wcześniej tylko `subspaceSizeOption` → drzewa dziedziczyły domyślne
> `HoeffdingTree` (grace=200, **δ=1e-7**) → płytkie → kolaps do klasy większościowej na NHTS (κ=0).
> Teraz ustawia grace=50/δ=0.01 (jak MOA ARF), wystawione jako `daarf_tree_grace_period`/
> `daarf_tree_split_confidence` + `setTreeParams()` (rebuild). (2) Podprzestrzeń ⌈√d⌉ była za wąska
> na realne dane; domyślnie **0.5·d** (`daarf_subspace_fraction`). Po obu poprawkach DA-ARF-ABC jest
> konkurencyjny z ARF na 7/8 zbiorów (szczegóły: `THESIS_IMPROVEMENT_PLAN.md` sekcja A).
- **Komponent A — kanał wewnętrzny** (per-learner ADWIN): każdy learner ma parę ADWIN
  (`warningDelta=1e-4`, `driftDelta=1e-5`). Na *warning* → spawn `background` learner z
  świeżo próbkowaną podprzestrzenią; na *drift* → background podmienia foreground
  (`bkgPromotions++`) lub full reset (`newLearner`). Sterowane błędem treningowym learnera.
- **Komponent A — kanał zewnętrzny** (`externalKeepOrFull` na globalny alarm):
  1. `unstable = lowImportanceDriftingFeatures(drifting)` — dryfujące cechy o **niskim**
     importance (poniżej kwantyla `unstableImportanceQuantile=0.5`); cechy dryfujące ale
     *ważne* są zachowane (nadal predykcyjne).
  2. jeśli `unstable` puste lub `externalResetFraction ≤ 0` → cały ensemble `KEEP`
     (`extKeepCount += N`).
  3. inaczej: kandydaci = learnery z `subspace ∩ unstable ≠ ∅`; `resetLimit =
     ⌈N · externalResetFraction=0.2⌉`; resetujemy `resetLimit` **najsłabszych** kandydatów
     (`chooseLowestAccuracyCandidates`) przez `newLearner(avoid=unstable)`; reszta KEEP.
     Liczniki `extFullCount`/`extKeepCount`. Domyślnie **RESET** (bez SURGICAL); tryb `SURGICAL`
     istnieje (`daarf_external_mode`), ale empirycznie szkodzi (patrz sekcja 6), więc nie jest domyślny.
- **Komponent B — sampling**: `sharpenAndBlend(importance, power, beta)` →
  `WeightedSubspaceSampler` (Efraimidis–Spirakis). Bez importance → uniform.
- **Komponent C — voting**: `predictProba` rankuje learnery po recent-accuracy, top-`K`
  głosuje wagami `(K−r)`, z fallbackiem do uniformu jeśli top-K nic nie zwróci.

> **Trening DA-ARF (per instancja):** online bagging `k ~ Poisson(lambda=6)` per learner;
> błąd treningowy karmi ADWIN-y; background (jeśli jest) trenuje równolegle. Na końcu,
> jeśli `driftAlarm` → `externalKeepOrFull`.

### 13.3. Warianty A/AB/ABC w configu (mapowanie ablacji)

| Wariant | co włączone | kluczowe pola JSON |
|---|---|---|
| `DA-SRP-A` | tylko A | `tau=0.5` (imp=null, alpha=0) |
| `DA-SRP-AB` | A+B | `+ w1,importance_power=2,sampling_beta=0.7` |
| `DA-SRP-ABC` | A+B+C | `+ topk_fraction=0.3, correction_alpha=0.15, max_blend_alpha=0.5` |
| `DA-ARF-A` | A (bez B/C) | `sampling_beta=1.0` (=uniform), `topk_fraction=1.0`, `daarf_use_background=false` |
| `DA-ARF-AB` | A+B | `sampling_beta=0.7`, `topk_fraction=1.0`, `use_background=true` |
| `DA-ARF-ABC` | A+B+C | `sampling_beta=0.7`, `topk_fraction=0.5`, `use_background=true` |

To jest **czysta ablacja**: włączamy komponenty pojedynczo (A → AB → ABC), na tych samych
strumieniach i seedach, żeby zmierzyć **przyrostowy** wkład każdego pomysłu.

---

## 14. Poziom 2 — fabryka selektorów (`buildSelector`) i ścieżki S1..S4

`K = defaultK(d) = ⌈√d⌉`. Wszystkie adaptacyjne selektory dostają własny `PiDDiscretizer(d,C)`
i fabrykę `InformationGainRanker`.

| Selektor | Klasa | Kiedy re-selekcjonuje | Parametry |
|---|---|---|---|
| `NONE`/`ALL` | `NoFeatureSelection(d)` | nigdy (model widzi wszystkie `d` cech) | — |
| `S1` | `StaticFeatureSelector(d,C)` | tylko raz w `initialize` (`update`=no-op) | — |
| `S2` | `AlarmTriggeredSelector` | **po alarmie**: zbiera `max(50,wPostDrift=1000)` post-drift instancji do świeżego rankera, potem commit nowego top-k | `wPostDrift` |
| `S3` | `PeriodicSelector` | **co `max(100,periodicInterval)`** instancji z ring-buffera; max `⌈0.3·k⌉` swapów/cykl, `minTenure=100` | `periodicInterval` |
| `S4` | `DriftAwareSelector` | **okresowo + po alarmie** (alarm celuje w cechy dryfujące) | `periodicInterval`, `wPostDrift` |

**Flow S2 (alarm-triggered), krok po kroku:**
1. `update(feats,y,alarm,drifting)`; jeśli `alarm` i nie `collecting` → `startCollecting`
   (świeży ranker, `collected=0`).
2. Podczas `collecting` każda instancja jest dyskretyzowana i karmiona do rankera.
3. Gdy `collected ≥ wPostDrift` → `ranker.selectTopK(k, preferredOrder=poprzednia selekcja)`
   → commit; `collecting=false`; `reSelections++`.

**Flow S3 (periodic):** ring-buffer ostatnich `periodN` zdyskretyzowanych wierszy; co
`periodN` instancji przeliczamy ranking i podmieniamy **maks. `⌈0.3k⌉`** cech (każda musi
przetrwać `minTenure` instancji) — mechanizm anty-flickering.

**Flow S4 (drift-aware):** utrzymuje **oba** mechanizmy — długie okno okresowe **i**
per-alarm post-drift ranker. Po alarmie podmienia preferencyjnie te cechy z selekcji,
które faktycznie dryfowały (`whereSwap` względem `drifting`), liczniki
`alarmSwapEvents`/`periodicSwapEvents`.

**Stabilizacja selekcji** (`selectTopK(k, preferredOrder, tieEpsilon)`): sort desc po
score, a przy remisach (`< tieEpsilon`) wygrywa cecha wcześniejsza w `preferredOrder`
(zwykle poprzednia selekcja) → selekcja nie „miga" dla mikro-różnic score.

---

## 15. Poziom 2 — detektor dwupoziomowy (`buildDetector`)

`TwoLevelDriftDetector.Config(numFeatures)`; `level1Delta=detectorDelta=0.002`,
`kswinAlpha`, `kswinWindowSize=max(10,kswinWindow)`. Wybór Level-1:
- `ADWIN` / `HDDM_A` / `HDDM_W` → odpowiedni typ,
- `KSWIN` → Level-1 = **ADWIN z zacieśnionym `delta ≤ 1e-4`** (globalny alarm), a
  faktyczną **lokalizację** cech robi Level-2 (`PerFeatureKSWIN`). To jest niuans E5:
  „KSWIN detector" oznacza *KSWIN-driven localization przy ciasnym ADWIN L1*.

**Przebieg `detector.update(err, feats)`:**
1. Level-1 dostaje `err` (0/1). Gdy krzyknie → `globalAlarm`.
2. Level-2 (`PerFeatureKSWIN`) na każdej instancji aktualizuje per-cechowe okna KS;
   `getDriftingFeatureIndices()` zwraca cechy istotne **po korekcie Benjamini-Hochberg**
   (`bhQ=0.10`), żeby przy `d` równoległych testach nie generować `α·d` fałszywych alarmów.
3. `getLastPValues()` → wektor p-wartości KS per cecha (używany do odświeżenia importance).

---

## 16. Poziom 1 — co dokładnie mierzymy (`MetricsCollector` + `RunDetailedRecorder`)

`MetricsCollector` (okno 1000) agreguje na bieżąco: `CohenKappa`, `TemporalKappa`
(κ vs NoChange), `PrequentialAccuracy`, `RecoveryTime`, `RAMHours`
(`Runtime.totalMemory()-freeMemory()`, próbka clampowana do 0 przy ujemnych odczytach GC),
`FeatureStabilityRatio` (Jaccard kolejnych selekcji).

`RunDetailedRecorder` emituje **6 typów zdarzeń** → 6 CSV per blok:

| CSV | Granularność | Kluczowe kolumny |
|---|---|---|
| `windows.csv` | co 1000 instancji | acc, κ, κ_per, temporal_κ, RAM-h, peak_MB, throughput, drift_count_in_window |
| `drift_alarms.csv` | każdy globalny alarm | instance, lista drifting, num_drifting, acc przed/po |
| `feature_selections.csv` | każda zmiana selekcji | trigger (warmup/periodic/alarm), selected, changed, jaccard, stability |
| `feature_importance.csv` | warmup + każdy alarm | snapshot importance per cecha, rank, is_selected, is_drifting |
| `recovery_time.csv` | każdy alarm | recovery_length, baseline_acc_before, max_drop, AUC recovery |
| `adaptation_events.csv` | każdy event DA-SRP/DA-ARF | KEEP/SURGICAL/FULL/NO_REPL (SRP) lub ext_keep/ext_full (ARF) |

Plus dwa poziomy zbiorcze: `runs_raw.csv` (1 wiersz/run) i `master_summary.csv`/per-blok
(`mean±std` po seedach, w tym temporal_κ, recovery_time, stability, selection_changes,
drift_alarms, mean_selected_feature_count).

---

## 17. Poziom 3 — analiza statystyczna per blok (`BlockStatisticalAnalysis`)

Dla **każdego bloku** i **każdej metryki** (`kappa, accuracy, ram_hours_gb, throughput…`):
1. **Macierz** `datasets × variants` (wartości uśrednione po seedach).
2. **Friedman** (omnibus): χ² + Iman-Davenport F — „czy *którakolwiek* para różni się".
3. **Nemenyi** post-hoc: `CD = q_α·√(k(k+1)/(6N))`; eksport `cd_diagram_*.csv` (rysunek w Pythonie).
4. **Wilcoxon signed-rank** parami over `(dataset, seed)` — większa moc niż rangi.
5. **Holm** step-down: korekta p-wartości w obrębie metryki (kontrola FWER).

Trzy testy = **trzy poziomy ostrości**: omnibus → ranking średnich rang → pairwise.

---

## 18. Ścieżki eksperymentów E1–E5 — co, po co, i jak czytać wynik

> Każdy blok to inne **pytanie badawcze**. Poniżej: hipoteza, warianty, datasety,
> co obserwować w wynikach.

### E1 — Baselines (`E1_baselines.csv`)
- **Pytanie:** jaki jest referencyjny poziom κ/acc/RAM-h/throughput dla naiwnych i
  klasycznych modeli, z i bez selekcji S1?
- **Warianty (8):** `Majority`, `NoChange` (baseline poziom-0), `HT/ARF/SRP` (bez FS,
  `selector=NONE`), `HT+S1/ARF+S1/SRP+S1` (ze statyczną selekcją `⌈√d⌉`).
- **Datasety (8):** 5 syntetycznych (SEA, Hyperplane, RandomRBF, FeatureDrift, **LED**) +
  3 realne ARFF (YahooFinance, NYCTaxi, NHTS).
  > Zmiana: **generyczny STAGGER usunięty** z E1/E2/E3 (κ=1.000 dla wszystkich → zero dyskryminacji;
  > zostaje jako STAGGER-HiDyn/-Low w E4/E5). W zamian **LED** (LEDGeneratorDrift): 24 cechy = 7 istotnych
  > + 17 nieistotnych, 10 klas — benchmark FS ze znanym ground-truth cech nieistotnych.
- **Detektor:** ADWIN jest w configu, ale w baseline'ach służy głównie do zliczania
  alarmów — modele bez `S2/S3/S4/DA-*` nie reagują na cechy.
- **Jak czytać:** poważny model musi bić `Majority` i `NoChange`; `+S1` vs bez-FS pokazuje
  koszt/zysk redukcji wymiaru do `⌈√d⌉`.

### E2 — Adaptive feature selection (`E2_adaptive.csv`)
- **Pytanie:** czy adaptacyjna selekcja (S2/S3/S4 + detekcja) bije statyczną S1 dla tego
  samego modelu?
- **Warianty (10):** `{ARF,SRP} × {NONE, S1, S2, S3, S4}`, detektor ADWIN,
  `w_post_drift=1000`, `periodic_interval=1000`.
- **Datasety (5):** tylko syntetyczne (kontrolowany dryft, znane ground-truth).
- **Jak czytać:** porównuj `ARF+S2/S3/S4` do `ARF+S1` (i analogicznie SRP). Patrz na
  `feature_stability` (S3/S4 powinny być mniej stabilne, ale szybciej odzyskiwać) i
  `recovery_time` po alarmach.

### E3 — Ablacja i porównanie DA (`E3_ablation.csv`)
- **Pytanie:** ile wnosi każdy komponent (A/B/C) w DA-SRP i DA-ARF, i jak wypadają vs
  klasyczne baseline'y?
- **Warianty (10):** baseline'y (`SRP`, `SRP+S1`, `ARF`, `ARF+S2`) + `DA-SRP-{A,AB,ABC}` +
  `DA-ARF-{A,AB,ABC}`.
- **Datasety (8):** 5 syntetycznych + 3 ARFF.
- **Jak czytać:** przyrost `A → AB → ABC` (monotoniczny wkład komponentów). `adaptation_events.csv`
  pokazuje KEEP/SURGICAL/FULL (DA-SRP) i ext_keep/ext_full (DA-ARF).

### E4 — High dynamics / magnitude dryftu (`E4_high_dynamics.csv`)
- **Pytanie:** jak metody znoszą **częsty** dryft (10 zmian / 100k) vs rzadki (3)?
- **Warianty (6):** `ARF`, `ARF+S1`, `SRP`, `SRP+S1`, `DA-SRP-ABC`, `DA-ARF-ABC`.
- **Datasety (4):** `SEA-HiDyn`(10), `STAGGER-HiDyn`(10), `SEA-Low`(3), `STAGGER-Low`(3) —
  te same generatory, różna `num_drifts` (patrz `buildCyclicAbruptStream`).
- **Jak czytać:** różnica κ między HiDyn a Low pokazuje **odporność na tempo dryftu**;
  DA-* powinny tracić mniej niż plain ARF/SRP. `recovery_time` i `windows.csv` (spadki κ
  wokół punktów zmiany) są kluczowe.

### E5 — Studium detektorów (`E5_detectors.csv`)
- **Pytanie:** który detektor (ADWIN/HDDM_A/HDDM_W/KSWIN) daje najlepszą stabilność DA-ARF?
- **Warianty (8):** `DA-ARF × {ADWIN, HDDM_A, HDDM_W, KSWIN}` + baseline'y `ARF/SRP(+S1)+ADWIN`.
- **Datasety (4):** `SEA-HiDyn`, `STAGGER-HiDyn`, `Hyperplane`, `RandomRBF`.
- **Metryki-klucz:** `ext_full_count` + `drift_count` — mierzą jak „nerwowy" jest detektor
  (za dużo full-resetów = niestabilność). KSWIN używa ciasnego ADWIN L1 + KSWIN L2 do
  lokalizacji (sekcja 15).

---

## 19. Mapa „gdzie w kodzie" (szybki indeks)

| Etap | Klasa / metoda | Plik |
|---|---|---|
| Config + orkiestracja | `UnifiedStreamExperimentRunner.{main,run,expandWork,executeAll}` | `experiments/UnifiedStreamExperimentRunner.java` |
| Jeden run | `RunWorker.{call,openStream,collectWarmup,buildComponents,runPrequentialLoop}` | ↑ |
| Strumienie + dryft | `SyntheticStreamFactory.{createSEA,createMultiDriftSEA,createSTAGGER,createHyperplane,createRandomRBF,createCustomFeatureDrift,createLEDDrift,addNoiseFeatures}` | `pipeline/SyntheticStreamFactory.java` |
| Dyskretyzacja | `PiDDiscretizer` / `FeatureDiscretizer` / `Layer1Histogram` / `Layer2Merger` | `discretization/` |
| Rankery | `AbstractFrequencyRanker` + `InformationGain/MutualInformation/ChiSquaredRanker` | `selection/` |
| Selektory | `NoFeatureSelection`, `StaticFeatureSelector`, `AlarmTriggeredSelector`, `PeriodicSelector`, `DriftAwareSelector` | `selection/` |
| Detekcja | `TwoLevelDriftDetector`, `PerFeatureKSWIN`, `KSWINSingleFeature`, `ADWIN/HDDMChangeDetector` | `detection/` |
| Modele | `HT/ARF/SRPWrapper`, `DriftAwareSRP`, `DAARFWrapper`, `FeatureImportance`, `WeightedSubspaceSampler` | `models/` |
| Metryki | `MetricsCollector` + `CohenKappa/TemporalKappa/RAMHours/RecoveryTime/FeatureStabilityRatio` | `evaluation/` |
| Detaliczne CSV | `RunDetailedRecorder` | `experiments/RunDetailedRecorder.java` |
| Statystyka | `BlockStatisticalAnalysis` + `FriedmanTest/NemenyiPostHoc/WilcoxonSignedRank/StatisticalTests` | `experiments/`, `evaluation/` |
| Config macierzy | `master_experiments.json` | `experiments/master_experiments.json` |

