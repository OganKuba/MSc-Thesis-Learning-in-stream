# METHODOLOGY.md — analiza metodologiczna projektu

> Kompanion do `CODE_MAP.md`. Tam masz „co jest w kodzie", tu masz „dlaczego to tak działa".
>
> **Jak czytać ten dokument.** Kolejność sekcji odpowiada **drodze, którą przebywają dane**:
> surowa instancja → dyskretyzacja → ranking cech → rozgałęzienie na selektory (S1–S4) →
> rozgałęzienie na modele (klasyczne / DA-SRP / DA-ARF) → metryki. Każdy selektor i każdy
> wariant DA ma **własną, samodzielną sekcję** — świadomie z powtórzeniami, żeby dało się
> przeczytać tylko jedną i zrozumieć całą ścieżkę bez skakania po dokumencie.
>
> Wzory pojawiają się **dopiero w sekcji, do której należą**, nie z góry.

---

# CZĘŚĆ I — Jak system przetwarza dane, krok po kroku

## 1. Zarys: co ten system robi

### Problem

Uczenie na strumieniu, w którym rozkład danych **zmienia się w czasie** (concept drift), przy
dwóch ograniczeniach: pamięć stała (nie wolno trzymać całego zbioru) i jeden przebieg
(każdą instancję widzimy raz).

Do tego dochodzi pytanie właściwe tej pracy: **czy w takim strumieniu opłaca się selekcjonować
cechy, a jeśli tak, to kiedy je przebudowywać?**

### Główna intuicja

Dryf nie jest jednorodny. Czasem zmienia się granica decyzyjna, a zestaw istotnych cech zostaje
ten sam (SEA). Czasem zmienia się **to, które cechy w ogóle niosą sygnał** (FeatureDrift,
Hyperplane). Metoda ma sens tylko w tym drugim przypadku — i cała konstrukcja eksperymentów
służy rozdzieleniu tych dwóch sytuacji.

### Pełna ścieżka jednej instancji

```
  Instance x
      │
      ├─► [2] PiD — dyskretyzacja ciągłych wartości na biny
      │        Layer 1 (64 biny) → Layer 2 (b2 binów)
      │
      ├─► [3] Ranker — tensor kontyngencji [F][B][C] → score Information Gain
      │
      ├─► [4] Detektor dwupoziomowy
      │        L1: ADWIN/HDDM na strumieniu błędu → globalny alarm
      │        L2: per-cechowy KSWIN + BH-FDR    → które cechy dryfują
      │
      ├─► [5] Wektor importance (tylko dla modeli DA)
      │        relevance z rankera  +  stability z p-value detektora
      │
      ├─► [6] SELEKTOR — rozgałęzienie: S1 / S2 / S3 / S4
      │        różnią się WYŁĄCZNIE tym, kiedy przebudowują zestaw cech
      │
      ├─► [7] MODEL — rozgałęzienie: HT/ARF/SRP  albo  DA-SRP / DA-ARF
      │        klasyczne: uczą się na cechach od selektora
      │        DA-*:      ignorują selektor, mają własne podprzestrzenie per learner
      │
      └─► [8] Metryki: κ, accuracy, RAM-Hours, recovery, stabilność selekcji
```

Kolejne sekcje idą dokładnie tą ścieżką.

---

## 2. Krok 1 — dyskretyzacja PiD

**Po co.** Ranker filtrowy liczy Information Gain z tabeli częstości. Tabela częstości wymaga
wartości dyskretnych, a większość cech w tych strumieniach jest ciągła. PiD (*Partition
Incremental Discretization*) robi to inkrementalnie, bez trzymania danych.

**Dwie warstwy** (`PiDDiscretizer`, `Layer1Histogram`, `Layer2Merger`):

| warstwa | rozmiar | rola |
|---|---|---|
| Layer 1 | **64 biny** (`b1`) | drobny histogram: „gdzie w ogóle leżą wartości tej cechy" |
| Layer 2 | **`b2` binów** (domyślnie 8) | gruboziarnista mapa: „gdzie są progi istotne dla klasy" |

Layer 2 powstaje przez **chciwe sklejanie sąsiednich binów Layer-1**, tak żeby łączyć te
o podobnym rozkładzie warunkowym klasy (odległość TV + wygładzanie Laplace'a). Przebudowa
co `recomputeEvery = 1000` instancji, po rozgrzewce `warmupN = 500`.

**Dlaczego dwie warstwy, a nie jedna.** Layer 1 sam w sobie jest zbyt drobny — 64 biny na
kilkaset instancji dają puste komórki i zawyżony IG. Layer 2 sam w sobie wymagałby znajomości
progów z góry. Rozdzielenie pozwala **adaptować progi bez odrzucania historii**: Layer 1 zbiera
dalej, Layer 2 tylko przelicza sklejenia.

**Parametry produkcyjne** (`UnifiedStreamExperimentRunner.newPid`):
`b1 = 64`, `b2` z configu (`pid_b2`, domyślnie 8), `warmupN = 500`, `recomputeEvery = 1000`.

**Adaptacja zakresu Layer-1.** Początkowy zakres min/max jest wyznaczany z warmupu z marginesem
10%, ale nie jest zamrożony na zawsze. Po warmupie `FeatureDiscretizer.update(...)` sprawdza dwa
warunki:

1. jeśli nowa wartość leży poza aktualnym `[min, max)`, zakres jest natychmiast rozszerzany tak,
   żeby objąć starą skalę, nową wartość i dodatkowy margines 20%;
2. jeśli zbyt duża część próbek trafia do skrajnych binów histogramu, `Layer1Histogram.shouldExpand`
   uznaje krawędzie za nasycone i również wymusza rozszerzenie zakresu.

Technicznie rozszerzenie robi `Layer1Histogram.rebin(...)`: stare liczniki są przenoszone do
nowej siatki binów na podstawie środków dawnych przedziałów. Dzięki temu PiD nie musi trzymać
historii instancji, ale zachowuje przybliżoną informację z dotychczasowego histogramu.

> **Ograniczenie do odnotowania.** Liczba binów Layer-1 pozostaje stała (`b1 = 64`). Gdy zakres
> mocno się rozszerzy, każdy bin obejmuje szerszy przedział, więc spada rozdzielczość
> dyskretyzacji. Mechanizm chroni przed upychaniem nowych wartości w skrajnych binach, ale nie
> daje tak dokładnej przebudowy jak ponowne policzenie histogramu z pełnej historii danych.

**`UNKNOWN_BIN = -1`.** Gdy dyskretyzator nie jest jeszcze gotowy albo wartość jest niepoprawna
(NaN/Inf), zwracany jest `-1` zamiast zgadywanego binu. Ranker takie wiersze **pomija**.
Alternatywa — wrzucanie do binu 0 — cicho zniekształcałaby tabelę częstości.

---

## 3. Krok 2 — ranking cech (Information Gain)

### Struktura, z której liczony jest score

Wszystko opiera się na jednym tensorze utrzymywanym inkrementalnie
(`AbstractFrequencyRanker`):

```
joint[F][B][C]   — ile razy (cecha f == bin b) wystąpiło razem z klasą c
```

Z niego wyprowadzane są rozkłady brzegowe i warunkowe. Nic więcej nie jest przechowywane —
stąd stała pamięć.

### Score: Information Gain — i tylko on

```
IG(f) = H(Y) − H(Y | X_f)        [w bitach]
```

**Wysoki IG** = znajomość binu cechy mocno redukuje niepewność co do klasy.
**IG ≈ 0** = cecha nie mówi nic ponad rozkład brzegowy klas.

> **Uwaga historyczna.** Wcześniejsza wersja tego dokumentu opisywała **trzy** rankery
> (Information Gain, Mutual Information, Chi-Squared) z tabelą porównawczą. To już nieprawda:
> `MutualInformationRanker` i `ChiSquaredRanker` **zostały usunięte z kodu**, bo żaden wariant
> w `master_experiments.json` ich nie wybierał. Została **jedna implementacja**,
> `InformationGainRanker`, i jest ona wspólna dla wszystkich selektorów (S1–S4) oraz dla
> wektora importance modeli DA. Ranker nie jest w tej pracy zmienną eksperymentalną.

### Konsekwencje wyboru podejścia filtrowego

Zalety, dla których zostało wybrane:

1. Score liczony **niezależnie od modelu** — tanio, bez retreningu, i ten sam ranking można
   podać dowolnemu klasyfikatorowi.
2. Działa **online i z zapominaniem** — reaguje na dryf bez przebudowy od zera.

Ograniczenia, które trzeba nazwać wprost w pracy:

3. **Ignoruje interakcje cech.** Cecha może mieć IG ≈ 0 samodzielnie, a być kluczowa w parze
   z inną (klasyczny XOR). Filtr tego nie zobaczy — to wpisane w metodę, nie do naprawienia
   strojeniem.
4. **Bias przy małych próbkach.** Empiryczna entropia jest przeszacowywana, więc świeżo
   zresetowany ranker chwilowo zawyża IG. Stąd próg `minSamplesReady` — poniżej niego ranker
   zgłasza „nie gotowy" i selektor nie przebudowuje selekcji.
5. **Wrażliwość na liczbę binów.** Cecha rozdrobniona na więcej binów wypada sztucznie wyżej.
   Dlatego `b2` jest **wspólne dla wszystkich cech**.

### Wybór top-k ze stabilizacją

```
selectTopK(k, preferredOrder, tieEpsilon)
```

- sortowanie malejąco po score,
- **przy remisie** (score kwantyzowany co `tieEpsilon = 0.01`) wygrywa cecha wcześniejsza
  w `preferredOrder` — czyli zwykle **poprzednia selekcja**.

To jest mechanizm **przeciw migotaniu selekcji**. Bez niego różnice rzędu 0.001 w IG
przerzucałyby cechy w tę i z powrotem, a `HoeffdingTreeWrapper` resetuje drzewo przy każdej
zmianie selekcji — czyli szum w rankingu kosztowałby realną jakość.

> Czego tie-break **nie** robi: nie utrzymuje cechy, której score realnie spadł. Jeśli cecha
> zjeżdża o pełne wiadro `tieEpsilon` niżej, zostaje odrzucona — i tak ma być. Oba zachowania
> są pokryte testami w `AlarmTriggeredSelectorSmokeTest`.

**Budżet cech:** `K = ⌈√d⌉` dla wszystkich selektorów. Dla d = 25 → 5 cech, dla d = 24 (LED) → 5.
To ustawienie jest **wspólne**, żeby S1–S4 różniły się tylko momentem przebudowy.

---

## 4. Krok 3 — detektor dwupoziomowy

Detektor daje **dwie różne informacje**, i to rozróżnienie jest kluczowe dla zrozumienia S2/S4
oraz modeli DA.

### Poziom 1 — globalny alarm („czy coś się zmieniło")

Karmiony **strumieniem błędu** klasyfikatora (0/1 na instancję). Do wyboru w configu:

| detektor | uwagi |
|---|---|
| `ADWIN` | domyślny; okno adaptacyjne na średniej błędu |
| `HDDM_A` | test Hoeffdinga na średnich |
| `HDDM_W` | wersja z ważeniem EWMA — **wymaga serializacji**, patrz §11 |
| `KSWIN` | tu: ciasny ADWIN na L1, a KSWIN pracuje na poziomie 2 |

Wyjście: **jeden bit** — `isGlobalDriftDetected()`.

### Poziom 2 — lokalizacja („które cechy się zmieniły")

Per-cechowy KSWIN (`PerFeatureKSWIN`) porównuje okno referencyjne z bieżącym dla **każdej
cechy osobno**, testem Kołmogorowa-Smirnowa. Przy `d` równoległych testach naiwny próg `p < α`
daje `α·d` fałszywych alarmów, więc stosowana jest **korekta Benjamini-Hochberg (FDR)** —
kontroluje odsetek fałszywych odkryć wśród odrzuceń, a nie prawdopodobieństwo choćby jednego
(jak Bonferroni, który przy `d = 25` byłby zbyt konserwatywny).

Nie ma więc jednego progu typu `p < 0.05` dla każdej cechy. P-value z `d` testów są sortowane
rosnąco:

```
p_(1) <= p_(2) <= ... <= p_(d)
```

a potem wybierany jest największy indeks `i`, dla którego:

```
p_(i) <= (i / d) · q,      q = 0.10
```

Wszystkie cechy z p-value nie większym niż ten próg trafiają do zbioru cech dryfujących. Dla
pierwszej, najmocniejszej cechy próg wynosi przykładowo `0.0125` przy `d = 8` (SEA), `0.005`
przy `d = 20` (Hyperplane/RandomRBF), `0.004` przy `d = 25` (FeatureDrift) i `0.00417` przy
`d = 24` (LED).

Wyjście: **zbiór indeksów cech dryfujących** + wektor p-value.

> **Ograniczenie, które trzeba opisać w pracy.** W E2 **91.6% alarmów nie wskazuje żadnej
> dryfującej cechy** — poziom 2 nie radzi sobie z dryfem stopniowym. W takich przypadkach S2/S4
> i modele DA schodzą do ścieżki „dryf niezlokalizowany" i działają globalnie zamiast celować.
> To realne ograniczenie mechanizmu lokalizacji, nie usterka.

---

## 5. Krok 4 — wektor importance (używany tylko przez modele DA)

Ta sekcja dotyczy **wyłącznie** wariantów DA-SRP i DA-ARF. Selektory S1–S4 jej nie używają.

### Do czego służy

Modele DA losują podprzestrzenie cech dla swoich learnerów. Losowanie ma być **ważone**: chcemy
częściej wybierać cechy, które są jednocześnie **informatywne** i **spokojne**. Wektor importance
jest właśnie tą wagą.

### Wzór

```
importance[f] = w1 · relevance[f]  +  w2 · stability[f]        (w1 = 0.7, w2 = 0.3)

gdzie:
  relevance[f] = IG cechy f z pełnoprzestrzennego rankera, znormalizowane do maksimum
  stability[f] = 1 / ((1 − p_f) + ε),  znormalizowane do maksimum
                 p_f = p-value testu KS dla cechy f z poziomu 2 detektora
```

Wynik jest normalizowany do sumy 1. Przy zdegenerowanym wejściu (same zera, wartości
niefinite) następuje **fallback do rozkładu jednostajnego** — zliczany w
`getDegenerateUniformFallbacks()`, żeby nie przemilczeć takich sytuacji.

**Odczytanie składników:**

- **relevance** — „ile ta cecha mówi o klasie". Wysokie dla cech niosących sygnał.
- **stability** — „jak spokojna jest ta cecha". Cecha bez śladu dryfu ma `p ≈ 1`, więc
  `(1 − p) ≈ 0` i stabilność wychodzi wysoka. Cecha mocno dryfująca ma `p ≈ 0` i stabilność
  spada.

**Dlaczego oba naraz.** Sama relevance wybrałaby cechy, które zaraz się zmienią pod nogami.
Sama stability wybrałaby cechy spokojne, bo nieinformatywne. Sens ma dopiero iloczyn intencji:
*informatywne i jednocześnie stabilne*.

> **Nazewnictwo w kodzie.** Parametry `FeatureImportance.update` nazywają się `miScores`
> i `ksStatistics`. To nazwy historyczne: pierwszy dostaje **IG** (nie MI — ranker MI nie
> istnieje), drugi **1 − p-value** (nie surową statystykę KS). Semantyka jest jak we wzorze
> wyżej.

### Modulacja przed losowaniem

Importance nie idzie prosto do samplera — najpierw dwa pokrętła:

```
weight[f] = (1 − samplingBeta) · importance[f]^importancePower  +  samplingBeta · uniform[f]
```

| parametr | domyślnie | działanie |
|---|---|---|
| `importancePower` | 2.0 | wyostrzenie. > 1 wzmacnia różnice między cechami; = 1 to proporcjonalność |
| `samplingBeta` | 0.7 | domieszka rozkładu jednostajnego. 0 = tylko importance, 1 = czysty los |

**Po co.** Bez `samplingBeta` wszystkie learnery zbiegłyby do tych samych top-cech i zespół
straciłby różnorodność — a różnorodność jest jedynym powodem, dla którego zespół bije pojedyncze
drzewo. Bez `importancePower` różnice IG rzędu 0.01 vs 0.05 byłyby dla samplera niewidoczne.
Razem dają „kontrolowaną zachłanność".

Losowanie bez zwracania robi `WeightedSubspaceSampler` algorytmem Efraimidisa-Spirakisa
(klucze `u^(1/w)`), z obsługą listy wykluczeń i fallbackiem, gdy wszystkie wagi są zerowe.

---

## 6. Krok 5 — selektory cech: S1, S2, S3, S4

Cztery strategie różnią się **wyłącznie jednym czynnikiem: kiedy przebudowują zestaw cech.**
Wszystko inne jest identyczne — budżet `K = ⌈√d⌉`, dyskretyzator PiD, ranker Information Gain,
stabilizacja tie-break. To celowe: bez tego nie dałoby się przypisać różnic samej adaptacyjności.

Do kompletu jest jeszcze `NONE` (`NoFeatureSelection`) — model widzi wszystkie `d` cech.
To **punkt odniesienia**, wobec którego mierzy się, czy selekcja w ogóle się opłaca.

---

### 6.1. S1 — `StaticFeatureSelector` (statyczna)

**Kiedy przebudowuje:** raz, na końcu rozgrzewki. Potem `update(...)` jest **pustą operacją**.

**Ścieżka danych:**

```
rozgrzewka (1500 instancji) → PiD → ranker IG → selectTopK(K) → ZAMROŻONE do końca
```

**Parametry:** brak (poza `K`).

**Rola w pracy.** To baseline pokazujący **koszt statycznej redukcji wymiaru**. Nie jest
konkurencyjną metodą — jest dowodem, że problem istnieje: średnia strata względem modelu bez
selekcji to **−0.202 κ** dla ARF na zbiorach syntetycznych, a na Hyperplane katastrofalne
0.712 → 0.193.

**Kiedy zawodzi:** zawsze, gdy istotność cech zmienia się po rozgrzewce. Wybór z pierwszych
1500 instancji zostaje na kolejne 98 500.

---

### 6.2. S2 — `AlarmTriggeredSelector` (sterowana zdarzeniem)

**Kiedy przebudowuje:** tylko **po alarmie dryfu**. S2 nie ma harmonogramu okresowego: jeśli
detektor nie zgłosi alarmu, selekcja zostaje taka sama jak po warmupie.

**Ścieżka danych, krok po kroku:**

```
0. Warmup:
      PiD + IG na pierwszych 1500 instancjach → początkowe top-K.

1. Normalna instancja bez alarmu:
      • PiD selektora aktualizuje histogramy,
      • ranker nie przebudowuje selekcji,
      • model dalej widzi dotychczasowe top-K.

2. Przychodzi alarm dryfu:
      • alarmsObserved++,
      • jeśli S2 już zbiera okno po poprzednim alarmie → alarm jest ignorowany
        (alarmsIgnoredWhileBusy++),
      • jeśli S2 nie jest zajęty → alarm jest przyjęty (alarmsAccepted++).

3. Po przyjętym alarmie:
      • `applyDriftDecay(drifting)` robi soft reset PiD:
          - jeśli Level-2 wskazał cechy dryfujące, resetowane są tylko one,
          - jeśli Level-2 nic nie wskazał, resetowane są wszystkie cechy;
      • ranker IG jest resetowany do zera,
      • `collecting = true`, `collected = 0`.

4. Zbieranie danych po dryfcie:
      przez `wPostDrift = 1000` kolejnych instancji S2 karmi świeży ranker
      nowymi obserwacjami. Stara selekcja nadal obowiązuje, bo nowy ranking
      nie jest jeszcze wiarygodny.

5. Re-selekcja:
      gdy `collected >= wPostDrift`:
        • liczony jest ranking IG z danych po alarmie,
        • wybierane jest `selectTopK(K, poprzednia_selekcja, tieEpsilon)`,
        • jeśli score'y są prawie równe, preferowana jest stara cecha,
        • nowa selekcja zostaje zatwierdzona,
        • `lastSelectionTrigger = "drift_alarm"`.
```

**Co znaczą parametry i stany:**

- `wPostDrift = 1000` — ile świeżych instancji po alarmie potrzeba, zanim wolno zmienić top-K.
  To opóźnia reakcję, ale chroni przed wyborem cech z kilku przypadkowych próbek.
- `driftDecay = 0.05` — soft reset PiD. Liczniki histogramu dla dryfujących cech są mnożone przez
  0.05, więc stary koncept nie znika absolutnie, ale jego wpływ jest silnie osłabiony.
- `collecting` — stan „czekam na wystarczająco dużo danych po alarmie". W tym stanie kolejne
  alarmy są ignorowane, żeby nie restartować zbierania w nieskończoność.
- `tieEpsilon = 0.01` — próg stabilizacji. Nowa cecha musi być lepsza od starej o więcej niż
  0.01 score'u IG, inaczej selektor zachowuje starą cechę.

**Kluczowa decyzja projektowa:** ranking po alarmie jest liczony ze **świeżego rankera**, więc
opiera się na danych po dryfcie, a nie na średniej starego i nowego konceptu. Ceną jest opóźnienie
o `wPostDrift` instancji.

**Wynik.** Najlepszy z adaptacyjnych: średnio **−0.023 κ** względem modelu bez selekcji, przy
**2.4× wyższej przepustowości i 2.8× niższych RAM-Hours**. To jedyna konfiguracja, w której
selekcja cech w tej pracy się broni — nie jakością, lecz kosztem.

---

### 6.3. S3 — `PeriodicSelector` (sterowana harmonogramem)

**Kiedy przebudowuje:** **co `max(100, periodicInterval = 1000)` instancji**, niezależnie od
tego, czy cokolwiek się zmieniło. Alarmy detektora są ignorowane.

**Ścieżka danych, krok po kroku:**

```
0. Warmup:
      PiD + IG na pierwszych 1500 instancjach → początkowe top-K.
      Te same zdyskretyzowane obserwacje trafiają też do bufora kołowego.

1. Każda kolejna instancja:
      • PiD aktualizuje histogramy,
      • instancja jest dyskretyzowana do binów,
      • para `(biny, klasa)` trafia do ring buffer,
      • rośnie licznik `updates`,
      • rośnie `tenure` cech aktualnie wybranych.

2. Ring buffer:
      przechowuje tylko ostatnie `periodN` zdyskretyzowanych instancji.
      Gdy jest pełny, nowa instancja nadpisuje najstarszą. Dzięki temu S3
      liczy ranking z ostatniego fragmentu strumienia, a nie z całej historii.

3. Tick okresowy:
      jeśli `updates % periodN == 0`, uruchamia się re-selekcja.
      Alarm dryfu nie ma tu znaczenia.

4. Re-selekcja:
      • tworzony jest nowy ranker IG,
      • ranker jest karmiony zawartością ring buffera,
      • cechy aktualnie wybrane sortowane są od najsłabszej do najlepszej,
      • cechy spoza selekcji sortowane są od najlepszej do najsłabszej.

5. Ograniczona wymiana:
      dla każdej pary `najsłabsza_wybrana` vs `najlepsza_niewybrana`:
        • wymień tylko wtedy, gdy nowa cecha ma score większy o `tieEpsilon`,
        • nie wyrzucaj cechy, która ma `tenure < minTenure`,
        • wykonaj maksymalnie `maxSwapsPerCycle = ceil(0.3 * K)` podmian.

6. Commit:
      jeśli doszło do podmiany, selekcja zostaje zatwierdzona,
      a `lastSelectionTrigger = "periodic"`.
```

**Co znaczą parametry i stany:**

- `periodN = 1000` — rytm przeglądu selekcji. Co 1000 instancji S3 pyta: „czy ostatnie okno
  danych sugeruje inne cechy?".
- `ring buffer` — pamięć ostatnich `periodN` obserwacji w postaci binów. To lokalne okno rankingu.
- `minTenure = 100` — minimalny czas życia cechy w selekcji. Nowo dodana cecha nie może wypaść
  natychmiast po kilku próbkach.
- `maxSwapsPerCycle = ceil(0.3 * K)` — maksymalna liczba cech, które wolno wymienić w jednym
  ticku. Dla `K = 5` daje to najwyżej 2 podmiany.
- `tieEpsilon = 0.01` — nowa cecha musi być wyraźnie lepsza, nie tylko minimalnie lepsza.

**Po co ograniczenia.** Gdyby S3 co 1000 instancji wybierał całe top-K od zera, selekcja mogłaby
ciągle przeskakiwać przez szum w rankingu. `maxSwapsPerCycle`, `minTenure` i `tieEpsilon`
wymuszają stopniową zmianę zamiast pełnej przebudowy.

**Wynik.** Robi **37.4 re-selekcji na przebieg** wobec 7.4 dla S2 — i wypada **gorzej**
(−0.057 vs −0.023 κ). To jest argument empiryczny, że selekcja ma być **sterowana zdarzeniem,
a nie zegarem**: przebudowa bez powodu kosztuje (reset modelu), a nic nie wnosi.

---

### 6.4. S4 — `DriftAwareSelector` (harmonogram + zdarzenie, z celowaniem)

**Kiedy przebudowuje:** **okresowo (jak S3) i dodatkowo po alarmie**, przy czym ścieżka
alarmowa **celuje w konkretne cechy** wskazane przez Level-2 detektora.

**Ścieżka danych — dwa niezależne kanały:**

```
0. Warmup:
      PiD + IG → początkowe top-K.
      Jak w S3, obserwacje trafiają też do ring buffera.

1. Każda instancja:
      • jeśli przyszedł alarm, najpierw działa kanał alarmowy,
      • PiD aktualizuje histogramy,
      • biny i klasa trafiają do ring buffera,
      • jeśli trwa zbieranie po alarmie, biny trafiają też do `postDriftRanker`,
      • co `periodN` instancji działa kanał okresowy.
```

**Kanał okresowy — taki jak w S3:**

```
1. Co `periodN = 1000` instancji budowany jest ranking IG z ring buffera.
2. Kandydaci do wyrzucenia: obecnie wybrane cechy, które przeżyły `minTenure`.
3. Kandydaci do dodania: cechy spoza selekcji.
4. Wymiana zachodzi tylko, jeśli nowa cecha jest lepsza o więcej niż `tieEpsilon`.
5. Liczba wymian jest ograniczona przez `maxSwapsPerCycle`.
6. Jeśli selekcja się zmieniła, trigger = "periodic".
```

**Kanał alarmowy — nowy względem S3:**

```
1. Przychodzi `alarm = true` i zbiór `driftingFeatures`.

2. Walidacja alarmu:
      • jeśli S4 już zbiera dane po poprzednim alarmie:
          alarmsIgnoredWhileBusy++;
          alarm odrzucony;
      • jeśli `driftingFeatures` jest puste:
          alarmsIgnoredNoTargets++;
          alarm odrzucony;
      • jeśli żadna wskazana cecha nie jest aktualnie w selekcji:
          alarmsIgnoredAllStable++;
          alarm odrzucony.

3. Jeśli alarm jest przyjęty:
      • dzielimy bieżące top-K na:
          driftingSelected = wybrane cechy wskazane jako dryfujące,
          stableSelected   = wybrane cechy niewskazane jako dryfujące;
      • PiD robi soft reset wskazanych cech (`driftDecay = 0.05`);
      • tworzony jest świeży `postDriftRanker`;
      • `collecting = true`, `collected = 0`.

4. Zbieranie danych po alarmie:
      przez `wPostDrift = 1000` instancji `postDriftRanker` uczy się
      z danych po dryfcie. Selekcja jeszcze się nie zmienia.

5. Alarmowa wymiana cech:
      po zebraniu `wPostDrift` instancji S4 próbuje wymienić tylko cechy
      z `driftingSelected`.

      Kandydaci do wyrzucenia:
          obecnie wybrane cechy, które są w `driftingFeatures`
          i mają `tenure >= minTenure`.

      Kandydaci do dodania:
          cechy spoza selekcji, które NIE są w `driftingFeatures`.

      Wymiana zachodzi tylko wtedy, gdy kandydat do dodania ma score
      większy o więcej niż `tieEpsilon`.

6. Commit:
      jeśli doszło do wymiany, trigger = "drift_alarm".
```

**Co znaczą liczniki alarmów:**

- `alarmsObserved` — ile alarmów dostał selektor.
- `alarmsAccepted` — ile alarmów uruchomiło zbieranie danych po dryfcie.
- `alarmsIgnoredWhileBusy` — alarm przyszedł, gdy trwało już zbieranie po poprzednim alarmie.
- `alarmsIgnoredNoTargets` — detektor globalny zgłosił dryft, ale Level-2 nie wskazał żadnej
  cechy.
- `alarmsIgnoredAllStable` — Level-2 wskazał cechy dryfujące, ale żadna z nich nie była w obecnym
  top-K, więc selektor nie ma czego usuwać.

**Co znaczą parametry:**

- `periodicInterval = 1000` — rytm kanału okresowego, identyczny jak w S3.
- `wPostDrift = 1000` — ile próbek po alarmie potrzeba do alarmowej wymiany.
- `minTenure = 100` — cecha musi pozostać w selekcji przez co najmniej 100 aktualizacji, zanim
  wolno ją wyrzucić.
- `driftDecay = 0.05` — soft reset PiD dla cech wskazanych przez Level-2.
- `maxSwapsPerCycle = ceil(0.3 * K)` — ograniczenie liczby wymian w pojedynczym kroku.
- `tieEpsilon = 0.01` — próg, który chroni przed wymianą na podstawie minimalnej różnicy score'u.

**Najważniejsza różnica względem S2 i S3.** S2 reaguje na alarm, ale po alarmie może zmienić
całe top-K. S3 zmienia selekcję okresowo, ale nie używa informacji o dryfujących cechach. S4
łączy oba pomysły: okresowo odświeża ranking jak S3, a po alarmie próbuje usunąć tylko te cechy
z bieżącego top-K, które Level-2 oznaczył jako dryfujące.

**Wynik.** Połączenie obu kanałów **nie pomaga** (−0.061 κ, gorzej niż samo S2). Wyjaśnienie
jest w ograniczeniu poziomu 2: skoro 91.6% alarmów nie wskazuje żadnej cechy, kanał celowany
prawie nigdy się nie uruchamia i zostaje sam harmonogram — czyli S3 z dodatkowym narzutem.

---

### 6.5. Zestawienie S1–S4

| | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| klasa | `StaticFeatureSelector` | `AlarmTriggeredSelector` | `PeriodicSelector` | `DriftAwareSelector` |
| wyzwalacz | raz, po rozgrzewce | alarm dryfu | zegar | zegar + alarm |
| celuje w cechy dryfujące | — | nie | nie | **tak** |
| re-selekcji / przebieg | 0 | 7.4 | 37.4 | 37.6 |
| Δκ vs model bez selekcji | **−0.202** | **−0.023** | −0.057 | −0.061 |

---

## 7. Krok 6 — modele klasyczne (HT, ARF, SRP)

**Ścieżka danych:** model dostaje instancję **przefiltrowaną do cech wybranych przez selektor**
(`FilteredHeaderBuilder` buduje zredukowany nagłówek, `useHardFilter = true`).

```
selektor.getCurrentSelection() → filtr instancji → model.train(...)
```

Wrappery (`HoeffdingTreeWrapper`, `ARFWrapper`, `SRPWrapper`) cache'ują selekcję i **wykrywają
jej zmianę**; przy `resetOnSelectionChange = true` przebudowują model. To dlatego migotanie
selekcji jest kosztowne i dlatego istnieje stabilizacja tie-break.

> **Ograniczenie twardej reselekcji.** Reselekcja zmienia mapowanie kolumn widzianych przez
> model. Przed zmianą kolumna 2 w zredukowanej instancji może oznaczać oryginalną cechę `f7`,
> a po zmianie — `f12`. Dla drzewa to nadal „atrybut 2", więc istniejące splity mogą przez pewien
> czas interpretować nową cechę strukturą zbudowaną dla starej. W obecnych eksperymentach
> klasyczne wrappery są budowane z `resetOnSelectionChange = false`, więc zmiana selekcji nie
> zeruje od razu całego HT/ARF/SRP. ARF i SRP mają własne mechanizmy warning/drift/background
> learner i mogą później wymienić słabe składowe, jeśli jakość spadnie, ale nie jest to
> natychmiastowy reset wywołany samą zmianą selekcji. To jest koszt twardej selekcji: mniej cech
> i szybszy model w zamian za ryzyko zaburzenia już nauczonych podziałów.

**Ważne:** wrappery **nie wołają** `selector.update(...)`. Robi to runner, dokładnie raz na
instancję, przed `model.train(...)`. Gdyby robiły to obie strony, każda instancja trafiałaby do
PiD selektora dwukrotnie.

**Ziarno losowe.** `setRandomSeed(seed)` musi być wywołane **przed** `prepareForUse()` — MOA
inicjalizuje `classifierRandom` właśnie w `resetLearning()` wywoływanym z `prepareForUse()`.
Wcześniejszy błąd polegał na odwrotnej kolejności, przez co pięć „seedów" dawało pięć
identycznych przebiegów.

---

## 8. Krok 6 (wariant) — DA-SRP: `NativeDriftAwareSRP`

### Idea i różnica względem klasycznych modeli

`NativeDriftAwareSRP` jest własnym zespołem w stylu Streaming Random Patches. Nie opakowuje MOA
`StreamingRandomPatches`, tylko sam utrzymuje tablicę learnerów:

```
ensemble[0..N-1]
    learner = ARFHoeffdingTree + jawna podprzestrzeń cech + ADWIN-y + opcjonalny background
```

Najważniejsza różnica względem `HT/ARF/SRP + S1..S4` jest taka, że DA-SRP **nie robi jednej
twardej selekcji top-K dla całego modelu**. Model ma dostęp do pełnej puli `d` cech, ale każdy
learner widzi tylko własną podprzestrzeń:

```
learner 0: {f1, f3, f7, f10, ...}
learner 1: {f0, f2, f8, f12, ...}
learner 2: {f4, f5, f7, f14, ...}
```

Selektor przekazany do konstruktora jest zachowany dla wspólnego interfejsu `ModelWrapper`, ale
nie steruje bezpośrednio wejściem modelu. Dlatego dla DA-SRP `feature_selections.csv` raportuje
unię podprzestrzeni learnerów, a nie top-K z selektora.

Metodologicznie to są dwa różne typy redukcji:

```
SRP+S1/S2/S3/S4:
    jedna twarda lista cech dla całego zespołu

DA-SRP:
    miękka selekcja przez rozkład wielu podprzestrzeni wewnątrz zespołu
```

> **Jedna implementacja.** Istniała wcześniej klasa `DriftAwareSRP`, która opakowywała MOA
> `StreamingRandomPatches` i sięgała **refleksją** do jego prywatnych tablic podprzestrzeni.
> Została **usunięta** — nie wybierał jej żaden config. Wszystkie wyniki DA-SRP pochodzą
> z `NativeDriftAwareSRP`: własny zespół `ARFHoeffdingTree`, jawne podprzestrzenie, online
> bagging, ADWIN per learner z background learnerem, zero refleksji.

### Cykl życia learnera

Każdy learner ma:

- `subspace` — indeksy oryginalnych cech, które widzi,
- `reducedHeader` — nagłówek MOA dla tej podprzestrzeni,
- `ARFHoeffdingTree` — drzewo uczące,
- `ADWIN drift` — detektor dryfu na błędzie learnera,
- opcjonalny `ADWIN warning` i `background` — świeży learner przygotowywany po ostrzeżeniu,
- okno ostatniej dokładności (`accWindow = 1000`) do diagnostyki i głosowania.

Predykcja learnera zawsze zaczyna się od projekcji pełnej instancji do jego podprzestrzeni:

```
pełna instancja x
    → FilteredHeaderBuilder.filteredInstance(x, subspace)
    → drzewo learnera
```

Trening używa online baggingu:

```
dla każdego learnera:
    k ~ Poisson(lambda)
    jeśli k > 0: trenuj drzewo k razy na tej instancji
```

Domyślnie `lambda = 6.0`. Jeśli learner ma background, background trenuje się równolegle na tej
samej instancji.

### Kanał wewnętrzny — ADWIN per learner

Niezależnie od zewnętrznego detektora z runnera, każdy learner śledzi własny błąd:

```
err_learner = 0, jeśli learner trafił
err_learner = 1, jeśli learner się pomylił
```

Ten błąd trafia do ADWIN-ów learnera:

```
warning:
    jeśli pojawia się ostrzeżenie i nie ma background learnera:
        utwórz background z nową podprzestrzenią

drift:
    jeśli pojawia się dryf:
        jeśli background istnieje:
            promuj background na głównego learnera
        inaczej:
            zbuduj nowego learnera od zera
```

Ten kanał nie potrzebuje informacji o tym, które cechy dryfują. Reaguje na pogorszenie jakości
konkretnego learnera.

### Komponent A — zewnętrzna reakcja na dryf cech

Po alarmie, **dla każdego learnera osobno**, liczony jest *overlap*: ile świeżo wykrytych cech
dryfujących leży w jego podprzestrzeni.

```
overlap = |podprzestrzeń_learnera ∩ cechy_dryfujące|

overlap == 0                        → KEEP    (nie ruszamy — nic go nie dotyczy)
0 < overlap/|podprzestrzeń| < τ     → SURGICAL(podmień TYLKO cechy dryfujące)
overlap/|podprzestrzeń| ≥ τ         → FULL    (podprzestrzeń zbyt skażona: losuj od nowa + reset)
brak akceptowalnego zamiennika      → NO_REPLACEMENT
```

Ta ścieżka wymaga nie tylko alarmu, ale też score'ów cech (`importance` albo IG). W aktualnym
runnerze `DA-SRP-A` jest budowany z `importance = null`, więc zewnętrzna ścieżka oparta na
score'ach cech nie ma z czego korzystać. Praktycznie:

```
DA-SRP-A:
    natywny zespół + online bagging + wewnętrzne ADWIN/background,
    bez importance-weighted sampling i bez korekty głosowania;

DA-SRP-AB / DA-SRP-ABC:
    jak wyżej + zewnętrzna adaptacja podprzestrzeni oparta na importance/scores.
```

Dlatego KEEP/SURGICAL/FULL należy czytać jako mechanizm dostępny wtedy, gdy model ma wektor oceny
cech. To jest ważne przy interpretacji ablacji.

**Dlaczego próg `τ = 0.5`, a nie „zawsze surgical".** Podmiana kilku cech w drzewie, którego
większość podprzestrzeni zdezaktualizowała się, zostawia strukturę zbudowaną na nieaktualnych
podziałach. Powyżej połowy taniej jest zbudować learner od zera. Poniżej — reset kosztowałby
utratę wiedzy, która wciąż jest dobra.

**Dowód, że mechanizm celuje** (21 960 par zdarzenie-learner, `e3_action_vs_overlap`):

| dryfujące cechy w podprzestrzeni | KEEP | SURGICAL | FULL | NO_REPL |
|---|---|---|---|---|
| 0 | **86.2%** | 0.0% | 13.8% | 0.0% |
| 1 | 0.0% | **86.9%** | 0.0% | 13.1% |
| 2 | 0.0% | **91.5%** | 0.0% | 8.5% |
| 3+ | 0.0% | **98.0%** | 0.9% | 1.1% |

Separacja jest zerojedynkowa w ścieżce **dryfu zlokalizowanego**: learner bez dryfującej cechy
**nigdy** nie dostaje wymiany chirurgicznej, a learner z taką cechą **nigdy** nie zostaje
nietknięty. Wiersz `overlap = 0` miesza jednak dwie sytuacje:

1. Level-2 wskazał jakieś cechy, ale dany learner żadnej z nich nie ma w podprzestrzeni. Wtedy
   normalna reguła overlapu daje `KEEP`.
2. Level-1 zgłosił alarm, ale Level-2 nie wskazał żadnej cechy (`driftingFeatures = {}`). Wtedy
   overlap każdego learnera formalnie wynosi 0, ale kod nie używa normalnej reguły overlapu,
   tylko osobny fallback `handleUnlocalized(...)`.

Te 13.8% pełnych resetów przy `overlap = 0` pochodzi właśnie z fallbacku dla **dryfu
niezlokalizowanego**: skoro nie wiadomo, które cechy wymienić chirurgicznie, metoda odświeża
część najsłabszych learnerów pełnym resetem (`FULL`). Nie przeczy to regule „overlap 0 → KEEP"
w ścieżce zlokalizowanej, bo jest to osobna gałąź algorytmu.

To odpowiada na zarzut, że zysk metody mógłby brać się z samego dodatkowego resetowania:
**przy pokryciu 0 metoda w 86% przypadków nie robi nic.**

### Komponent B — losowanie podprzestrzeni ważone importance

Komponent B mówi, **jakie cechy mają wejść do nowych albo naprawianych podprzestrzeni**. Nie
decyduje sam, czy akcją jest `KEEP`, `SURGICAL` czy `FULL`; to robi komponent A. B działa wtedy,
gdy trzeba dobrać zamiennik:

```
SURGICAL:
    wymień dryfujące cechy w istniejącej podprzestrzeni

FULL:
    wylosuj całą nową podprzestrzeń
```

Bez komponentu B losowanie jest zasadniczo uniformowe. Z komponentem B cechy są losowane z wag:

```
weight[f] = (1 − samplingBeta) · importance[f]^importancePower
            + samplingBeta · uniform[f]
```

Interpretacja:

- `importance[f]` — wysoka, gdy cecha jest jednocześnie informatywna i stabilna,
- `importancePower = 2.0` — wyostrza różnice między cechami,
- `samplingBeta = 0.7` — dodaje losowość, żeby wszystkie learnery nie zbiegały do tych samych
  kilku cech.

To jest kompromis:

```
za mało importance:
    nowe podprzestrzenie są prawie przypadkowe

za dużo importance:
    ensemble traci różnorodność, bo learnery wybierają podobne cechy
```

Przy `SURGICAL` kandydat musi być jeszcze akceptowalny względem starej cechy
(`surgicalReplacementTolerance = 0.95`). Jeśli nie ma dobrego zamiennika, akcja kończy się jako
`NO_REPLACEMENT`.

**Efekt w wynikach:** to jedyny komponent, który realnie pomaga — ale nierówno:

| przejście | mean | median | wygrane | co go napędza |
|---|---|---|---|---|
| DA-ARF: A→AB | +0.017 | +0.004 | **5/8** | zbiory realne (NHTS +0.089) |
| DA-SRP: A→AB | +0.026 | **0.000** | 4/8 | wyłącznie SEA i Hyperplane |

Dla DA-SRP mediana wynosi **dokładnie 0.000** — średnią ciągną dwa zbiory syntetyczne.

### Komponent C — top-K rank-weighted voting

Komponent C zmienia **agregację predykcji**, nie sam trening.

Najpierw liczony jest zwykły głos zespołu:

```
baseProba = predykcja wszystkich learnerów
```

Potem, jeśli jest dostępne `importance` i `correctionAlpha > 0`, model buduje korektę:

```
1. dla każdego learnera policz średnią importance jego podprzestrzeni,
2. uszereguj learnery od najwyższej średniej importance,
3. weź top ceil(topKFraction · N),
4. waż ich głosy rangowo: w_r = K - r,
5. zmieszaj:

   final = (1 - alpha) · baseProba + alpha · topKProba
```

`alpha` jest ograniczane przez `maxBlendAlpha`, żeby korekta nie zdominowała normalnego głosu
zespołu.

**Efekt w wynikach: szkodzi.** DA-SRP AB→ABC wygrywa na **2/8** zbiorach (mediana −0.004),
DA-ARF na **3/8** (mediana −0.006). To wniosek odporny na wybór zbiorów i trzeba go napisać
wprost — ujemny wynik ablacji świadczy o rzetelności metodologii.

### Mapowanie wariantów w configu

| wariant | co włączone | parametry różnicujące |
|---|---|---|
| `DA-SRP-A` | natywny SRP + wewnętrzne ADWIN/background | `tau=0.5`, `importance=null`, `correctionAlpha=0` |
| `DA-SRP-AB` | A/B: adaptacja podprzestrzeni + ważone losowanie | `w1=0.7`, `importance_power=2.0`, `sampling_beta=0.7` |
| `DA-SRP-ABC` | AB + korekta głosowania | `topk_fraction=0.3`, `correction_alpha=0.15`, `max_blend_alpha=0.5` |

### Ograniczenia DA-SRP

1. **SURGICAL nie resetuje drzewa.** Wymienia część wejścia learnera, ale zostawia jego strukturę.
   To oszczędza wiedzę, lecz może zostawić splity dopasowane do starej cechy.
2. **FULL jest drogi poznawczo.** Resetuje drzewo i detektory, więc learner musi odbudować wiedzę.
3. **Fallback niezlokalizowany jest mniej precyzyjny.** Gdy `driftingFeatures = {}`, model nie
   wie, co wymienić, więc odświeża część słabszych learnerów pełnym resetem.
4. **Komponent C empirycznie szkodzi.** To nie błąd implementacji, tylko wynik ablacji: korekta
   głosowania nie dała stabilnego zysku.

---

## 9. Krok 6 (wariant) — DA-ARF: `DAARFWrapper`

### Idea i różnica względem klasycznego ARF

`DAARFWrapper` jest własnym zespołem drzew `ARFHoeffdingTree`, projektowanym jako drift-aware
odpowiednik ARF z jawnymi podprzestrzeniami. Tak jak DA-SRP, ignoruje twardą selekcję jako
wejście całego modelu: selektor jest potrzebny do interfejsu, ale model sam losuje
podprzestrzenie per learner.

W aktualnej konfiguracji DA-ARF używa szerszej podprzestrzeni niż `K = ⌈√d⌉`:

```
subspaceSize = ceil(daArfSubspaceFraction · d)
domyślnie daArfSubspaceFraction = 0.5
```

To była świadoma korekta metodologiczna: podprzestrzeń `√d` okazała się zbyt wąska dla
strojenia DA-ARF, szczególnie przy danych realnych i niezbalansowanych.

### Budowa learnera

Każdy learner ma:

- `subspace` — własny zbiór indeksów oryginalnych cech,
- `reducedHeader` — nagłówek MOA zbudowany dla tej podprzestrzeni,
- `ARFHoeffdingTree` — drzewo bazowe,
- okno ostatniej dokładności,
- ADWIN ostrzeżenia,
- ADWIN dryfu,
- opcjonalny background learner.

Drzewo bazowe jest strojone zgodnie z MOA ARF:

```
gracePeriod = 50
splitConfidence = 0.01
```

To ma znaczenie dla uczciwego porównania: wcześniejsze, zbyt konserwatywne ustawienia drzew
handicapowały DA-ARF względem baseline'u ARF.

### Predykcja

DA-ARF nie głosuje zawsze całym zespołem jednakowo. Dla predykcji:

```
1. każdy learner głosuje na swojej podprzestrzeni,
2. learnery są sortowane po niedawnej dokładności,
3. wybierane jest top K = ceil(topKFraction · ensembleSize),
4. głosy są ważone rangowo:
      najlepszy learner: K
      drugi:             K - 1
      ...
      ostatni z top-K:   1
5. wynik jest normalizowany do rozkładu po klasach.
```

Jeśli top-K nie daje poprawnych głosów, model spada do zwykłego głosu wszystkich learnerów.

### Trening

Dla każdej instancji DA-ARF wykonuje:

```
1. każdy learner przewiduje klasę na swojej podprzestrzeni,
2. wynik aktualizuje jego okno dokładności,
3. losowane jest k ~ Poisson(lambda),
4. jeśli k > 0, learner trenuje się k razy,
5. jeśli istnieje background learner, on też trenuje się równolegle,
6. błąd foreground learnera trafia do jego ADWIN warning/drift,
7. jeśli runner zgłosił external drift alarm, działa kanał zewnętrzny.
```

Domyślnie `lambda = 6.0`, jak w ARF/SRP.

### Kanał wewnętrzny — klasyczny mechanizm ARF

```
KANAŁ WEWNĘTRZNY (jak w klasycznym ARF):
    każde drzewo ma własny ADWIN na swoim błędzie
      ├─ ostrzeżenie (warnDelta) → uruchom background learner
      └─ dryf (driftDelta)       → promuj background w miejsce drzewa
    liczniki: getIntrinsicFullResetCount(), getBkgPromotions()
```

Znaczenie:

- `warning` — learner może zacząć tracić aktualność, więc tworzony jest background,
- `background` — świeży learner trenowany równolegle z foregroundem,
- `drift` — jeśli pogorszenie się potwierdzi, background zastępuje foreground,
- `intrinsicFullReset` — jeśli dryf wystąpił bez backgroundu, learner jest tworzony od zera.

Ten kanał odpowiada na pytanie:

```
czy konkretny learner zaczął się mylić?
```

Nie potrzebuje listy dryfujących cech.

### Kanał zewnętrzny — alarm z detektora i cechy dryfujące

```
KANAŁ ZEWNĘTRZNY (nowość DA):
    globalny alarm z detektora + cechy dryfujące
      └─ wybór learnerów, których podprzestrzenie zawierają niestabilne cechy
    liczniki: getExtKeepCount(), getExtSurgicalCount(), getExtFullCount()
```

Zewnętrzny kanał robi:

```
1. Weź `driftingFeatures` z Level-2.
2. Odfiltruj cechy uznane za naprawdę niestabilne:
      lowImportanceDriftingFeatures(...)
   czyli cechy dryfujące z importance poniżej kwantyla
      unstableImportanceQuantile = 0.50.
3. Dla każdego learnera policz, czy jego podprzestrzeń zawiera te cechy.
4. Learner z overlapem > 0 staje się kandydatem do odświeżenia.
5. Liczba odświeżeń jest ograniczona:
      externalResetFraction = 0.20.
6. Spośród kandydatów wybierane są te o najniższej ostatniej dokładności.
7. Wybrane learnery dostają:
      EXT_FULL w trybie RESET,
      albo EXT_SURGICAL w trybie SURGICAL.
```

W obecnym `master_experiments.json` DA-ARF używa domyślnego trybu:

```
ExternalActionMode.RESET
```

czyli zewnętrzna reakcja to głównie `EXT_KEEP` albo `EXT_FULL`. Tryb `SURGICAL` istnieje w kodzie,
ale nie jest główną ścieżką obecnych wyników.

**Ważna różnica względem DA-SRP.** DA-ARF nie ma zewnętrznego fallbacku dla pustego
`driftingFeatures`. W `train(...)` kanał zewnętrzny uruchamia się tylko wtedy, gdy:

```
driftAlarm == true && !driftingFeatures.isEmpty()
```

Jeśli Level-1 zgłosi alarm, ale Level-2 nie wskaże cech, zewnętrzny kanał DA-ARF nie ma celu i
nie wykonuje resetu. Nadal może działać kanał wewnętrzny per learner.

### Komponent B w DA-ARF

Komponent B steruje losowaniem podprzestrzeni dla:

```
nowych learnerów foreground,
background learnerów,
EXT_FULL resetów,
EXT_SURGICAL replacement, jeśli ten tryb jest włączony.
```

Używany jest ten sam wektor `FeatureImportance` co w DA-SRP: relevance z IG plus stability z
detektora cech, potem `importancePower` i `samplingBeta`. Jeśli importance jest niedostępne lub
zdegenerowane, sampler wraca do losowania równomiernego.

### Komponent C w DA-ARF

W DA-ARF komponent C to top-K rank-weighted voting po niedawnej dokładności learnerów. Jest
prostszy niż w DA-SRP: nie ma mieszania `baseProba` przez `correctionAlpha`; model od razu
agreguje top `ceil(topKFraction · N)` learnerów z wagami rangowymi.

Empirycznie ten komponent nie poprawił jakości:

```
DA-ARF AB → ABC:
    mediana zmiany κ = -0.006
    wygrane na 3/8 zbiorach
```

Dlatego wniosek jest negatywny: ważenie głosowania po ostatniej jakości learnera nie dało
stabilnego zysku.

### Po co dwa kanały adaptacji

Kanały odpowiadają na inne pytania:

```
kanał wewnętrzny:
    czy ten konkretny learner zaczął gorzej przewidywać?

kanał zewnętrzny:
    czy podprzestrzeń tego learnera zawiera cechy wskazane jako dryfujące?
```

Rozdzielenie jest potrzebne, bo learner może tracić jakość z powodów niewidocznych dla Level-2,
ale może też mieć w podprzestrzeni dryfującą cechę zanim jego własny błąd zdąży wzrosnąć.
Rozdzielone liczniki (`intrinsic*` i `ext*`) pozwalają później zobaczyć, który mechanizm
rzeczywiście wykonywał pracę.

### Różnica mechanizmu widoczna w danych

DA-SRP działa przez **SURGICAL i KEEP**, prawie nie robiąc pełnych resetów. DA-ARF w ogóle nie
używa ścieżki chirurgicznej — wyłącznie zewnętrznej: **EXT_KEEP i EXT_FULL**. To argument
jakościowy, którego nie widać w samej κ.

> **Zastrzeżenie o liczbie zdarzeń.** DA-SRP adaptuje się o **rząd wielkości rzadziej**: od 0
> (LED) do 512 (NHTS) zdarzeń na 8 zbiorach, wobec 31–4431 dla DA-ARF. Na syntetykach DA-SRP
> schodzi do kilkunastu zdarzeń, więc proporcje akcji opierają się tam na bardzo małej próbie.
> Figura `e*_adaptation_actions` podaje `n=` nad każdym słupkiem właśnie po to.

### Wynik: rozjazd między rodzinami

- `DA-SRP-AB` bije surowy SRP na **5/8** zbiorach,
- `DA-ARF-ABC` przegrywa z surowym ARF na **6/8**.

Metoda oparta na SRP się broni, oparta na ARF — nie. Na syntetykach `DA-SRP-ABC` jest najlepsze
ze wszystkiego (0.758 vs ARF 0.755), ale na zbiorach realnych spada do 0.347 wobec 0.432 dla
ARF. Przewaga powstaje tam, gdzie założenia metody są spełnione (zlokalizowany dryf istotności
cech), i znika na danych rzeczywistych.

### Ograniczenia DA-ARF

1. **Dwa kanały mogą sobie przeszkadzać.** Jeśli kanał wewnętrzny już przygotowuje background,
   zewnętrzny reset może wejść w ten sam obszar adaptacji. W kodzie istnieje
   `gateExternalOnPendingBackground`, ale domyślnie jest wyłączone.
2. **Zewnętrzny kanał zależy od Level-2.** Gdy lokalizacja cech nie działa, DA-ARF traci część
   przewagi drift-aware i zostaje głównie klasycznym ARF z jawnymi podprzestrzeniami.
3. **EXT_FULL traci wiedzę.** W domyślnym trybie `RESET` learner jest budowany od zera. To jest
   semantycznie czystsze niż chirurgiczne przepięcie cech, ale kosztuje utratę nauczonego drzewa.
4. **Szersze podprzestrzenie zwiększają koszt.** `0.5 · d` pomaga względem zbyt wąskiego `√d`,
   ale zwiększa pamięć, czas predykcji i szansę, że wiele learnerów obejmie dryfujące cechy.

---

## 10. Krok 7 — co jest mierzone

### Metryki jakości

| metryka | po co |
|---|---|
| **κ Cohena** | metryka wiodąca. Na NHTS klasa mniejszościowa to ~0.6%, więc accuracy jest bezużyteczna (0.98 przy κ ≈ 0) |
| κ temporalna | odniesienie do klasyfikatora „powtórz poprzednią etykietę" — łapie autokorelację (NYCTaxi, ceny) |
| accuracy | tylko pomocniczo, do załącznika |
| `recovery_max_drop` | głębokość spadku po alarmie. **Jedyna metryka recovery, która osiąga istotność** |
| `recovery_time` | zmierzona poprawnie, ale nieistotna we wszystkich blokach — nie cytować |

### Metryki kosztu

| metryka | definicja |
|---|---|
| **RAM-Hours** | gigabajty trzymane przez **model**, całkowane po czasie. Mierzone przez `ModelSize.of()` → MOA `measureByteSize()` przez agenta `sizeofag` |
| `peak_mb` | szczytowy głęboki rozmiar modelu |
| `throughput` | pełny krok prequential (predykcja + detekcja + selekcja + trening) |
| `predict_latency_us` | sama inferencja, rozdzielona od powyższego |

**Dlaczego rozmiar modelu, a nie sterta JVM.** Wcześniejsza implementacja próbkowała
`Runtime.totalMemory() − freeMemory()`, czyli **całą stertę dzieloną przez 12 wątków** — wartość
zależała od tego, ile sąsiednich przebiegów akurat działało (`MajorityClassWrapper`, alokujący
jedną tablicę `long[numClasses]`, dostawał 3.5 GB na NHTS). Obecna ścieżka jest zgodna
z definicją RAM-Hours (Bifet i in., 2010).

Gdy agent `sizeofag` nie jest załadowany, `measureByteSize()` zwraca −1. `RAMHours` propaguje
to jako **unavailable** → `NaN`, zamiast klampować do zera albo podstawiać odczyt sterty. Lepiej
brak liczby niż liczba zmyślona.

> **Jednostka w tabelach.** RAM-Hours liczone z rozmiaru modelu wychodzą rzędu **10⁻⁶ GB·h**
> (modele mają 0.03–2.0 MB, przebiegi trwają minuty). Tabele podają je z mnożnikiem
> (`analysis/config.py: RAMH_SCALE`) — bez tego cała kolumna drukowała się jako `0.00`.
> W tabelach per-blok wartości są dodatkowo **normalizowane do strumienia 100k instancji**, bo
> surowa metryka całkuje po czasie i bez normalizacji odzwierciedlałaby głównie długość
> strumienia (NHTS ma 1.99 mln instancji wobec 100k dla syntetyków).

### Zapominanie i pamięć stała

System nigdy nie trzyma całego zbioru:

- **okna przesuwne** — accuracy, KSWIN, dokładność per learner,
- **multiplikatywny decay** — tabele częstości rankera są mnożone przez współczynnik < 1,
  co wykładniczo wygasza stare obserwacje,
- **event-driven forgetting** — po alarmie `softReset` czyści statystyki dryfujących cech,
  zamiast czekać, aż decay je wypłucze.

---

## 11. Pułapki i decyzje, które trzeba znać (weryfikowane w kodzie)

1. **`feature_selections.csv` zapisuje zbiór cech, którego używa MODEL, nie selektor.**
   Dla HT/ARF/SRP to jedno i to samo. Dla DA-* model ignoruje selektor i losuje własne
   podprzestrzenie — kolumna pokazuje wtedy **sumę podprzestrzeni zespołu**, a zmiany mają
   `trigger_type = subspace_change`. Wcześniej zapisywany był wybór selektora, przez co każdy
   przebieg DA wyglądał jak zamrożone S1 (`feature_stability_mean = 1.000`,
   `selection_changes_mean = 0`, `mean_selected_feature_count = ⌈√d⌉`). Naprawione — po
   poprawce DA mają stabilność ≈ 0.98 i ≈ 24.7 cech. Szczegół per learner jest
   w `adaptation_events.csv` (`per_learner_subspace`).

2. **`selector.update(...)` woła runner, nie wrapper modelu.** Wcześniejsze wersje
   dokumentacji twierdziły inaczej. W `runPrequentialLoop` jest to jawne wywołanie przed
   `model.train(...)`, dokładnie raz na instancję.

3. **Ranking cech jest liczony poza regionem mierzonym czasowo** (`updateFullFeatureRanker`
   przed `stepStart`), żeby nie obciążać baseline'ów pracą, z której korzystają tylko modele DA.
   Dlatego `throughput` jest porównywalny między wariantami.

4. **`HDDM_W` był niedeterministyczny przy wielu wątkach — naprawione serializacją.**
   MOA `HDDM_W_Test` deklaruje pięć pól stanu (`sample1/2_IncrMonitoring`,
   `sample1/2_DecrMonitoring`, `total`) jako **`private static`**, więc wszystkie instancje
   detektora w JVM dzielą jeden zestaw liczników. Przy 12 wątkach przebiegi nadpisywały sobie
   stan: dwa identyczne przebiegi `DA-ARF+HDDM_W` na Hyperplane zgadzały się na **0 z 5**
   seedów, a liczba alarmów wahała się od 3 do 12. Jednowątkowo ta sama konfiguracja odtwarza
   się dokładnie (5/5, także między osobnymi JVM), co dowodzi, że `resetLearning()` czyści
   statyki — psuje wyłącznie współbieżność. `HDDM_A_Test` i `ADWIN` nie mają pól statycznych.
   **Naprawa:** `usesSharedStateDetector()` wykrywa takie przebiegi i wykonuje je pod wspólnym
   zamkiem, z ostrzeżeniem przy starcie. **Skutek dla wyników:** liczba alarmów HDDM_W wzrosła
   z 30 do **77** — wcześniejsza teza o „oszczędnym detektorze" była artefaktem wyścigu.

5. **Etykieta `initial` była nadawana dwa razy na przebieg.** `onInitialSelection` nie ustawiało
   `lastSelectionChangeInstance`, więc pierwsza realna re-selekcja też dostawała `initial`
   zamiast swojego triggera — w E3 dawało to **75 wierszy `initial` przy 40 przebiegach**
   `ARF+S2`. Naprawione; dotyczyło wszystkich bloków.

6. **Usunięty martwy kod.** `StreamPipeline`, `RecordingMetrics`, `StreamMetrics`,
   `ExperimentRunner` i `Shims` nie leżały na ścieżce produkcyjnej (runner ma własną pętlę
   prequential, a fabryki w `Shims` rzucały `UnsupportedOperationException`). Skasowane wraz
   z refleksyjnym `DriftAwareSRP` — z 82 plików Java zostały 73. Testy sprawdzające
   `FeatureImportance` i `WeightedSubspaceSampler` przeniesiono do
   `ImportanceSamplerSmokeTest`, zamiast usunąć razem z gospodarzem.

7. **Modele deterministyczne** (HT, Majority, NoChange) mają `kappa_std = 0` na zbiorach ARFF —
   to właściwość algorytmu (`isRandomizable() == false`), nie brak losowości eksperymentu.

8. **κ na NHTS przyjmuje wartości z małego zbioru dyskretnego.** Przy klasie mniejszościowej
   0.6% model albo kolapsuje do zera, albo trafia garść instancji mniejszościowych — stąd
   powtarzające się wartości między seedami (np. `ARF+S1`: 0, 0, 0, 0.0988, 0.0988). To **nie
   jest** kolizja ziaren: 0 z 42 konfiguracji stochastycznych na zbiorach realnych ma wszystkie
   5 seedów identycznych, a 33 z nich mają 5 różnych wartości. Konsekwencja praktyczna: NHTS ma
   duże `kappa_std` i nie rozstrzyga sam z siebie sporu „DA-ARF vs ARF".

---

## 11b. Wyciek etykiety w NYC Taxi — znaleziony i naprawiony (2026-09-01)

**Mechanizm.** Etykieta `demand_level` jest liczona z `trip_count` bieżącej godziny:

```
HIGH  ⟺  trip_count_t > mediana_historyczna_strefy
```

Dlatego `trip_count` był celowo wyrzucany z ARFF (`DROP_COLS`). To nie wystarczyło — w zbiorze
zostały dwie kolumny powiązane tożsamością:

```
delta_trip_count = trip_count_t − trip_count_lag1     (definicja w features.py)
⟹ trip_count_t  = delta_trip_count + trip_count_lag1  (dokładnie, nie w przybliżeniu)
```

Model dodawał dwie cechy, odzyskiwał usuniętą wielkość i porównywał ją z `rolling_mean_24h`
(też cecha), odtwarzając regułę decyzyjną etykiety.

**Skala.** Zmierzona reguła z dwóch cech: **κ = 0.772**. Pełny ARF z 20 cechami: **0.784**.
Czyli ~98% raportowanej jakości na tym zbiorze pochodziło z odtworzenia definicji etykiety,
a nie z predykcji popytu.

**Naprawa.** `delta_trip_count` dodane do `DROP_COLS` (19 cech zamiast 20). `trip_count_lag1`
zostaje — popyt z **poprzedniej** godziny to legalna historia, a nie wielkość, z której liczona
jest etykieta.

**Skutek.** Średnia κ na NYCTaxi: **0.594 → 0.507**. Przepaść syntetyczne/realne pogłębiła się
(ARF 0.755 vs **0.432**, wcześniej 0.480) — czyli wniosek „zbiory rzeczywiste są trudniejsze"
stał się mocniejszy. Główna teza E3 **zyskała na odporności**: `DA-SRP-AB vs SRP` wygrywa teraz
na 6/8 zbiorach (było 5/8), a test leave-one-out utrzymuje dodatni znak we wszystkich ośmiu
przypadkach (wcześniej usunięcie SEA odwracało go na −0.007).

**Przy okazji, NHTS.** Usunięto `edition_boundary` — flagę równą 1 w dokładnie dwóch wierszach
na 1.99 mln, dokładnie tam, gdzie zmienia się edycja badania. W pracy o **wykrywaniu** dryfu
podawanie modelowi znacznika „tutaj zmienia się rozkład" jest tym, co detektor ma znaleźć sam.
Średnia κ na NHTS: 0.252 → 0.224. Zbiór ma teraz 16 cech.

**Czego naprawa nie załatwia.** Zadanie NYC Taxi nadal dotyczy godziny **bieżącej**, nie
następnej — cechy `avg_trip_*` i `neighbor_avg_demand` też pochodzą z godziny `t`. Opis w pracy
musi więc mówić o **klasyfikacji bieżącego poziomu popytu** (nowcasting), a nie o prognozie
na następną godzinę.

**Kontrola po przeliczeniu E1 i E3:** zbiory syntetyczne wyszły bit w bit identycznie
(450/450 przebiegów), YahooFinance bez zmian — czyli zmiana dotknęła wyłącznie tego, co miała.

---

## 12. Ograniczenia metody

### Kiedy działa dobrze

- Dryf **zlokalizowany w cechach** — zmienia się, które cechy niosą sygnał (FeatureDrift,
  Hyperplane). Tu adaptacyjna selekcja i mechanizm chirurgiczny mają czego się uchwycić.
- Cechy **warunkowo niezależne** — filtr nie widzi interakcji, więc im mniej ich w danych, tym
  lepiej dla metody.
- Strumienie, gdzie **koszt ma znaczenie** — S2 oddaje 0.023 κ za 2.4× przepustowości.

### Kiedy zawodzi

- **Dryf globalny** (SEA) — zmienia się granica decyzyjna, nie istotność cech. Adaptacja
  selekcji nie ma czego naprawiać.
- **Dryf stopniowy** — poziom 2 go nie lokalizuje (91.6% alarmów bez wskazanej cechy), więc
  mechanizm celowany nie startuje.
- **Bardzo intensywny dryf w wielu cechach naraz** — przy 10 dryfujących cechach z 20 wymiana
  chirurgiczna nie ma dokąd uciec. Widać to w E4: DA-* degraduje się **bardziej** niż baseline
  (−0.075/−0.080 wobec −0.057/−0.054).
- **Mały budżet cech przy wielu nieistotnych** — LED ma 7 istotnych z 24, a `K = ⌈√24⌉ = 5`.
  Selekcja strukturalnie nie może złapać wzorca; wszystkie warianty tracą tam podobnie.
- **Dane rzeczywiste** — przewaga DA-SRP znika (0.347 vs 0.432 dla ARF).

### Biasy do nazwania w pracy

1. **Ewaluacja prequential** faworyzuje modele szybko się uczące — każdy błąd na początku waży
   tyle samo co późniejszy.
2. **Zbiory syntetyczne mają znany ground truth**, więc metody celujące w cechy mają tam
   przewagę strukturalną. Kontrast syntetyczne/realne jest w pracy celowy właśnie po to.
3. **`K = ⌈√d⌉` jest arbitralne.** Heurystyka z lasów losowych, nie optymalizowana pod te dane.
   Wyniki S1 zależą od niej silnie.
4. **Ranker jest ustalony** (Information Gain). Praca nie testuje, czy inny score dałby inne
   wnioski — to świadome zawężenie zakresu, nie przeoczenie.

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

## 13. Poziom 0 — orkiestracja: od `main()` do CSV

### 13.1. Wejście i parsowanie configu

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

### 13.2. Rozwinięcie macierzy pracy (`expandWork`)

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
| E4 | 4 × 6 × 5 | 120 |
| E5 | 8 × 4 × 5 | 160 |
| **Σ** | | **~1250** WorkItem-ów (minus brakujące ARFF) |

### 13.3. Wykonanie równoległe (`executeAll`)

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

### 13.4. Deterministyczne domknięcie i zapisy

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

## 14. Poziom 1 — cykl życia jednego runu (`RunWorker.call`)

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

### 14.1. `openStream` — budowa strumienia

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
>   (STAGGER). Dla `numDrifts=10, n=100k` → **dryft co 10k instancji**. To jest „high dynamics"
>   w rodzinie SEA; w E4 pozostałe dwie rodziny skalują intensywność inaczej — FeatureDrift
>   liczbą dryfujących cech (2 → 10), a RandomRBF prędkością centroidów (0.001 → 0.010).
> - **Hyperplane** — 15 atr., 5 dryfujących, `magChange=sigma` → *gradual* dryft ciągły.
> - **RandomRBF** — 15 atr., 20 centroidów, 5 dryfujących, `speedChange=speed` → gradual.
> - **CustomFeatureDrift** — Hyperplane 20 atr. z `numDriftFeatures` dryfującymi — celowo
>   testuje *lokalizację cech* (czy KSWIN wskaże właściwe kolumny).

### 14.2. `collectWarmup` — bufor rozgrzewki

Drenuje **`warmup=1500`** pierwszych instancji do `warmupWindow[1500][d]` +
`warmupLabels[1500]`, ekstrahując cechy przez `space.extractFeatures`. Jeśli strumień
skończy się wcześniej — bufor przycinany do `collected`. `warmupCollected` zapamiętane.

### 14.3. `buildComponents` — budowa wszystkich obiektów runu

W tej **dokładnej** kolejności (są zależności!):

1. **`selector = buildSelector(v, d, C)`** → `selector.initialize(warmupWindow, warmupLabels)`
   (patrz sekcja 14 — S1..S4/NONE). Selektor **od razu** ma pierwszą selekcję top-k.
2. **`importance = new FeatureImportance(d)`** — na razie pusty.
3. **`buildFullFeatureRankerFromWarmup()`** — buduje **osobny, pełnowymiarowy** ranker
   (nie ten w selektorze!): własny `PiDDiscretizer(d,C)` + `InformationGainRanker`
   nad **wszystkimi** `d` cechami. Karmi go całym warmupem; jeśli `isReady()` →
   **seeduje `importance`** wartościami IG z pełnoprzestrzennego rankera (składnik stability = wektor zer). Ten ranker służy potem
   do dwóch rzeczy: (a) zasilania `importance` po alarmie, (b) dostarczania score'ów do
   `NativeDriftAwareSRP.handleDrift` / `DAARFWrapper` przy chirurgicznej podmianie cech.
4. **`model = buildModel(v, selector, header, C, seed, importance)`** (sekcja 15).
5. **`detector = buildDetector(v, d)`** → `TwoLevelDriftDetector` (sekcja 17).
6. **`metrics = new MetricsCollector(C, windowSize=1000, logEvery=0, ramSampleEvery=200)`**.

### 14.4. `attachRecorder`

`RunDetailedRecorder` dostaje tożsamość runu i:
- `onInitialSelection(warmupCollected, selector.getCurrentSelection())`,
- `onImportanceSnapshot(...)` — snapshot importance po warmupie,
- jeśli `model instanceof NativeDriftAwareSRP` → podpina `driftListener` przekazujący
  `DriftActionSummary` do `recorder.onDASRPEvent`. (Dla DA-ARF liczniki `extKeep/extFull`
  są odczytywane różnicowo w pętli — patrz 12.5.)

### 14.5. `runPrequentialLoop` — pętla per-instancja (dokładna kolejność)

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
`extKeepCount/extFullCount`; dla `NativeDriftAwareSRP` → `totalKept/Surgical/Full/NoReplacement`.

---

## 15. Poziom 2 — fabryka modeli (`buildModel`) i warianty drift-aware

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

### 15.1. `NativeDriftAwareSRP` — ablacja A / AB / ABC (bez refleksji)

> **Jedyna implementacja DA-SRP** to `NativeDriftAwareSRP` — **własny ensemble bez
> refleksji**. Usunięta już wersja (`DriftAwareSRP`) owijała MOA `StreamingRandomPatches` i refleksją
> nadpisywała jego prywatne podprzestrzenie (krucha, nieobronna). Nowa klasa jest własnym
> ensemblem: rzutowane patche `ARFHoeffdingTree` (strojone jak MOA: grace=50, δ=0.01) + online
> bagging + per-learner ADWIN, a podprzestrzenie są **jawne** w kodzie. Domyślny patch = **60%
> cech** (jak MOA SRP; wąskie ⌈√d⌉ kolapsuje na realnych). Domyślne `daSrpNative=true` — stara
> refleksyjna klasa nie jest już używana w żadnym eksperymencie. Walidacja head-to-head: natywny
> ≈ refleksyjny (lepszy na syntetyce, minimalnie słabszy na NYCTaxi/NHTS, bez kolapsów).

`newDASRP` deleguje do `newNativeDASRP`, budując `NativeDriftAwareSRP(selector, header, numClasses,
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

### 15.2. `DAARFWrapper` — DA-ARF (trzy komponenty, dwa kanały)

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

### 15.3. Warianty A/AB/ABC w configu (mapowanie ablacji)

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

## 16. Poziom 2 — fabryka selektorów (`buildSelector`) i ścieżki S1..S4

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

## 17. Poziom 2 — detektor dwupoziomowy (`buildDetector`)

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

## 18. Poziom 1 — co dokładnie mierzymy (`MetricsCollector` + `RunDetailedRecorder`)

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

## 19. Poziom 3 — analiza statystyczna per blok (`BlockStatisticalAnalysis`)

Dla **każdego bloku** i **każdej metryki** (`kappa, accuracy, ram_hours_gb, throughput…`):
1. **Macierz** `datasets × variants` (wartości uśrednione po seedach).
2. **Friedman** (omnibus): χ² + Iman-Davenport F — „czy *którakolwiek* para różni się".
3. **Nemenyi** post-hoc: `CD = q_α·√(k(k+1)/(6N))`; eksport `cd_diagram_*.csv` (rysunek w Pythonie).
4. **Wilcoxon signed-rank** parami over `(dataset, seed)` — większa moc niż rangi.
5. **Holm** step-down: korekta p-wartości w obrębie metryki (kontrola FWER).

Trzy testy = **trzy poziomy ostrości**: omnibus → ranking średnich rang → pairwise.

---

## 20. Ścieżki eksperymentów E1–E5 — co, po co, i jak czytać wynik

> Każdy blok to inne **pytanie badawcze**. Poniżej: hipoteza, warianty, datasety,
> co obserwować w wynikach.

### E1 — Baselines (`E1_baselines.csv`)
- **Pytanie:** jaki jest referencyjny poziom κ/acc/RAM-h/throughput dla naiwnych i
  klasycznych modeli, z i bez selekcji S1?
- **Warianty (8):** `Majority`, `NoChange` (baseline poziom-0), `HT/ARF/SRP` (bez FS,
  `selector=NONE`), `HT+S1/ARF+S1/SRP+S1` (ze statyczną selekcją `⌈√d⌉`).
- **Datasety (8):** 5 syntetycznych (SEA, Hyperplane, RandomRBF, FeatureDrift, **LED**) +
  3 realne ARFF (YahooFinance, NYCTaxi, NHTS).
  > Zmiana: **generyczny STAGGER usunięty** z E1/E2/E3 (κ=1.000 dla wszystkich → zero dyskryminacji).
  > Wypadł potem także z E4 (nasycony po usunięciu wariantów `+S1` — patrz sekcja E4); zostaje
  > wyłącznie jako `STAGGER-HiDyn` w E5. W zamian **LED** (LEDGeneratorDrift): 24 cechy = 7 istotnych
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

### E4 — Intensywność dryftu: Low vs HiDyn (`E4_high_dynamics.csv`)
- **Pytanie:** czy przewaga metod autorskich utrzymuje się, gdy dryf staje się intensywniejszy?
- **Warianty (4):** `ARF`, `SRP`, `DA-SRP-ABC`, `DA-ARF-ABC`. Warianty `+S1` zostały usunięte
  z tego bloku — 4 metody × 6 zbiorów dają **CD = 1.91 zamiast 3.77**, więc E4 jako jedyny blok
  ma Nemenyi'ego, który cokolwiek rozstrzyga (pozostałe mają CD 3.7–6.1).
- **Datasety (6) — trzy rodziny w parach Low/HiDyn**, w każdej parze zmieniany jest **jeden**
  parametr intensywności, a przestrzeń cech jest identyczna:

  | rodzina | oś intensywności | Low → HiDyn | parametr |
  |---|---|---|---|
  | SEA | liczba nagłych zmian | 3 → 10 | `num_drifts` |
  | FeatureDrift | ile cech dryfuje | 2 → 10 | `drift_features` |
  | RandomRBF | prędkość centroidów | 0.001 → 0.010 | `speed` |

- **Odrzucone osie dynamiki** (warto zachować jako wynik metodologiczny): **Hyperplane** —
  przemiatanie magnitudy 0.001–0.3 zmieniało średnie κ niemonotonicznie i w złą stronę;
  **STAGGER** — obie jego wersje są nasycone (κ = 1.000), gdy z bloku wypadną warianty `+S1`,
  bo to wyłącznie ich kolaps przy K = ⌈√3⌉ = 2 je wcześniej różnicował.
- **Jak czytać:** różnica κ **wewnątrz pary** Low → HiDyn izoluje wpływ samej intensywności
  (reszta jest identyczna) — to najczystszy kontrast w całej pracy. `recovery_time` i
  `windows.csv` (spadki κ wokół punktów zmiany) uzupełniają obraz.
- **Wynik przeczy hipotezie:** zakładano, że DA-* stracą mniej niż zwykłe ARF/SRP. Średni spadek
  Low → HiDyn wynosi ARF −0.057, SRP −0.054, ale **DA-ARF-ABC −0.075 i DA-SRP-ABC −0.080** —
  metody drift-aware degradują się *bardziej*, najsilniej na FeatureDrift (−0.167 / −0.178),
  czyli w scenariuszu, pod który były projektowane. Prawdopodobna przyczyna: przy 10 dryfujących
  cechach chirurgiczna wymiana nie ma dokąd uciec — brakuje stabilnych cech na nową
  podprzestrzeń (por. wzrost NO\_REPL w E3). To ma trafić do pracy wprost.

### E5 — Studium detektorów (`E5_detectors.csv`)
- **Pytanie:** który detektor (ADWIN/HDDM_A/HDDM_W/KSWIN) daje najlepszą stabilność DA-ARF?
- **Warianty (8):** `DA-ARF × {ADWIN, HDDM_A, HDDM_W, KSWIN}` + baseline'y `ARF/SRP(+S1)+ADWIN`.
- **Datasety (4):** `SEA-HiDyn`, `STAGGER-HiDyn`, `Hyperplane`, `RandomRBF`.
- **Metryki-klucz:** `ext_full_count` + `drift_count` — mierzą jak „nerwowy" jest detektor
  (za dużo full-resetów = niestabilność). KSWIN używa ciasnego ADWIN L1 + KSWIN L2 do
  lokalizacji (sekcja 17).

---

## 21. Mapa „gdzie w kodzie" (szybki indeks)

| Etap | Klasa / metoda | Plik |
|---|---|---|
| Config + orkiestracja | `UnifiedStreamExperimentRunner.{main,run,expandWork,executeAll}` | `experiments/UnifiedStreamExperimentRunner.java` |
| Jeden run | `RunWorker.{call,openStream,collectWarmup,buildComponents,runPrequentialLoop}` | ↑ |
| Strumienie + dryft | `SyntheticStreamFactory.{createSEA,createMultiDriftSEA,createSTAGGER,createHyperplane,createRandomRBF,createCustomFeatureDrift,createLEDDrift,addNoiseFeatures}` | `pipeline/SyntheticStreamFactory.java` |
| Dyskretyzacja | `PiDDiscretizer` / `FeatureDiscretizer` / `Layer1Histogram` / `Layer2Merger` | `discretization/` |
| Rankery | `FilterRanker` → `AbstractFrequencyRanker` → `InformationGainRanker` (jedyna implementacja; `MutualInformationRanker` i `ChiSquaredRanker` zostały usunięte — żaden wariant w `master_experiments.json` ich nie używał) | `selection/` |
| Selektory | `NoFeatureSelection`, `StaticFeatureSelector`, `AlarmTriggeredSelector`, `PeriodicSelector`, `DriftAwareSelector` | `selection/` |
| Detekcja | `TwoLevelDriftDetector`, `PerFeatureKSWIN`, `KSWINSingleFeature`, `ADWIN/HDDMChangeDetector` | `detection/` |
| Modele | `HT/ARF/SRPWrapper`, `NativeDriftAwareSRP`, `DAARFWrapper`, `FeatureImportance`, `WeightedSubspaceSampler`, `DriftActionSummary`, `DriftEvent`, `ModelSize` | `models/` |
| Metryki | `MetricsCollector` + `CohenKappa/TemporalKappa/RAMHours/RecoveryTime/FeatureStabilityRatio` | `evaluation/` |
| Detaliczne CSV | `RunDetailedRecorder` | `experiments/RunDetailedRecorder.java` |
| Statystyka | `BlockStatisticalAnalysis` + `FriedmanTest/NemenyiPostHoc/WilcoxonSignedRank/StatisticalTests` | `experiments/`, `evaluation/` |
| Config macierzy | `master_experiments.json` | `experiments/master_experiments.json` |
