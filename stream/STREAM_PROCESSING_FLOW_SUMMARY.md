# Stream processing, selekcja cech i adaptacja modeli

Ten dokument podsumowuje flow projektu w formie metodologicznej: co dzieje sie z pojedyncza instancja, jak dziala PiD, jak wykrywany jest dryft globalny i dryft cech, jak dzialaja selektory `S1`-`S4`, oraz jak zwykle modele (`HT`, `ARF`, `SRP`) roznia sie od wariantow `DA-SRP` i `DA-ARF`.

Opis dotyczy aktualnego kodu projektu, szczegolnie:

- `stream/src/main/java/thesis/experiments/UnifiedStreamExperimentRunner.java`
- `stream/src/main/java/thesis/pipeline/SyntheticStreamFactory.java`
- `stream/src/main/java/thesis/discretization/*`
- `stream/src/main/java/thesis/detection/*`
- `stream/src/main/java/thesis/selection/*`
- `stream/src/main/java/thesis/models/*`

## 1. Glowna idea systemu

Projekt implementuje uczenie online na strumieniu danych. Dane nie sa przetwarzane batchowo, tylko instancja po instancji.

Dla kazdej instancji model najpierw wykonuje predykcje, potem poznaje prawdziwa klase, a dopiero potem aktualizowane sa detektory, selektory, metryki i model. Jest to podejscie prequential:

```text
predict -> observe true label -> update/train
```

Glowna idea metodologiczna jest taka:

```text
model ma uczyc sie online,
detektor ma wykrywac kiedy zmienil sie strumien,
KSWIN ma lokalizowac ktore cechy zmienily rozklad,
selekcja cech ma adaptowac wejscie lub podprzestrzenie,
ensemble ma reagowac bez resetowania wszystkiego naraz.
```

Projekt ma dwa style selekcji:

```text
HT / ARF / SRP + S1-S4:
    twarda selekcja cech, czyli model dostaje tylko wybrane top-K cech

DA-SRP / DA-ARF:
    miekka selekcja cech, czyli model ma dostep do pelnej puli cech,
    ale podprzestrzenie learnerow sa sterowane importance i dryftem
```

## 2. Flow pojedynczej instancji

W `UnifiedStreamExperimentRunner.runPrequentialLoop()` pojedyncza instancja przechodzi przez nastepujacy proces:

```text
1. Pobranie instancji ze strumienia.
2. Wyciagniecie cech `feats` i etykiety `yTrue`.
3. Aktualizacja globalnego PiD i globalnego rankera cech.
4. Predykcja modelu `yHat`.
5. Wyliczenie bledu `err = 0/1`.
6. Aktualizacja detektora dryftu: Level-1 i Level-2 buffers.
7. Jesli jest alarm, aktualizacja FeatureImportance.
8. Aktualizacja metryk.
9. Aktualizacja selektora cech.
10. Trening modelu na aktualnej instancji.
11. Zapis artefaktow do recorderow.
```

W uproszczeniu:

```java
double[] feats = space.extractFeatures(x);
int yTrue = (int) x.classValue();

updateFullFeatureRanker(feats, yTrue);

int yHat = model.predict(x);
double err = (yHat == yTrue) ? 0.0 : 1.0;

detector.update(err, feats);
boolean alarm = detector.isGlobalDriftDetected();
Set<Integer> drifting = alarm ? detector.getDriftingFeatureIndices() : Set.of();

if (alarm) updateFeatureImportanceFromDetector();

metrics.update(yTrue, yHat, elapsed);
if (alarm) metrics.onDriftAlarm();

selector.update(feats, yTrue, alarm, drifting);
model.train(x, yTrue, alarm, drifting);
```

Wazny niuans: globalny ranker cech jest aktualizowany przed predykcja tej instancji. Nie trenuje to bezposrednio modelu, ale aktualizuje statystyki pomocnicze. Sama predykcja modelu nadal jest wykonana przed treningiem modelu na tej instancji.

## 3. Globalny PiD i ranker cech

Przed predykcja wykonywane jest:

```java
updateFullFeatureRanker(feats, yTrue);
```

Ta metoda aktualizuje globalna wiedze o tym, ktore cechy sa informatywne.

Proces:

```text
surowe cechy -> PiD -> biny dyskretne -> ranker IG -> score cech
```

### 3.1. Po co PiD

Ranker czestosciowy (`InformationGainRanker`, jedyna implementacja `AbstractFrequencyRanker`)
pracuje na wartosciach dyskretnych — liczy IG z tensora kontyngencji `[cecha][bin][klasa]`.
Dane wejsciowe moga byc ciagle, wiec trzeba je zdyskretyzowac.

PiD robi to osobno dla kazdej cechy:

```text
wartosc ciagla cechy -> bin Layer-1 -> bin Layer-2
```

W aktualnym kodzie domyslne parametry `PiDDiscretizer` to:

```text
Layer-1: 64 biny
Layer-2: 8 binow
warmupN: 500 wartosci na ceche
recomputeEvery: 1000 aktualizacji
expandThreshold: 0.20
decayFactor: 1.0
```

### 3.2. Layer-1

Layer-1 to histogram rownej szerokosci. Dla kazdej cechy po warmupie wyznaczany jest zakres `min/max`, a potem dzielony na 64 biny.

Przy kazdej nowej instancji dla kazdej cechy:

```text
1. Znajdz bin Layer-1.
2. Zwieksz licznik binu.
3. Zwieksz licznik klasy w tym binie.
```

Przyklad:

```text
f0 = 2.4, y = 1
f0 trafia do Layer-1 bin 12

binCounts[12] += 1
classCounts[12][1] += 1
```

### 3.3. Rozszerzanie zakresu

Jesli po warmupie zakres cechy byl np.:

```text
0..100
```

a pozniej przyjdzie wartosc:

```text
500
```

kod rozszerzy zakres histogramu:

```java
if (value < l1.getMin() || value >= l1.getMax()) {
    expandRange(value);
}
```

Nowy zakres obejmie stara skale, nowa wartosc i margines okolo 20%.

Rozszerzenie moze tez nastapic, gdy za duzo masy trafia na krawedzie histogramu:

```java
if (l1.shouldExpand(expandThreshold)) {
    expandRange(value);
}
```

Sens: jesli wiele wartosci zaczyna kleic sie do krawedzi, to rozklad prawdopodobnie przesuwa sie poza pierwotny zakres. Rozszerzenie zakresu zapobiega temu, ze duzo obserwacji bedzie upychanych w skrajnych binach.

Trade-off: zakres sie rozszerza, ale liczba binow Layer-1 zostaje stala. Jesli zakres mocno urosnie, pojedynczy bin obejmuje wiekszy przedzial, czyli spada rozdzielczosc dyskretyzacji.

### 3.4. Layer-2

Layer-2 grupuje 64 biny Layer-1 w 8 wiekszych binow. Mapowanie:

```text
Layer-1 bin -> Layer-2 bin
```

jest okresowo przeliczane.

Layer-2 nie scala binow losowo. Scalanie opiera sie na podobienstwie rozkladow klas w sasiednich binach.

Dla dwoch sasiednich binow liczymy rozklady klas:

```text
bin A: P(class)
bin B: Q(class)
```

i odleglosc total variation:

```text
TV(P, Q) = 0.5 * sum_c |P(c) - Q(c)|
```

Im mniejsza odleglosc, tym bardziej podobne sa biny pod wzgledem klas i tym bardziej sensownie je polaczyc.

Algorytm jest zachlanny:

```text
1. Start: 64 grupy Layer-1.
2. Znajdz pare sasiednich grup o najmniejszym koszcie polaczenia.
3. Scal je.
4. Powtarzaj az zostanie 8 grup.
```

### 3.5. Kiedy liczy sie Layer-2 i ranker

Layer-1 aktualizuje sie przy kazdej instancji.

Layer-2 przelicza sie:

```text
co recomputeEvery = 1000 aktualizacji danej cechy
```

Po aktualizacji PiD aktualna instancja jest dyskretyzowana:

```java
fullRankingPid.discretizeAll(feats)
```

a ranker aktualizuje swoje liczniki:

```text
joint[f][layer2Bin][class] += 1
```

Wazny niuans: przeliczenie Layer-2 nie przebudowuje historycznych licznikow rankera od zera. Nowe instancje beda mapowane wedlug nowego Layer-2, ale ranker dziala inkrementalnie.

## 4. Information Gain jako score cechy

W glownym runnerze uzywany jest przede wszystkim `InformationGainRanker`.

`Information Gain` mierzy, ile znajomosc danej cechy zmniejsza niepewnosc co do klasy:

```text
IG(Y, X_f) = H(Y) - H(Y | X_f)
```

Intuicja:

```text
wysokie IG:
    biny cechy dobrze rozdzielaja klasy

niskie IG:
    niezaleznie od binu rozklad klas jest podobny
```

Przyklad:

```text
f2 bin 0 -> prawie zawsze klasa 0
f2 bin 7 -> prawie zawsze klasa 1
```

Taka cecha ma wysokie IG.

Jesli:

```text
kazdy bin f5 -> 50% klasa 0, 50% klasa 1
```

to cecha ma niskie IG.

## 5. Detekcja dryftu: dwa poziomy

Projekt stosuje detekcje dwupoziomowa:

```text
Level-1:
    globalny dryft na strumieniu bledow modelu

Level-2:
    lokalizacja, ktore cechy zmienily rozklad
```

### 5.1. Level-1: ADWIN/HDDM

Po predykcji liczony jest blad:

```java
double err = (yHat == yTrue) ? 0.0 : 1.0;
```

Do Level-1 trafia strumien:

```text
0, 0, 1, 0, 1, 1, ...
```

ADWIN/HDDM nie wie, jaka byla klasa ani jaka byla predykcja. Widzi tylko, czy model zaczyna popelniac wiecej bledow.

Jesli Level-1 wykryje zmiane, oznacza to:

```text
model globalnie zaczal zachowywac sie inaczej
```

### 5.2. Level-2: dryft cech

Dryft cechy oznacza:

```text
rozkład wartosci danej cechy zmienil sie w czasie
```

To nie jest to samo co wyrzucenie cechy z selekcji. `driftingFeatures` to cechy wskazane przez detektor KS + BH, a nie cechy usuniete przez selektor.

Proces w aktualnym `TwoLevelDriftDetector`:

```text
1. Kazda instancja aktualizuje rolling buffer cech.
2. Level-1 monitoruje blad modelu.
3. Jesli Level-1 wykryje globalny dryft:
   - snapshot ostatnich 200 wartosci cech jako reference window,
   - zaczyna sie zbieranie post-drift window.
4. Przez kolejne 200 instancji zbierane jest post window.
5. Dla kazdej cechy porownuje sie reference vs post testem KS.
6. Wyniki p-value sa korygowane przez Benjamini-Hochberg FDR.
7. Powstaje set `driftingFeatures`.
```

Domyslnie:

```text
kswinWindowSize = 200
bhQ = 0.10
```

### 5.3. KS bez dyskretyzacji

Dryft cech liczony jest na surowych wartosciach numerycznych. Nie ma tutaj PiD.

Dla kazdej cechy:

```text
reference[f] = stare 200 wartosci cechy f
post[f]      = nowe 200 wartosci cechy f
```

Test KS liczy:

```text
D = max_x |F_reference(x) - F_post(x)|
```

oraz p-value.

Interpretacja:

```text
niskie p-value:
    rozklady prawdopodobnie sa rozne

wysokie p-value:
    brak silnego dowodu na zmiane rozkladu
```

### 5.4. Benjamini-Hochberg FDR

Poniewaz test KS jest robiony dla wielu cech naraz, potrzebna jest korekta wielokrotnego testowania.

Bez korekty, przy 100 cechach i `alpha = 0.05`, oczekiwalibysmy okolo:

```text
100 * 0.05 = 5
```

falszywych alarmow nawet bez realnego dryftu.

Benjamini-Hochberg kontroluje FDR, czyli oczekiwana frakcje falszywych odkryc wsrod cech oznaczonych jako dryfujace.

W kodzie:

```text
1. Posortuj p-value rosnaco.
2. Dla rangi i policz prog: (i / F) * bhQ.
3. Znajdz najwieksza range, gdzie p_i <= prog.
4. Wszystkie cechy do tej rangi uznaj za dryfujace.
```

Domyslnie:

```text
bhQ = 0.10
```

To nie znaczy, ze 10% cech moze dryfowac. To znaczy, ze wsrod wykryc kontrolujemy oczekiwana frakcje falszywych wskazan.

## 6. FeatureImportance

`FeatureImportance` laczy informatywnosc i stabilnosc cechy.

W uproszczeniu:

```text
importance[f] = w1 * normalizedScore[f] + w2 * stability[f]
```

Domyslnie:

```text
w1 = 0.7
w2 = 0.3
epsilon = 1e-6
```

W aktualnym kodzie do `FeatureImportance` trafia score z `InformationGainRanker`, chociaz pole historycznie nazywa sie `miScores`.

Po alarmie:

```java
double[] scores = fullRanker.getFeatureScores();
double[] ksProxy = invertPValues(p, numFeatures);
importance.update(scores, ksProxy);
```

`ksProxy` jest liczony jako:

```text
1 - pValue
```

Niskie p-value oznacza silny sygnal zmiany, wiec daje wysoki proxy sygnal niestabilnosci.

Po co importance:

```text
1. Do wazonego losowania nowych podprzestrzeni w DA-SRP i DA-ARF.
2. Do oceny, ktore dryfujace cechy sa nisko wazne.
3. Do korekty glosowania w DA-SRP-ABC.
```

## 7. Selekcja cech S1-S4

Wszystkie strategie `S1`-`S4` wybieraja:

```text
K = ceil(sqrt(liczba_cech))
```

i uzywaja:

```text
PiD + Information Gain
```

Roznia sie tym, kiedy i jak zmieniaja selekcje.

### 7.1. S1: StaticFeatureSelector

S1 wybiera cechy raz na warmupie:

```text
warmup -> PiD -> IG -> top-K -> selekcja zamrozona
```

Po warmupie S1 nie aktualizuje selekcji.

Zalety:

```text
stabilnosc,
prostota,
brak migotania cech.
```

Trade-offy:

```text
brak reakcji na dryft,
cechy dobre na starcie moga stac sie slabe pozniej.
```

### 7.2. S2: AlarmTriggeredSelector

S2 reaguje na globalny alarm dryftu.

Proces:

```text
1. Normalnie trzyma aktualna selekcje.
2. Po alarmie:
   - akceptuje alarm, jesli nie jest juz w trybie zbierania,
   - robi decay/soft reset PiD dla dryfujacych cech,
   - resetuje ranker,
   - zaczyna zbierac okno post-drift.
3. Przez `wPostDrift` instancji zbiera nowe dane.
4. Po oknie liczy ranking IG.
5. Wybiera nowe top-K z preferencja starej selekcji przy malych roznicach.
```

Zalety:

```text
selekcja zmienia sie tylko gdy jest sygnal dryftu,
ranking opiera sie na danych po zmianie.
```

Trade-offy:

```text
zalezy od jakosci detektora,
reaguje z opoznieniem, bo czeka na okno post-drift,
falszywy alarm moze wywolac niepotrzebny reranking.
```

### 7.3. S3: PeriodicSelector

S3 dziala okresowo, niezaleznie od alarmow.

Proces:

```text
1. Kazda instancja aktualizuje PiD i ring buffer.
2. Co `periodN` instancji:
   - ranker liczony jest z ostatniego okna,
   - kandydaci do wyjscia sortowani sa od najslabszych,
   - kandydaci do wejscia sortowani sa od najlepszych,
   - wykonywana jest ograniczona liczba swapow.
```

Domyslnie limit swapow:

```text
maxSwapsPerCycle = ceil(0.3 * K)
```

Zalety:

```text
nie zalezy od detektora dryftu,
moze adaptowac sie do powolnych zmian.
```

Trade-offy:

```text
moze zmieniac selekcje bez realnego dryftu,
moze zareagowac dopiero przy nastepnym cyklu.
```

### 7.4. S4: DriftAwareSelector

S4 laczy podejscie okresowe i alarmowe. Najwazniejsza roznica: uzywa `driftingFeatures`.

Po alarmie:

```text
1. Dostaje `driftingFeatures`, np. {2, 7, 11}.
2. Czysci indeksy spoza zakresu.
3. Sprawdza, czy dryfujace cechy sa w aktualnej selekcji.
4. Jesli zadna dryfujaca cecha nie jest aktualnie wybrana, ignoruje alarm.
5. Dla dryfujacych cech robi soft reset PiD.
6. Zbiera okno post-drift.
7. Liczy ranking IG na danych po dryfcie.
8. Probuje wymienic tylko dryfujace cechy z aktualnej selekcji.
```

Przyklad:

```text
aktualna selekcja = {1, 2, 5, 7}
driftingFeatures  = {2, 7, 11}

dryfujace i wybrane = {2, 7}
```

S4 probuje wymienic `2` i `7`, ale nie rusza `11`, bo `11` nie bylo uzywane.

Kandydaci do wyrzucenia:

```text
cechy w selekcji,
ktore sa dryfujace,
ktore maja wystarczajacy tenure.
```

Kandydaci do wejscia:

```text
cechy poza selekcja,
ktore nie sa oznaczone jako dryfujace.
```

Swap jest wykonany tylko jesli:

```text
score(in) > score(out) + tieEpsilon
```

Sens:

```text
nie wymieniaj cech przez minimalny szum w score,
nie wymieniaj wszystkiego naraz,
celuj w konkretne cechy dotkniete dryftem.
```

Trade-offy:

```text
zalezy od poprawnej lokalizacji dryftu przez KS/BH,
bardziej zlozony mechanizm,
moze zignorowac dryft, jesli dryfujace cechy nie przecinaja aktualnej selekcji.
```

## 8. Stabilizacja selekcji: preferredOrder i tieEpsilon

Przy rerankingu uzywany jest mechanizm stabilizacji:

```java
selectTopK(k, preferredOrder, tieEpsilon)
```

`preferredOrder` to zwykle poprzednia selekcja.

Cel:

```text
jesli dwie cechy maja bardzo podobny score,
zostaw ceche, ktora juz byla wybrana.
```

Bez tego ranking moglby migotac:

```text
{1, 3, 5} -> {1, 7, 5} -> {1, 3, 5}
```

Taki flickering destabilizuje model, bo model dostaje ciagle inna przestrzen wejsc.

## 9. Zwykle modele HT, ARF, SRP z S1-S4

Dla zwyklych modeli selekcja dziala jako hard filter.

Model nie dostaje pelnej instancji:

```text
[f0, f1, f2, f3, f4]
```

tylko wybrane cechy:

```text
[f1, f3]
```

### 9.1. HT

`HoeffdingTreeWrapper` filtruje instancje przez aktualna selekcje.

Jesli selekcja sie zmieni, wrapper buduje nowy `reducedHeader`, ale w aktualnej konfiguracji:

```text
resetOnSelectionChange = false
```

czyli drzewo nie jest resetowane.

Konsekwencja:

```text
drzewo moze byc semantycznie niespojne,
bo historyczne splity mogly byc uczone na starej interpretacji kolumn.
```

Przyklad:

```text
przed: kolumna 0 modelu = f2
po:    kolumna 0 modelu = f7
```

Trade-off:

```text
plus:
    brak kosztu pelnego resetu

minus:
    pojedyncze drzewo moze gorzej znosic zmiane przestrzeni cech
```

### 9.2. ARF

`ARFWrapper` tez uzywa hard filtera, jesli selektor nie jest `NONE`.

W aktualnej konfiguracji:

```text
useHardFilter = true
resetOnSelectionChange = false
```

Po zmianie selekcji ARF dostaje nowy zestaw cech, ale caly las nie jest resetowany przez wrapper.

ARF ma wewnetrzne mechanizmy adaptacyjne MOA. Jesli wewnetrzny detektor uzna, ze learner dziala zle, moze z czasem wymienic drzewo juz w nowej przestrzeni cech.

Trade-off:

```text
plus:
    ensemble amortyzuje zmiane lepiej niz pojedyncze drzewo

minus:
    stare learnery moga przez pewien czas byc niespojne z nowa selekcja
```

### 9.3. SRP

`SRPWrapper` analogicznie stosuje hard filter dla zwyklego `SRP + S1/S2/S3/S4`.

Po zmianie selekcji:

```text
SRP dostaje nowa przefiltrowana przestrzen,
ale nie jest resetowany przez wrapper.
```

Zwykly MOA `StreamingRandomPatches` ma swoje weighted voting. Lepsze learnery moga miec wiekszy wplyw na glosowanie, ale jest to mechanizm bazowy MOA, nie oparty na `FeatureImportance` projektu.

### 9.4. Wynik empiryczny: modele bez selekcji cech

W wynikach projektu warianty bez selekcji (`selector = NONE`) sa bardzo waznym punktem odniesienia. Oznaczaja:

```text
HT / ARF / SRP:
    model widzi wszystkie cechy

HT+S1 / ARF+S1 / SRP+S1:
    model widzi tylko statyczne top-K z warmupu
```

Empirycznie wynik jest mocny: w wielu datasetach modele bez selekcji sa lepsze niz `S1`.

Przyklady z `master_summary.csv`, metryka `kappa_mean`, E1:

```text
FeatureDrift:
    ARF = 0.667 vs ARF+S1 = 0.380
    SRP = 0.584 vs SRP+S1 = 0.257

Hyperplane:
    ARF = 0.718 vs ARF+S1 = 0.194
    SRP = 0.617 vs SRP+S1 = 0.121

NYCTaxi:
    ARF = 0.783 vs ARF+S1 = 0.452
    SRP = 0.759 vs SRP+S1 = 0.462

RandomRBF:
    ARF = 0.937 vs ARF+S1 = 0.846
    SRP = 0.929 vs SRP+S1 = 0.792
```

Srednio w E1:

```text
ARF       = 0.666
ARF+S1    = 0.482

HT        = 0.605
HT+S1     = 0.429

SRP       = 0.617
SRP+S1    = 0.399
```

Interpretacja:

```text
statyczna selekcja S1 czesto za mocno obcina przestrzen cech.
```

S1 wybiera top-K tylko raz na warmupie. Jesli pozniej wazne staja sie inne cechy, model ich nie widzi. Dla modeli ensemble (`ARF`, `SRP`) pelna przestrzen cech czesto jest korzystna, bo ich wewnetrzna losowosc i adaptacja potrafia same ignorowac czesc slabych cech.

To nie znaczy, ze selekcja cech jest bez sensu. Wyniki E2 pokazuja, ze adaptacyjna selekcja potrafi odzyskac jakosc:

```text
Hyperplane:
    ARF = 0.718
    ARF+S1 = 0.194
    ARF+S2 = 0.754

FeatureDrift:
    ARF = 0.667
    ARF+S1 = 0.380
    ARF+S2 = 0.704
```

Czyli wniosek nie brzmi "selekcja cech jest zla", tylko:

```text
statyczna, globalna selekcja top-K moze byc zbyt agresywna,
natomiast selekcja adaptacyjna albo miekka selekcja ensemblowa
ma wiekszy sens w strumieniu z dryftem.
```

Trade-off:

```text
bez selekcji:
    plus: model nie traci potencjalnie waznych cech
    plus: ARF/SRP moga same wykorzystac roznorodnosc cech
    minus: wiekszy koszt obliczeniowy i pamieciowy
    minus: wiecej szumu dla slabszych modeli

z S1:
    plus: mniej cech, nizszy koszt, prostszy model
    minus: ryzyko wyrzucenia cech, ktore beda wazne po dryfcie

z S2/S3/S4:
    plus: mozliwosc odzyskania cech po zmianie
    minus: wieksza zlozonosc i zaleznosc od detekcji/rerankingu
```

Ten wynik jest wazny metodologicznie, bo pokazuje, ze w streamingu sama redukcja wymiaru nie jest automatycznie korzystna. Selekcja musi byc albo adaptacyjna, albo miekka i rozproszona po learnerach, jak w `DA-SRP`/`DA-ARF`.

## 10. DA-SRP

`DA-SRP` to adaptacyjna wersja SRP, ale nie dziala jak zwykly `SRP + S4`.

Najwazniejsze:

```text
DA-SRP ma dostep do wszystkich cech.
Selektor nie jest hard-filterem wejscia.
Adaptacja dzieje sie przez podprzestrzenie learnerow.
```

> **AKTUALIZACJA (Option B): DA-SRP to teraz `NativeDriftAwareSRP` — wlasny ensemble bez
> refleksji.** Poprzednia wersja (`DriftAwareSRP`) opakowywala MOA `StreamingRandomPatches`
> i **refleksja** grzebala w jego prywatnych podprzestrzeniach (krucha, nieobronna). Nowa wersja
> jest wlasnym ensemblem — rzutowane patche `ARFHoeffdingTree` (strojone jak MOA: grace=50,
> δ=0.01) + online bagging + per-learner ADWIN — a podprzestrzenie sa **jawne** w kodzie, wiec
> nic nie jest czytane/pisane refleksja. Domyslny patch = **60% cech** (jak MOA SRP; wąskie sqrt
> kolapsuje na realnych danych). Stara refleksyjna klasa zostaje w kodzie, ale **nie jest juz
> uzywana w zadnym eksperymencie** (`da_srp_native=true` domyslnie). Walidacja head-to-head:
> natywny ≈ refleksyjny (lepszy na syntetyce, minimalnie slabszy na NYCTaxi/NHTS, bez kolapsow).

### 10.1. Dlaczego DA-SRP widzi wszystkie cechy

Gdyby DA-SRP widzial tylko globalne top-K, ensemble straciloby roznorodnosc. SRP z natury opiera sie na podprzestrzeniach cech.

DA-SRP robi selekcje miekka:

```text
cechy wazne i stabilne maja wieksza szanse trafic do podprzestrzeni,
cechy dryfujace i slabe sa wymieniane lub unikane,
ale zadna cecha nie jest globalnie odcieta raz na zawsze.
```

To jest zasadne, bo po dryfcie cecha spoza aktualnego top-K moze stac sie informatywna.

### 10.2. DriftingFeatures vs selectedFeatures

Trzeba rozroznic:

```text
driftingFeatures:
    cechy wskazane przez KS + BH jako zmieniajace rozklad

selectedFeatures:
    cechy wybrane przez selektor S1-S4

removedFromSelection:
    cechy wyrzucone przez selektor po rerankingu
```

DA-SRP reaguje przede wszystkim na:

```text
driftingFeatures ∩ subspace learnera
```

a nie na cechy wyrzucone z selekcji.

### 10.3. Reakcja DA-SRP po dryfcie

Kazdy learner SRP ma swoja podprzestrzen:

```text
L0: {0, 1, 3}
L1: {2, 4, 8}
L2: {2, 7, 11}
L3: {5, 6, 9}
```

Jesli:

```text
driftingFeatures = {2, 7, 11}
```

DA-SRP liczy overlap:

```text
L0 overlap = 0 -> KEEP
L1 overlap = 1 -> mozliwe SURGICAL
L2 overlap = 3 -> mozliwe FULL
L3 overlap = 0 -> KEEP
```

Akcje:

```text
KEEP:
    learner nie uzywa problematycznych cech

SURGICAL:
    learner uzywa niewielu problematycznych cech,
    wiec wymieniane sa tylko te cechy w subspace

FULL:
    duza czesc subspace jest problematyczna,
    wiec learner dostaje nowa podprzestrzen i jest resetowany

NO_REPLACEMENT:
    kod chcial wymienic ceche, ale nie znalazl dobrego zamiennika
```

Granica dla `SURGICAL` vs `FULL`:

```text
overlap / subspaceSize < tau -> SURGICAL
overlap / subspaceSize >= tau -> FULL
```

Domyslnie:

```text
tau = 0.5
```

### 10.4. DA-SRP-A, DA-SRP-AB, DA-SRP-ABC

Warianty sa narastajace.

```text
DA-SRP-A:
    adaptacja po dryfcie przez KEEP/SURGICAL/FULL

DA-SRP-AB:
    A + FeatureImportance
    nowe podprzestrzenie sa losowane z preferencja cech waznych/stabilnych

DA-SRP-ABC:
    AB + korekta glosowania
    learnery z lepszymi podprzestrzeniami maja dodatkowy wplyw na predykcje
```

W `ABC` predykcja (w natywnym ensemblu):

```text
1. Wlasny ensemble liczy zwykly glos (baseProba) — usredniony po wszystkich learnerach.
2. DA-SRP liczy srednia importance podprzestrzeni kazdego learnera.
3. Bierze top-K learnerow po tym score.
4. Agreguje ich glosy rangowo.
5. Miesza wynik z bazowym glosem ensembla:

final = (1 - correctionAlpha) * baseProba
        + correctionAlpha * weightedTopKProba
```

Domyslnie:

```text
correctionAlpha = 0.15
```

Czyli `ABC` nie zastepuje SRP, tylko delikatnie koryguje glosowanie na podstawie quality podprzestrzeni.

## 11. DA-ARF

`DA-ARF` to wlasna implementacja drift-adaptive ARF oparta na `ARFHoeffdingTree`.

Podobnie jak DA-SRP:

```text
DA-ARF ma dostep do pelnej puli cech,
ale kazdy learner widzi tylko swoja podprzestrzen.
```

Selektor jest zachowany dla kompatybilnosci interfejsu, ale nie dziala jako globalny hard filter.

### 11.1. Budowa learnera

Kazdy learner ma:

```text
subspace:
    zestaw cech

tree:
    ARFHoeffdingTree

warning ADWIN:
    detektor ostrzezenia

drift ADWIN:
    detektor dryftu

background:
    opcjonalny learner zapasowy

recentAccuracy:
    okno trafien do rankingu learnerow
```

### 11.2. Podprzestrzenie

Jesli jest `FeatureImportance`, nowe podprzestrzenie sa losowane wazenie:

```text
wysoka importance -> wieksza szansa trafienia do subspace
```

Jesli nie ma importance, losowanie jest uniformowe.

> **AKTUALIZACJA (naprawa DA-ARF): dwie poprawki bazowego lasu.** (1) **Strojenie drzew** —
> `newTree()` ustawia teraz grace=50, δ=0.01, maxByteSize=2e6 (jak MOA `AdaptiveRandomForest`);
> wczesniej drzewa dziedziczyly domyslne `HoeffdingTree` (grace=200, **δ=1e-7**) → plytkie drzewa
> → kolaps do klasy wiekszosciowej na niezbalansowanym NHTS (κ=0). (2) **Szersza podprzestrzen** —
> domyslnie **0.5·d** (`daarf_subspace_fraction`) zamiast ⌈√d⌉; wąska projekcja za mocno obcina
> sygnal na realnych danych, a strojone (glebsze) drzewa na wąskiej podprzestrzeni przeuczaja sie
> pod dryf ciagly. Obie poprawki potrzebne razem.

To jest miekka selekcja cech:

```text
cechy wazne pojawiaja sie czesciej,
ale model nie odcina globalnie reszty cech.
```

### 11.3. Predykcja DA-ARF

DA-ARF nie glosuje wszystkimi learnerami rowno.

Proces:

```text
1. Dla kazdego learnera obliczana jest recentAccuracy.
2. Learnery sa sortowane od najlepszych do najslabszych.
3. Brane jest K = ceil(topKFraction * ensembleSize).
4. Glosy top-K learnerow sa wazone ranga:
   najlepszy dostaje najwieksza wage.
5. Wynik jest normalizowany.
```

Sens:

```text
bardziej ufamy learnerom, ktore ostatnio lepiej dzialaly.
```

### 11.4. Trening DA-ARF

Dla kazdej instancji i kazdego learnera:

```text
1. Losowane jest k ~ Poisson(lambda).
2. Learner robi predykcje.
3. Liczony jest lokalny blad learnera.
4. Aktualizowane jest recentAccuracy.
5. Jesli k > 0, learner trenuje z waga k.
6. Background learner, jesli istnieje, trenuje rownolegle.
7. Blad trafia do prywatnych ADWIN-ow learnera.
```

### 11.5. Wewnetrzny drift per learner

Kazdy learner ma swoje ADWIN-y:

```text
warning ADWIN:
    jesli wykryje warning, tworzony jest background learner

drift ADWIN:
    jesli wykryje drift:
        jesli background istnieje, zastapi foreground,
        inaczej learner jest resetowany od nowa
```

To daje adaptacje lokalna:

```text
nie resetujemy calego lasu,
tylko learner, ktory sam zaczal dzialac zle.
```

### 11.6. Zewnetrzny drift z pipeline

DA-ARF dostaje tez:

```text
driftAlarm
driftingFeatures
```

Jesli alarm jest prawdziwy i `driftingFeatures` nie jest puste, odpala:

```java
externalKeepOrFull(drifting)
```

Proces:

```text
1. Z dryfujacych cech wybierz tylko nisko wazne:
   lowImportanceDriftingFeatures.

2. Znajdz learnerow, ktorych podprzestrzen przecina te niestabilne cechy.

3. Ogranicz liczbe resetow:
   externalResetFraction = 0.20

4. Wsrod kandydatow wybierz learnerow z najnizsza recentAccuracy.

5. Zresetuj ich i daj im nowe podprzestrzenie, unikajac unstable cech.

6. Reszta zostaje KEEP.
```

Domyslnie DA-ARF stosuje konserwatywny schemat `KEEP albo FULL reset` (bez `SURGICAL`).

```text
KEEP albo FULL reset
```

> **Uwaga (Option A / diagnoza):** dodano opcjonalny tryb `SURGICAL` dla external layer
> (`daarf_external_mode`), ale runy diagnostyczne pokazaly, ze na DA-ARF **szkodzi** (NYCTaxi
> κ 0.81→0.71) — chirurgia cech w drzewie ARF psuje semantyke podzialow. Dlatego domyslny tryb
> to `RESET`. Ustalono tez, ze external layer prawie nie odpala sie na syntetykach (intrinsic
> ADWIN robi cala prace przez promocje background) — patrz poprawiona sekcja 11.8.

### 11.7. DA-ARF-A, DA-ARF-AB, DA-ARF-ABC

W konfiguracji projektu:

```text
DA-ARF-A:
    samplingBeta = 1.0
    topKFraction = 1.0
    background learners wylaczone

DA-ARF-AB:
    importance-weighted sampling wlaczone
    background learners wlaczone
    topKFraction = 1.0

DA-ARF-ABC:
    importance-weighted sampling wlaczone
    background learners wlaczone
    topKFraction = 0.5
```

Interpretacja:

```text
A:
    bazowa adaptacja per learner

AB:
    A + importance jako miekka selekcja cech

ABC:
    AB + top-K rank-weighted voting
```

### 11.8. Wynik empiryczny DA-ARF: pozorny negative result → ZDIAGNOZOWANY i NAPRAWIONY

**Wczesniejszy wniosek (nieaktualny):** w E3 DA-ARF prawie zawsze byl slabszy niz `ARF`/`ARF+S2`,
co interpretowano jako "brak bezpiecznej chirurgii cech w ARF" i podawano jako *negative result*.

**Diagnoza (sekcja A planu, runy diagnostyczne) obalila te hipoteze:**

1. **Nie ma podwojnego resetu ani problemu z chirurgia.** Intrinsic ADWIN prawie zawsze promuje
   cieply background (nigdy nie robi destrukcyjnego full-resetu), a external layer na syntetykach
   prawie w ogole sie nie odpala. Wersja `SURGICAL` sprawdzona empirycznie — **pogarsza**.
2. **Ablacja B/C: komponenty POMAGAJA** (usuniecie importance sampling lub top-K voting szkodzi).
   DA-ARF-ABC to najlepsza konfiguracja i na FeatureDrift **bije** ARF.
3. **Prawdziwa przyczyna = dwa ukryte handicapy BAZOWEGO lasu**, ktore falszowaly porownanie z ARF:
   - **nietrojone drzewa** (grace=200/δ=1e-7 zamiast MOA 50/0.01) → plytkie drzewa → kolaps na NHTS;
   - **za waska podprzestrzen** (⌈√d⌉) → za malo cech na realne dane.

**Po naprawie** (strojone drzewa + podprzestrzen 0.5·d — patrz 11.2) DA-ARF-ABC przechodzi z
"przegrywa 7/8" na **konkurencyjny/lepszy na 7/8** zbiorow (jedyny kolaps: NHTS przy skrajnym
niezbalansowaniu). Reprezentatywnie (30k, seed 1–2): Hyperplane 0.72, RandomRBF 0.92,
FeatureDrift 0.74 (> ARF), NYCTaxi 0.90, LED 0.71 (= pelny ARF).

**Wlasciwy wniosek do pracy** (mocniejszy niz stary negative result): *slabosc autorskiej metody
lezala w wiernosci odtworzenia bazowego ensembla (strojenie drzew, szerokosc podprzestrzeni), a
NIE w idei drift-aware.* Rygorystyczna sciezka diagnostyczna (odrzucanie kolejnych hipotez danymi)
jest sama w sobie wartosciowym materialem.

> **UWAGA: liczby w sekcji 15 ponizej pochodza ze STAREGO runu (przed naprawa DA-ARF i przed
> natywnym DA-SRP, z STAGGER, ze starymi nazwami baseline'ow).** Sa nieaktualne dla DA-ARF —
> zostawione jako referencja do czasu pelnego re-runu. Po `bash stream/run_experiments.sh` +
> `python -m analysis` nalezy je podmienic.

## 12. Co daje to podejscie, zalety i ograniczenia

### 12.1. Zalety

System rozdziela trzy problemy:

```text
1. Czy cos sie zmienilo?
   -> Level-1 ADWIN/HDDM na bledach modelu

2. Gdzie sie zmienilo?
   -> Level-2 KS + BH na cechach

3. Jak zareagowac?
   -> S2/S4, DA-SRP, DA-ARF
```

To daje bardziej precyzyjna adaptacje niz globalny reset modelu.

Zalety:

```text
lepsza reakcja na concept drift,
mniej niepotrzebnych resetow,
mozliwosc lokalizacji problematycznych cech,
miekka selekcja w ensemble,
zachowanie roznorodnosci learnerow,
ograniczenie falszywych wskazan przez BH-FDR.
```

### 12.2. Trade-offy, ograniczenia i threats to validity

Podejscie jest bardziej zlozone i ma kilka ryzyk. To jest tez lista "limitations",
ktora trzeba umiec wymienic na interview.

**Ograniczenia detekcji i rankingu:**

1. **KS wykrywa zmiane rozkladu cechy P(X), nie P(y|X).** Cecha moze zmienic rozklad
   bez zmiany predykcyjnej roli (virtual drift). Dlatego KS jest laczony z importance/IG.
2. **Filter ranking jest univariate.** Nie widzi interakcji (XOR): cecha z MI≈0
   samodzielnie moze byc kluczowa w parze. Wrapper approach bylby drozszy.
3. **S4 zalezy od jakosci lokalizacji `driftingFeatures`.** Jesli KS/BH nie wskaze
   dobrej cechy, reakcja jest zbyt slaba.

**Ograniczenia dyskretyzacji i pamieci modelu:**

4. **PiD Layer-1 min/max sa zamrozone w warmupie.** Range drift → wartosci trafiaja do
   `UNKNOWN_BIN` i sa pomijane w rankingu → mozliwe zanizenie score cechy.
5. **PiD Layer-2 zmienia mapowanie binow, ale ranker nie przebudowuje historii.**
   Szybki online, ale nie idealny batchowy recompute.
6. **Zwykle HT/ARF/SRP z hard-filterem nie resetuja sie po zmianie selekcji**
   (`resetOnSelectionChange=false`) → mozliwa niespojnosc historycznych splitow
   (kolumna 0 modelu moze znaczyc inna ceche po zmianie).

**Ograniczenia protokolu i ewaluacji:**

7. **Globalny ranker jest aktualizowany przed predykcja.** Nie trenuje modelu
   bezposrednio, ale w rygorystycznej interpretacji prequential mozna by chciec
   przesunac to po predykcji.
8. **Delayed labels pominiete.** Prequential zaklada natychmiastowa etykiete; w realu
   error-based detektor reagowalby pozniej.
9. **Ground truth dryftu w danych realnych jest wywnioskowany**, nie znany —
   czesciowo skonfundowany zmianami zewnetrznymi/metodologicznymi.
10. **Mala liczba datasetow** → testy nieparametryczne; ostroznosc z twierdzeniami o
    dominacji (E3 Friedman p=0.24 — patrz sekcja 15.3).

**Ograniczenia interpretacyjne:**

11. **DA-SRP i DA-ARF sa bardziej zlozone interpretacyjnie.** Selekcja jest miekka i
    rozproszona po learnerach, a nie jednym top-K.
12. **(DO WERYFIKACJI po re-runie)** Na STARYM runie komponenty B/C w DA-SRP nie zawsze
    poprawialy srednia. Uwaga: po naprawie DA-ARF ablacja B/C pokazala, ze **B i C POMAGAJA**
    (usuniecie szkodzi) — stary wniosek byl czesciowo artefaktem zepsutego DA-ARF i refleksyjnego
    DA-SRP. Ostateczna ocena B/C dopiero po pelnym re-runie z naprawionym kodem.

## 13. Najkrotsze podsumowanie

Flow systemu:

```text
instancja
 -> globalny PiD/ranker aktualizuje statystyki cech
 -> model przewiduje klase
 -> blad 0/1 idzie do Level-1 detektora
 -> po globalnym alarmie Level-2 lokalizuje dryfujace cechy przez KS + BH
 -> importance laczy informatywnosc i stabilnosc
 -> S1-S4 moga zmienic globalna selekcje cech
 -> zwykle HT/ARF/SRP dostaja nowy hard-filter cech
 -> DA-SRP/DA-ARF uzywaja pelnej puli cech i adaptuja podprzestrzenie learnerow
 -> model trenuje sie na instancji
```

Najwazniejsze rozroznienie:

```text
S1-S4 dla zwyklych modeli:
    globalna, twarda selekcja top-K

DA-SRP / DA-ARF:
    miekka, ensemblowa selekcja przez importance, dryft i podprzestrzenie
```

To podejscie probuje uniknac dwoch skrajnosci:

```text
1. Nie ignorowac dryftu i nie trzymac statycznych cech na zawsze.
2. Nie resetowac brutalnie calego modelu po kazdej zmianie.
```

Zamiast tego system robi adaptacje lokalna:

```text
wykryj zmiane,
zlokalizuj cechy,
zmien tylko to, co prawdopodobnie trzeba zmienic.
```

---

# Eksperymenty, wyniki i ewaluacja

> Sekcje 1-13 opisuja *jak dziala kod*. Sekcje 14-19 opisuja *jak jest testowany*:
> eksperymenty, faktyczne liczby (zweryfikowane z `stream/results/*.csv`), metryki,
> statystyke, dane i architekture runnera. Zrodla prawdy dla liczb:
> `E1_baselines.csv`, `E2_adaptive.csv`, `E3_ablation.csv`, `E4_high_dynamics.csv`,
> `E5_detectors.csv`, `E3/E3_stats_report.txt`. Wszystkie liczby to `kappa_mean`
> po **5 seedach**, chyba ze zaznaczono inaczej.
>
> **Przy konfliktach ze starszym tekstem pracy / notatkami zrodlem prawdy jest kod
> i te CSV.** Konkretnie: wczesniejsze notatki podawaly, ze `DA-SRP-ABC` jest
> najlepszy i osiaga ~0.797 na NYC Taxi — aktualne wyniki tego nie potwierdzaja
> (patrz sekcja 15.3). Uzywaj liczb stad.
>
> **⚠️ LICZBY W SEKCJI 15 SA PRZED-NAPRAWCZE (pending re-run).** Pochodza z runu sprzed:
> (a) naprawy DA-ARF (strojone drzewa + podprzestrzen 0.5·d), (b) natywnego DA-SRP, (c) usuniecia
> STAGGER i dodania LED, (d) zmiany nazw baseline'ow. **DA-ARF jest tu jeszcze zepsuty** — po
> pelnym re-runie (`bash stream/run_experiments.sh` → `python -m analysis`) bedzie konkurencyjny
> (patrz sekcja 11.8). Tabele ponizej podmienic po re-runie.

## 14. Projekt eksperymentow: bloki E1-E5 i pytania badawcze

Cala macierz jest w jednym pliku `master_experiments.json`, uruchamiana przez
`UnifiedStreamExperimentRunner` (12 watkow, 5 seedow domyslnie). Kazdy blok
odpowiada jednemu pytaniu badawczemu (RQ).

| Blok | RQ | Co testuje | Warianty | Datasety |
|---|---|---|---|---|
| **E1** | RQ1 | Baseline'y modelowe: jak dzialaja modele bez selekcji i co robi statyczne S1? | HT/ARF/SRP bez FS, HT+S1/ARF+S1/SRP+S1, Majority, NoChange | 8 (5 syntetycznych + Yahoo/NYC/NHTS) |
| **E2** | RQ2 | Czy adaptacyjna FS (S2/S3/S4) poprawia statyczna S1 i/lub raw model bez FS? | ARF/SRP × {NONE,S1,S2,S3,S4} | 5 syntetycznych (znany ground truth) |
| **E3** | RQ3 | Ablacja drift-aware: A / AB / ABC dla SRP i ARF | SRP+S1, ARF+S2, DA-SRP-{A,AB,ABC}, DA-ARF-{A,AB,ABC} | 8 |
| **E4** | RQ4 | Reakcja pelnego wariantu na high- vs low-dynamics | ARF+S1, SRP+S1, DA-SRP-ABC, DA-ARF-ABC | SEA/STAGGER × {Low, HiDyn} |
| **E5** | RQ5 | Ktory detektor gra najlepiej z DA-ARF? | DA-ARF × {ADWIN, HDDM_A, HDDM_W, KSWIN} + ARF/SRP+ADWIN | SEA-HiDyn, STAGGER-HiDyn, Hyperplane, RandomRBF |

**Kluczowa mysl konstrukcyjna:** E1 ustawia punkt odniesienia, E2 izoluje wplyw
*selekcji cech*, E3 izoluje wplyw *poszczegolnych komponentow* metody drift-aware,
E4 sprawdza *odpornosc na tempo dryftu*, E5 sprawdza *wrazliwosc na wybor detektora*.
To jest klasyczna ablacja: nie "czy caly system dziala", tylko "ktory kawalek daje
zysk i w jakich warunkach".

> **AKTUALIZACJA zbioru datasetow:** z E1/E2/E3 **usunieto generyczny STAGGER** (κ=1.000 dla
> wszystkich modeli → zero dyskryminacji; STAGGER-HiDyn/-Low zostaje w E4/E5). W zamian dodano
> **LED (LEDGeneratorDrift)** — 24 cechy = **7 istotnych + 17 nieistotnych**, 10 klas, dryf
> gradualny; idealny do pokazania FS (znany ground truth cech nieistotnych) i mocno dyskryminujacy.
> Nazwy baseline'ow w E3 uproszczono (`SRP`, `SRP+S1`, `ARF`, `ARF+S2` zamiast `*_baseline`).
> Warianty ablacyjne naprawy DA-ARF (`-untuned`/`-narrow`) i porownawczy `DA-SRP-ABC-reflect`
> byly jednorazowe — nie sa w glownej macierzy (liczby w `THESIS_IMPROVEMENT_PLAN.md`).

---

## 15. Faktyczne wyniki liczbowe (zweryfikowane z CSV)

### 15.1. E1 — baseline'y modelowe: bez selekcji cech (kappa, 5 seedow)

W tej sekcji slowo **baseline** oznacza przede wszystkim model bez zewnetrznej
selekcji cech (`selector = NONE`): `HT`, `ARF`, `SRP`. Warianty `HT+S1`,
`ARF+S1`, `SRP+S1` to nie glowny baseline modelowy, tylko statyczna redukcja
top-K z warmupu. To rozroznienie jest wazne, bo w wynikach S1 czesto pogarsza
jakosc wzgledem modelu widzacego wszystkie cechy.

| Dataset | HT (NONE) | HT+S1 | ARF (NONE) | ARF+S1 | SRP (NONE) | SRP+S1 | Uwaga |
|---|---:|---:|---:|---:|---:|---:|---|
| RandomRBF | 0.885 | 0.784 | **0.937** | 0.842 | 0.922 | 0.792 | pelna przestrzen wygrywa |
| LED | **0.713** | 0.592 | 0.710 | 0.589 | 0.705 | 0.595 | 7 istotnych z 24; K=5 nie wystarcza |
| SEA | 0.611 | 0.603 | 0.745 | **0.754** | 0.403 | 0.535 | jedyny zbior, gdzie S1 pomaga |
| NYCTaxi | **0.797** | 0.464 | 0.784 | 0.468 | 0.778 | 0.450 | S1 mocno obcina sygnal |
| FeatureDrift | 0.614 | 0.358 | **0.672** | 0.389 | 0.622 | 0.305 | raw modele duzo lepsze niz S1 |
| YahooFinance | 0.069 | 0.043 | **0.280** | 0.204 | 0.152 | 0.043 | trudny dataset, ale NONE > S1 |
| Hyperplane | 0.637 | 0.190 | **0.712** | 0.193 | 0.647 | 0.156 | S1 bardzo szkodliwe |
| NHTS | 0.225 | -0.007 | 0.377 | 0.040 | **0.410** | 0.077 | acc ~0.98 przy kappa ~0 dla S1 |

> **STAGGER zniknal z E1** (kappa = 1.000 dla kazdego wariantu → zero dyskryminacji);
> w zamian doszedl **LED**. Wczesniejsza wersja tej tabeli zawierala liczby sprzed naprawy
> propagacji seeda — najwieksza roznica to **NHTS/ARF: 0.188 → 0.377** (stara wartosc byla
> jednym pechowym losowaniem powielonym piec razy) oraz **NHTS/SRP+S1: 0.000 → 0.077**,
> czyli slynne „kappa = 0" nie bylo kolapsem modelu, tylko artefaktem.

Srednie po E1:

```text
ARF bez FS = 0.652  vs  ARF+S1 = 0.435
HT  bez FS = 0.569  vs  HT+S1  = 0.378
SRP bez FS = 0.580  vs  SRP+S1 = 0.369
```

Wnioski do interview:

1. **Baseline'em modelowym powinien byc wariant bez selekcji cech**, bo pokazuje,
   ile daje sam model (`HT/ARF/SRP`) bez dodatkowego filtera.
2. **S1 czesto jest za agresywne**, bo wybiera top-K tylko raz na warmupie i
   moze wyrzucic cechy, ktore pozniej staja sie wazne.
3. **NHTS pokazuje, czemu accuracy nie wystarcza**: acc okolo 0.98 moze isc w parze
   z kappa bliska 0 przy silnym imbalance klasy wiekszosciowej.
4. **Yahoo Finance jest trudny predykcyjnie**, ale nawet tam `NONE` jest zwykle
   lepsze niz statyczne `S1`.
5. **Adaptacja ma sens dopiero jako korekta slabosci S1**, szczegolnie tam, gdzie
   relevance cech sie zmienia (patrz E2: Hyperplane, FeatureDrift).

### 15.2. E2 — adaptacyjna selekcja cech pomaga, GDY zmienia sie relevance

| Dataset | ARF+S1 | ARF+S2 | zysk S2 vs S1 | Interpretacja |
|---|---|---|---|---|
| Hyperplane | 0.193 | **0.752** | **+0.559** | relevance sie zmienia → ogromny zysk |
| FeatureDrift | 0.389 | **0.722** | **+0.333** | dryft dotyka konkretnych cech → duzy zysk |
| RandomRBF | 0.842 | 0.842 | 0.000 | relevance stabilna → brak zysku |
| SEA | 0.754 | 0.756 | +0.002 | dryft globalny, nie feature-specific |
| LED | 0.589 | 0.589 | 0.000 | K=5 z 24 cech — selekcja strukturalnie nie lapie wzorca |

> **Uwaga o punkcie odniesienia.** Zysk liczony wzgledem S1 jest zawyzony, bo S1 jest
> najslabsza mozliwa referencja. Wzgledem modelu **bez selekcji** (raw ARF) ten sam `ARF+S2`
> zyskuje na Hyperplane tylko **+0.040**, a nie +0.559 — reszta to odrobienie szkody
> wyrzadzonej przez statyczny podzbior. Patrz `tab_e2_delta_vs_raw`.

Dla SRP podobnie (Hyperplane 0.121→0.636 dla S2). **S2 zwykle >= S4** — bardziej
zlozony S4 nie byl konsekwentnie lepszy (np. FeatureDrift ARF: S2=0.704 > S4=0.647;
S3=0.673). To potwierdza teze: *adaptacja jest warunkowa, nie darmowa*.

> Teza: A static feature subset fails badly exactly when the *identity* of the
> informative features changes (Hyperplane, FeatureDrift). When relevance is
> stable (RandomRBF) or drift is global rather than feature-specific (SEA), the
> extra adaptation buys nothing and can only add variance and cost.

### 15.3. E3 — ablacja DA-SRP / DA-ARF

DA-SRP, kappa (8 zbiorow, 5 seedow):

| Dataset | SRP+S1 | DA-SRP-A | DA-SRP-AB | DA-SRP-ABC | ARF+S2 | ARF (raw) |
|---|---|---|---|---|---|---|
| RandomRBF | 0.792 | 0.921 | 0.932 | 0.928 | 0.842 | **0.937** |
| LED | 0.595 | 0.694 | 0.718 | **0.720** | 0.589 | 0.710 |
| SEA | 0.535 | 0.610 | 0.753 | 0.749 | **0.756** | 0.745 |
| Hyperplane | 0.156 | 0.631 | 0.717 | 0.714 | **0.752** | 0.712 |
| FeatureDrift | 0.305 | 0.676 | 0.661 | 0.680 | **0.722** | 0.672 |
| NYCTaxi | 0.450 | 0.708 | 0.687 | 0.678 | 0.474 | **0.784** |
| NHTS | 0.077 | **0.388** | 0.378 | 0.313 | 0.378 | 0.377 |
| YahooFinance | 0.043 | 0.103 | 0.092 | 0.064 | 0.226 | **0.280** |

- **Srednie rangi (kappa, 10 metod):** ARF=4.00 < DA-SRP-AB=4.375 < ARF+S2=4.50 <
  DA-ARF-AB=4.75 < DA-ARF-A=5.00 < DA-SRP-ABC=5.375 < DA-ARF-ABC=5.50 < SRP=5.625 <
  DA-SRP-A=6.125 < SRP+S1=9.75. Nizsza ranga = lepiej.
- **Friedman kappa: p = 0.0072** — istotny. Nemenyi ma tu jednak CD = 4.79 przy skali 1-10,
  wiec rozstrzyga tylko skrajnosci; wnioski opieraj na Wilcoxonie z poprawka Holma
  (11/45 par istotnych).
- **Najlepszy wariant DA to AB, nie A i nie ABC.** Komponent B (importance-weighted
  sampling) pomaga, komponent C (top-K voting) szkodzi w obu rodzinach.

> **Uwaga: ta sekcja byla wczesniej oparta o stary przebieg** (sprzed naprawy propagacji
> seeda) i miala odwrotne wnioski — „najlepszy wariant to A", „Friedman p = 0.243",
> „DA-ARF slabszy przez bugi bazowego lasu". Zadne z tych zdan nie obowiazuje.

Ablacja komponentow (srednia kappa po 8 zbiorach):

| metoda | A | AB | ABC |
|---|---|---|---|
| DA-ARF | 0.6007 | 0.6175 (**+0.017**) | 0.6134 (**-0.004**) |
| DA-SRP | 0.5914 | 0.6171 (**+0.026**) | 0.6058 (**-0.011**) |

**Czytaj to ostroznie — srednia po zbiorach jest krucha.** Dla `DA-SRP-AB vs SRP` calosc
dodatniej sredniej (+0.037) pochodzi z **jednego zbioru**: SEA wnosi +0.044, a po jego
usunieciu srednia spada do **-0.007**. Powod jest znany — raw SRP ma na SEA kappa 0.403
przy ARF 0.745, wiec DA-SRP naprawia tam konkretna patologie SRP, a nie poprawia go ogolnie.
Mediana (+0.011) i liczba wygranych (5/8) sa odporne i to je nalezy cytowac.
Szczegoly w `tab_e3_ablation_deltas` (kolumny mean / median / std / wins / worst).

**Rozjazd miedzy rodzinami — nie zacieraj go:**

- `DA-SRP-AB` bije surowy SRP na **5/8** zbiorow (mediana +0.011),
- `DA-ARF-ABC` przegrywa z surowym ARF na **6/8** (mediana -0.015).

Metoda oparta na SRP sie broni, oparta na ARF — nie. Na zbiorach syntetycznych
`DA-SRP-ABC` jest najlepsze ze wszystkiego (0.758 vs ARF 0.755), ale na realnych spada do
0.347 wobec 0.432 dla ARF: przewaga powstaje tam, gdzie zalozenia metody sa spelnione
(zlokalizowany dryft istotnosci cech), i znika na danych rzeczywistych.

**Dowod, ze adaptacja jest celowana** (`e3_action_vs_overlap`, 21 960 par zdarzenie-learner):
learner, w ktorego podprzestrzeni nie ma zadnej swiezo dryfujacej cechy, dostaje KEEP w 86%
przypadkow i **nigdy** wymiany chirurgicznej; learner z co najmniej jedna taka cecha dostaje
SURGICAL w 87-98% i **nigdy** nie zostaje nietkniety. To odpowiada na zarzut, ze zysk
komponentu B moglby wynikac z samego dodatkowego resetowania.

### 15.4. E4 — intensywnosc dryftu (Low vs HiDyn)

Blok przebudowany: **4 metody x 6 zbiorow** w trzech rodzinach (STAGGER usuniety jako
nasycony, warianty `+S1` usuniete). Dzieki temu Nemenyi ma CD = 1.91 zamiast 3.77 — to
jedyny blok, w ktorym ten test cokolwiek rozstrzyga.

| Dataset | ARF | SRP | DA-ARF-ABC | DA-SRP-ABC |
|---|---|---|---|---|
| SEA-Low | 0.731 | 0.489 | **0.737** | 0.736 |
| SEA-HiDyn | 0.733 | 0.536 | 0.749 | **0.749** |
| FeatureDrift-Low | 0.754 | 0.745 | 0.783 | **0.794** |
| FeatureDrift-HiDyn | **0.642** | 0.593 | 0.617 | 0.617 |
| RandomRBF-Low | **0.937** | 0.922 | 0.909 | 0.928 |
| RandomRBF-HiDyn | **0.875** | 0.866 | 0.839 | 0.853 |

Osie intensywnosci: SEA = liczba nagłych zmian (3 → 10), FeatureDrift = ile cech dryfuje
(2 → 10), RandomRBF = predkosc centroidow (0.001 → 0.010). W kazdej parze przestrzen cech
jest identyczna, wiec roznica **wewnatrz pary** izoluje sam wplyw intensywnosci.

**Wynik przeczy hipotezie.** Zakladano, ze DA-* straci mniej niz zwykle ARF/SRP. Sredni
spadek Low → HiDyn: ARF **-0.057**, SRP **-0.054**, ale DA-ARF-ABC **-0.075** i
DA-SRP-ABC **-0.080**. Najostrzej na FeatureDrift (-0.167 / -0.178), czyli dokladnie
w scenariuszu, pod ktory metoda byla projektowana. Prawdopodobna przyczyna: przy 10
dryfujacych cechach wymiana chirurgiczna nie ma dokad uciec — brakuje stabilnych cech na
nowa podprzestrzen.

**Osobny wniosek metodologiczny:** na SEA **wszystkie** metody zyskuja przy wiekszej liczbie
dryfow (SRP nawet +0.047). Sama czestotliwosc dryfu nie utrudnia zadania — SEA cyklicznie
wraca do wczesniejszych konceptow, wiec 10 zmian to wiecej powtorzen tego samego. Utrudnia
dopiero **zakres** dryfu: ile cech sie zmienia (FeatureDrift) albo jak szybko (RandomRBF).

### 15.5. E5 — wybor detektora ma znaczenie (dla DA-ARF)

| Dataset | ADWIN | HDDM_A | HDDM_W | KSWIN |
|---|---|---|---|---|
| Hyperplane | 0.710 | 0.720 | 0.722 | **0.734** |
| RandomRBF | **0.909** | 0.905 | 0.905 | 0.901 |
| SEA-HiDyn | **0.749** | 0.741 | 0.733 | 0.744 |
| STAGGER-HiDyn | **1.000** | 0.928 | 0.878 | 0.989 |

Wartosci dotycza `DA-ARF` z danym detektorem. Rozpietosc miedzy detektorami jest **mala**
(0.01-0.12 kappa), znacznie mniejsza niz roznica miedzy modelami — dla porownania
`ARF+ADWIN` = 0.712 wobec `SRP+S1+ADWIN` = 0.156 na Hyperplane.

**Sama liczba alarmow nie rozstrzyga — liczy sie ich skutecznosc.** Mierzac zmiane accuracy
w oknie po alarmie (`tab_e5_alarm_effectiveness`):

| wariant | alarmy | srednia dacc | uzyteczne (>= 1 pp) | szkodliwe (<= -1 pp) |
|---|---|---|---|---|
| DA-ARF+HDDM_W | 30 | +0.017 | **66.7%** | 20.0% |
| DA-ARF+ADWIN | 125 | +0.004 | 45.6% | 26.4% |
| ARF+S1+ADWIN | 136 | **-0.006** | 37.5% | **46.3%** |

HDDM_W strzela najrzadziej, ale kupuje najwiecej za jeden alarm — jest oszczedny, a nie
martwy. Warianty z S1 maja wiecej alarmow szkodliwych niz uzytecznych i ujemna srednia:
dla modelu na okrojonej przestrzeni cech adaptacja po alarmie statystycznie *szkodzi*.

> **Uwaga: poprzednia wersja tej sekcji byla ze starego przebiegu** i podawala np.
> Hyperplane/ADWIN = 0.209 wobec HDDM_A = 0.451, z wnioskiem o „ponad 2x lepszej kappie".
> W aktualnych danych rozpietosc na Hyperplane to 0.710-0.734, wiec ten wniosek nie
> obowiazuje.

Wniosek: **ADWIN nie jest uniwersalnie najlepszy.** Na gradualnym Hyperplane
HDDM_A daje ponad 2× lepsza kappe niz ADWIN dla DA-ARF. To dobra amunicja na
pytanie "Why ADWIN?" — odpowiedz: bylo domyslne, ale E5 pokazal, ze dla wolnych,
gradualnych zmian detektory typu HDDM reaguja lepiej.

---

## 16. Metodologia ewaluacji (metryki + statystyka)

### 16.1. Protokol: prequential (test-then-train)

Dla kazdej instancji: **predict → zapisz blad → naucz**. Nigdy nie patrzymy w
przyszlosc, strumien nie jest tasowany. To symuluje produkcyjny online learning i
uniemozliwia uczenie sie na biezacej etykiecie przed predykcja. Konsekwencja:
klasyczny IID cross-validation nie ma sensu; wyniki sa czasowo zalezne.

### 16.2. Metryki (klasy w `thesis.evaluation`)

- **Accuracy (sliding window 1000)** — myli przy imbalance (NHTS: acc 0.98, kappa 0).
- **Cohen's Kappa** — koryguje zgodnosc o przypadek; **glowna metryka pracy**.
- **Temporal Kappa (`kappa_per`)** — kappa vs baseline "NoChange" (przewiduj poprzednia
  etykiete). Wazna przy autokorelacji czasowej (NYC Taxi, ceny). Uwaga: temporal kappa
  bywa ujemna (Yahoo SRP+S1 = -0.32) — model gorszy niz naiwne "tak samo jak wczoraj".
- **RAM-Hours (GB·h)** — trapezoidalna calka `uzyta_pamiec × czas`. Traktowana jako
  czesc jakosci modelu, nie afterthought. Ujemne probki (2 nie-atomiczne odczyty
  Runtime przy 12 watkach + GC) sa clampowane do 0.
- **Recovery time** — po alarmie liczba instancji do powrotu kappa ≥ (pre-drift kappa − tol),
  cap = 10000. Mierzy jak szybko model sie odbudowuje.
- **Feature stability ratio** — Jaccard kolejnych selekcji (jak bardzo migocze selekcja).

### 16.3. Testy statystyczne (`thesis.evaluation.StatisticalTests`)

Trzy poziomy ostrosci na macierzy `datasety × warianty` (usrednionej po seedach):

1. **Friedman** (omnibus) — czy *cokolwiek* sie rozni. χ² + Iman-Davenport F.
2. **Nemenyi** (post-hoc) — ktore *srednie rangi* sie roznia; `CD = q·sqrt(k(k+1)/6N)`.
3. **Wilcoxon signed-rank** (pairwise) — pary wariantow, wieksza moc (na surowych
   wynikach, nie rangach), z **korekta Holm** (kontrola family-wise error).

5 seedow → powtarzalnosc dla stochastycznych generatorow. Testy nieparametryczne, bo
liczba datasetow mala i normalnosc nieuzasadniona. **Nie twierdzimy dominacji, gdy
Holm/Friedman jej nie potwierdza**. W E3 Friedman na kappie jest istotny (p = 0.0072), ale
Nemenyi ma tam CD = 4.79 przy skali 1-10, wiec rozstrzygaja dopiero pary z Wilcoxona
(11/45 istotnych).

### 16.4. Multiple testing w Level-2: Benjamini-Hochberg FDR

Przy `d` rownoleglych testach KS na cechach, naiwne `p<alpha` daje `alpha·d`
falszywych alarmow. BH kontroluje **oczekiwana frakcje falszywych odkryc** wsrod
oznaczonych jako dryfujace (`bhQ=0.10`). Dlaczego nie Bonferroni? Bonferroni
kontroluje family-wise error i jest bardziej konserwatywny; przy lokalizacji cech
zalezalo na czulosci → FDR to lepszy kompromis. **Nie mowic**, ze q=0.10 gwarantuje
"max 10% FP w kazdym alarmie" — to kontrola *w oczekiwaniu*, pod zalozeniami.

---

## 17. Dane: syntetyczne (ground truth) + realne (realizm)

### 17.1. Generatory syntetyczne (MOA, `SyntheticStreamFactory`)

- **SEA** — 4 segmenty `ConceptDriftStream`, abrupt w n/4, n/2, 3n/4, noise 10%.
- **STAGGER** — reguły dyskretne, abrupt; trywialnie separowalny (kappa 1.0).
  **Usuniety z E1/E2/E3** (nasycony, zero dyskryminacji); zostaje jako STAGGER-HiDyn/-Low w E4/E5.
- **Hyperplane** — 15 atr., 5 dryfujacych, dryft **gradualny** (`sigma`).
- **RandomRBF** — 20 centroidow, 5 dryfujacych, gradual (`speed`); relevance stabilna.
- **CustomFeatureDrift** — Hyperplane z jawnie dryfujacymi cechami `{0..k-1}`;
  sluzy do testu **lokalizacji** (czy KSWIN wskaze wlasciwe kolumny).
- **LED (LEDGeneratorDrift)** — 24 cechy = **7 istotnych (idx 0–6) + 17 nieistotnych (idx 7–23)**,
  10 klas, dryf gradualny (`numberAttributesDrift`), noise 10%. **Idealny benchmark FS**: znany
  ground truth cech nieistotnych (czy selektor je odrzuca?) i mocno dyskryminujacy — pelny ARF
  ~0.72, ale FS do K=⌈√24⌉=5 spada do ~0.59 (K<7 istotnych → strukturalnie za maly budzet).
- `NoiseAugmentedStream` — dokłada `nNoise` kolumn U(0,1) (cechy nieinformatywne)
  → test odpornosci selekcji.

Zaleta syntetyki: znamy czas/typ/cechy dryftu → mozna mierzyc detection delay,
recovery, precision/recall lokalizacji (blok E4 to robi).

### 17.2. Dane realne (pipeline `preproccessing/`, → CSV + ARFF dla MOA)

Cechy sa **kauzalne** (tylko przeszlosc/terazniejszosc; rolling/lag bez zagladania
w przyszlosc). Kauzalnosc **nie wystarcza** — patrz notka o NYC Taxi ponizej: cecha moze byc
w pelni kauzalna, a mimo to pozwalac odtworzyc etykiete.

- **Yahoo Finance** — 80 tickerow, 7 sektorow, 2015-2025, dzienne bary. Cechy:
  SMA/EMA/MACD/RSI/Bollinger (`ta`). Target: ruch ceny (up/flat/down, prog ±0.5%).
  **Bardzo trudny** (kappa baseline ~0.03-0.23) — rynki blisko random walk.
- **NYC Taxi (TLC)** — 20 stref Manhattanu, 2022-2024 (36 plikow miesiecznych), 405 306
  instancji, **19 cech**. Cechy: cykliczne czas (sin/cos hour, dow), weekend, swieta, lagi
  i srednie kroczace popytu, `neighbor_avg_demand`. Target: `demand_level` = czy `trip_count`
  w **biezacej** godzinie przekracza mediane historyczna strefy. To **nowcasting**, nie prognoza
  na t+1 — do poprawienia w opisie w pracy.
  > **Naprawiony wyciek etykiety (2026-09-01).** `delta_trip_count` = `trip_count − lag1`, wiec
  > dodanie dwoch cech odtwarzalo `trip_count`, z ktorego liczona jest etykieta. Regula
  > z dwoch cech dawala κ = 0.772 wobec 0.784 dla pelnego ARF — ~98% wyniku bylo odtwarzaniem
  > definicji etykiety. Cecha usunieta z ARFF; srednia κ na tym zbiorze spadla z 0.594 do 0.507.
- **NHTS** — National Household Travel Survey, edycje 2009/2017/2022 (naturalny dryft
  miedzy edycjami), 1 985 822 instancje, **16 cech**. Target: srodek transportu
  (Private Vehicle / Transit / Active). **97.3% Private Vehicle** → accuracy myli, kappa jest
  wlasciwa metryka. Dryft jest **skokowy i trzykrokowy** (granice edycji), a nie ciagly.
  > Z cech usunieto `edition_boundary` — flage rowna 1 w dokladnie dwoch wierszach na 2 mln,
  > tam gdzie zmienia sie edycja. W pracy o **wykrywaniu** dryftu podawanie modelowi znacznika
  > „tutaj zmienia sie rozklad" jest tym, co detektor ma znalezc sam.

Wniosek z pracy: **mocny wynik na syntetyce nie przenosi sie automatycznie na
real-world.** Traktowac synthetic-to-real transfer jako osobne pytanie badawcze.

---

## 18. Architektura runnera (dla pytan o eksperyment infra / XManager)

`UnifiedStreamExperimentRunner`:

```text
main → Cfg.load(master_experiments.json)   # globalne + bloki E1-E5
     → expandWork()                          # kartezjan bloki×datasety×warianty×seedy
     → ExecutorService(12 watkow)            # kazdy WorkItem = niezalezny RunWorker
     → ConcurrentLinkedQueue(RunArtifacts)
     → sortuj deterministycznie (block→dataset→variant→seed)
     → single-thread zapis CSV + per-block Friedman/Nemenyi/Wilcoxon-Holm
```

Decyzje warte wspomnienia:
- **Zero wspoldzielonego mutable state w hot-pathie** → brak lockow. Kazdy worker
  buduje wlasny stream/selector/model/detector/metrics/recorder.
- **Odpornosc na FAIL** — pojedynczy run konczacy sie wyjatkiem zwraca `status=FAIL`
  i nie wywala calej macierzy; zapisany jako wiersz w `runs_raw.csv`.
- **Determinizm zapisu** — sortowanie po kluczu przed zapisem → CSV bit-w-bit
  powtarzalne niezaleznie od kolejnosci watkow.
- **6 typow detalicznych CSV per blok**: `windows`, `drift_alarms`,
  `feature_selections`, `feature_importance`, `recovery_time`, `adaptation_events`.

To jest odpowiednik "small-scale experiment orchestration": specyfikacja z JSON,
parameter sweep (dataset×variant×seed), izolacja per-worker, retry/FAIL handling,
reprodukowalne artefakty. Mozna to mapowac na koncepty XManager (packaging,
parameter sweeps, scheduling, tracking, reproducibility).

Analiza (`analysis/`, Python) czyta te CSV i generuje tabele LaTeX + wykresy
(matplotlib) + CD-diagramy. Java liczy dane i surowa statystyke; Python robi
wizualizacje.

> Ograniczenia calego podejscia (threats to validity) sa zebrane w sekcji 12.2 —
> tam jest pelna lista (P(X) vs P(y|X), range drift w PiD, univariate filter,
> delayed labels, maly N datasetow, brak zysku z komponentow B/C, itd.).

---

## 19. Mapa: fakt → pytanie interview (z poprawnymi liczbami)

| Pytanie z prep-doc | Fakt oparty na kodzie/wynikach |
|---|---|
| WHEN vs WHERE | Level-1 ADWIN/HDDM na strumieniu bledu (globalny, ciagly, tani). Po alarmie Level-2 KSWIN per-cecha + BH-FDR. Trade-off: koszt O(d) vs opoznienie/pokrycie. |
| Surgical adaptation DA-SRP | overlap subspace ∩ drifting: 0→KEEP, <tau→SURGICAL, ≥tau→FULL; tau=0.5. Dziala tylko dzieki **jawnym** subspace SRP. |
| Dlaczego DA-ARF bywal slabszy | **NIE** z powodu braku SURGICAL (obalone — surgical w ARF wrecz szkodzi). Prawdziwa przyczyna: nietrojone drzewa (grace=200/δ=1e-7) + waska podprzestrzen ⌈√d⌉ w bazowym lesie. Po naprawie (grace=50/δ=0.01 + 0.5·d) DA-ARF jest konkurencyjny — sekcja 11.8. |
| Importance-weighted sampling | `w = (1-beta)·importance^power + beta·uniform`; beta=0.7, power=2.0. Balans exploitation vs diversity ensembla. **Ablacja B/C (po naprawie): komponent B POMAGA** (usuniecie szkodzi) — wczesniejszy wniosek "B nie pomaga" byl na zepsutym DA-ARF. |
| S1-S4 | S1 statyczny; S2 alarm-triggered re-select; S3 periodic swap (≤30%/cykl); S4 periodic+alarm z targetowaniem drifting features. **S2 zwykle ≥ S4.** |
| Kiedy adaptacja pomaga | E2 wzgledem S1: Hyperplane +0.559, FeatureDrift +0.333, RandomRBF/SEA/LED ~0 → warunkowa. **Wzgledem modelu bez selekcji** te same zyski to tylko +0.040 i +0.050 — reszta to odrobienie szkody S1. |
| Kappa vs accuracy | NHTS acc 0.98, kappa 0 (98% majority). Temporal kappa dla autokorelacji (NYC/ceny), bywa ujemna (Yahoo -0.32). |
| Czy zyski nie sa losowe | 5 seedow, Friedman+Nemenyi+Wilcoxon+Holm. **E3 Friedman kappa p = 0.0072** (istotny), ale Nemenyi CD = 4.79 nie rozstrzyga — wnioski z Wilcoxona, 11/45 par. Srednia Δκ jest krucha: dla DA-SRP-AB vs SRP caly dodatni wynik pochodzi z SEA (leave-one-out odwraca znak), wiec cytuj mediane i wins. |
| Leakage | strict temporal order, brak shuffle, cechy kauzalne (rolling/lag bez future), test-then-train, statystyki online. |
| Why ADWIN | domyslny globalny trigger na bledzie. **E5: wybor detektora wazy malo** — rozpietosc kappa miedzy 4 detektorami przy tym samym DA-ARF to 0.01-0.12, wobec 0.45+ miedzy modelami. Roznica jest w *skutecznosci* alarmu: HDDM_W strzela 30 razy i 67% alarmow daje >= 1 pp accuracy, ADWIN 125 razy przy 46%. |
| Co zawiodlo (zaktualizowane) | (1) DA-ARF < ARF NIE bylo wina architektury — to byly bugi bazowego lasu (nietrojone drzewa + waska podprzestrzen); **po naprawie DA-ARF jest konkurencyjny** (sekcja 11.8). (2) DA-SRP zrefaktorowany na wersje natywna bez refleksji. (3) B/C w DA-SRP na starym runie nie zawsze pomagaly — do weryfikacji po re-runie (w DA-ARF B/C pomagaja). |
| Business value | selektywna adaptacja zmniejsza niepotrzebne resety i skraca recovery gdy tylko czesc reprezentacji sie starzeje; aktywowac warunkowo, bo zlozonosc ma koszt. |

**Zlota zasada spojna z danymi:** *Adaptation is a conditional mechanism, not a free
boost; more machinery is not automatically better; and the right adaptation
granularity depends on the base model's architecture.*
