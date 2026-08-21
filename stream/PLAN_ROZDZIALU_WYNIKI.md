
# Plan rozdziału „Wyniki" — co umieścić, dlaczego i jaki z tego wniosek

Dokument opisuje **rekomendowany zestaw tabel, wykresów i testów** do rozdziału z wynikami,
z uzasadnieniem dla każdej pozycji i gotowym wnioskiem opartym na liczbach z przebiegu
**2026-08-21** (1250 runów, 5 seedów, po naprawach z `bledy.md`).

Wszystkie liczby w tym pliku pochodzą z `stream/results/master_summary.csv` i `E*/stat_tests/`.
Nazwy artefaktów odpowiadają plikom w `stream/results/{tables,figures}/`.

**Rekomendowany budżet:** ok. **23 tabele i 24 figury** w rozdziale głównym, reszta do załącznika
lub do pominięcia. Pipeline generuje obecnie 59 tabel i 82 figury — poniżej jest wskazane, które
z nich niosą argument, a które są duplikatem.

**Aktualizacja (analiza czasowa).** Wcześniejsza wersja planu opierała się prawie wyłącznie na
średnich z całego przebiegu. Doszły trzy grupy artefaktów pokazujących **zachowanie w czasie**:
skuteczność alarmów w E5 (§5), pełny zestaw figur czasowych dla E5, który wcześniej był
wyłączony konfiguracją, oraz widok per-learner dla DA-SRP w E3 (§3, dane przeliczone — §11).

**Dwie figury zostały naprawione, bo wprowadzały w błąd.** `e*_drift_alarm_counts` miało oś
liniową, na której wszystkie zbiory syntetyczne znikały (§1, wniosek 6), a
`e*_feature_importance_heatmap` uśredniało ważność cech po zbiorach i po wariantach, co nie
miało sensu na czterech niezależnych poziomach (§7 pkt 9). Jeśli w tekście pracy są już zdania
oparte na starych wersjach tych figur, wymagają przepisania.

---

## 0. Zasady przekrojowe

**Metryka wiodąca to κ Cohena, nie accuracy.** Na NHTS klasa mniejszościowa to ~0.6%, więc
accuracy jest tam bezużyteczna. Accuracy podawaj wyłącznie jako tabelę pomocniczą w załączniku.

**Nie cytuj `recovery_time`.** Po naprawie metryka jest poprawnie zmierzona, ale **nie jest
istotna w żadnym bloku** (p = 0.87 / 0.079 / 0.52 / 0.92; w E5 nie dało się jej policzyć).
Do wniosków o adaptacji używaj `recovery_max_drop` — głębokości spadku κ po alarmie — która
jest istotna w E2 (p = 0.0007), E3 (p = 0.0045) i E4 (p = 0.024).

**Uważaj z Nemenyi'm.** Krytyczna różnica rang zależy od liczby metod i zbiorów:

| blok | metod | zbiorów | CD | skala rang | użyteczny? |
|---|---|---|---|---|---|
| E1 | 8 | 8 | 3.71 | 1–8 | tylko skrajne różnice |
| E2 | 10 | 5 | **6.06** | 1–10 | **nie** |
| E3 | 10 | 8 | 4.79 | 1–10 | ledwie |
| E4 | 4 | 6 | **1.91** | 1–4 | **tak** |
| E5 | 8 | 4 | **5.25** | 1–8 | **nie** |

W E2, E3 i E5 opieraj wnioski o **Wilcoxona z poprawką Holma** (pary dataset × seed), a
CD-diagram traktuj jako ilustrację, nie dowód. To jest do zaznaczenia w metodologii.

**Podawaj odchylenia standardowe.** Po naprawie propagacji seeda wariancja jest realna i
miejscami duża — np. `ARF+S1` na FeatureDrift ma κ = 0.389 ± **0.328**. Sama średnia jest tu myląca.

---

## 1. E1 — poziom odniesienia

### Tabele

| artefakt | co pokazuje | wniosek do napisania |
|---|---|---|
| `tab_e1_baselines` | macierz κ (8 zbiorów × 8 wariantów) | **ARF jest najmocniejszym baseline'em** (średnia ranga 1.50). Wygrywa na 4/8 zbiorów, HT na 2, SRP na 1, ARF+S1 na 1. |
| `tab_e1_resources` | RAM-Hours (w jednostkach 10⁻⁶ GB·h) i przepustowość | Koszt rośnie o **dwa rzędy wielkości**: HT 147 tys. inst./s przy 0.064 MB, ARF 4.9 tys. inst./s przy 0.112 MB. Za wzrost κ z 0.69 do 0.76 (syntetyki) płaci się 30× wolniejszym przetwarzaniem. **Uwaga o jednostce:** RAM-Hours liczone z rozmiaru modelu wychodzą rzędu 10⁻⁶ GB·h, więc tabela je skaluje — bez tego cała kolumna drukowała się jako `0.00`. Mnożnik jest w podpisie tabeli, trzeba go przenieść do tekstu. |
| `tab_e1_friedman` | test omnibus dla 6 metryk | Istotne: accuracy, κ, κ temporalna, RAM-hours. |
| `tab_e1_wilcoxon_kappa` | pary z poprawką Holma | **26 z 28 par istotnych** — najmocniejszy statystycznie blok. |

### Figury

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `e1_kappa_by_dataset` | κ per zbiór i wariant | Jedna figura zastępuje czytanie tabeli; pokazuje przepaść między wariantami z S1 i bez. |
| `e1_kappa_timeseries_{SEA,Hyperplane,NHTS,YahooFinance}` | κ w czasie, 4 reprezentatywne zbiory | Dobór celowy: SEA = dryf nagły, Hyperplane = ciągły, NHTS = niezbalansowany realny, YahooFinance = zbiór, gdzie naiwny baseline wygrywa. |
| `e1_drift_alarm_counts` | liczba alarmów per wariant, **oś Y logarytmiczna** | Podstawa wniosków 5 i 6 (rozpiętość syntetyki vs zbiory realne, degeneracja detektora przy `Majority`). **Uwaga:** oś musi zostać logarytmiczna — dane obejmują trzy rzędy wielkości (1.0 → 1329) i na skali liniowej wszystkie zbiory syntetyczne rysują się jako płaska linia przy zerze, co czyta się jak „nie wykryto żadnego dryfu". |
| `e1_feature_importance_heatmap` | ważność cech **per zbiór**, w jednostkach udziału równomiernego (1.0 = cecha niesie dokładnie swój udział 1/d); ramka = cechy szumowe | Podstawa wniosku 7. Każdy blok szumu jest niebieski, czyli poniżej udziału równomiernego — ranker odsuwa od nich ważność. **Uwaga:** figura była wcześniej pivotem wariant × indeks cechy z uśrednieniem po zbiorach; nie wolno do tego wracać, powody w §7 pkt 9. |

### Kluczowe wnioski E1

1. **Statyczna selekcja S1 kosztuje bardzo dużo.** Średnia strata względem pełnego modelu na
   zbiorach syntetycznych: ARF 0.755 → 0.554, SRP 0.660 → 0.477. Na Hyperplane katastrofalnie:
   0.712 → 0.194.
2. **Zbiory realne są wyraźnie trudniejsze niż syntetyczne** — ARF: 0.755 (synt.) vs 0.480 (real.).
   To argument przeciwko wnioskowaniu wyłącznie z generatorów.
3. **Na YahooFinance `NoChange` (κ = 0.248) bije wszystko poza pełnym ARF (0.280).** Wymaga
   akapitu — to zbiór o silnej autokorelacji etykiet, nie zepsuty eksperyment.
4. **Na NHTS selekcja S1 niemal zabija model** (ARF 0.377 → 0.040) przy K = ⌈√17⌉ = 5 i klasie
   mniejszościowej 0.6%. Uwaga: to **nie** jest zerowy kolaps — stara wersja wyników pokazywała
   κ = 0.0000, ale to był artefakt braku propagacji seeda.
5. **Detektor strzela rzadziej na syntetykach niż na realnych — o dwa rzędy wielkości.** Średnia
   liczba alarmów na przebieg: SEA 3–5, Hyperplane 6–8, FeatureDrift 7–8, LED 1–5, wobec
   NYCTaxi 33–77, YahooFinance 37–105, NHTS 89–116. Na SEA i FeatureDrift są **3 zaplanowane
   dryfy GT** (25k / 50k / 75k), więc 5 alarmów na SEA to detekcja z niewielkim nadmiarem, a 7.8
   na FeatureDrift to już wyraźne przestrzeliwanie. Na LED cztery warianty (`HT+S1`, `SRP+S1`,
   `Majority`, `NoChange`) **nie strzelają ani razu** przez cały przebieg.
6. **`Majority` na NYCTaxi generuje 1329 alarmów na przebieg — i to jest wynik, nie usterka.**
   Model o stałej predykcji nie potrafi się uczyć, więc jego strumień błędów odwzorowuje wprost
   zmiany rozkładu klas; ADWIN, podpięty pod taki model, degeneruje się do licznika szumu.
   Kontrast jest jednoznaczny: ten sam `Majority` daje 2.0 alarmu na SEA i **zero** na
   Hyperplane, RandomRBF, FeatureDrift i LED, gdzie priory klas są stacjonarne. Wniosek do
   napisania wprost: **liczba alarmów mierzy własność pary (model, strumień), a nie jakość
   detektora** — dlatego wnioski o detektorach z E5 opieraj na skuteczności alarmu
   (`tab_e5_alarm_effectiveness`), a nie na jego liczbie. To ta sama patologia, która w E5
   pojawia się przy wariantach z S1.
7. **Ranker odsuwa ważność od cech szumowych — 2.7 do 5.5×** (`e1_feature_importance_heatmap`).
   Średnia ważność cech sygnałowych wobec wstrzykniętych szumowych, liczona **osobno dla
   każdego zbioru** (inaczej się nie da — patrz §7 pkt 9):

   | dataset | sygnał | szum | iloraz |
   |---|---|---|---|
   | RandomRBF | 0.0629 | 0.0114 | **5.50×** |
   | SEA | 0.2485 | 0.0509 | **4.88×** |
   | Hyperplane | 0.0596 | 0.0212 | 2.81× |
   | FeatureDrift | 0.0457 | 0.0172 | 2.66× |
   | LED | 0.0493 | 0.0385 | **1.28×** |

   NHTS, NYCTaxi i YahooFinance nie mają wstrzykniętego szumu, więc nie da się dla nich podać
   ilorazu. **LED jest wyjątkiem wartym zdania w tekście:** przy 17 cechach nieistotnych na 24
   separacja niemal zanika, co współgra z pozostałymi obserwacjami o tym zbiorze (detektor
   strzela tam najrzadziej, część wariantów ani razu).

---

## 2. E2 — adaptacyjna selekcja cech (S1–S4)

To blok, w którym pada najważniejszy negatywny wynik pracy. Warto go wyeksponować, nie ukryć.

### Tabele

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `tab_e2_kappa` | κ dla ARF/SRP × {—, S1, S2, S3, S4} | Podstawa całego bloku. |
| `tab_e2_delta_vs_raw` | Δκ każdego selektora względem **modelu bez selekcji** (surowy ARF / SRP) | Zgodne z wnioskami poniżej: żaden selektor nie bije surowego modelu średnio, a S2 podchodzi najbliżej (−0.023). Odniesienie do S1 pozostaje odzyskiwalne jako różnica wiersza i wiersza S1 tego samego modelu. **Uwaga:** wcześniej tabela mierzyła względem S1, co dawało dwa wiersze zer i zawyżało zasługę adaptacji — patrz §7 pkt 10. Nazwa pliku i etykiety zostały bez zmian (`tab_e2_delta_vs_raw`), żeby nie psuć `\input` w `main.tex`; przy okazji redakcji warto przemianować na `tab_e2_delta_vs_raw`. |
| `tab_e2_stability` | stabilność selekcji, liczba zmian | S2 zmienia selekcję 6.7 razy na przebieg, S3/S4 — 36.5 razy, przy tej samej stabilności Jaccarda (~0.745). |
| `tab_e2_drift_response` | reakcja na dryf | Wiąże liczbę alarmów z jakością. |
| `tab_e2_wilcoxon_kappa` | 19/45 par istotnych | **Test wiodący dla tego bloku** — Nemenyi ma tu CD = 6.06 i nie rozstrzyga niczego. |

### Figury

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `e2_kappa_heatmap` | mapa κ (zbiór × wariant) | Natychmiast widać wzorzec: kolumna S1 jest ciemna, S2 jasna. |
| `e2_adaptive_vs_static` | adaptacyjne vs statyczne | Bezpośrednia ilustracja głównej tezy bloku. |
| `e2_selection_timeline_{FeatureDrift,Hyperplane}` | które cechy są wybrane w czasie | Pokazuje **mechanizm**: S3/S4 przełączają cechy stale, S2 tylko po alarmie. |
| `e2_feature_selection_overview` | statystyki selekcji | Uzupełnienie ilościowe timeline'ów. |

### Kluczowe wnioski E2

**Średnia strata κ względem pełnego modelu (ARF, 5 zbiorów syntetycznych):**

| selektor | Δκ | interpretacja |
|---|---|---|
| S1 statyczny | **−0.202** | selekcja raz na starcie jest destrukcyjna |
| S2 alarmowy | **−0.023** | praktycznie bez kosztu |
| S3 okresowy | −0.057 | gorszy od S2 mimo 5× częstszych przebudów |
| S4 alarm+okresowy | −0.061 | połączenie nie pomaga |

1. **Główny wniosek: adaptacyjna selekcja odzyskuje prawie całą stratę statycznej, ale nie
   przewyższa modelu bez selekcji.** S2 jest o 0.18 κ lepszy od S1, ale wciąż 0.023 poniżej ARF.
   Wartość selekcji leży więc w koszcie (S1 daje 13.1 tys. inst./s vs 4.9 tys. dla ARF), nie w jakości.
2. **Więcej przebudów ≠ lepiej.** S3/S4 robią 36.5 re-selekcji na przebieg, S2 tylko 6.7, a wypadają
   gorzej. To argument, że selekcja ma być **sterowana zdarzeniem**, a nie harmonogramem.
3. **Na LED wszystkie selektory tracą tak samo** (−0.12 do −0.17). LED ma 7 istotnych cech,
   a K = ⌈√24⌉ = 5 — selekcja **strukturalnie nie może** złapać wzorca. Mocny argument o budżecie cech.
4. **Ograniczenie do zaznaczenia:** w E2 **91.6% alarmów nie wskazuje żadnej dryfującej cechy**,
   więc S2/S4 najczęściej działają fallbackiem „zdecayuj wszystko". Lokalizacja Level-2 (KSWIN+BH-FDR)
   nie radzi sobie z dryfem stopniowym. To ograniczenie metody, trzeba je nazwać wprost.

---

## 3. E3 — ablacja metod autorskich (DA-SRP, DA-ARF)

### Tabele

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `tab_e3_ablation` | κ dla A / AB / ABC obu metod + baseline'y | Rdzeń rozdziału o wkładzie własnym. |
| `tab_e3_ablation_deltas` | przyrosty między wariantami | Izoluje wkład każdego komponentu. |
| `tab_e3_adaptation_actions` | liczniki KEEP / SURGICAL / FULL / EXT_* | Pokazuje, **czym różnią się mechanizmy** obu metod. |
| `tab_e3_wilcoxon_kappa` | 11/45 par istotnych | Test wiodący (CD = 4.79 przy 10 metodach jest zbyt duże). |

### Figury

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `e3_ablation_bar` | słupki κ per wariant | Główna ilustracja bloku. |
| `e3_adaptation_actions` | proporcje akcji adaptacji | **Najmocniejsza figura mechanizmu** — patrz niżej. |
| `e3_importance_evolution_{FeatureDrift,Hyperplane,NHTS}` | ranking ważności cech w czasie | Pokazuje, że komponent B faktycznie śledzi zmiany istotności. |
| `e3_kappa_timeseries_{FeatureDrift,Hyperplane,NYCTaxi,YahooFinance}` | κ w czasie | Dobór: 2 syntetyki + 2 realne. |
| **`e3_causality_{FeatureDrift,NYCTaxi}`** *(propozycja dodania)* | overlay alarm → akcja → recovery | Łańcuch przyczynowy w jednym obrazie; obecnie niecytowane. |
| **`e3_action_vs_overlap`** *(nowe)* | rozkład akcji learnera wobec liczby dryfujących cech w jego podprzestrzeni | **Dowód, że komponent B celuje, a nie strzela na oślep**: przy overlap = 0 dominuje KEEP, przy overlap ≥ 1 akcją jest SURGICAL. To jest ilościowa wersja twierdzenia o „chirurgicznej" adaptacji — dotąd opisywanego wyłącznie słowami. |
| **`e3_learner_lanes_{FeatureDrift,Hyperplane,NYCTaxi,YahooFinance}`** *(nowe)* | raster: 1 pas na learnera; lewy panel = podjęta akcja, prawy = liczba dryfujących cech w jego podprzestrzeni | Ta sama teza w ujęciu czasowym i per-learner: widać, że reset trafia w te konkretne składowe zespołu, których podprzestrzeń zawiera dryfującą cechę, a reszta pozostaje nietknięta. |

### Kluczowe wnioski E3 — uczciwie, także negatywne

**Ablacja (średnia κ po 8 zbiorach):**

| metoda | A | AB | ABC |
|---|---|---|---|
| DA-ARF | 0.6007 | 0.6175 (**+0.017**) | 0.6134 (**−0.004**) |
| DA-SRP | 0.5914 | 0.6171 (**+0.026**) | 0.6058 (**−0.011**) |

1. **Komponent B (importance-weighted sampling) działa** — daje +0.017 i +0.026 κ.
2. **Komponent C (top-K voting) NIE pomaga** — pogarsza obie metody. To trzeba napisać wprost;
   ujemny wynik ablacji jest wartościowy i pokazuje rzetelność metodologii.
3. **DA-SRP-ABC bije SRP na 5/8 zbiorów (średnio +0.026), DA-ARF-ABC przegrywa z ARF na 6/8
   (średnio −0.039).** Metoda oparta na SRP się broni, oparta na ARF — nie. Nie zacieraj tego.
4. **Obie metody adaptują się inaczej** i to widać w licznikach: DA-SRP działa przez
   **SURGICAL (127) i KEEP (119)**, prawie nie robiąc pełnych resetów (20); DA-ARF w ogóle nie
   używa ścieżki chirurgicznej, tylko zewnętrznej: **EXT_KEEP 170, EXT_FULL 38**. To jest
   argument jakościowy, którego nie widać w samej κ.
5. **Adaptacja jest celowana, nie losowa** (nowe, z `e3_action_vs_overlap`). Rozkład akcji
   w zależności od tego, ile świeżo wykrytych cech dryfujących leży w podprzestrzeni danego
   learnera — 21 960 par (zdarzenie, learner), wszystkie zbiory i seedy:

   | dryfujące cechy w podprzestrzeni | KEEP | SURGICAL | FULL | NO_REPL |
   |---|---|---|---|---|
   | 0 | **86.2%** | 0.0% | 13.8% | 0.0% |
   | 1 | 0.0% | **86.9%** | 0.0% | 13.1% |
   | 2 | 0.0% | **91.5%** | 0.0% | 8.5% |
   | 3+ | 0.0% | **98.0%** | 0.9% | 1.1% |

   Separacja jest zerojedynkowa: **learner bez dryfującej cechy w podprzestrzeni nigdy nie
   dostaje wymiany chirurgicznej, a learner z taką cechą nigdy nie zostaje nietknięty.**
   Pełne resety (13.8% przy pokryciu 0) to wyłącznie ścieżka dryfu **niezlokalizowanego** —
   detektor zgłasza dryf, ale nie wskazuje cech, więc mechanizm chirurgiczny nie ma się czego
   uchwycić. To odpowiada wprost na zarzut, że zysk komponentu B mógłby wynikać z samego
   dodatkowego resetowania: przy pokryciu 0 metoda w 86% przypadków **nie robi nic**.
   Malejący udział NO_REPL (13.1% → 8.5% → 1.1%) ma osobną interpretację: przy jednej cesze
   dryfującej częściej brakuje sensownego zamiennika w rankingu ważności niż przy trzech.
   *Zakres:* kolumny per-learner emituje tylko DA-SRP; w danych są warianty **AB i ABC**
   (DA-SRP-A nie generuje zdarzeń adaptacji w ogóle — to stan sprzed tej zmiany, nie regresja).

---

## 4. E4 — intensywność dryftu

Blok został przebudowany (6 zbiorów × 4 metody, CD spadło z 3.77 do 1.91), więc **opis w pracy
wymaga aktualizacji składu**: STAGGER został usunięty jako nasycony, doszły pary Low/HiDyn dla
FeatureDrift i RandomRBF, a wewnątrz każdej pary przestrzeń cech jest identyczna.

### Tabele

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `tab_e4_kappa` | κ (6 zbiorów × 4 metody) | Podstawa. |
| `tab_e4_dynamics_sensitivity` | wrażliwość na intensywność dryftu | Rdzeń bloku. |
| `tab_e4_friedman` | testy omnibus | Istotne tylko κ temporalna (p = 0.0069) i `recovery_max_drop` (p = 0.024). |
| `tab_e4_adaptation_actions` | akcje adaptacji przy różnej dynamice | Czy metody reagują proporcjonalnie do intensywności. |

### Figury

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `e4_kappa_by_dynamics` | κ w podziale Low/HiDyn | Główna ilustracja. |
| `e4_kappa_timeseries_{SEA-Low,SEA-HiDyn,FeatureDrift-Low,FeatureDrift-HiDyn}` | κ w czasie | **Uwaga: zastępuje figury STAGGER-*, które już nie istnieją.** |
| `e4_recovery_depth` | głębokość spadku po alarmie | Metryka, która tu faktycznie różnicuje. |

### Kluczowe wnioski E4

**Kontrast Low → HiDyn (średnia κ po metodach):**

| rodzina | Low | HiDyn | Δ |
|---|---|---|---|
| SEA (liczba nagłych dryftów 3 → 10) | 0.673 | 0.692 | **+0.019** |
| FeatureDrift (2 → 10 dryfujących cech) | 0.769 | 0.617 | **−0.152** |
| RandomRBF (prędkość 0.001 → 0.010) | 0.924 | 0.858 | **−0.066** |

1. **Nie każda „intensywność dryftu" jest tym samym zjawiskiem.** Zwiększenie *liczby nagłych
   dryftów* prawie nie zmienia średniej κ — adaptacyjne ensemble odbudowują się szybko między
   zmianami. Dopiero zwiększenie *zakresu* dryftu (ile cech dryfuje, jak szybko przesuwa się
   koncept) realnie utrudnia zadanie. To jest samodzielny wniosek metodologiczny.
2. **Sama κ nie rozstrzyga** (p = 0.138), rozstrzyga κ temporalna i głębokość spadku. Napisz to
   otwarcie zamiast przedstawiać blok jako niekonkluzywny.
3. **Ranking metod:** ARF = DA-SRP-ABC (2.00) < DA-ARF-ABC (2.50) < SRP (3.50), przy CD = 1.91 —
   więc różnica ARF/DA-SRP vs SRP jest istotna, a między ARF a DA-SRP-ABC nie.

---

## 5. E5 — wpływ detektora dryftu

### Tabele

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `tab_e5_kappa` | κ dla DA-ARF × {ADWIN, HDDM_A, HDDM_W, KSWIN} + baseline'y | Podstawa. |
| `tab_e5_detector_ranking` | ranking detektorów | Zbiorcze podsumowanie. |
| `tab_e5_friedman` | testy | **Istotne tylko accuracy i RAM-hours** — κ nie (p = 0.084). |
| **`tab_e5_alarm_effectiveness`** *(nowe)* | per wariant: liczba alarmów, średnia i mediana Δaccuracy, % alarmów użytecznych i szkodliwych | **Ilościowy rdzeń nowej narracji bloku** — patrz wniosek 5. Odróżnia detektor, który strzela rzadko i celnie, od takiego, który strzela często i bez efektu. |

### Figury

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `e5_kappa_heatmap` | κ detektor × zbiór | Podstawa. |
| `e5_alarms_vs_kappa` | liczba alarmów vs jakość | Kluczowa figura tego bloku. |
| `e5_recovery_vs_kappa` | recovery vs jakość | Uzupełnienie. |
| `e5_drift_alarm_counts` | liczby alarmów | Dowód na martwe detektory. |
| **`e5_alarm_effectiveness`** *(nowe)* | rozkład Δaccuracy na alarm per wariant + % alarmów użytecznych | Pokazuje, że sama liczba alarmów nic nie mówi — liczy się ich skuteczność. |
| **`e5_alarm_effect_timeline_{Hyperplane,SEA-HiDyn}`** *(nowe)* | oś czasu: każdy alarm jako słupek o wysokości równej odzyskanej accuracy | Ujęcie czasowe tego samego: widać, czy alarmy trafiają w momenty rzeczywistego dryfu (linie GT), czy rozkładają się losowo. |
| **`e5_kappa_timeseries_{Hyperplane,SEA-HiDyn}`** *(nowe)* | κ w czasie | Blok nie miał **żadnej** figury czasowej — konfiguracja wyłączała je dla E5. Dobór: dryf ciągły vs nagły o wysokiej intensywności. |
| **`e5_adaptation_timeline_{Hyperplane,SEA-HiDyn}`** *(nowe)* | akcje adaptacji w czasie | j.w.; pokazuje, jak różne detektory rozkładają adaptacje wzdłuż strumienia. |

### Kluczowe wnioski E5

1. **Wybór detektora ma mniejsze znaczenie niż wybór modelu.** Rozpiętość κ między czterema
   detektorami przy tym samym DA-ARF wynosi 0.01–0.11, podczas gdy różnica ARF vs SRP+S1 to 0.45.
2. **HDDM_A i HDDM_W są całkowicie martwe na RandomRBF** — 0.0 alarmów, w efekcie identyczna
   κ = 0.9052 dla obu. Na SEA-HiDyn HDDM_W strzela średnio 0.4 razy na 100 tys. instancji.
   To istotne ograniczenie, warte osobnego akapitu.
3. **Więcej alarmów nie znaczy lepiej.** Na RandomRBF `ARF+S1+ADWIN` generuje 9.2 alarmu i ma
   κ = 0.842, a `ARF+ADWIN` 2.2 alarmu i κ = 0.937.
4. **Blok ma najsłabszą moc statystyczną w całej pracy** (CD = 5.25 przy skali 1–8). Wnioski
   opieraj na Wilcoxonie (13/28 par istotnych), nie na rangach.
5. **Skuteczność alarmu, nie ich liczba, różnicuje detektory** (nowe, `tab_e5_alarm_effectiveness`).
   Mierząc zmianę accuracy w oknie następującym po alarmie:

   | wariant | alarmy | średnia Δacc | % użytecznych (≥ 1 pp) | % szkodliwych (≤ −1 pp) |
   |---|---|---|---|---|
   | DA-ARF+HDDM_W | 30 | +0.017 | **66.7%** | 20.0% |
   | DA-ARF+HDDM_A | 58 | +0.005 | 50.0% | 20.7% |
   | DA-ARF+KSWIN | 87 | +0.008 | 46.0% | 28.7% |
   | DA-ARF+ADWIN | 125 | +0.004 | 45.6% | 26.4% |
   | ARF+S1+ADWIN | 136 | **−0.006** | 37.5% | **46.3%** |

   Dwa wnioski do napisania wprost. Po pierwsze, **HDDM_W kupuje najwięcej za jeden alarm** —
   to łagodzi zarzut z pkt 2, że jest „martwy": jest oszczędny, a nie bezużyteczny, i te dwie
   rzeczy trzeba w tekście rozdzielić. Po drugie, **warianty z S1 mają więcej alarmów
   szkodliwych niż użytecznych i ujemną średnią Δacc** — dla modelu na okrojonej przestrzeni
   cech adaptacja po alarmie statystycznie *szkodzi*. To niezależne potwierdzenie głównego
   negatywnego wniosku z E1 i E2, uzyskane na zupełnie innej metryce.

---

## 6. Synteza przekrojowa

| artefakt | co pokazuje | wniosek |
|---|---|---|
| `tab_cross_best_methods` | najlepsza metoda per zbiór | Zbiorcze zestawienie. |
| `tab_cross_resource_vs_kappa` | κ vs koszt | Podstawa dyskusji o kompromisie. |
| `cross_rq_summary` | przegląd pytań badawczych | **Dobra figura otwierająca rozdział.** |
| `cross_pareto_front` | front Pareto jakość–koszt | Pokazuje, że HT+S1 i ARF leżą na froncie, a warianty SRP+S1 są zdominowane. |
| `cross_synthetic_vs_real` | syntetyczne vs realne | Ilustruje lukę 0.755 vs 0.480 dla ARF. |

---

## 7. Sekcja metodologiczna — co koniecznie odnotować

Te punkty wynikają ze zmian w pomiarach i **muszą** znaleźć się w opisie, bo inaczej liczby są
nieporównywalne z wcześniejszymi wersjami pracy:

1. **RAM-hours mierzy rozmiar modelu**, nie zużycie JVM (przez agenta `sizeofag`). Wcześniejsza
   implementacja próbkowała całą stertę dzieloną przez 12 wątków i była nieinterpretowalna.
2. **`throughput` obejmuje pełny krok prequential**; koszt samej inferencji jest raportowany
   osobno jako `predict_latency_us`. Liczby bezwzględne podawaj z przebiegu jednowątkowego —
   różnica między 1 a 6 wątkami wynosi ~18%.
3. **Recovery mierzone w instancjach**, dwufazowo (najpierw realny spadek, potem powrót), z
   czterema kategoriami wyniku. W E1 **43% epizodów kończy się jako CANCELLED** (nowy alarm przed
   domknięciem poprzedniego), więc `recovery_time` opiera się tam na 23.5% epizodów.
4. **κ temporalna występuje w dwóch agregacjach** — `kappa_temporal_final` (ostatnie okno) i
   `kappa_temporal_windowed` (średnia po oknach). W tabelach używana jest ta druga.
5. **Modele deterministyczne** (HT, Majority, NoChange) mają `kappa_std = 0` na zbiorach ARFF —
   to właściwość algorytmu (`isRandomizable() == false`), nie brak losowości eksperymentu.
6. **5 seedów, 1250 runów**, pełna odtwarzalność potwierdzona dwoma niezależnymi przebiegami.
7. **RAM-Hours raportowane w 10⁻⁶ GB·h.** Metryka liczona z głębokiego rozmiaru modelu daje
   wartości rzędu 10⁻⁶, więc tabele podają je z mnożnikiem (`analysis/config.py: RAMH_SCALE`).
   Jednostkę trzeba podać w tekście, inaczej liczby są nieinterpretowalne.
8. **Definicja skuteczności alarmu** (nowa metryka w E5): Δaccuracy = accuracy okna
   następującego po alarmie minus accuracy okna w momencie alarmu; „użyteczny" = Δ ≥ 1 pp,
   „szkodliwy" = Δ ≤ −1 pp. Próg 1 pp leży powyżej szumu okna 1000-instancyjnego
   (≈ 0.3 pp błędu standardowego przy acc 0.9). Obie wartości pochodzą z kolumn
   `window_accuracy_before` / `window_accuracy_after`, uzupełnianych przez recorder dopiero po
   przetworzeniu pełnego kolejnego okna — nie są więc dostępne „w momencie alarmu".
9. **Ważności cech NIE WOLNO uśredniać po zbiorach** — i nie zależą one od modelu. Cztery
   niezależne powody, każdy sam w sobie wystarczający (dawna wersja
   `e*_feature_importance_heatmap` łamała wszystkie cztery naraz):
   1. Indeks *i* oznacza inną zmienną w każdym strumieniu (SEA ma 8 cech, YahooFinance 36).
   2. Ważność jest normalizowana do sumy 1 na snapshot, więc jej skala to ~1/d — zbiory
      niskowymiarowe automatycznie przeważają (SEA: 1/8 = 0.125 wobec YahooFinance 1/36 = 0.028).
   3. Snapshoty są wyzwalane alarmami, więc średnia zbiorcza jest ważona częstotliwością
      alarmów: NYCTaxi wnosi 1955 snapshotów, LED 33.
   4. **Estymator ważności jest karmiony strumieniem, nie modelem.** W tym samym punkcie
      strumienia wszystkie warianty raportują wartości bit w bit identyczne — łącznie
      z `Majority` i `NoChange`, które w ogóle się nie uczą. Oś wariantów nie niosła więc
      żadnej informacji o modelu, a różnice między wierszami wynikały wyłącznie z tego, że
      każdy wariant miał inną liczbę snapshotów.

   Konsekwencja dla tekstu: o ważności cech pisz **per zbiór**, w jednostkach udziału
   równomiernego (1.0 = 1/d), i nie przypisuj różnic w ważności poszczególnym modelom.
10. **Punktem odniesienia dla selekcji cech jest model bez selekcji, nie S1.** S1 jest najsłabszą
    możliwą referencją: na Hyperplane ma κ = 0.193 wobec 0.712 surowego ARF. Mierząc względem
    S1, `ARF+S2` „zyskuje" tam +0.559 — ale 0.518 z tego to samo odrobienie szkody wyrządzonej
    przez statyczny podzbiór, a realny zysk z adaptacji to +0.040. Skoro E1 pokazało, że
    najmocniejszym baseline'em jest model pełny, to jego selektor musi pobić, żeby być wart
    swojego kosztu. Odniesienie do S1 zostaje odzyskiwalne przez odjęcie wiersza S1.
11. **Rejestracja per-learner dla DA-SRP.** `adaptation_events.csv` ma trzy dodatkowe kolumny
   (`per_learner_action`, `per_learner_overlap`, `per_learner_subspace`), kodowane znakiem `|`,
   po jednym wpisie na składową zespołu. Dane pochodzą z `DriftActionSummary`, który liczył je
   od początku — wcześniej były zwijane do czterech liczników zbiorczych przy zapisie. Wiersze
   DA-ARF mają te kolumny puste: ta metoda raportuje wyłącznie przyrosty zbiorcze.

---

## 8. Czego NIE umieszczać

| co | dlaczego |
|---|---|
| `recovery_time` w jakiejkolwiek postaci | nieistotne we wszystkich blokach; użyj `recovery_max_drop` |
| tabele `avg_ranks_*` dla metryk innych niż κ | duplikat `friedman` + `nemenyi` |
| CD-diagramy dla E2 i E5 jako dowód | CD = 6.06 i 5.25 — nie rozstrzygają niczego |
| osobne wykresy accuracy obok κ | ta sama informacja; accuracy do załącznika |
| per-datasetowe rastry alarmów **dla E1–E4** | agregat `e*_drift_alarm_counts` mówi to samo. **Wyjątek: E5** — tam raster niesie teraz Δaccuracy na alarm, czego żaden agregat nie pokazuje, i jest włączony celowo |
| wykresy `*_importance_noise` | duplikat `e*_feature_importance_heatmap` |

---

## 9. Zmiany wymuszone w obecnym tekście pracy

Sześć odwołań w `main.tex` wskazuje na pliki, które już nie istnieją:

| obecnie | zamiennik | co poprawić w tekście |
|---|---|---|
| `e1_recovery_length.pdf` | `e1_recovery_depth.pdf` | oś to głębokość spadku κ, nie liczba okien |
| `e2_recovery_length.pdf` | `e2_recovery_depth.pdf` | j.w. |
| `e5_recovery_length.pdf` | `e5_recovery_depth.pdf` | j.w. |
| `e4_kappa_timeseries_STAGGER-Low.pdf` | `e4_kappa_timeseries_FeatureDrift-Low.pdf` | zmiana składu E4 |
| `e4_kappa_timeseries_STAGGER-HiDyn.pdf` | `e4_kappa_timeseries_FeatureDrift-HiDyn.pdf` | j.w. |
| `e1_feature_selection_overview.pdf` | — usunąć | selekcja w E1 jest statyczna, wykres pokazywał stałą |

Dodatkowo: wszystkie liczby w tekście pochodzą z przebiegu majowego i **wymagają weryfikacji**.
Zmiany bywają duże — np. κ dla ARF na NHTS wzrosło z 0.188 do **0.377** po naprawie propagacji
seeda (stara wartość była jednym pechowym losowaniem powielonym pięć razy).

---

## 10. Propozycje rozszerzeń (opcjonalne, artefakty już istnieją)

| artefakt | dlaczego warto |
|---|---|
| `e*_stat_tests/e*_cd_kappa.svg` | CD-diagramy to standard w tej literaturze (Demšar 2006); obecnie generowane, ale niecytowane. Sensowne szczególnie dla E1 i E4. |
| `e*_cd_recovery_max_drop.svg` | jedyna metryka recovery, która osiąga istotność |
| `e3_causality_{FeatureDrift,NYCTaxi}` | łańcuch alarm → akcja → odbudowa κ w jednym obrazie |
| `e3_adaptation_timeline_*` | pokazuje różnicę mechanizmów DA-SRP (SURGICAL) vs DA-ARF (EXT_*) |
| `e2_selection_timeline_{FeatureDrift,Hyperplane}` | które cechy są wybrane w kolejnych etapach strumienia — mechanizm S2/S3/S4 w jednym obrazie |
| `e3_selection_timeline_*` | to samo dla wariantów DA-*, z zacieniowanym pasem cech szumowych |

**Nadal niewykorzystane, choć dane są w CSV** (wymagałyby tylko nowego generatora w `analysis/`):

| pomysł | źródło danych | co by pokazało |
|---|---|---|
| zmienność selekcji w czasie | `feature_selections.csv`: `jaccard_to_previous`, `selected_feature_count` | jak gwałtownie selektor przełącza cechy — timeline pokazuje *które*, ale nie *jak bardzo*; tabele podają tylko średnią |
| trafność selekcji wobec dryfu | `feature_importance.csv`: `is_selected` × `is_drifting` | czy selektor wybiera akurat te cechy, które właśnie dryfują |
| tempo resetów (krzywa) | `adaptation_events.csv` | raster przy 25 tys. zdarzeń w E3 wizualnie się zapycha; krzywa tempa czyta się lepiej |

---

## 11. Status danych per-learner

**Zrobione.** Blok E3 został przeliczony w całości (400 runów, 8 zbiorów × 10 wariantów ×
5 seedów) i `stream/results/E3/adaptation_events.csv` zawiera kolumny `per_learner_*`.
Liczby w §3 pkt 5 pochodzą z tego pełnego przebiegu — są gotowe do cytowania.

**Kontrola spójności przed podmianą pliku.** Przeliczenie poszło do katalogu tymczasowego,
a do `stream/results/` trafił wyłącznie `E3/adaptation_events.csv`. Porównanie starego i nowego
przebiegu potwierdziło determinizm: `drift_alarms.csv` i `feature_importance.csv` są **bit
w bit identyczne**, w `E3_ablation.csv` różnią się wyłącznie kolumny zależne od zegara
(`wall_ms_mean`, `throughput_mean`, `ram_hours_gb_*`, `peak_mb_mean`), a w samym
`adaptation_events.csv` wszystkie 15 wcześniejszych kolumn jest identycznych — doszły tylko
3 nowe. Reszta wyników w `stream/results/` pozostaje z pierwotnego przebiegu i jest z nim
spójna.

**Pozostałe bloki.** E4 zawiera DA-SRP-ABC, więc analogiczne figury dałoby się tam uzyskać po
przeliczeniu E4 (ten sam schemat: przebieg do katalogu tymczasowego, podmiana jednego pliku).
E5 nie zawiera wariantów DA-SRP, więc go to nie dotyczy. DA-ARF nie raportuje danych
per-learner w ogóle — `onDAARFEvent` dostaje wyłącznie przyrosty zbiorcze i rozszerzenie go
wymagałoby zmian w `DAARFWrapper`, nie w samym recorderze.
