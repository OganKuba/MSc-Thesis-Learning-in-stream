🔴 Krytyczne

1. Seed nie jest propagowany do ARF / SRP / HT — 5 „seedów" to 5 ident
2. ycznych kopii tego samego przebiegu

ARFWrapper.newARF(), SRPWrapper.newSRP(), HoeffdingTreeWrapper.newTree() ustawiają tylko ensembleSize/lambda/gracePeriod — nigdzie nie ma randomSeedOption. buildModel() w ogóle nie przekazuje seed do baseline'ów. Na zbiorach ARFF (deterministyczny strumień z pliku) daje to bitowo identyczne wyniki:

NYCTaxi / ARF / E1:   seed 1..5 → kappa 0.782667 (×5), acc 0.896 (×5), drift_count 39 (×5)

┌──────────┬───────────────────────┬──────────────┐
│  model   │ komórek REAL ze std=0 │ komórek REAL │
├──────────┼───────────────────────┼──────────────┤
│ ARF      │ 9                     │ 9            │
├──────────┼───────────────────────┼──────────────┤
│ SRP      │ 6                     │ 6            │
├──────────┼───────────────────────┼──────────────┤
│ HT       │ 6                     │ 6            │
├──────────┼───────────────────────┼──────────────┤
│ DA-ARF   │ 0                     │ 9            │
├──────────┼───────────────────────┼──────────────┤
│ DA-SRP-* │ 0                     │ 9            │
└──────────┴───────────────────────┴──────────────┘

Czyli na NHTS / NYCTaxi / YahooFinance baseline'y mają n=1, nie n=5, a DA-* mają prawdziwe n=5. Wilcoxon/Friedman porównują metodę z realną wariancją z metodą o wariancji zerowej — to zawyża istotność na korzyść baseline'ów i jest nieobronne na obronie. Na syntetykach seed zmienia strumień, więc wariancja jest, ale to
wariancja danych, nie modelu.

2. RAM-hours i peak_mb mierzą całą JVM, nie model — przy 12 wątkach to śmieci

RAMHours.sampleFromRuntime() czyta globalne Runtime.totalMemory()-freeMemory(), a 12 runów dzieli jedną JVM. Dowód:

Majority / SEA          → peak_mb =  164.3      (klasyfikator większościowy!)
Majority / NHTS         → peak_mb = 3554.9
korelacja ram_hours vs wall_ms = 0.9978

Majority nie alokuje niczego — 3.5 GB to pamięć sąsiednich wątków. ram_hours to praktycznie przeskalowany czas ścienny. Mimo to jest liczony Friedman + CD diagram dla ram_hours_gb w każdym bloku (E1: χ²=306.6, p≈0 — „istotność" wynikająca z tego, że wolniejsze metody dostają wyższy rank). Ta metryka i wszystkie 5
CD-diagramów ram_hours_gb do wyrzucenia albo do przemiaru w osobnym, jednowątkowym runie.

3. recovery_time jest zdegenerowany (potwierdza się to, co plan już zauważył)

recovery_time_mean ∈ [1.0000, 1.2589], 14 × NaN
E1/recovery_time.csv: recovery_length==1 → 64.2% przypadków, ==-1 → 35.4%
E3/recovery_time.csv: recovery_length==1 → 89.6%
Metryka ma dwie wartości: „natychmiast" albo „nigdy". max_drop/area_under_recovery_curve też są w 66% (E1) zerowe. Wszystkie tab_e*_recovery, avg_ranks_recovery_time i CD-diagramy recovery są zbudowane na tym.

🟠 Poważne

4. temporal_kappa w windows.csv to kolumna w 100% NaN

123 280/123 280 (E1) i 154 100/154 100 (E3) wierszy = NaN. To placeholder z RunDetailedRecorder:125. Natomiast temporal_kappa_mean w summary jest liczone z kappaPer, więc kolumna w summary działa — ale nie jest to ta sama metryka co w nazwie: temporal_kappa_mean vs kappa_per_mean korelują 0.959, a 90/250 wierszy różni
się o <0.01. W tabelach masz de facto dwa razy tę samą wielkość pod dwiema nazwami.

5. Dwie różne definicje „throughput" w jednym zestawie wyników

- master_summary.throughput_mean = n / wall_seconds → max 383 tys./s (sensowne)
- windows.csv.throughput = 1e6 / avgUpdateMicros, gdzie avgUpdateMicros mierzy tylko model.predict(), bez detektora/selektora/treningu

W efekcie w E1/windows.csv 58 848 z 123 280 wierszy (48%) ma throughput > 1 mln inst/s, maksymalnie 39.7 mln/s. Jeśli któryś wykres czasowy bierze throughput z windows, pokazuje bzdurę.

6. Level-2 (KSWIN + BH-FDR) prawie nic nie lokalizuje w E2

E2: 91.8% alarmów ma num_drifting_features = 0
E3: 34.8%
E1: 19.5%
W E2 — bloku, który cały jest o adaptacyjnej selekcji — S2/S4 w 92% przypadków nie dostają żadnej cechy i lecą fallbackiem „zdecayuj wszystko". To osłabia interpretację całego E2.

7. 14 konfiguracji z drift_count = 0 — detektor nie strzelił ani razu przez 100k instancji:

E1/E2/E3 LED:  HT+S1, SRP+S1, SRP+S2, DA-SRP-ABC, Majority, NoChange
E1 RandomRBF:  Majority, NoChange
E5 RandomRBF:  DA-ARF+HDDM_A, DA-ARF+HDDM_W   ← oba dają identyczną kappa 0.905201
HDDM_A i HDDM_W na RandomRBF są kompletnie martwe → warstwa external DA-ARF nigdy się nie uruchamia → identyczny wynik. W E5 (blok o detektorach) to znaczy, że 2 z 4 badanych detektorów na jednym z 4 zbiorów nie robią nic.

8. Nasycenie STAGGER w E4/E5 — 12 wariantów z κ = 1.000000 i acc = 1.000000

E4 STAGGER-Low:   ARF, ARF+S1, SRP, SRP+S1, DA-ARF-ABC, DA-SRP-ABC → wszystkie 1.0
E4 STAGGER-HiDyn: ARF, SRP, DA-ARF-ABC → 1.0
Stąd Friedman w E4 nie wykrywa niczego: accuracy/kappa/kappa_per p = 0.845, recovery p = 0.852, temporal p = 0.643. Istotny jest tylko ram_hours_gb (p = 0.0001) — czyli jedyny „wynik" E4 pochodzi z metryki opisanej w pkt 2. E4 w obecnej formie nie ma mocy statystycznej. (STAGGER został odfiltrowany z E1–E3 przez
SATURATED_DATASETS_BY_BLOCK, ale w E4/E5 świadomie został — tyle że tam stanowi połowę zbiorów.)

🟡 Warte uwagi

9. Bałagan w results/ — trzy pokolenia plików obok siebie

┌───────────────────────────────────────────────────────────────────────────┬───────────────┬──────────────────────────────────────────────────────┐
│                                    co                                     │     data      │                        status                        │
├───────────────────────────────────────────────────────────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
│ E{1..5}/*.csv, master_summary, runs_raw                   a        │      │ 2026-08-10    │ ✅  aktualne                                          │
├───────────────────────────────────────────────────────────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
│ results/E1_baselines.csv … (root)                         s               │ 2026-05-16    │ ❌ stary duplikat, inna treść  (E1: 40 vs 64 wiersze) │
├───────────────────────────────────────────────────────────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
│ E2/E2_results.csv, E3/E3_window.csv, E4/E4_ranking.csv, E2/summary.csv, … │ 2026-05-03/05 │ ❌  legacy z poprzedniego runnera                     │
├───────────────────────────────────────────────────────────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
│ 56 figur + 9 tabel                                                        │ 2026-05-05    │ ❌  nieregenerowane sieroty                           │
├───────────────────────────────────────────────────────────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
│ 18 figur *_STAGGER* w E1/E2/E3                                            │ 2026-07-31    │ ❌  STAGGER już nie istnieje w E1–E3                  │
├───────────────────────────────────────────────────────────────────────────┼───────────────┼──────────────────────────────────────────────────────┤
│ pozostałe 316 figur + 84 tabele                                           │ 2026-08-20    │ ✅  dzisiejsze                                        │
└───────────────────────────────────────────────────────────────────────────┴───────────────┴──────────────────────────────────────────────────────┘

analysis/loaders.py czyta z podkatalogów bloków, więc stare pliki nie zatruwają wykresów — ale duplikat results/E1_baselines.csv z inną liczbą wariantów to pułapka przy ręcznym sprawdzaniu wyników. Osierocone tabele w tables/ (np. tab_e4_min_detectable.tex, tab_e3_pairwise_wilcoxon.tex) wciąż tam leżą i wskoczą do
pracy, jeśli zaciągniesz katalog hurtem — a opisują nieistniejącą już konfigurację.

10. NHTS: S1 kolapsuje do klasy większościowej
    NHTS: ARF+S1 κ=0.0000   SRP+S1 κ=0.0000   HT+S1 κ=-0.0071   (vs ARF 0.188, SRP 0.413)
    K=⌈√17⌉=5 na zbiorze z ~0.6% klasy mniejszościowej. To ten sam mechanizm, który plan opisał dla DA-ARF (A7), ale tutaj dotyczy baseline'ów z selekcją i nie jest nigdzie odnotowany.

11. YahooFinance: wszystkie modele przegrywają z NoChange
    NoChange κ=0.2482 |  ARF 0.291  SRP 0.104  HT 0.069  ARF+S1 0.227  SRP+S1 0.033
    Tylko pełny ARF minimalnie bije naiwny baseline. Przy kappa_per (temporal) sytuacja jest jeszcze gorsza — to zbiór, gdzie „predykcja = poprzednia etykieta" jest prawie nie do pobicia. Wymaga komentarza w pracy, inaczej wygląda jak zepsuty eksperyment.

12. Drobne
- mean_selected_feature_count_std = 0.0 w 250/250 wierszy (stałe K — poprawne, ale kolumna bezużyteczna).
- rank_matrix_*.csv anonimizuje zbiory do D1..D8 — nie da się odtworzyć, który to który.
- W E2 tylko 4 z 4372 rekordów selekcji mają trigger drift_alarm — reszta re-selekcji S2/S4 jest etykietowana jako selection_change (zmiana widoczna dopiero po wPostDrift), więc wykres „co wywołało zmianę selekcji" jest mylący.
- E1/adaptation_events.csv i E2/adaptation_events.csv są puste (177 B, sam nagłówek) — to poprawne, w tych blokach nie ma modeli DA.

✅ Co jest zdrowe

- 1250/1250 runów status=OK, zero błędów, zero brakujących seedów (250 komórek × 5).
- Zero NaN w runs_raw poza kolumną error (z definicji pusta).
- feature_importance.csv: zero snapshotów ze zdegenerowanym uniform 1/d (19 403 w E1, 12 755 w E3) — fallback degenerateUniformFallbacks nigdy nie zadziałał.
- Liczby instancji spójne we wszystkich seedach (NHTS 1 985 822, NYCTaxi 405 306, YahooFinance 198 899, syntetyki 100 000).
- ARF+Sx ≠ SRP+Sx na wszystkich zbiorach — brak podejrzanego kopiowania wyników między modelami.
- Zero nan/inf w wygenerowanych .tex.
- Majority daje κ=0.0000 wszędzie tam, gdzie powinien — sanity check Level-1 przechodzi.

Kolejność naprawy wg kosztu: pkt 9 (sprzątanie, minuty) → pkt 1 i 2 (poprawki w Javie, wymagają re-runu — zrób razem) → pkt 3/4/5 (metryki, też Java, do tego samego re-runu) → pkt 8 (zamiana STAGGER w E4/E5 na coś nienasyconego, np. LED-HiDyn). Chcesz, żebym zaczął od któregoś?


Punkty 1–3 zamknięte. Z pozostałych dziewięciu tylko część to faktycznie usterki do naprawy — reszta to wyniki do opisania. Rozdzielenie tych dwóch grup jest tu najważniejsze.

Warto poprawić — kod, do tego samego re-runu

Pkt 4 (temporal_kappa) — sprawdziłem: kappa_per i temporal_kappa to ta sama metryka (TemporalKappa), różnią się wyłącznie agregacją. kappa_per_mean to wartość z ostatniego okna,
temporal_kappa_mean to średnia po wszystkich oknach. Kolumna temporal_kappa w windows.csv jest czystym placeholderem NaN (RunDetailedRecorder:125). Naprawa: usunąć martwą kolumnę i
przemianować obie na coś, co mówi prawdę (kappa_temporal_final / kappa_temporal_windowed). Inaczej w pracy masz dwie tabele i dwa CD-diagramy tej samej wielkości pod różnymi nazwami —
to pytanie na obronie, na które nie ma dobrej odpowiedzi. ~30 min.

Pkt 5 (throughput) — dwie definicje pod jedną nazwą. windows.csv liczy 1e6/avgUpdateMicros, gdzie mierzony jest tylko model.predict(), stąd 39.7 mln inst/s. Albo mierzyć pełny krok
pętli, albo przemianować kolumnę na predict_latency_us i nie udawać, że to przepustowość. ~20 min.

Pkt 12, dwie z czterech pozycji:
- rank_matrix_*.csv anonimizuje zbiory do D1..D8 (BlockStatisticalAnalysis:391, jedna linia) — nie da się odtworzyć, który wiersz to który zbiór. Trywialne, a bez tego macierze rang
  są bezużyteczne w załączniku.
- Etykietowanie triggera selekcji w E2 (4 z 4372 rekordów jako drift_alarm) — S2/S4 zmieniają selekcję dopiero po wPostDrift, więc zapisuje się selection_change. Wystarczy przenieść
  etykietę z alarmu na moment re-selekcji.

Pozostałe dwie pozycje z pkt 12 zostawiłbym: mean_selected_feature_count_std = 0 jest poprawne (K jest stałe), a puste adaptation_events.csv w E1/E2 to prawidłowe zachowanie.

Warto poprawić — bez kodu, do zrobienia od ręki

Pkt 9 (bałagan w results/) — największy stosunek zysku do kosztu w całej liście. Do usunięcia: 5 nieaktualnych duplikatów w roocie results/, ~20 plików legacy z 3–5 maja w katalogach
bloków, 56 osieroconych figur i 9 tabel, 18 figur *_STAGGER* z E1–E3. Ryzyko jest realne: tab_e4_min_detectable.tex czy tab_e3_pairwise_wilcoxon.tex opisują konfigurację, która już
nie istnieje, i wejdą do pracy, jeśli zaciągniesz katalog hurtem. ~10 min, bez re-runu.

Wymaga Twojej decyzji — projekt eksperymentu

Pkt 8 (nasycenie STAGGER w E4/E5) — merytorycznie najpoważniejszy z pozostałych. E4 nie ma obecnie mocy statystycznej: Friedman dla accuracy/kappa/kappa_per p = 0.845, dla recovery p
= 0.852. Jedyny „istotny" wynik pochodził z ram_hours_gb, czyli z metryki, którą właśnie naprawiliśmy jako niemierzącą tego, co deklarowała. STAGGER to 3 cechy binarne i κ = 1.000000
dla 9 z 24 wariantów — nie ma czego mierzyć.

To nie jest bug do załatania, tylko wybór zbiorów do zmiany, i dlatego pytam zamiast działać: zastąpić STAGGER w E4/E5 czymś nienasyconym (LED z wieloma dryfami jest naturalnym
kandydatem — 24 cechy, 10 klas, ~74% Bayesa, już masz generator), czy zostawić i opisać E4 jako blok bez rozstrzygnięcia? Pierwsze daje rozdział z wynikiem, drugie jest tańsze.

Nie naprawiać — to są wyniki, nie usterki

Pkt 7 (detektor milczy na LED / HDDM martwe na RandomRBF), 10 (S1 kolapsuje na NHTS przy K=⌈√17⌉=5 i klasie 0.6%), 11 (na YahooFinance NoChange bije prawie wszystko) — to poprawnie
zmierzone zachowanie metod na trudnych danych. Pkt 10 to zresztą dokładnie ten sam argument o budżecie cech, który plan już rozwinął dla DA-ARF w sekcji A7, tylko dla baseline'ów —
mocny materiał, nie wstyd. Jedyne, czego wymagają, to akapit komentarza; „naprawianie" ich polegałoby na dobieraniu konfiguracji aż wyjdzie ładniej.

Pkt 6 (92% alarmów w E2 bez zlokalizowanej cechy) jest na granicy. Wszystkie zbiory w E2 mają dryf stopniowy, a Level-2 porównuje okna oddalone o 200 instancji — możliwe, że to
właściwość metody, a nie parametrów. Zrobiłbym jeden przebieg wrażliwości na kswin_window i bhQ, i niezależnie od wyniku opisał; jeśli lokalizacja nie działa na dryfie stopniowym, to
jest ograniczenie metody i trzeba je nazwać wprost.

Proponowana kolejność

1. Pkt 9 — teraz, 10 minut, zero ryzyka.
2. Decyzja co do pkt 8.
3. Pkt 4 + 5 + 12 — jeden commit, ~1 h.
4. Jeden re-run E1–E5 zbierający pkt 1, 2, 3, 4, 5, 12 (+8, jeśli zmieniamy zbiory).
5. Pkt 6 jako osobny mały eksperyment wrażliwości.
6. Pkt 7, 10, 11 → akapity w rozdziale z wynikami.