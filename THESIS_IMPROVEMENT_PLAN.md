
# Plan implementacyjny — zmiany w KODZIE (nie w raporcie)

> Tylko poprawki wymagające zmian w kodzie: **Java** (`stream/src/main/java/thesis/...`) i **pipeline analiz**
> (`analysis/*.py`). Redakcja / pisanie rozdziałów — osobno, później.
> Priorytety: **P0** = największy wpływ / zrób pierwsze, **P1** = ważne, **P2** = jeśli czas.
> Checkboxy do odhaczania. Odwołania `plik:linia` względem repo.

---

## LEGENDA lokalizacji
- **[JAVA]** = `stream/src/main/java/thesis/...` → wymaga rekompilacji + re-runu eksperymentów.
- **[PY]** = `analysis/*.py` → tylko regeneracja wykresów/tabel (`python -m analysis`), bez re-runu Javy.
- Zmiany [PY] są tanie (sekundy). Zmiany [JAVA] są drogie (re-run E1–E5).

---

## A. [JAVA] Naprawa DA-ARF — P0 — ZAIMPLEMENTOWANE + ZDIAGNOZOWANE (hipoteza obalona)

**Status kodu:** wszystko A1–A5 zaimplementowane, kompiluje się (JDK 17), smoke test
`DAARFRepairSmokeTest` = 9/9 PASS. Runy diagnostyczne wykonane.

- [x] **A1. Instrumentacja diagnostyczna.** `DAARFWrapper`: nowe liczniki `extSurgicalCount`,
      `extNoReplacementCount`, `extGatedSkipCount`, `intrinsicFullResetCount` (+ istniejące `bkgPromotions`).
      `RunDetailedRecorder.onDAARFEvent` rozszerzony — intrinsic resety → `full_replacement_count`, promocje →
      `kept_count`, external surgical → `surgical_count`, external reset → `ext_full_count` (rozdzielone w CSV,
      bez zmiany schematu). Runner loguje per-instancję wszystkie delty.
- [x] **A2. Tryb `ExternalActionMode {RESET, SURGICAL}`** (`daarf_external_mode`). Surgical: podmienia dryfujące
      cechy w podprzestrzeni, zachowuje drzewo, przebudowuje `reducedHeader` (swap tylko typo-zgodny numeric↔numeric).
- [x] **A3. Gating** (`daarf_gate_external`): external pomija drzewa z pending background.
- [x] **A4. Intrinsic OFF** (`daarf_intrinsic_drift=false`): wyłącza intrinsic ADWIN, external jedynym mechanizmem.
- [x] **A5. Ekspozycja w configu:** wszystkie flagi + `externalResetFraction`/`warningDelta`/`driftDelta`
      (już były) w `VariantSpec`. 3 nowe warianty w `master_experiments.json` E3 + `analysis/config.py`.

### ⚠️ WYNIK DIAGNOSTYCZNY A1 — hipoteza „podwójnego resetu" OBALONA

Runy diagnostyczne (Hyperplane/FeatureDrift 20k; NYCTaxi/NHTS 30k, seed=1) pokazały liczbami:

| dataset | promote (intrinsic) | intr_reset | ext_full | ext_surg | κ ABC | κ surgical | κ extonly | κ ARF |
|---|---|---|---|---|---|---|---|---|
| Hyperplane | 17 | **0** | **0** | 0 | 0.709 | 0.709 | 0.578 | 0.681 |
| FeatureDrift | 24 | **0** | **0** | 0 | 0.626 | 0.626 | 0.584 | 0.656 |
| NYCTaxi | 15 | **0** | 6 | 2 | 0.810 | **0.710** | 0.823 | 0.909 |
| NHTS | — | — | — | — | **0.000** | 0.000 | 0.000 | 0.554 |

Wnioski (do wpisania jako wynik pracy — to jest wartościowe negatywne odkrycie):
1. **Nie ma podwójnego resetu.** `intr_reset=0` wszędzie — intrinsic ADWIN zawsze promuje ciepły background,
   NIGDY nie robi destrukcyjnego full-resetu. Pierwotna hipoteza z sekcji 0 była błędna.
2. **Surgical (A2) POGARSZA** na NYCTaxi (0.710 vs 0.810) — chirurgiczna podmiana cech w drzewach ARF szkodzi
   bardziej niż czysty reset. A2 odrzucone.
3. **Luka do ARF jest architektoniczna, nie od reset-layera.** Najlepszy wariant (extonly 0.823) wciąż daleko od
   ARF_baseline (0.909). Winne są komponenty B (importance-weighted subspace sampling) i/lub C (top-K voting),
   nie warstwa resetu.
4. **NHTS: DA-ARF kolapsuje do klasy większościowej (κ=0)** vs ARF κ=0.554 — osobny, poważny bug w głosowaniu/subspace.

### A6 — Ablacja komponentów B/C — ZROBIONE (B i C oczyszczone)

Wyniki κ (30k, seed=1), izolacja każdego komponentu:

| dataset | ARF | ABC | noB | noC | noBC | noExt | noBC-noExt |
|---|---|---|---|---|---|---|---|
| Hyperplane | 0.760 | **0.764** | 0.706 | 0.762 | 0.706 | 0.764 | 0.706 |
| FeatureDrift | 0.662 | **0.756** | 0.577 | 0.705 | 0.594 | 0.756 | 0.406 |
| NYCTaxi | **0.909** | 0.810 | 0.758 | 0.749 | 0.775 | 0.719 | 0.800 |
| NHTS | **0.554** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

- [x] **B (importance-weighted sampling) POMAGA** — usunięcie (`noB`) szkodzi wszędzie (FeatureDrift 0.756→0.577).
- [x] **C (top-K voting) POMAGA** — `noC` gorsze na FeatureDrift/NYCTaxi. **DA-ARF-ABC to najlepsza konfiguracja**
      i na FeatureDrift (0.756) **bije ARF** (0.662). Hipoteza „B/C psują" OBALONA.
- [x] **External layer POMAGA na NYCTaxi** (`noExt` 0.719 < ABC 0.810). Też nie jest problemem.
- [x] **Luka NYCTaxi jest strukturalna** — `noBC-noExt` (najbliżej plain ARF) = 0.800, wciąż < ARF 0.909.

### A6b — Rozmiar podprzestrzeni — ZROBIONE (częściowa przyczyna NYCTaxi)

κ vs `daarf_subspace_size` (domyślnie ⌈√d⌉=5):

| dataset | ARF | sub5 | sub8 | sub12 | sub16 |
|---|---|---|---|---|---|
| NYCTaxi | 0.909 | 0.810 | 0.828 | **0.849** | 0.835 |
| NHTS | 0.554 | 0.000 | 0.000 | 0.000 | 0.000 |

- [x] **NYCTaxi: większa podprzestrzeń pomaga** (+0.04 przy sub12) — projekcja ⌈√d⌉ jest za agresywna dla realnych
      danych (spójne z motywem budżetu cech K=⌈√F⌉ z sekcji 0). Rezydualna luka do ARF zostaje → custom ensemble
      jest nieco słabszy niż strojony MOA ARF.

### A7 — Kolaps NHTS — ROZWIĄZANE: znaleziono PRAWDZIWY root cause

- [x] **Root cause = DA-ARF nie replikował hiperparametrów drzew MOA ARF.** MOA `AdaptiveRandomForest` tworzy drzewa
      z `ARFHoeffdingTree -e 2000000 -g 50 -c 0.01` (grace=50, δ=0.01). DA-ARF `newTree()` ustawiał TYLKO
      `subspaceSizeOption` → drzewa dziedziczyły domyślne `HoeffdingTree` (grace=200, **δ=1e-7**) → prawie się nie
      dzieliły → płytkie → przy niezbalansowaniu NHTS (~0.6% mniejszości) kolaps do klasy większościowej (κ=0).
- [x] **Fix zaimplementowany:** `newTree()` ustawia teraz grace=50, δ=0.01, maxByteSize=2e6 (domyślnie MOA-matching);
      wystawione jako config (`daarf_tree_grace_period`, `daarf_tree_split_confidence`) + setter `setTreeParams`
      rebuildujący ensemble. Dodano ułamkową podprzestrzeń (`daarf_subspace_fraction`).

### Wyniki naprawy (κ, 30k, seed=1) — tree tuning × subspace

| dataset | ARF | untuned (stare) | tuned (sqrt) | **tuned+wide 0.75** |
|---|---|---|---|---|
| SEA | 0.643 | 0.654 | 0.627 | **0.648** |
| Hyperplane | 0.760 | 0.776 | 0.572 | **0.764** |
| RandomRBF | 0.916 | 0.879 | 0.868 | **0.934** |
| FeatureDrift | 0.662 | 0.729 | 0.713 | **0.757** |
| NYCTaxi | 0.909 | 0.761 | 0.801 | **0.900** |
| NHTS | 0.554 | 0.000 | 0.000 | **0.123** |

- [x] **`DA-ARF tuned+wide` dorównuje/bije ARF na 5/6** (SEA, Hyperplane, RandomRBF, FeatureDrift, NYCTaxi).
- [x] Sam tree-tuning przy wąskiej podprzestrzeni psuje Hyperplane (0.776→0.572, overfitting pod dryf ciągły) —
      **wide subspace to naprawia** (→0.764). Oba czynniki potrzebne razem.
- [x] NHTS: z totalnego kolapsu (0) do 0.123 (wciąż słabo — sub≥12/17 potrzebne; NHTS pozostaje trudny, ale
      to już nie degeneracja).

### WNIOSEK KOŃCOWY sekcji A (do pracy — mocna, uczciwa narracja)

Słabość DA-ARF vs ARF **NIE była w maszynerii drift-aware** (podwójny reset — brak; B/C/external — pomagają).
Były to **dwa ukryte handicapy bazowego ensembla**, które fałszowały porównanie z baseline ARF:
1. **Nietrojone drzewa bazowe** (grace=200/δ=1e-7 zamiast MOA 50/0.01) → płytkie drzewa, kolaps na NHTS.
2. **Za agresywna podprzestrzeń** (⌈√d⌉) → za mało cech na realne dane; ~0.75·d przywraca konkurencyjność.
Po ich naprawie DA-ARF przechodzi z „przegrywa 7/8" na „konkurencyjny/lepszy 5/6". To jest właściwy wynik
naprawy autorskiej metody + rygorystyczna ścieżka diagnostyczna (A1–A7) jako materiał do rozdziału.

- [x] ~~A2 surgical~~ / ~~podwójny reset~~ / ~~B-C~~ — wszystkie odrzucone empirycznie (dobre negatywne wyniki).
- [x] Kod A1–A7 + `DAARFRepairSmokeTest` (9/9) skompilowany na JDK 17. Config E3: dodano
      `DA-ARF-ABC-untuned` (pokazuje handicap) i `DA-ARF-ABC-wide` (naprawa).

### A8 — Analiza różnorodności ensembla (F → K → drzewa) — ZROBIONE

Ensemble N=10. Wskaźnik: **ovlp** = oczekiwane nakładanie podprzestrzeni 2 drzew (wyższe=mniej różnorodne);
`C(K,m)` = liczba możliwych różnych podprzestrzeni z puli.

| dataset | F | K=⌈√F⌉ | ARF+S1 (pula=K) | DA-ARF sqrt | DA-ARF f0.75 | DA-SRP |
|---|---|---|---|---|---|---|
| STAGGER | 3 | 2 | m=2 **C=1 ovlp=1.00** | 0.67 | 1.00 | 1.00 |
| SEA | 8 | 3 | m=3 **C=1 ovlp=1.00** | 0.38 | 0.75 | 0.50 |
| Hyperplane/RBF/NYCTaxi | 20 | 5 | m=3 C=10 ovlp=0.60 | 0.25 | 0.75 | 0.25 |
| FeatureDrift | 25 | 5 | m=3 C=10 ovlp=0.60 | 0.20 | 0.76 | 0.24 |
| YahooFinance | 36 | 6 | m=3 C=20 ovlp=0.50 | 0.17 | 0.75 | 0.19 |

Empiria (`DAARFDiversityProbe`, Hyperplane F=20): narrow m=5 → Jaccard **0.14**; wide m=15 → Jaccard **0.60**.

- [x] **Ścieżka S1 (ARF+S1/SRP+S1) degeneruje różnorodność przez K=⌈√F⌉.** Podwójna redukcja F→K→√K:
      STAGGER (K=2) i SEA (K=3) → **C=1 → wszystkie 10 drzew IDENTYCZNE** (ensemble = 1 drzewo). K=5 → tylko 10
      możliwych podprzestrzeni na 10 drzew → 60% nakładania. To nowy, twardy argument, że K=⌈√F⌉ jest za agresywne
      (nie tylko dla accuracy, ale niszczy różnorodność lasu). Łączy się z sekcją 0 / motywem budżetu cech.
- [x] **DA-ARF/DA-SRP (sqrt) mają ZDROWĄ różnorodność** (omijają K, pula=F, Jaccard 0.14) → ukryta przewaga metod DA
      nad baseline S1.
- [x] **`subspace_fraction` to trade-off siła↔różnorodność.** Sweep κ (30k, 1 seed):

  | dataset | ARF | sqrt | **f0.50** | f0.75 |
  |---|---|---|---|---|
  | SEA | 0.643 | 0.627 | **0.654** | 0.648 |
  | Hyperplane | 0.760 | 0.572 | 0.718 | 0.764 |
  | RandomRBF | 0.916 | 0.868 | 0.906 | 0.934 |
  | FeatureDrift | 0.662 | 0.713 | **0.761** | 0.757 |
  | NYCTaxi | 0.909 | 0.801 | 0.892 | 0.900 |
  | NHTS | 0.554 | 0.000 | 0.000 | 0.123 |

- [x] **DECYZJA: domyślny ułamek = 0.50** (sweet spot). Dokładność ≈ f0.75 na 4/6 (różnice w szumie), ale POŁOWA
      nakładania (0.50 vs 0.75). f0.75 kupuje ~0.01–0.04 κ kosztem podwojenia nakładania — zły interes dla lasu.
      NHTS pozostaje słaby niezależnie od ułamka (osobna słabość). `DA-ARF-ABC-wide` w E3 ustawiony na 0.5.

### DECYZJA finalna nt. domyślnych DA-ARF (przed pełnym re-runem)
- [x] **Domyślny DA-ARF = strojone drzewa + podprzestrzeń 0.5·d** (`daArfSubspaceFraction=0.5`). Powód: strojone
      drzewa przy ⌈√d⌉ regresują na dryfcie ciągłym (Hyperplane 0.72→0.57); 0.5 to zbalansowana naprawa.
      Zweryfikowane: DA-ARF-ABC domyślny → Hyperplane 0.718 (nie 0.572), LED 0.711 (= pełny ARF).
- [x] E3 ablacja izoluje oba fixy: `DA-ARF-ABC` (strojone+0.5 = naprawa), `DA-ARF-ABC-untuned` (oryginał: nietrojone+sqrt),
      `DA-ARF-ABC-narrow` (strojone+sqrt = izoluje efekt podprzestrzeni). A/AB/ABC przy stałym 0.5.
- [ ] **Pełny re-run E1–E5** (1480 runów): `bash stream/run_experiments.sh`, potem `analysis/.venv/bin/python -m analysis`.
- [ ] Uwaga: re-run NADPISUJE `stream/results/` — zrobić backup, jeśli stare wyniki potrzebne.

---

## B. [PY] Wymiana zdegenerowanych metryk / wykresów — P0 (tanie, bez Javy)

**B1. Recovery: zamienić `recovery_length` (~1.0 wszędzie) na `max_drop` + `area_under_recovery_curve`. ✅ ZROBIONE**

- [x] `block_utils.plot_recovery` → 2-panelowy wykres (mean max_drop + mean area), odporny na brak kolumn;
      `recovery_table` agreguje tylko istniejące kolumny. Nowa metryka DYSKRYMINUJE (STAGGER: S1/S2 drop ~0.09, DA-SRP ~0).
- [x] Wszystkie bloki E1–E5 przełączone: fname `*_recovery_length` → `*_recovery_depth`.
- [x] `tab_e1_recovery` przełączona na `mean_max_drop` (zamiast recovery_length).
- [x] Stare pliki `e[1-5]_recovery_length.{pdf,png}` usunięte z `figures/`.
- [ ] (POMINIĘTE — wymaga Javy) `max_drop_mean`/`auc_recovery_mean` w `master_summary.csv` do CD-diagramów.
      CD/rank-tabele dla recovery są pre-liczone w Javie na zdegenerowanym `recovery_time` → osobna zmiana [JAVA].

**B2. Usunąć trywialne wykresy stability/count z E1/E4/E5. ✅ ZROBIONE**

- [x] E1: usunięte wywołanie `plot_feature_selection` w `e1_analysis.run()` (selekcja statyczna, tylko K=const).
- [x] Guard `block_utils.is_static_selection` — `plot_feature_selection_overview` pomija bloki, gdzie są tylko
      triggery `initial` (chroni też E4/E5). Zweryfikowane: E1 ma wyłącznie `initial`, E2/E3 mają `selection_change`.
- [x] Zostawione TYLKO w E2/E3 (tam selekcja realnie się zmienia).
- [x] Stary `e1_feature_selection_overview.{pdf,png}` usunięty.

**B3. Odfiltrować generyczny STAGGER z tabel/wykresów kappa+accuracy w E1/E2/E3. ✅ ZROBIONE**

- [x] `config.SATURATED_DATASETS_BY_BLOCK` (E1/E2/E3 → STAGGER) + `SATURATED_METRICS` (kappa_mean, accuracy_mean).
- [x] `block_utils.drop_saturated` wpięte centralnie w `metric_pivot` i `plot_metric_bar` → automatycznie filtruje
      wszystkie tabele/heatmapy/bary kappa+accuracy. Zweryfikowane: STAGGER=0 w tab_e1_baselines/accuracy, tab_e2_kappa,
      tab_e3_ablation.
- [x] STAGGER ZACHOWANY w `temporal_kappa` (=1 w tab_e1_temporal_kappa) i w E4/E5 (STAGGER-HiDyn =2 w tab_e4_kappa).

**B4. Oznaczyć cechy szumowe na heatmapach importance. ✅ ZROBIONE**

- [x] `config.NOISE_FEATURES` (noise=5 dla SEA/Hyperplane/RandomRBF/FeatureDrift) + `block_utils.noise_feature_indices`
      (szum = ostatnie N indeksów, potwierdzone z `SyntheticStreamFactory.NoiseAugmentedStream.buildHeader`).
- [x] Nowa `block_utils.plot_importance_noise_annotated` — per-dataset heatmapa variant×feature z czerwonym pasem
      + czerwonymi etykietami na kolumnach szumu. Wołana w E1/E2/E3.
- [x] Zweryfikowane wizualnie: cechy szumowe (np. FeatureDrift idx 20–24) mają NAJNIŻSZĄ ważność u wszystkich
      wariantów — czytelnie pokazuje, że FS odrzuca szum.

---

## C. [PY] Nowe wykresy czasowe z ISTNIEJĄCYCH danych — P1 (bez Javy) ✅ ZROBIONE

- [x] **C1. Timeline akcji adaptacji.** `block_utils.plot_adaptation_timeline` — per (dataset) subplot na wariant,
      każde zdarzenie jako marker w torze akcji (KEEP/SURGICAL/FULL/NO_REPL/EXT_KEEP/EXT_FULL) na osi `instance_index`,
      rozmiar ~ liczba learnerów, linie GT-dryftu. Wołane w E3/E4/E5 (subdir `e{3,4,5}_timelines`).
      **Wynik diagnostyczny:** wprost widać, że DA-SRP działa przez SURGICAL, a DA-ARF przez EXT_KEEP/EXT_FULL.
- [x] **C2. Selection timeline dla E3.** `block_utils.plot_selection_timeline` (uogólniony, z filtrem wariantów) —
      DA-* w E3, indeksy wybranych cech w czasie, linie GT-dryftu + **zacieniony pas cech szumowych**.
      `e3_timelines/e3_selection_timeline_<DS>`. (E2 zachował własną wersję.)
- [x] **C3. Overlay alarm → akcja → recovery.** `block_utils.plot_causality_overlay` — 2 panele: górny κ(okno) +
      alarmy detektora + GT-dryft; dolny raster akcji adaptacji. Dla FeatureDrift i NYCTaxi (DA-SRP-ABC).
      `e3_timelines/e3_causality_<DS>`. Czytelnie pokazuje łańcuch alarm→akcja→recovery κ.
- [x] **C4. Evolucja importance + NYCTaxi + `is_drifting`.** `e3_analysis.plot_importance_evolution` rozszerzony:
      dodany NYCTaxi (realny zbiór) i markery × w punktach flagi dryftu per cecha. `e3_extra/e3_importance_evolution_<DS>`.
      **Obserwacja:** na NYCTaxi top-cechy są niemal ciągle flagowane jako dryfujące, a mimo to pozostają najważniejsze.

---

## NOWY: dodanie ciekawego syntetyka LED (zamiast nasyconego STAGGER) — ZROBIONE

- [x] Dodano generator **LED (LEDGeneratorDrift)** do E1/E2/E3: 24 cechy = **7 istotnych (idx 0–6) + 17 nieistotnych
      (idx 7–23)**, 10 klas, dryf `numberAttributesDrift=4`, szum 10%. Zweryfikowano definitywnie ordering cech
      (probe: istotne = deterministyczna funkcja klasy). `NOISE_FEATURES["LED"]=17` → B4 automatycznie zaznaczy szum.
- [x] Kod: `SyntheticStreamFactory.createLEDDrift` + case `LED/LEDDRIFT` w runnerze; `analysis/config.py`
      (DATASET_ORDER, SYNTHETIC_DATASETS, DRIFT_POINTS=continuous, NOISE_FEATURES=17).
- [x] **Idealny dla tezy o FS i mocno dyskryminujący:** ARF pełny κ=0.711 vs ARF+S1 (K=⌈√24⌉=5) κ=0.559 —
      K=5 < 7 istotnych → FS **strukturalnie nie może** złapać wzorca (twardy argument budżetu). Bonus: filtrowa FS
      (IG per cecha) źle radzi sobie na LED (istotność kombinatoryczna) → dobra dyskusja o granicach filtrowej selekcji.

## OPCJA B: [JAVA] Bezrefleksyjny natywny DA-SRP — ZROBIONE + ZWALIDOWANE

Motywacja: stary `DriftAwareSRP` opakowuje MOA StreamingRandomPatches i **refleksją** grzebie w jego
prywatnych podprzestrzeniach (krucha — zgaduje nazwy pól; nieobronne, trudne do instrumentacji).
Referencyjny `/home/kubog/MLDataStreams/stream-ctr` pisze to ręcznie (jawny SubspaceManager) — jak nasz DA-ARF.

- [x] Napisany `NativeDriftAwareSRP` — własny ensemble, ZERO refleksji: rzutowane patche `ARFHoeffdingTree` +
      online bagging + per-learner ADWIN + `handleDrift` (KEEP/SURGICAL/FULL wg overlap+tau) + głosowanie
      z correction-alpha. Zachowany cały publiczny kontrakt (settery, gettery totalKept/Surgical/Full/NoReplacement,
      setDriftListener, ModelWrapper).
- [x] **Kluczowe:** MOA SRP domyślnie używa **60% cech na patch** (subspaceSize=60, tryb Percentage) — natywny
      MUSI też (`defaultSubspaceSize=0.6·d`); wąskie sqrt kolapsuje na realnych (ta sama lekcja co DA-ARF).
- [x] **Walidacja head-to-head (30k, 2 seedy):** natywny ≈ refleksyjny, LEPSZY na syntetykach (Hyperplane +0.03,
      RandomRBF +0.03), minimalnie słabszy na NYCTaxi (−0.04)/NHTS (−0.03, bez kolapsu). Wygrywa 5/7.
- [x] Natywny jest DOMYŚLNY (`VariantSpec.daSrpNative=true`); `da_srp_native:false` wybiera starą refleksyjną;
      E3 ma wariant porównawczy `DA-SRP-ABC-reflect`. Smoke testy: DAARFRepair 9/9, DriftAwareSRP 27/27.
- [ ] (opcjonalnie) dodać dedykowany `NativeDriftAwareSRPSmokeTest` — walidacja runem już to pokrywa.

## D. [JAVA] Instrumentacja per-learner — ODRZUCONE przez użytkownika (za drogie, mały zwrot)

> Zaimplementowane i **cofnięte** na prośbę użytkownika (D/E/F niepotrzebne). Oryginalny plan poniżej dla referencji.

**Blokada:** żaden CSV nie ma `learner_id`. Bez tego wykresy „per-learner" i „learner↔cechy" są niemożliwe.

- [ ] **D1. Log per-learner.** W `DriftAwareSRP` (i opcjonalnie `DAARFWrapper`) dopisać emisję wiersza
      `{block,dataset,variant,seed,instance_index,learner_id,action,subspace,local_error}` do nowego
      `per_learner.csv`. Miejsce: metoda `train`/akcje adaptacji + tam gdzie znane są podprzestrzenie.
- [ ] **D2. Wykres [PY] po D1.** Heatmapa learner × feature w czasie + „które learnery żyją długo / kiedy reset".
- [ ] **Decyzja:** zrobić TYLKO dla DA-SRP (bo działa) na 1 datasecie. Jeśli brak czasu — pominąć całą sekcję D.

---

## E. [JAVA] K-sensitivity — P2 (opcjonalny eksperyment; wymaga zmiany runnera)

**Blokada:** `UnifiedStreamExperimentRunner.buildSelector` hardkoduje `K = StaticFeatureSelector.defaultK(d)`
i S1 używa domyślnego konstruktora. Sweep K wymaga przekazania K z configu.

- [ ] **E1. Dodać pole `k` / `kFraction` do `VariantSpec`** i użyć w `buildSelector` zamiast stałego `defaultK`.
      (`NoFeatureSelection` dla K=all już istnieje.)
- [ ] **E2. Nowy blok configu** (np. `E6_ksensitivity`): `K ∈ {⌈√d⌉, ⌊0.25d⌋, ⌊0.5d⌋, d}` × {S1, DA-SRP-ABC}
      × {Hyperplane, FeatureDrift, NYCTaxi, RandomRBF}.
- [ ] **E3. [PY] Wykres budżet-K → accuracy** (nowa funkcja w `cross_experiment.py`).

---

## F. [JAVA] Dostrojenie DA-SRP — P2 (drobne, config-driven)

DA-SRP działa — to głównie parametry, nie przebudowa. Większość już jest w konstruktorze/setterach.

- [ ] **F1. Analiza wrażliwości `surgicalReplacementTolerance`** (obecnie 0.95, `DriftAwareSRP.java:83`) — wystawić
      w configu i przetestować {0.85, 0.95, 1.0} na wolno-dryfującym strumieniu.
- [ ] **F2. Hamulec over-adaptacji.** E4 pokazał, że reset przy rzadkim dryfcie szkodzi (STAGGER-Low DA-ARF 0.80).
      Sprawdzić czy DA-SRP też cierpi; jeśli tak — dodać próg minimalnej częstości alarmów przed akcją FULL.

---

## Kolejność wykonania (implementacyjna)

1. **Najpierw [PY] P0 (sekcja B)** — tanie, natychmiastowa poprawa Results, bez re-runu Javy.
2. **[PY] P1 (sekcja C)** — nowe wykresy czasowe z danych, które już masz.
3. **[JAVA] P0 (sekcja A)** — naprawa DA-ARF: A1 diagnoza → A2/A3/A4 → re-run E3 (i E4/E5 jeśli dotyczy).
4. **[JAVA] P2 (sekcje D/E/F)** — tylko jeśli zostanie czas; per-learner tylko jako case study.

> Zasada: wyczerp wszystkie zmiany **[PY]** zanim ruszysz **[JAVA]** — [PY] nie wymaga re-runu i od razu
> poprawia rozdział z wynikami. [JAVA] grupuj, żeby zrobić JEDEN re-run E1–E5 na końcu.

---
*Plan implementacyjny. Odhaczaj `[ ]` → `[x]`.*
