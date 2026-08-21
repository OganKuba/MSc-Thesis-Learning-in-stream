# Inwentarz wyników analizy

Spis wszystkich tabel (`stream/results/tables/*.tex`) i rysunków
(`stream/results/figures/**/*.pdf` lub `.svg`) generowanych przez pipeline
`python -m analysis`. Plik zawiera **wyłącznie opis czego dotyczy dany
artefakt** – bez interpretacji wyników – żeby można było na jego podstawie
zbudować prompt do napisania rozdziału „Wyniki" pracy.

## Konwencje wspólne dla wszystkich bloków (E1–E5)

- Każdy blok ma swój katalog `stream/results/E{1..5}/` z surowymi CSV oraz
  podkatalog `stat_tests/` z gotowymi testami statystycznymi.
- Metryki agregowane przez runner (sufix `_mean` / `_std` w summary CSV):
  `accuracy`, `kappa`, `kappa_per`, `temporal_kappa`, `recovery_time`,
  `ram_hours_gb`, `throughput`, `peak_mb`, `drift_count`, `drift_alarms`,
  `feature_stability`, `selection_changes`, `mean_selected_feature_count`,
  oraz liczniki akcji DA-SRP / DA-ARF: `da_kept`, `da_surgical`, `da_full`,
  `da_no_replacement`, `ext_keep_count`, `ext_full_count`.
- Każdy zestaw `stat_tests/` zawiera Friedmana + Nemenyi'ego + Wilcoxona oraz
  CD diagramy dla 6 metryk: `accuracy`, `kappa`, `kappa_per`,
  `temporal_kappa`, `recovery_time`, `ram_hours_gb`.
- Format eksportu: tabele → LaTeX `.tex` (booktabs), wykresy → `.pdf` + `.png`,
  CD diagramy → `.svg` (kopiowane z `stat_tests/`).

Skróty wariantów występujące w plikach:
- **S1** – statyczna selekcja cech (baseline),
- **S2** – adaptacyjna selekcja sterowana detektorem driftu,
- **S3** – adaptacyjna selekcja okresowa,
- **S4** – adaptacyjna selekcja sterowana ważnościami cech,
- **DA-ARF-A / -AB / -ABC**, **DA-SRP-A / -AB / -ABC** – progresywne
  warianty drift-aware ARF / SRP (A = wykrycie, AB = + chirurgiczna wymiana,
  ABC = + pełna wymiana / waga uczących),
- **Majority / NoChange** – klasyfikatory referencyjne (Level-1 validation).

---

## E1 – Baselines (HT / ARF / SRP + Majority / NoChange) na 8 zbiorach

Datasety: `SEA`, `STAGGER`, `Hyperplane`, `RandomRBF`, `FeatureDrift`,
`YahooFinance`, `NYCTaxi`, `NHTS`.
Warianty: `HT+S1`, `ARF+S1`, `SRP+S1`, `Majority`, `NoChange`.

### Tabele (`stream/results/tables/`)

| Plik | Co zawiera |
|---|---|
| `tab_e1_baselines.tex` | Macierz **mean κ** (dataset × wariant); bold = best per dataset. |
| `tab_e1_accuracy.tex` | Macierz **mean accuracy** (dataset × wariant); bold = best per row. |
| `tab_e1_temporal_kappa.tex` | Macierz **mean temporal κ** (dataset × wariant). |
| `tab_e1_resources.tex` | RAM-Hours (w jednostkach 10⁻⁶ GB·h — patrz `config.RAMH_SCALE`) i throughput (instances/sec) dla baseline'ów `HT+S1`/`ARF+S1`/`SRP+S1` per dataset. |
| `tab_e1_recovery.tex` | Średnia długość recovery (liczba okien) per (dataset × wariant). |
| `tab_e1_friedman.tex` | Wynik testu Friedmana (χ², p-value, liczba algorytmów/datasetów, istotność) dla 6 metryk. |
| `tab_e1_avg_ranks_{accuracy,kappa,kappa_per,temporal_kappa,recovery_time,ram_hours_gb}.tex` | Średnie rangi metod dla danej metryki, posortowane rosnąco. |
| `tab_e1_nemenyi_kappa.tex` | Tabela Nemenyi'ego po kappie (rank_diff, critical_difference, significant). |
| `tab_e1_wilcoxon_kappa.tex` | Pary Wilcoxona po kappie (n, statystyka, p, p_adjusted (Holm), istotność, effect size). |

### Rysunki (`stream/results/figures/`)

| Plik | Typ wykresu | Co przedstawia |
|---|---|---|
| `e1_kappa_by_dataset.pdf` | grouped bar | mean κ na dataset z grupowaniem po wariancie. |
| `e1_accuracy_by_dataset.pdf` | grouped bar | mean accuracy na dataset z grupowaniem po wariancie. |
| `e1_temporal_kappa.pdf` | grouped bar | mean temporal κ na dataset z grupowaniem po wariancie. |
| `e1_drift_alarm_counts.pdf` | grouped bar | średnia liczba alarmów driftu per (dataset, wariant). |
| `e1_recovery_length.pdf` | grouped bar | średnia długość recovery per (dataset, wariant). |
| `e1_feature_selection_overview.pdf` | 2 panele bar | (a) mean liczba wybranych cech, (b) mean stability ratio – per (dataset, wariant). |
| `e1_feature_importance_heatmap.pdf` | heatmap | ważność cech per (dataset, indeks cechy), w jednostkach udziału równomiernego (wartość × d, więc 1.0 = udział 1/d); czarna ramka = wstrzyknięte cechy szumowe; puste komórki = zbiór nie ma takiej cechy. Oś wariantów celowo usunięta — estymator ważności nie zależy od modelu. |
| `e1_timeseries/e1_kappa_timeseries_<DS>.pdf` (8 datasetów) | line plot | κ (po oknach) w funkcji `end_instance` per wariant; pionowe linie = ground-truth drifty. |
| `e1_timeseries/e1_accuracy_timeseries_<DS>.pdf` (8 datasetów) | line plot | accuracy po oknach w funkcji `end_instance` per wariant. |
| `e1_alarms/e1_alarms_<DS>.pdf` (8 datasetów) | timeline | sub-plot na wariant: czerwone linie = alarmy detektora, czarne kreskowane = GT drift. |
| `e1_stat_tests/e1_cd_{accuracy,kappa,kappa_per,temporal_kappa,recovery_time,ram_hours_gb}.svg` | CD diagram | krytyczna różnica + grupy metod nieróżniących się istotnie, na metryce. |

---

## E2 – Adaptacyjna selekcja cech (S1 vs S2/S3/S4)

Datasety: `SEA`, `STAGGER`, `Hyperplane`, `RandomRBF`, `FeatureDrift`.
Warianty (ARF i SRP): `+S1`, `+S2`, `+S3`, `+S4`.

### Tabele

| Plik | Co zawiera |
|---|---|
| `tab_e2_kappa.tex` | Macierz **mean κ** (wariant × dataset); bold = best per row. |
| `tab_e2_accuracy.tex` | Macierz **mean accuracy** (wariant × dataset). |
| `tab_e2_temporal_kappa.tex` | Macierz **mean temporal κ** (wariant × dataset). |
| `tab_e2_delta_vs_raw.tex` | Δκ wariantu względem **modelu bez selekcji tej samej rodziny**: `ARF+S*` wobec surowego ARF, `SRP+S*` wobec surowego SRP, per dataset. Wiersze referencyjne pominięte (byłyby zerami). Δ wobec S1 = różnica wiersza i wiersza S1 tego samego modelu. |
| `tab_e2_stability.tex` | `feature_stability_mean` (Jaccard kolejnych selekcji) per (wariant × dataset). |
| `tab_e2_drift_response.tex` | Per wariant: średnia liczba alarmów, liczba zmian selekcji, średnia wielkość selekcji, stability. |
| `tab_e2_friedman.tex`, `tab_e2_avg_ranks_*.tex` (6 metryk), `tab_e2_nemenyi_kappa.tex`, `tab_e2_wilcoxon_kappa.tex` | Pakiet testów statystycznych jak w E1. |

### Rysunki

| Plik | Typ | Co przedstawia |
|---|---|---|
| `e2_kappa_heatmap.pdf` | heatmap | κ dla pełnej macierzy wariant × dataset z anotacjami. |
| `e2_adaptive_vs_static.pdf` | scatter z łączącymi liniami | per dataset: najlepsze S1 vs najlepsze S2/S3/S4 (κ). |
| `e2_feature_selection_overview.pdf` | 2 panele bar | (a) mean liczba wybranych cech, (b) mean stability ratio. |
| `e2_feature_importance_heatmap.pdf` | heatmap | jak `e1_feature_importance_heatmap`: dataset × indeks cechy, jednostki udziału równomiernego, ramka na cechach szumowych. |
| `e2_timelines/e2_selection_timeline_<DS>.pdf` (5 datasetów) | scatter po czasie | dla 1 ziarna i 4 wariantów adaptacyjnych: indeksy aktualnie wybranych cech w funkcji instancji. |
| `e2_drift_alarm_counts.pdf` | grouped bar | mean liczba alarmów per (dataset, wariant). |
| `e2_alarms/e2_alarms_<DS>.pdf` (5 datasetów) | timeline | sub-plot na wariant: alarmy detektora vs GT drift. |
| `e2_timeseries/e2_kappa_timeseries_<DS>.pdf` (5 datasetów) | line plot | κ po oknach per wariant (top-6 w danym datasecie). |
| `e2_recovery_length.pdf` | grouped bar | średnia długość recovery per (dataset, wariant). |
| `e2_stat_tests/e2_cd_<metric>.svg` (6 metryk) | CD diagram | jak w E1. |

---

## E3 – Ablation DA-SRP / DA-ARF

Datasety: `SEA`, `STAGGER`, `Hyperplane`, `RandomRBF`, `FeatureDrift`,
`YahooFinance`, `NYCTaxi`, `NHTS` (8).
Warianty: `SRP+S1_baseline`, `ARF+S2_baseline`, `DA-ARF-A/-AB/-ABC`,
`DA-SRP-A/-AB/-ABC`.

### Tabele

| Plik | Co zawiera |
|---|---|
| `tab_e3_ablation.tex` | Macierz **mean κ** (dataset × wariant) z **bold = best** per row. Pod tabelą **mean Δκ** każdego wariantu względem `SRP+S1_baseline`. |
| `tab_e3_accuracy.tex` | Macierz **mean accuracy** (dataset × wariant). |
| `tab_e3_temporal_kappa.tex` | Macierz **mean temporal κ** (dataset × wariant). |
| `tab_e3_adaptation_actions.tex` | Sumy zdarzeń DA per (dataset × wariant): KEEP / SURGICAL / FULL / NO\_REPL / EXT\_KEEP / EXT\_FULL. |
| `tab_e3_friedman.tex`, `tab_e3_avg_ranks_*.tex` (6 metryk) | Friedman + średnie rangi. |
| `tab_e3_nemenyi_{kappa,temporal_kappa,recovery_time}.tex`, `tab_e3_wilcoxon_{kappa,temporal_kappa,recovery_time}.tex` | Testy post-hoc dla 3 kluczowych metryk. |

### Rysunki

| Plik | Typ | Co przedstawia |
|---|---|---|
| `e3_ablation_bar.pdf` | grouped bar | mean κ per (dataset, wariant). |
| `e3_temporal_kappa_bar.pdf` | grouped bar | mean temporal κ per (dataset, wariant). |
| `e3_adaptation_actions.pdf` | stacked bar (panel per wariant) | proporcje akcji KEEP/SURGICAL/FULL/NO\_REPL/EXT\_KEEP/EXT\_FULL na dataset. |
| `e3_adaptation_event_counts.pdf` | grouped bar | całkowita liczba zdarzeń adaptacji per (dataset, wariant), wszystkie ziarna. |
| `e3_extra/e3_importance_evolution_<DS>.pdf` (3 datasety: FeatureDrift, NHTS, Hyperplane) | line plot | dla najbogatszego DA-wariantu i ziarna 1: ważność top-5 cech w funkcji instancji + pionowe linie driftu. |
| `e3_timeseries/e3_kappa_timeseries_<DS>.pdf` (8 datasetów) | line plot | κ po oknach per wariant. |
| `e3_action_vs_overlap.pdf` | stacked bar (panel per wariant) | rozkład akcji learnera (KEEP/SURGICAL/FULL) wobec liczby dryfujących cech w jego podprzestrzeni. **Wymaga kolumn `per_learner_*` — patrz niżej.** |
| `e3_timelines/e3_learner_lanes_<DS>.pdf` | raster 2-panelowy | pas na learnera: lewy panel = akcja, prawy = liczba dryfujących cech w podprzestrzeni tego learnera (kolorowany overlap). **Wymaga kolumn `per_learner_*`.** |
| `e3_recovery_length.pdf` | grouped bar | mean recovery length per (dataset, wariant). |
| `e3_drift_alarm_counts.pdf` | grouped bar | mean liczba alarmów per (dataset, wariant). |
| `e3_stat_tests/e3_cd_<metric>.svg` (6 metryk) | CD diagram | jak w E1. |

---

## E4 – High-dynamics (Low vs HiDyn) na SEA i STAGGER

Datasety: `SEA-Low`, `SEA-HiDyn`, `STAGGER-Low`, `STAGGER-HiDyn`.
Warianty: `ARF+S1`, `SRP+S1`, `DA-ARF-ABC`, `DA-SRP-ABC`.

### Tabele

| Plik | Co zawiera |
|---|---|
| `tab_e4_kappa.tex` | Macierz **mean κ** (dataset × wariant); bold = best per row. |
| `tab_e4_accuracy.tex` | Macierz **mean accuracy** (dataset × wariant). |
| `tab_e4_temporal_kappa.tex` | Macierz **mean temporal κ** (dataset × wariant). |
| `tab_e4_dynamics_sensitivity.tex` | mean κ per wariant w podziale na `Low` vs `HiDyn` (uśrednione po generatorze). |
| `tab_e4_adaptation_actions.tex` | Sumy zdarzeń DA per (dataset × wariant). |
| `tab_e4_friedman.tex`, `tab_e4_avg_ranks_*.tex` (6 metryk) | Friedman + średnie rangi. |
| `tab_e4_nemenyi_{kappa,temporal_kappa,recovery_time}.tex`, `tab_e4_wilcoxon_{kappa,temporal_kappa,recovery_time}.tex` | Post-hoc. |

### Rysunki

| Plik | Typ | Co przedstawia |
|---|---|---|
| `e4_kappa_by_dynamics.pdf` | panele bar (panel per generator) | mean κ na osi `Low`/`HiDyn`, grupowanie po wariancie. |
| `e4_timeseries/e4_kappa_timeseries_<DS>.pdf` (4 datasety) | line plot | κ po oknach per wariant z pionowymi liniami driftu. |
| `e4_timeseries/e4_accuracy_timeseries_<DS>.pdf` (4 datasety) | line plot | accuracy po oknach per wariant. |
| `e4_drift_alarm_counts.pdf` | grouped bar | mean liczba alarmów per (dataset, wariant). |
| `e4_alarms/e4_alarms_<DS>.pdf` (4 datasety) | timeline | sub-plot na wariant: alarmy vs GT drift. |
| `e4_recovery_length.pdf` | grouped bar | mean recovery length per (dataset, wariant). |
| `e4_adaptation_actions.pdf` | stacked bar (panel per wariant) | proporcje akcji DA na dataset. |
| `e4_stat_tests/e4_cd_<metric>.svg` (6 metryk) | CD diagram | jak w E1. |

---

## E5 – Porównanie detektorów driftu (w ramach DA-ARF)

Datasety: `SEA-HiDyn`, `STAGGER-HiDyn`, `Hyperplane`, `RandomRBF`.
Warianty: `ARF+ADWIN`, `SRP+ADWIN`,
`DA-ARF+ADWIN`, `DA-ARF+HDDM_A`, `DA-ARF+HDDM_W`, `DA-ARF+KSWIN`.

### Tabele

| Plik | Co zawiera |
|---|---|
| `tab_e5_kappa.tex` | Macierz **mean κ** (wariant × dataset); bold = best per row. |
| `tab_e5_accuracy.tex` | Macierz **mean accuracy** (wariant × dataset). |
| `tab_e5_temporal_kappa.tex` | Macierz **mean temporal κ** (wariant × dataset). |
| `tab_e5_detector_ranking.tex` | Ranking 4 detektorów (DA-ARF): mean κ, accuracy, temporal κ, recovery, mean liczba alarmów. |
| `tab_e5_alarm_effectiveness.tex` | Per (wariant, detektor): liczba alarmów, mean/median Δaccuracy w oknie po alarmie, % alarmów użytecznych (≥ 1 pp) i szkodliwych (≤ −1 pp). |
| `tab_e5_friedman.tex`, `tab_e5_avg_ranks_*.tex` (6 metryk) | Friedman + średnie rangi. |
| `tab_e5_nemenyi_{kappa,temporal_kappa,recovery_time}.tex`, `tab_e5_wilcoxon_{kappa,temporal_kappa,recovery_time}.tex` | Post-hoc. |

### Rysunki

| Plik | Typ | Co przedstawia |
|---|---|---|
| `e5_kappa_heatmap.pdf` | heatmap | mean κ dla wariant × dataset z anotacjami. |
| `e5_alarms_vs_kappa.pdf` | scatter | mean liczba alarmów (X) vs mean κ (Y) na wariant; lewa-góra = „cichy i dokładny". |
| `e5_recovery_vs_kappa.pdf` | scatter | mean recovery (X) vs mean κ (Y) na wariant; prawa-dół = wolne recovery przy dobrej κ. |
| `e5_timeseries/e5_kappa_timeseries_<DS>.pdf` (Hyperplane, SEA-HiDyn) | line plot | κ po oknach per wariant z liniami driftu. |
| `e5_drift_alarm_counts.pdf` | grouped bar | mean liczba alarmów per (dataset, wariant). |
| `e5_alarm_effectiveness.pdf` | boxplot + strip | rozkład Δaccuracy na alarm per wariant; etykieta = % alarmów dających ≥ 1 pp. |
| `e5_timelines/e5_alarm_effect_timeline_<DS>.pdf` (Hyperplane, SEA-HiDyn) | timeline | sub-plot na wariant: każdy alarm jako słupek o wysokości = odzyskana accuracy; kropka = moment alarmu, linia przerywana = GT drift. |
| `e5_timelines/e5_adaptation_timeline_<DS>.pdf` (Hyperplane, SEA-HiDyn) | raster | akcje adaptacji w osi instancji, pas na typ akcji. |
| `e5_recovery_length.pdf` | grouped bar | mean recovery length per (dataset, wariant). |
| `e5_adaptation_actions.pdf` | stacked bar (panel per wariant) | proporcje akcji DA na dataset. |
| `e5_stat_tests/e5_cd_<metric>.svg` (6 metryk) | CD diagram | jak w E1. |

---

## Syntheza międzyeksperymentalna (`cross_experiment.py`)

Źródło: `stream/results/master_summary.csv` (z fallbackiem do konkatenacji
summary z bloków E1–E5).

### Tabele

| Plik | Co zawiera |
|---|---|
| `tab_cross_best_methods.tex` | Najlepszy wariant per dataset (po wszystkich blokach) + Δκ względem `HT+S1` z E1. |
| `tab_cross_resource_vs_kappa.tex` | Per wariant: mean κ, mean RAM-Hours (10⁻⁶ GB·h), mean throughput, liczba pomiarów (n). |

### Rysunki

| Plik | Typ | Co przedstawia |
|---|---|---|
| `cross_pareto_front.pdf` | scatter w skali log-X | mean κ (Y) vs mean RAM-Hours w 10⁻⁶ GB·h (X, log) per wariant; gwiazdki = Pareto-optimal, kreskowana linia = front Pareto. |
| `cross_synthetic_vs_real.pdf` | scatter z y=x | mean κ na zbiorach syntetycznych (X) vs mean κ na realnych (Y) per wariant. |
| `cross_rq_summary.pdf` | siatka 2×3 paneli | (RQ1) baselines E1, (RQ2) S1 vs S2/S3/S4 z E2, (RQ3) ablation z E3, (RQ4) κ vs dynamics z E4, (RQ5) ranking wariantów detektorów z E5, (Summary) zbiorczy bar mean κ po blokach. |

---

## Surowe CSV i pomocnicze pliki

| Plik / katalog | Co zawiera |
|---|---|
| `stream/results/master_summary.csv` | wszystkie wiersze summary z E1–E5 razem (klucz: block, dataset, variant…). |
| `stream/results/runs_raw.csv` | każde uruchomienie (per seed) – nieuśrednione, ze statusem `OK/FAIL`. |
| `stream/results/E{1..5}/windows.csv` | metryki per okno (`window_id`, `start_instance`, `end_instance`, accuracy/kappa/kappa_per/temporal_kappa, RAM, throughput, drift_count_in_window). |
| `stream/results/E{1..5}/drift_alarms.csv` | każde wzbudzenie detektora: `instance_index`, `global_alarm`, `drifting_features`, `num_drifting_features`, `error_at_alarm`, accuracy okna przed/po. |
| `stream/results/E{1..5}/recovery_time.csv` | per drift: `drift_instance`, `recovered_instance`, `recovery_length`, baseline accuracy przed driftem, threshold, `max_drop`, `area_under_recovery_curve`. |
| `stream/results/E{1..5}/feature_selections.csv` | każda zmiana selekcji: `trigger_type` (`initial`/`periodic`/`drift`), lista wybranych cech, count, zmienione cechy, Jaccard do poprzedniej selekcji, stability_ratio. |
| `stream/results/E{1..5}/feature_importance.csv` | snapshot ważności cech: `feature_index`, `importance`, `rank`, `is_selected`, `is_drifting`. |
| `stream/results/E{1..5}/adaptation_events.csv` | każde zdarzenie DA-SRP / DA-ARF: `event_type`, liczniki kept/surgical/full/no\_replacement/ext\_keep/ext\_full oraz — tylko DA-SRP — `per_learner_action` (K/S/F/N), `per_learner_overlap`, `per_learner_subspace`, kodowane `\|`, po jednym wpisie na składową zespołu. Kolumny per-learner pojawiają się dopiero w plikach z przebiegów po ich dodaniu; starsze CSV-ki ich nie mają, a generatory je wtedy pomijają. |
| `stream/results/E{1..5}/stat_tests/` | gotowe `friedman.csv`, `nemenyi.csv`, `wilcoxon.csv`, `ranks.csv`, `cd_diagram.csv`, oraz per metryka: `avg_ranks_<m>.csv`, `rank_matrix_<m>.csv`, `pairwise_significance_<m>.csv`, `cd_diagram_<m>.svg`/`.tex`, `warnings.txt`. |

## Mapowanie blok → pytanie badawcze

- **E1 / RQ1** – baseline'y standardowych modeli na 8 zbiorach (czy ma sens
  pchać się dalej?).
- **E2 / RQ2** – czy adaptacyjna selekcja cech (S2/S3/S4) bije statyczną (S1)?
- **E3 / RQ3** – ablacja drift-aware (komponenty A/AB/ABC) dla SRP i ARF.
- **E4 / RQ4** – jak wariant pełny (`DA-*-ABC`) reaguje na high-dynamics
  vs low-dynamics?
- **E5 / RQ5** – który detektor (ADWIN / HDDM\_A / HDDM\_W / KSWIN) najlepiej
  współpracuje z DA-ARF?
- **Cross** – Pareto κ-RAM, generalizacja synthetic ↔ real, podsumowanie RQ.
