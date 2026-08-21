FAZA 0 — Kręgosłup (przeczytaj NAJPIERW, pobieżnie)

Zanim wejdziesz w komponenty, zobacz szkielet, który je spina.

1. experiments/UnifiedStreamExperimentRunner.java → tylko metoda runPrequentialLoop() (~50 linii, ~570–615)

▎ To jest mapa-klucz: predict → detector.update → selector.update → model.train, instancja po instancji. Wszystko inne to komponent wołany stąd. Resztę tego pliku (1315 linii) zostaw na FAZĘ 8.

2. Kontrakty (interfejsy — małe, poznaj słownictwo):
- models/ModelWrapper.java, selection/FeatureSelector.java, selection/FilterRanker.java, detection/DriftDetector.java

  ---
FAZA 1 — Dyskretyzacja (ciągłe wartości → biny)

Cały „filter approach" stoi na tym, że cechy ciągłe są binowane, żeby liczyć IG.

3. discretization/Layer1Histogram.java → update() + shouldExpand() — fine 64-binowy histogram + klampowanie/rozszerzanie zakresu
4. discretization/Layer2Merger.java → merge() — nadzorowany agglomeratywny merge 64→8 binów (łączy sąsiednie o podobnym rozkładzie klas)
5. discretization/FeatureDiscretizer.java → update() + recomputeLayer2() — spina Layer-1+2 per cecha, warmup, softReset (event-driven forgetting)
6. discretization/PiDDiscretizer.java — cienki wrapper: tablica FeatureDiscretizerów

  ---
FAZA 2 — Ranking cech (która cecha coś mówi o klasie)

7. selection/AbstractFrequencyRanker.java → getFeatureScores() — tabela kontyngencji cecha×bin×klasa
8. selection/InformationGainRanker.java → computeScore() — IG z tabeli (jedyny używany ranker)

  ---
FAZA 3 — Selekcja cech S1–S4

9. selection/StaticFeatureSelector.java → initialize() — S1: raz na warmupie wybiera top-K=⌈√d⌉, potem zamrożone
10. selection/AlarmTriggeredSelector.java → update() — S2: re-rank po alarmie dryftu + softReset dryfujących cech
11. selection/PeriodicSelector.java — S3: re-rank co N instancji
12. selection/DriftAwareSelector.java — S4: S2+S3 połączone

▎ (NoFeatureSelection.java = ścieżka „NONE", model widzi wszystkie cechy.)
  
---
FAZA 4 — Detekcja dryftu (kiedy + gdzie)

13. detection/ADWINChangeDetector.java — Level-1 (globalny dryft na błędzie), wrapper na MOA ADWIN
14. detection/FeatureBuffers.java — rolling/reference/post bufory wartości cech
15. detection/PerFeatureKSWIN.java → testWindows() — Level-2: KS reference-vs-post per cecha + BH-FDR
16. detection/TwoLevelDriftDetector.java → update() — spina Level-1+2 (alarm → snapshot → zbierz post → lokalizuj cechy)

  ---
FAZA 5 — Hydraulika modeli (wspólne klocki)

17. models/FeatureSpace.java + FilteredHeaderBuilder.java — rzutowanie instancji na podprzestrzeń cech
18. models/FeatureImportance.java — wektor ważności (relevance × stability) sterujący DA
19. models/WeightedSubspaceSampler.java — losowanie podprzestrzeni ważone importance
20. models/DriftActionSummary.java — liczniki KEEP/SURGICAL/FULL

  ---
FAZA 6 — Modele bazowe (punkt odniesienia)

21. models/HoeffdingTreeWrapper.java → ARFWrapper.java → SRPWrapper.java — HT/ARF/SRP owinięte w ModelWrapper
22. models/MajorityClassWrapper.java + NoChangeWrapper.java — sanity baseline'y (κ musi je bić)

  ---
FAZA 7 — Autorskie metody (rdzeń pracy) ⭐

23. models/DAARFWrapper.java → train(), handleIntrinsicDrift(), predictProba() (710 l.) — DA-ARF: własny las, per-tree ADWIN + background, importance sampling, top-K voting, strojone drzewa + 0.5·d
24. models/NativeDriftAwareSRP.java → train(), handleDrift(), predictProba() (740 l.) — DA-SRP (natywny, bez refleksji): patche 60%, KEEP/SURGICAL/FULL po overlap+tau, correction-alpha voting

▎ POMIŃ models/DriftAwareSRP.java — stara wersja refleksyjna, nieużywana (zostaje tylko dla DriftEvent).

  ---

FAZA 8 — Powrót do orkiestratora (teraz wszystko się składa)

25. experiments/UnifiedStreamExperimentRunner.java — reszta: buildModel(), buildSelector(), buildDetector(), expandWork(), pisarze CSV. Teraz zrozumiesz każdą fabrykę.
26. experiments/RunDetailedRecorder.java — co dokładnie ląduje w windows/drift_alarms/adaptation_events/...csv
27. pipeline/SyntheticStreamFactory.java — generatory (SEA/Hyperplane/RandomRBF/FeatureDrift/LED + noise)

  ---
FAZA 9 — Ewaluacja i statystyka

28. evaluation/MetricsCollector.java — κ / temporal-κ / recovery / RAM-h per okno
29. evaluation/{CohenKappa,TemporalKappa,RecoveryTime}.java — pojedyncze metryki
30. evaluation/StatisticalTests.java + experiments/BlockStatisticalAnalysis.java — Friedman → Nemenyi → Wilcoxon+Holm

  ---
FAZA 10 — Pipeline analiz (Python, osobny świat)

  
---
⛔ POMIŃ (martwe/legacy — nie marnuj czasu)

- pipeline/ExperimentRunner.java + Shims.java — stary runner, Shims rzucają UnsupportedOperationException.
- models/DriftAwareSRP.java — refleksyjny DA-SRP, zastąpiony natywnym.
- models/DAARFDiversityProbe.java — jednorazowy probe do analizy różnorodności.

  ---

  ---
FAZA 10 — Pipeline analiz (Python, osobny świat)

31. analysis/loaders.py → analysis/config.py → analysis/e3_analysis.py (jako wzór) → cross_experiment.py. Wejście: python -m analysis.

  ---
⛔  POMIŃ (martwe/legacy — nie marnuj czasu)

- pipeline/ExperimentRunner.java + Shims.java — stary runner, Shims rzucają UnsupportedOperationException.
- models/DriftAwareSRP.java — refleksyjny DA-SRP, zastąpiony natywnym.

  ---
Zasada nawigacji: przy każdym pliku czytaj tylko wskazaną metodę + konstruktor, resztę skanuj. Najcięższe są 3 pliki (runner 1315, DA-ARF 710, DA-SRP 740) — do nich wracaj wielokrotnie, nie czytaj liniowo.

Chcesz, żebym zapisał to jako stream/READING_MAP.md (żebyś miał pod ręką), i/lub rozrysował diagram przepływu jednej instancji przez te komponenty?


- czzy muismy dyskretzyowac kazda ceche?
- unkown bin
- usuneice nie potzrbenych metod/ kodu nie uzywnaych