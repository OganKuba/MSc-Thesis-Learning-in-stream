from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "stream" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

E1_DIR = RESULTS_DIR / "E1"
E2_DIR = RESULTS_DIR / "E2"
E3_DIR = RESULTS_DIR / "E3"
E4_DIR = RESULTS_DIR / "E4"
E5_DIR = RESULTS_DIR / "E5"

BLOCK_DIRS = {
    "E1": E1_DIR,
    "E2": E2_DIR,
    "E3": E3_DIR,
    "E4": E4_DIR,
    "E5": E5_DIR,
}

SUMMARY_FILES = {
    "E1": "E1_baselines.csv",
    "E2": "E2_adaptive.csv",
    "E3": "E3_ablation.csv",
    "E4": "E4_high_dynamics.csv",
    "E5": "E5_detectors.csv",
}

MASTER_SUMMARY_FILE = RESULTS_DIR / "master_summary.csv"
RUNS_RAW_FILE = RESULTS_DIR / "runs_raw.csv"

STAT_TESTS_SUBDIR = "stat_tests"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_DATASETS = {
    "SEA", "STAGGER", "Hyperplane", "RandomRBF",
    "FeatureDrift", "CustomFeatureDrift", "LED",
    "SEA-Low", "SEA-HiDyn",
    "FeatureDrift-Low", "FeatureDrift-HiDyn",
    "RandomRBF-Low", "RandomRBF-HiDyn",
    # STAGGER-HiDyn survives only in E5 (there k=8 still includes the +S1 variants, whose
    # collapse at K=ceil(sqrt(3))=2 is what makes the stream discriminate at all).
    "STAGGER-HiDyn",
}
REAL_DATASETS = {"YahooFinance", "NYCTaxi", "NHTS"}

DATASET_ORDER = [
    "SEA", "STAGGER", "Hyperplane", "RandomRBF", "FeatureDrift", "LED",
    "SEA-Low", "SEA-HiDyn",
    "FeatureDrift-Low", "FeatureDrift-HiDyn",
    "RandomRBF-Low", "RandomRBF-HiDyn",
    "STAGGER-HiDyn",
    "YahooFinance", "NYCTaxi", "NHTS",
]

GENERATOR_ORDER = ["SEA", "STAGGER", "Hyperplane", "RandomRBF", "CustomFeatureDrift"]
MODEL_ORDER = ["HT", "ARF", "SRP", "DA-ARF", "DA-SRP", "MAJORITY", "NOCHANGE"]
DETECTOR_ORDER = ["ADWIN", "HDDM_A", "HDDM_W", "KSWIN"]
DYNAMICS_ORDER = ["Low", "HiDyn"]

E1_VARIANT_ORDER = ["HT", "ARF", "SRP", "HT+S1", "ARF+S1", "SRP+S1", "Majority", "NoChange"]
E2_VARIANT_ORDER = [
    "ARF",
    "ARF+S1", "ARF+S2", "ARF+S3", "ARF+S4",
    "SRP",
    "SRP+S1", "SRP+S2", "SRP+S3", "SRP+S4",
]
E3_VARIANT_ORDER = [
    "SRP", "SRP+S1",
    "ARF", "ARF+S2",
    # DA-ARF-ABC = repaired default (tuned trees + 0.5 subspace). The untuned/narrow repair
    # ablation was a one-off diagnostic (numbers in THESIS_IMPROVEMENT_PLAN.md), not kept here.
    "DA-ARF-A", "DA-ARF-AB", "DA-ARF-ABC",
    # DA-SRP-* run on the reflection-free native ensemble (Option B).
    "DA-SRP-A", "DA-SRP-AB", "DA-SRP-ABC",
]
# k=4, not 6: Nemenyi CD scales with k(k+1) but only with sqrt(N), so dropping the two +S1
# variants (which belong to E2, the feature-selection block) buys more power than adding
# datasets would. CD goes 3.77 -> 1.91, i.e. below half the 1..4 rank scale.
E4_VARIANT_ORDER = ["ARF", "SRP", "DA-ARF-ABC", "DA-SRP-ABC"]
E5_VARIANT_ORDER = [
    "ARF+ADWIN", "SRP+ADWIN", "ARF+S1+ADWIN", "SRP+S1+ADWIN",
    "DA-ARF+ADWIN", "DA-ARF+HDDM_A", "DA-ARF+HDDM_W", "DA-ARF+KSWIN",
]

VARIANT_ORDER_BY_BLOCK = {
    "E1": E1_VARIANT_ORDER,
    "E2": E2_VARIANT_ORDER,
    "E3": E3_VARIANT_ORDER,
    "E4": E4_VARIANT_ORDER,
    "E5": E5_VARIANT_ORDER,
}

DRIFT_POINTS_E1E3 = {
    "SEA": [25000, 50000, 75000],
    "STAGGER": [20000, 40000, 60000],
    "Hyperplane": "continuous",
    "RandomRBF": "continuous",
    "FeatureDrift": [25000, 50000, 75000],
    "LED": "continuous",
}

# Recomputed from SyntheticStreamFactory.buildCyclicAbruptStream:
#   segments = num_drifts + 1, step = n // segments, drift k at k*step  (n = 100_000).
# The previous values were wrong for every entry — SEA-HiDyn has 10 drifts, not 3, and none of
# the marked positions matched where the generator actually switches concepts.
DRIFT_POINTS_E4E5 = {
    "SEA-Low": [25000, 50000, 75000],
    "SEA-HiDyn": [9090, 18180, 27270, 36360, 45450,
                  54540, 63630, 72720, 81810, 90900],
    "STAGGER-HiDyn": [9090, 18180, 27270, 36360, 45450,
                      54540, 63630, 72720, 81810, 90900],
    "FeatureDrift-Low": "continuous",
    "FeatureDrift-HiDyn": "continuous",
    "RandomRBF-Low": "continuous",
    "RandomRBF-HiDyn": "continuous",
    "Hyperplane": "continuous",
    "RandomRBF": "continuous",
}

# --- Saturated datasets (B3) --------------------------------------------
# On the low-dynamics blocks (E1/E2/E3) the generic STAGGER stream saturates:
# kappa == accuracy == 1.000 for every serious model, so those rows/bars are
# empty. We drop STAGGER from kappa/accuracy tables & plots for those blocks,
# but keep it for kappa_temporal (which does discriminate). STAGGER-Low was removed from E4
# entirely (all 6 variants scored exactly 1.0000, a rank row of 3.5s carrying zero information);
# STAGGER-HiDyn remains only in E5 and is left untouched there.
SATURATED_DATASETS_BY_BLOCK = {
    "E1": {"STAGGER"},
    "E2": {"STAGGER"},
    "E3": {"STAGGER"},
}
SATURATED_METRICS = {"kappa_mean", "accuracy_mean"}

# --- Noise-feature ground truth (B4) ------------------------------------
# Synthetic streams are augmented with `noise_features` random columns that
# are appended AFTER the signal features (see SyntheticStreamFactory.
# NoiseAugmentedStream.buildHeader), so the noise indices are always the LAST
# N feature columns. Keyed by dataset name; datasets not listed (real ARFFs and STAGGER) have
# no known noise ground truth. Values track master_experiments.json.
NOISE_FEATURES = {
    "SEA": 5,
    "Hyperplane": 5,
    "RandomRBF": 5,
    "FeatureDrift": 5,
    "CustomFeatureDrift": 5,
    # Every E4 stream now carries the same 5 noise features, so a Low/HiDyn contrast varies
    # drift intensity ONLY. Previously SEA-Low had 0 and SEA-HiDyn had 5, which confounded
    # drift frequency with feature-space size (3 vs 8 features, K=2 vs K=3).
    "SEA-Low": 5,
    "SEA-HiDyn": 5,
    "FeatureDrift-Low": 5,
    "FeatureDrift-HiDyn": 5,
    "RandomRBF-Low": 5,
    "RandomRBF-HiDyn": 5,
    "STAGGER": 0,
    "STAGGER-HiDyn": 0,
    # LED: 7 relevant segment attributes (idx 0-6) + 17 irrelevant (idx 7-23),
    # so the irrelevant ones are the last 17 — exactly the noise-annotation convention.
    "LED": 17,
}

# Metrics with pre-computed stat_tests/ outputs from UnifiedStreamExperimentRunner.
STAT_METRICS = [
    "accuracy",
    "kappa",
    # Temporal kappa averaged over all windows. Formerly listed twice, as "kappa_per" and
    # "temporal_kappa" — both were the same TemporalKappa metric, differing only in aggregation
    # (final window vs mean over windows), so each block emitted two redundant CD diagrams.
    "kappa_temporal",
    "recovery_time",
    # Depth of the post-alarm accuracy dip. Defined for every variant that saw an alarm, whereas
    # recovery_time is NaN whenever no episode ever closed — so this keeps the Friedman test on
    # the full set of datasets instead of the subset where something happened to recover.
    "recovery_max_drop",
    "ram_hours_gb",
]

# --- RAM-Hours reporting units -------------------------------------------
# RAM-Hours are integrated over the DEEP SIZE OF THE MODEL (thesis.models.ModelSize via MOA's
# sizeofag agent), not over the JVM heap. Learners here hold 0.03-0.26 MB and run for minutes,
# so the raw metric sits around 1e-6 GB-h and rounds to "0.00" at any sane table precision.
# Tables and the Pareto figure therefore report RAMh * RAMH_SCALE, with the unit in the caption.
RAMH_SCALE = 1e6
RAMH_UNIT_TEX = r"$10^{-6}$ GB$\cdot$h"

# --- Output budget -------------------------------------------------------
# The pipeline used to emit 413 artefacts while the thesis cited 39 figures and 40 tables.
# The three settings below cut that to roughly what is actually used, without removing any
# generator: switch a value back and the corresponding output returns.

# Formats written by plot_utils.save_fig. The thesis pulls only .pdf via \includegraphics, so
# writing a .png twin for every figure doubled the directory for nothing. Add "png" for slides.
FIGURE_FORMATS = ["pdf"]

# Datasets that get their OWN figure from the per-dataset generators (window timeseries, alarm
# timeline, adaptation timeline, selection timeline, noise-annotated importance). Those five
# functions loop over every dataset in the block and were responsible for ~120 of the unused
# figures; the aggregate views (e*_drift_alarm_counts, e*_feature_importance_heatmap) already
# cover the rest. Listed datasets mirror what main.tex cites. Empty list = aggregates only.
PER_DATASET_FIGURES = {
    "E1": ["SEA", "Hyperplane", "NHTS", "YahooFinance"],
    "E2": ["FeatureDrift", "Hyperplane"],
    "E3": ["FeatureDrift", "Hyperplane", "NYCTaxi", "YahooFinance"],
    # E4 lost STAGGER (saturated); FeatureDrift-* is the replacement contrast in the thesis.
    "E4": ["SEA-Low", "SEA-HiDyn", "FeatureDrift-Low", "FeatureDrift-HiDyn"],
    # E5 used to be empty, which left the detector block without a single temporal figure —
    # even though "WHEN does each detector fire" is exactly its research question. Two datasets
    # give the abrupt/continuous contrast: SEA-HiDyn (abrupt, frequent) vs Hyperplane
    # (continuous). RandomRBF duplicates Hyperplane's regime and STAGGER-HiDyn saturates.
    "E5": ["Hyperplane", "SEA-HiDyn"],
}

# Metrics that get an avg_ranks table. Previously every metric in STAT_METRICS did, i.e. 6 per
# block = 30 tables, none of which the thesis cited — it uses friedman + nemenyi_kappa +
# wilcoxon_kappa instead.
RANK_TABLE_METRICS = ["kappa"]

# Per-dataset generators switched off, per figure and per block. These produced one file per
# dataset per block and were mostly uncited: the accuracy timeseries duplicates the kappa one and
# the noise-annotated importance panel duplicates e*_feature_importance_heatmap.
#
# Value is either "all" (off everywhere) or a list of blocks it is off for. Use
# figure_disabled(name, block) rather than reading this directly.
DISABLED_FIGURES = {
    "accuracy_timeseries": "all",
    "importance_noise": "all",
    # The alarm timeline is redundant with the aggregate e*_drift_alarm_counts wherever the
    # question is only "how many alarms". In E5 the question is "when, and did the alarm buy
    # anything" — the timeline now colours each alarm by the accuracy it recovered, which no
    # aggregate view shows, so it stays on for that block alone.
    "alarm_timeline": ["E1", "E2", "E3", "E4"],
}


def figure_disabled(name: str, block: str | None = None) -> bool:
    """True when figure generator `name` is switched off (globally, or for this block)."""
    rule = DISABLED_FIGURES.get(name)
    if rule is None:
        return False
    if rule == "all":
        return True
    return block in rule

# CD diagrams exported to figures/. Kappa is the headline metric; recovery_max_drop is included
# because it is the only recovery metric that reaches significance (E2/E3/E4) now that
# recovery_time turned out to discriminate nowhere.
CD_DIAGRAM_METRICS = ["kappa", "recovery_max_drop"]

PALETTE = "colorblind"
SNS_STYLE = "whitegrid"
FONT_SCALE = 1.05
FIG_DPI = 300
FONT_FAMILY = "DejaVu Sans"

PLOT_RC = {
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "figure.titlesize": 13,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

ALPHA = 0.05
