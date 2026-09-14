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
    "DA-ARF-A", "DA-ARF-AB", "DA-ARF-ABC",
    # DA-SRP-* run on the reflection-free native ensemble (Option B).
    "DA-SRP-A", "DA-SRP-AB", "DA-SRP-ABC",
]
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
    "NHTS": [1041075, 1957756],
    "SEA": [25000, 50000, 75000],
    "STAGGER": [20000, 40000, 60000],
    "Hyperplane": "continuous",
    "RandomRBF": "continuous",
    "FeatureDrift": [25000, 50000, 75000],
    "LED": "continuous",
}

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

SATURATED_DATASETS_BY_BLOCK = {
    "E1": {"STAGGER"},
    "E2": {"STAGGER"},
    "E3": {"STAGGER"},
}
SATURATED_METRICS = {"kappa_mean", "accuracy_mean"}

NOISE_FEATURES = {
    "SEA": 5,
    "Hyperplane": 5,
    "RandomRBF": 5,
    "FeatureDrift": 5,
    "CustomFeatureDrift": 5,
    "SEA-Low": 5,
    "SEA-HiDyn": 5,
    "FeatureDrift-Low": 5,
    "FeatureDrift-HiDyn": 5,
    "RandomRBF-Low": 5,
    "RandomRBF-HiDyn": 5,
    "STAGGER": 0,
    "STAGGER-HiDyn": 0,
    "LED": 17,
}

# Metrics with pre-computed stat_tests/ outputs from
STAT_METRICS = [
    "accuracy",
    "kappa",
    "kappa_temporal",
    "recovery_time",
    "recovery_max_drop",
    "ram_hours_gb",
]

RAMH_SCALE = 1e6
RAMH_UNIT_TEX = r"$10^{-6}$ GB$\cdot$h"


FIGURE_FORMATS = ["pdf"]

PER_DATASET_FIGURES = {
    "E1": ["SEA", "Hyperplane", "NHTS", "YahooFinance"],
    "E2": ["FeatureDrift", "Hyperplane"],
    "E3": ["SEA", "FeatureDrift", "Hyperplane", "NYCTaxi", "YahooFinance"],
    "E4": ["SEA-Low", "SEA-HiDyn", "FeatureDrift-Low", "FeatureDrift-HiDyn"],
    "E5": ["Hyperplane", "SEA-HiDyn"],
}

RANK_TABLE_METRICS = ["kappa"]

DISABLED_FIGURES = {
    "accuracy_timeseries": "all",
    "importance_noise": "all",
    "alarm_timeline": ["E1", "E2", "E3", "E4"],
}


def figure_disabled(name: str, block: str | None = None) -> bool:
    """True when figure generator `name` is switched off (globally, or for."""
    rule = DISABLED_FIGURES.get(name)
    if rule is None:
        return False
    if rule == "all":
        return True
    return block in rule

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

ARFF_FILES = {
    "NYCTaxi": ROOT / "preproccessing" / "nyc_taxi" / "data" / "arff" / "nyc_taxi.arff",
    "NHTS": ROOT / "preproccessing" / "data" / "arff" / "nhts.arff",
    "YahooFinance": ROOT / "preproccessing" / "yahoo_finance" / "data" / "arff" / "yahoo_finance.arff",
}
