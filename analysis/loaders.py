from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from . import config


def _safe_read(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"missing file: {path}")
        return None

    try:
        df = pd.read_csv(path, comment="#", **kwargs)
    except pd.errors.EmptyDataError:
        warnings.warn(f"empty file: {path}")
        return None

    if df is None or len(df) == 0:
        warnings.warn(f"no rows: {path}")
        return None

    return df


def filter_ok(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None

    if "status" in df.columns:
        return df[df["status"].astype(str).str.upper() == "OK"].copy()

    return df.copy()


def parse_pipe(s) -> list[str]:
    if s is None or pd.isna(s) or str(s).strip() == "":
        return []

    return [x.strip() for x in str(s).split("|") if x.strip() != ""]


def parse_pipe_int(s) -> list[int]:
    """
    Parses strings like:
      1|2|3
      1.0|2.0|3.0
      1:0.099|2:0.087

    For feature-score pairs, only the feature id before ':' is returned.
    """
    values: list[int] = []

    for x in parse_pipe(s):
        if ":" in x:
            x = x.split(":", 1)[0]

        values.append(int(float(x)))

    return values


def parse_pipe_float(s) -> list[float]:
    values: list[float] = []

    for x in parse_pipe(s):
        if ":" in x:
            x = x.split(":", 1)[1]

        values.append(float(x))

    return values


# --- Per-block loaders ---------------------------------------------------

_PER_BLOCK_FILES = {
    "windows": "windows.csv",
    "drift_alarms": "drift_alarms.csv",
    "feature_selections": "feature_selections.csv",
    "feature_importance": "feature_importance.csv",
    "recovery_time": "recovery_time.csv",
    "adaptation_events": "adaptation_events.csv",
}


def load_block(block: str) -> dict:
    """Load every CSV produced by UnifiedStreamExperimentRunner for one block."""
    if block not in config.BLOCK_DIRS:
        raise ValueError(f"unknown block: {block}")

    base = config.BLOCK_DIRS[block]
    summary_name = config.SUMMARY_FILES[block]
    summary = _safe_read(base / summary_name)

    data = {"summary": summary}
    for key, fname in _PER_BLOCK_FILES.items():
        data[key] = _safe_read(base / fname)

    data["stat_tests"] = load_stat_tests(block)
    return data


def load_e1():
    return load_block("E1")


def load_e2():
    return load_block("E2")


def load_e3():
    return load_block("E3")


def load_e4():
    return load_block("E4")


def load_e5():
    return load_block("E5")


# --- Top-level files -----------------------------------------------------

def load_master_summary() -> pd.DataFrame | None:
    return _safe_read(config.MASTER_SUMMARY_FILE)


def load_runs_raw() -> pd.DataFrame | None:
    return filter_ok(_safe_read(config.RUNS_RAW_FILE))


# --- stat_tests/ loaders -------------------------------------------------

def load_stat_tests(block: str) -> dict:
    """Read the contents of <block>/stat_tests/. Returns a dict with per-metric tables."""
    base = config.BLOCK_DIRS[block] / config.STAT_TESTS_SUBDIR
    if not base.exists():
        warnings.warn(f"missing stat_tests dir: {base}")
        return {}

    out: dict = {
        "friedman": _safe_read(base / "friedman.csv"),
        "nemenyi": _safe_read(base / "nemenyi.csv"),
        "wilcoxon": _safe_read(base / "wilcoxon.csv"),
        "cd_diagram": _safe_read(base / "cd_diagram.csv"),
        "ranks": _safe_read(base / "ranks.csv"),
        "warnings": (base / "warnings.txt").read_text(encoding="utf-8")
        if (base / "warnings.txt").exists() else None,
    }

    out["per_metric"] = {}
    for metric in config.STAT_METRICS:
        out["per_metric"][metric] = {
            "avg_ranks": _safe_read(base / f"avg_ranks_{metric}.csv"),
            "rank_matrix": _safe_read(base / f"rank_matrix_{metric}.csv"),
            "pairwise": _safe_read(base / f"pairwise_significance_{metric}.csv"),
            "cd_diagram_svg": base / f"cd_diagram_{metric}.svg",
            "cd_diagram_tex": base / f"cd_diagram_{metric}.tex",
        }

    return out


# --- Inventory -----------------------------------------------------------

def data_inventory():
    print("=" * 72)
    print("DATA INVENTORY")
    print("=" * 72)

    blocks = [
        ("E1", load_e1()),
        ("E2", load_e2()),
        ("E3", load_e3()),
        ("E4", load_e4()),
        ("E5", load_e5()),
    ]

    for name, data in blocks:
        print(f"\n[{name}]")

        for key, value in data.items():
            if value is None:
                print(f"  {key}: MISSING")
            elif isinstance(value, pd.DataFrame):
                cols_preview = list(value.columns)[:6]
                suffix = "..." if len(value.columns) > 6 else ""
                print(f"  {key}: {len(value)} rows  cols={cols_preview}{suffix}")
            elif isinstance(value, dict):
                stat_keys = [k for k, v in value.items() if v is not None and not isinstance(v, dict)]
                print(f"  {key}: dict ({len(stat_keys)} non-null tables)")
            elif isinstance(value, str):
                print(f"  {key}: text  {len(value)} chars")
            else:
                print(f"  {key}: {type(value).__name__}")

    master = load_master_summary()
    raw = load_runs_raw()
    print()
    print(f"master_summary: {'MISSING' if master is None else f'{len(master)} rows'}")
    print(f"runs_raw:       {'MISSING' if raw is None else f'{len(raw)} rows'}")
    print("=" * 72)


if __name__ == "__main__":
    data_inventory()
