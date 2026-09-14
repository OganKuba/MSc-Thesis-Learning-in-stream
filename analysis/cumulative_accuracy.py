"""Whole-stream cumulative accuracy, reconstructed offline from."""
from __future__ import annotations
import pandas as pd
from pathlib import Path

from . import config

OUT_DIR = config.RESULTS_DIR / "cumulative_analysis"


def cumulative_accuracy_for_block(block: str) -> pd.DataFrame | None:
    windows_path = config.BLOCK_DIRS[block] / "windows.csv"
    if not windows_path.exists():
        print(f"  [cumulative] {block}: no windows.csv, skipping")
        return None
    df = pd.read_csv(windows_path)
    df["window_size"] = df["end_instance"] - df["start_instance"]
    df["weighted_correct"] = df["accuracy"] * df["window_size"]

    group_cols = ["block", "dataset", "variant", "model", "selector", "detector", "seed"]
    per_run = (
        df.groupby(group_cols, as_index=False)
        .agg(total_correct=("weighted_correct", "sum"), total_instances=("window_size", "sum"))
    )
    per_run["cumulative_accuracy"] = per_run["total_correct"] / per_run["total_instances"]

    agg_cols = ["block", "dataset", "variant", "model", "selector", "detector"]
    per_variant = (
        per_run.groupby(agg_cols, as_index=False)["cumulative_accuracy"]
        .agg(cumulative_accuracy_mean="mean", cumulative_accuracy_std="std", n_seeds="count")
    )
    return per_variant


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_blocks = []
    for block in ["E1", "E2", "E3", "E4", "E5"]:
        result = cumulative_accuracy_for_block(block)
        if result is None:
            continue
        out_path = OUT_DIR / f"{block}_cumulative_accuracy.csv"
        result.to_csv(out_path, index=False)
        print(f"  [cumulative] {block}: wrote {out_path.relative_to(config.ROOT)} "
              f"({len(result)} dataset/variant rows)")
        all_blocks.append(result)
    if all_blocks:
        combined = pd.concat(all_blocks, ignore_index=True)
        combined_path = OUT_DIR / "all_blocks_cumulative_accuracy.csv"
        combined.to_csv(combined_path, index=False)
        print(f"  [cumulative] wrote combined -> {combined_path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    run()
