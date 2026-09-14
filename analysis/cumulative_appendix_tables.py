"""Whole-stream cumulative kappa/accuracy appendix tables (uwaga #71)."""
from __future__ import annotations
from . import loaders, block_utils

BLOCK_LABELS = {
    "E1": ("baseline", "sec:results-e1"),
    "E2": ("E2 adaptive-selection", "sec:results-e2"),
    "E3": ("E3 ablation", "sec:results-e3"),
    "E4": ("E4 Low/HiDyn", "sec:results-e4"),
    "E5": ("E5 detector-comparison", "sec:results-e5"),
}


def run():
    loaders_by_block = {
        "E1": loaders.load_e1, "E2": loaders.load_e2, "E3": loaders.load_e3,
        "E4": loaders.load_e4, "E5": loaders.load_e5,
    }
    for block, loader in loaders_by_block.items():
        data = loader()
        summary = data.get("summary")
        if summary is None or len(summary) == 0:
            print(f"  [cumulative-appendix] {block}: no summary, skipping")
            continue
        desc, sec_label = BLOCK_LABELS[block]

        pv_k = block_utils.metric_pivot(block, summary, "kappa_cumulative_mean",
                                         index="dataset", columns="variant")
        block_utils.write_metric_table(
            f"tab_{block.lower()}_cumulative_kappa", pv_k,
            caption=(rf"{block} {desc}: whole-stream cumulative Cohen's $\kappa$ per "
                     r"(dataset, variant), accumulated over every post-warmup instance with no "
                     r"eviction (as opposed to the final-window $\kappa$ used as the primary "
                     rf"metric in Section~\ref{{{sec_label}}}). Bold = best per row."),
            label=f"tab:{block.lower()}_cumulative_kappa",
        )

        pv_a = block_utils.metric_pivot(block, summary, "accuracy_cumulative_mean",
                                         index="dataset", columns="variant")
        block_utils.write_metric_table(
            f"tab_{block.lower()}_cumulative_accuracy", pv_a,
            caption=(rf"{block} {desc}: whole-stream cumulative accuracy per "
                     r"(dataset, variant), accumulated over every post-warmup instance with no "
                     r"eviction. Bold = best per row."),
            label=f"tab:{block.lower()}_cumulative_accuracy",
        )
        print(f"  [cumulative-appendix] {block}: wrote cumulative kappa/accuracy tables")


if __name__ == "__main__":
    run()
