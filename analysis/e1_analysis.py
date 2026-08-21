from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, loaders, latex_tables, plot_utils, block_utils

BLOCK = "E1"

#: RAM-Hours are reported in units of 1e-6 GB-h — see config.RAMH_SCALE for why.
RAMH_SCALE = config.RAMH_SCALE


def table_baselines(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e1_baselines", pv,
        caption=r"E1 baselines: mean $\kappa$ per (dataset, variant). Bold = best per row.",
        label="tab:e1_baselines",
    )

    pv_acc = block_utils.metric_pivot(BLOCK, summary, "accuracy_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e1_accuracy", pv_acc,
        caption="E1 baselines: mean accuracy per (dataset, variant). Bold = best per row.",
        label="tab:e1_accuracy",
    )

    pv_temp = block_utils.metric_pivot(BLOCK, summary, "kappa_temporal_windowed_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e1_temporal_kappa", pv_temp,
        caption=r"E1 baselines: temporal $\kappa$ averaged over all evaluation windows, per (dataset, variant).",
        label="tab:e1_temporal_kappa",
    )


def table_resources(summary: pd.DataFrame):
    cols = ["ram_hours_gb_mean", "throughput_mean", "peak_mb_mean", "wall_ms_mean"]
    cols = [c for c in cols if c in summary.columns]
    if not cols:
        return
    variants = ["HT", "ARF", "SRP", "HT+S1", "ARF+S1", "SRP+S1"]
    base = summary[summary.variant.isin(variants)]
    agg = base.groupby(["dataset", "variant"], as_index=False)[cols].mean()

    rows = []
    for ds in block_utils.ordered_datasets(agg):
        row = {"Dataset": ds}
        for v in variants:
            d = agg[(agg.dataset == ds) & (agg.variant == v)]
            if len(d) == 0:
                row[f"{v}-RAMh"] = np.nan
                row[f"{v}-thr"] = np.nan
            else:
                # RAM-Hours are model-size based (fractions of a MB held for minutes), so the
                # raw figures sit around 1e-6 GB-h and round to 0.00 at table precision.
                # Report them in units of 1e-6 GB-h; the caption carries the multiplier.
                ram = d.ram_hours_gb_mean.iloc[0] if "ram_hours_gb_mean" in d else np.nan
                row[f"{v}-RAMh"] = ram * RAMH_SCALE
                thr = d.throughput_mean.iloc[0] if "throughput_mean" in d else np.nan
                # Pre-formatted: throughput needs no decimals, while the RAM columns need three
                # (a single ndigits for the whole table cannot serve both).
                row[f"{v}-thr"] = "-" if pd.isna(thr) else f"{thr:.0f}"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Dataset")
    body = latex_tables.df_to_booktabs(df, ndigits=3, index_name="Dataset")
    latex_tables.write_table(
        "tab_e1_resources", body,
        caption=(
            r"E1 resource usage: RAM-Hours (in units of " + config.RAMH_UNIT_TEX
            + r", columns \emph{-RAMh}) and throughput (instances/sec, columns \emph{-thr}) for raw and "
            r"S1 baseline models. RAM-Hours are measured on the deep size of the learner "
            r"itself (MOA \texttt{measureByteSize}), not on the JVM heap."
        ),
        label="tab:e1_resources",
    )


def plot_kappa_by_dataset(summary: pd.DataFrame):
    block_utils.plot_metric_bar(
        BLOCK, summary, "kappa_mean",
        fname="e1_kappa_by_dataset",
        ylabel=r"Cohen's $\kappa$ (mean)",
        title="E1: Baseline kappa per dataset and variant",
    )


def plot_accuracy_by_dataset(summary: pd.DataFrame):
    block_utils.plot_metric_bar(
        BLOCK, summary, "accuracy_mean",
        fname="e1_accuracy_by_dataset",
        ylabel="Accuracy (mean)",
        title="E1: Baseline accuracy per dataset and variant",
    )


def plot_temporal_kappa(summary: pd.DataFrame):
    block_utils.plot_metric_bar(
        BLOCK, summary, "kappa_temporal_windowed_mean",
        fname="e1_temporal_kappa",
        ylabel=r"Temporal $\kappa$ (mean)",
        title="E1: Temporal kappa per dataset and variant",
    )


def plot_windows_kappa(window: pd.DataFrame):
    block_utils.plot_window_timeseries(
        BLOCK, window, metric="kappa",
        drift_points_map=config.DRIFT_POINTS_E1E3,
        fname_prefix="e1_kappa_timeseries", subdir="e1_timeseries",
        ylabel=r"$\kappa$ (window)", title_prefix="E1",
    )


def plot_windows_accuracy(window: pd.DataFrame):
    block_utils.plot_window_timeseries(
        BLOCK, window, metric="accuracy",
        drift_points_map=config.DRIFT_POINTS_E1E3,
        fname_prefix="e1_accuracy_timeseries", subdir="e1_timeseries",
        ylabel="Accuracy (window)", title_prefix="E1",
    )


def plot_drift_alarms(drift_alarms: pd.DataFrame):
    block_utils.plot_alarm_counts(
        BLOCK, drift_alarms,
        fname="e1_drift_alarm_counts",
        title="E1: Mean drift alarms per run",
    )
    block_utils.plot_alarm_timeline(
        BLOCK, drift_alarms,
        fname_prefix="e1_alarms", subdir="e1_alarms",
        drift_points_map=config.DRIFT_POINTS_E1E3,
        title_prefix="E1",
    )


def plot_recovery(recovery: pd.DataFrame):
    block_utils.plot_recovery(
        BLOCK, recovery,
        fname="e1_recovery_depth",
        title="E1: Recovery depth per (dataset, variant)",
    )
    agg = block_utils.recovery_table(BLOCK, recovery)
    if not agg.empty and "mean_max_drop" in agg.columns:
        pv = agg.pivot(index="dataset", columns="variant", values="mean_max_drop")
        rows = block_utils.ordered_datasets(agg)
        cols = block_utils.ordered_variants(BLOCK, agg)
        pv = pv.reindex(index=rows, columns=cols)
        block_utils.write_metric_table(
            "tab_e1_recovery", pv,
            caption="E1 mean max accuracy drop after drift per (dataset, variant).",
            label="tab:e1_recovery",
            ndigits=3, bold_max_per_row=False,
        )


def plot_feature_selection(feat_sel: pd.DataFrame):
    agg = block_utils.feature_selection_summary(feat_sel)
    if agg.empty:
        return
    plot_utils.setup_style()
    agg["dataset"] = pd.Categorical(
        agg.dataset, categories=block_utils.ordered_datasets(agg), ordered=True,
    )
    agg["variant"] = pd.Categorical(
        agg.variant, categories=block_utils.ordered_variants(BLOCK, agg), ordered=True,
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=agg, x="dataset", y="mean_selected_count", hue="variant",
                ax=ax, errorbar=None)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean selected feature count")
    ax.set_title("E1: selected feature count after warm-up")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plot_utils.short_legend(ax, ncol=2, title="Variant")
    fig.tight_layout()
    plot_utils.save_fig(fig, "e1_feature_selection_overview")


def plot_feature_importance(feat_imp: pd.DataFrame):
    block_utils.plot_importance_heatmap(
        BLOCK, feat_imp,
        fname="e1_feature_importance_heatmap",
        title="E1: feature importance per dataset (1.0 = a uniform share of the total)",
    )
    block_utils.plot_importance_noise_annotated(
        BLOCK, feat_imp,
        fname_prefix="e1_importance_noise", subdir="e1_importance_noise",
        title_prefix="E1",
    )


def write_stat_tables(stat_tests: dict):
    block_utils.friedman_table(BLOCK, stat_tests)
    # Only the metrics the thesis actually cites; previously every STAT_METRICS entry got
    # its own avg_ranks table (6 per block = 30 unused files).
    for metric in config.RANK_TABLE_METRICS:
        block_utils.per_metric_rank_table(BLOCK, stat_tests, metric)
    block_utils.nemenyi_table(BLOCK, stat_tests, metric="kappa")
    block_utils.wilcoxon_table(BLOCK, stat_tests, metric="kappa")
    block_utils.export_cd_diagrams(BLOCK, stat_tests, subdir="e1_stat_tests")


def run():
    print("\n[E1] Baseline analysis")
    data = loaders.load_e1()
    summary = data["summary"]
    if summary is None or len(summary) == 0:
        print("  E1 summary missing, skipping")
        return
    table_baselines(summary)
    table_resources(summary)
    plot_kappa_by_dataset(summary)
    plot_accuracy_by_dataset(summary)
    plot_temporal_kappa(summary)
    plot_windows_kappa(data["windows"])
    plot_windows_accuracy(data["windows"])
    plot_drift_alarms(data["drift_alarms"])
    plot_recovery(data["recovery_time"])
    # B2: E1 selection is static (K=ceil(sqrt(d)) frozen at warm-up), so the
    # selected-count / stability overview carries no signal — intentionally skipped.
    plot_feature_importance(data["feature_importance"])
    write_stat_tables(data["stat_tests"])


if __name__ == "__main__":
    run()
