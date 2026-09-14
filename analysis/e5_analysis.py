from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, loaders, latex_tables, plot_utils, block_utils

BLOCK = "E5"


def _ordered_detectors(df, col="detector"):
    if df is None or col not in df.columns:
        return []
    present = [d for d in config.DETECTOR_ORDER if d in df[col].unique()]
    extras = [d for d in df[col].unique() if d not in present]
    return present + extras


def table_detector_kappa(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="variant", columns="dataset")
    block_utils.write_metric_table(
        "tab_e5_kappa", pv,
        caption=(r"E5 detectors: mean $\kappa$ per (variant, dataset). "
                 r"Bold = best variant per dataset (column)."),
        label="tab:e5_kappa",
        bold_max_per_row=False, bold_max_per_col=True,
    )

    pv_acc = block_utils.metric_pivot(BLOCK, summary, "accuracy_mean", index="variant", columns="dataset")
    block_utils.write_metric_table(
        "tab_e5_accuracy", pv_acc,
        caption=("E5 detectors: mean accuracy per (variant, dataset). "
                 "Bold = best variant per dataset (column)."),
        label="tab:e5_accuracy",
        bold_max_per_row=False, bold_max_per_col=True,
    )

    pv_tk = block_utils.metric_pivot(BLOCK, summary, "kappa_temporal_windowed_mean", index="variant", columns="dataset")
    block_utils.write_metric_table(
        "tab_e5_temporal_kappa", pv_tk,
        caption=(r"E5 detectors: temporal $\kappa$ averaged over all evaluation windows, per "
                 r"(variant, dataset). Bold = best variant per dataset (column)."),
        label="tab:e5_temporal_kappa",
        bold_max_per_row=False, bold_max_per_col=True,
    )


def table_detector_ranking(summary: pd.DataFrame):
    """Aggregate across datasets/seeds per detector (using detector."""
    if summary is None or len(summary) == 0:
        return
    sub = summary[summary.model == "DA-ARF"] if "model" in summary.columns else summary
    if len(sub) == 0:
        sub = summary
    agg = sub.groupby("detector").agg(
        n_datasets=("dataset", "nunique"),
        mean_kappa=("kappa_mean", "mean"),
        mean_accuracy=("accuracy_mean", "mean"),
        mean_temporal_kappa=("kappa_temporal_windowed_mean", "mean"),
        mean_recovery=("recovery_time_mean", "mean"),
        mean_alarms=("drift_alarms_mean", "mean"),
    ).reset_index().sort_values("mean_kappa", ascending=False)
    agg.insert(0, "rank", range(1, len(agg) + 1))
    df = agg.set_index("rank")
    body = latex_tables.df_to_booktabs(df, ndigits=4, index_name="Rank")
    latex_tables.write_table(
        "tab_e5_detector_ranking", body,
        caption=r"E5 detector ranking (DA-ARF variants): mean $\kappa$, accuracy, temporal $\kappa$, recovery, alarms.",
        label="tab:e5_detector_ranking",
    )


def plot_kappa_heatmap(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="variant", columns="dataset")
    if pv.empty:
        return
    plot_utils.setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pv, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={"label": r"$\kappa$"})
    ax.set_title("E5: kappa heatmap (variants × datasets)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Variant")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    plot_utils.save_fig(fig, "e5_kappa_heatmap")


def plot_detector_alarms_vs_kappa(summary: pd.DataFrame):
    if summary is None or len(summary) == 0:
        return
    plot_utils.setup_style()
    agg = summary.groupby("variant").agg(
        kappa=("kappa_mean", "mean"),
        alarms=("drift_alarms_mean", "mean"),
        recovery=("recovery_time_mean", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette(config.PALETTE, n_colors=len(agg))
    for color, (_, r) in zip(palette, agg.iterrows()):
        ax.scatter(r.alarms, r.kappa, color=color, s=140, edgecolor="black", linewidth=0.5)
        ax.annotate(r.variant, (r.alarms, r.kappa), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean drift alarms per run")
    ax.set_ylabel(r"Mean $\kappa$")
    ax.set_title("E5: alarms vs kappa per variant (lower-right = noisy, upper-left = quiet & accurate)")
    fig.tight_layout()
    plot_utils.save_fig(fig, "e5_alarms_vs_kappa")


def plot_recovery_vs_kappa(summary: pd.DataFrame):
    if summary is None or len(summary) == 0 or "recovery_time_mean" not in summary.columns:
        return
    plot_utils.setup_style()
    agg = summary.groupby("variant").agg(
        kappa=("kappa_mean", "mean"),
        recovery=("recovery_time_mean", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette(config.PALETTE, n_colors=len(agg))
    for color, (_, r) in zip(palette, agg.iterrows()):
        ax.scatter(r.recovery, r.kappa, color=color, s=140, edgecolor="black", linewidth=0.5)
        ax.annotate(r.variant, (r.recovery, r.kappa), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean recovery time (windows)")
    ax.set_ylabel(r"Mean $\kappa$")
    ax.set_title("E5: recovery vs kappa per variant (lower-right = fast recovery & accurate)")
    fig.tight_layout()
    plot_utils.save_fig(fig, "e5_recovery_vs_kappa")


def plot_windows_kappa(window: pd.DataFrame):
    block_utils.plot_window_timeseries(
        BLOCK, window, metric="kappa",
        drift_points_map=config.DRIFT_POINTS_E4E5,
        fname_prefix="e5_kappa_timeseries", subdir="e5_timeseries",
        ylabel=r"$\kappa$ (window)", title_prefix="E5",
    )


def plot_drift_alarms(drift_alarms: pd.DataFrame):
    block_utils.plot_alarm_counts(
        BLOCK, drift_alarms,
        fname="e5_drift_alarm_counts",
        title="E5: Mean drift alarms per run",
    )
    block_utils.plot_alarm_timeline(
        BLOCK, drift_alarms,
        fname_prefix="e5_alarm_effect_timeline", subdir="e5_timelines",
        drift_points_map=config.DRIFT_POINTS_E4E5,
        title_prefix="E5",
    )
    block_utils.plot_alarm_effectiveness(
        BLOCK, drift_alarms,
        fname="e5_alarm_effectiveness",
        title=(r"E5: accuracy recovered per alarm (box = distribution, "
               r"label = share of alarms gaining $\geq$ 1pp)"),
    )
    block_utils.alarm_effectiveness_table(
        BLOCK, drift_alarms,
        name="tab_e5_alarm_effectiveness",
        caption=("E5 alarm effectiveness: change in window accuracy one window after each "
                 "detector alarm. \\emph{useful\\_pct} / \\emph{harmful\\_pct} are the shares "
                 "of alarms followed by a gain / loss of at least 1 percentage point."),
        label="tab:e5_alarm_effectiveness",
    )


def plot_recovery(recovery: pd.DataFrame):
    block_utils.plot_recovery(
        BLOCK, recovery,
        fname="e5_recovery_depth",
        title="E5: Recovery depth per (dataset, variant)",
    )


def plot_action_proportions(events: pd.DataFrame):
    block_utils.plot_adaptation_stacked(
        BLOCK, events,
        fname="e5_adaptation_actions",
        title="E5: adaptation action proportions per variant",
    )


def plot_adaptation_timeline(events: pd.DataFrame):
    block_utils.plot_adaptation_timeline(
        BLOCK, events,
        fname_prefix="e5_adaptation_timeline", subdir="e5_timelines",
        drift_points_map=config.DRIFT_POINTS_E4E5, title_prefix="E5",
    )


def write_stat_tables(stat_tests: dict):
    block_utils.friedman_table(BLOCK, stat_tests)
    for metric in config.RANK_TABLE_METRICS:
        block_utils.per_metric_rank_table(BLOCK, stat_tests, metric)
    for metric in ["kappa", "recovery_time", "kappa_temporal"]:
        block_utils.nemenyi_table(BLOCK, stat_tests, metric=metric)
        block_utils.wilcoxon_table(BLOCK, stat_tests, metric=metric)
    block_utils.export_cd_diagrams(BLOCK, stat_tests, subdir="e5_stat_tests")


def run():
    print("\n[E5] Detector comparison")
    data = loaders.load_e5()
    summary = data["summary"]
    if summary is None or len(summary) == 0:
        print("  E5 summary missing, skipping")
        return
    table_detector_kappa(summary)
    table_detector_ranking(summary)
    block_utils.resource_table(
        BLOCK, summary, "tab_e5_resources", "tab:e5_resources",
        "E5 detector comparison")
    plot_kappa_heatmap(summary)
    plot_detector_alarms_vs_kappa(summary)
    plot_recovery_vs_kappa(summary)
    plot_windows_kappa(data["windows"])
    plot_drift_alarms(data["drift_alarms"])
    plot_recovery(data["recovery_time"])
    plot_action_proportions(data["adaptation_events"])
    plot_adaptation_timeline(data["adaptation_events"])
    write_stat_tables(data["stat_tests"])


if __name__ == "__main__":
    run()
