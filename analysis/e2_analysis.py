from __future__ import annotations
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, loaders, latex_tables, plot_utils, block_utils

BLOCK = "E2"


def table_comparison(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="variant", columns="dataset")
    block_utils.write_metric_table(
        "tab_e2_kappa", pv,
        caption=(r"E2 adaptive feature selection: mean $\kappa$ per (variant, dataset). "
                 r"Bold = best variant per dataset (column)."),
        label="tab:e2_kappa",
        bold_max_per_row=False, bold_max_per_col=True,
    )

    pv_acc = block_utils.metric_pivot(BLOCK, summary, "accuracy_mean", index="variant", columns="dataset")
    block_utils.write_metric_table(
        "tab_e2_accuracy", pv_acc,
        caption=("E2 adaptive feature selection: mean accuracy per (variant, dataset). "
                 "Bold = best variant per dataset (column)."),
        label="tab:e2_accuracy",
        bold_max_per_row=False, bold_max_per_col=True,
    )

    pv_tk = block_utils.metric_pivot(BLOCK, summary, "kappa_temporal_windowed_mean", index="variant", columns="dataset")
    block_utils.write_metric_table(
        "tab_e2_temporal_kappa", pv_tk,
        caption=(r"E2: temporal $\kappa$ averaged over all evaluation windows, per "
                 r"(variant, dataset). Bold = best variant per dataset (column)."),
        label="tab:e2_temporal_kappa",
        bold_max_per_row=False, bold_max_per_col=True,
    )

    if "model" not in summary.columns:
        return
    raw_lookup = (summary[summary.selector.isna() | (summary.selector == "NONE")]
                  .groupby(["dataset", "model"]).kappa_mean.mean().reset_index())
    if raw_lookup.empty:
        return
    rows = []
    for _, r in summary.iterrows():
        base = raw_lookup[(raw_lookup.dataset == r.dataset) & (raw_lookup.model == r.model)]
        if len(base) == 0:
            continue
        delta = r.kappa_mean - base.iloc[0].kappa_mean
        rows.append({"variant": r.variant, "dataset": r.dataset, "delta": delta})
    if not rows:
        return
    delta_df = pd.DataFrame(rows).pivot(index="variant", columns="dataset", values="delta")
    keep = [v for v in block_utils.ordered_variants(BLOCK, summary)
            if v in delta_df.index and not (delta_df.loc[v].abs() < 1e-12).all()]
    delta_df = delta_df.reindex(index=keep, columns=block_utils.ordered_datasets(summary))
    block_utils.write_metric_table(
        "tab_e2_delta_vs_raw", delta_df,
        caption=(r"E2 $\Delta\kappa$ of every feature-selection strategy against the "
                 r"\emph{same-model baseline without feature selection}: rows ARF+S$k$ are "
                 r"measured against raw ARF, rows SRP+S$k$ against raw SRP. E1 identified that "
                 r"baseline as the strongest reference. Negative means the selector costs more "
                 r"accuracy than it buys. The change relative to the static S1 strategy is the "
                 r"difference between a row and the S1 row of the same model."),
        label="tab:e2_delta_vs_raw",
        ndigits=3, bold_max_per_row=False,
    )


def table_stability(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "feature_stability_mean",
                                  index="variant", columns="dataset")
    block_utils.write_metric_table(
        "tab_e2_stability", pv,
        caption="E2 feature-set stability per (variant, dataset).",
        label="tab:e2_stability",
        ndigits=3, bold_max_per_row=False,
    )


def table_drift_response(summary: pd.DataFrame):
    rows = []
    for v in block_utils.ordered_variants(BLOCK, summary):
        sub = summary[summary.variant == v]
        rows.append({
            "Variant": v,
            "alarms": sub.drift_alarms_mean.mean() if "drift_alarms_mean" in sub else np.nan,
            "sel_changes": sub.selection_changes_mean.mean() if "selection_changes_mean" in sub else np.nan,
            "sel_count": sub.mean_selected_feature_count.mean() if "mean_selected_feature_count" in sub else np.nan,
            "stability": sub.feature_stability_mean.mean() if "feature_stability_mean" in sub else np.nan,
        })
    df = pd.DataFrame(rows).set_index("Variant")
    body = latex_tables.df_to_booktabs(df, ndigits=3, index_name="Variant")
    latex_tables.write_table(
        "tab_e2_drift_response", body,
        caption="E2 drift response per variant: mean drift alarms, selection changes, selected count, stability.",
        label="tab:e2_drift_response",
    )


def plot_kappa_heatmap(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="variant", columns="dataset")
    if pv.empty:
        return
    plot_utils.setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pv, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={"label": r"$\kappa$"})
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Variant")
    ax.set_title("E2: kappa heatmap (variants × datasets)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    plot_utils.save_fig(fig, "e2_kappa_heatmap")


def plot_adaptive_vs_static(summary: pd.DataFrame):
    """Best static vs best adaptive selection per dataset, against the."""
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="variant", columns="dataset")
    if pv.empty:
        return
    plot_utils.setup_style()
    s1_rows = [v for v in pv.index if v.endswith("S1")]
    adaptive_rows = [v for v in pv.index if re.search(r"\+S[234]$", v)]
    raw_rows = [v for v in pv.index if not re.search(r"\+S\d$", v)]
    if not s1_rows or not adaptive_rows:
        return
    datasets = list(pv.columns)
    x = np.arange(len(datasets))
    s1_best = pv.loc[s1_rows].max(axis=0)
    adaptive = pv.loc[adaptive_rows].max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    if raw_rows:
        raw_best = pv.loc[raw_rows].max(axis=0)
        ax.scatter(x, raw_best, color="C3", marker="_", s=420, linewidth=2.4, zorder=4,
                   label="no selection (raw ARF/SRP)")
        for xi, ds in enumerate(datasets):
            ax.plot([xi - 0.28, xi + 0.28], [raw_best[ds]] * 2, color="C3",
                    alpha=0.25, linewidth=1.0, zorder=1)
    ax.scatter(x, s1_best, color="grey", marker="o", s=80, zorder=3, label="best S1 (static)")
    ax.scatter(x, adaptive, color="C2", marker="^", s=90, zorder=3,
               label="best adaptive (S2/S3/S4)")
    for xi, ds in enumerate(datasets):
        ax.plot([xi, xi], [s1_best[ds], adaptive[ds]], color="black", alpha=0.3, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel(r"Best $\kappa$")
    ax.set_title("E2: no selection vs static (S1) vs adaptive (S2/S3/S4) — best-of per dataset")
    fig.tight_layout()
    plot_utils.legend_below(ax, ncol=3)
    plot_utils.save_fig(fig, "e2_adaptive_vs_static")


def plot_feature_selection(feat_sel: pd.DataFrame):
    block_utils.plot_feature_selection_overview(
        BLOCK, feat_sel,
        fname="e2_feature_selection_overview",
        title="E2: feature selection summary",
    )


def plot_feature_importance(feat_imp: pd.DataFrame):
    block_utils.plot_importance_heatmap(
        BLOCK, feat_imp,
        fname="e2_feature_importance_heatmap",
        title="E2: feature importance per dataset (1.0 = a uniform share of the total)",
    )
    block_utils.plot_importance_noise_annotated(
        BLOCK, feat_imp,
        fname_prefix="e2_importance_noise", subdir="e2_importance_noise",
        title_prefix="E2",
    )


def plot_selection_timeline(feat_sel: pd.DataFrame):
    """Per (variant, dataset, seed=1) plot the indices of selected."""
    if feat_sel is None or len(feat_sel) == 0:
        return
    plot_utils.setup_style()
    datasets = [d for d in block_utils.per_dataset_targets(BLOCK, feat_sel) if d in feat_sel.dataset.unique()]
    for ds in datasets:
        sub = feat_sel[feat_sel.dataset == ds]
        adaptive = [v for v in block_utils.ordered_variants(BLOCK, sub)
                    if pd.Series([v]).str.contains(r"\+S[234]$", regex=True).iloc[0]]
        if not adaptive:
            continue
        adaptive = adaptive[:4]
        fig, axes = plt.subplots(len(adaptive), 1, figsize=(11, 1.6 * len(adaptive) + 1.2),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]
        max_feat = 0
        seed_to_use = sorted(sub.seed.unique())[0] if "seed" in sub.columns else None
        for v, ax in zip(adaptive, axes):
            vd = sub[sub.variant == v]
            if seed_to_use is not None:
                vd = vd[vd.seed == seed_to_use]
            vd = vd.sort_values("instance_index")
            for _, r in vd.iterrows():
                feats = loaders.parse_pipe_int(r.selected_features)
                for f in feats:
                    ax.plot(r.instance_index, f, "s", color="C0", markersize=2)
                    max_feat = max(max_feat, f)
            ax.set_ylabel(f"{v}\nfeat idx", fontsize=8)
            ax.set_ylim(-0.5, max(max_feat + 0.5, 1))
        axes[-1].set_xlabel("Instance #")
        fig.suptitle(f"E2: Selection timeline (seed={seed_to_use}) — {ds}")
        fig.tight_layout()
        plot_utils.save_fig(fig, f"e2_selection_timeline_{ds}", subdir="e2_timelines")


def plot_drift_alarms(drift_alarms: pd.DataFrame):
    block_utils.plot_alarm_counts(
        BLOCK, drift_alarms,
        fname="e2_drift_alarm_counts",
        title="E2: Mean drift alarms per run",
    )
    block_utils.plot_alarm_timeline(
        BLOCK, drift_alarms,
        fname_prefix="e2_alarms", subdir="e2_alarms",
        drift_points_map=config.DRIFT_POINTS_E1E3,
        title_prefix="E2",
    )


def plot_windows_kappa(window: pd.DataFrame):
    block_utils.plot_window_timeseries(
        BLOCK, window, metric="kappa",
        drift_points_map=config.DRIFT_POINTS_E1E3,
        fname_prefix="e2_kappa_timeseries", subdir="e2_timeseries",
        ylabel=r"$\kappa$ (window)", title_prefix="E2",
        top_n=6,
    )


def plot_recovery(recovery: pd.DataFrame):
    block_utils.plot_recovery(
        BLOCK, recovery,
        fname="e2_recovery_depth",
        title="E2: Recovery depth per (dataset, variant)",
    )


def write_stat_tables(stat_tests: dict):
    block_utils.friedman_table(BLOCK, stat_tests)
    for metric in config.RANK_TABLE_METRICS:
        block_utils.per_metric_rank_table(BLOCK, stat_tests, metric)
    block_utils.nemenyi_table(BLOCK, stat_tests, metric="kappa")
    block_utils.wilcoxon_table(BLOCK, stat_tests, metric="kappa")
    block_utils.export_cd_diagrams(BLOCK, stat_tests, subdir="e2_stat_tests")


def run():
    print("\n[E2] Adaptive feature selection")
    data = loaders.load_e2()
    summary = data["summary"]
    if summary is None or len(summary) == 0:
        print("  E2 summary missing, skipping")
        return
    table_comparison(summary)
    table_stability(summary)
    table_drift_response(summary)
    block_utils.resource_table(
        BLOCK, summary, "tab_e2_resources", "tab:e2_resources",
        "E2 adaptive feature selection")
    plot_kappa_heatmap(summary)
    plot_adaptive_vs_static(summary)
    plot_feature_selection(data["feature_selections"])
    plot_feature_importance(data["feature_importance"])
    plot_selection_timeline(data["feature_selections"])
    plot_drift_alarms(data["drift_alarms"])
    plot_windows_kappa(data["windows"])
    plot_recovery(data["recovery_time"])
    write_stat_tables(data["stat_tests"])


if __name__ == "__main__":
    run()
