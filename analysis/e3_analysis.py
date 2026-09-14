from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, loaders, latex_tables, plot_utils, block_utils

BLOCK = "E3"
SRP_RAW_BASELINE = "SRP"
SRP_S1_BASELINE = "SRP+S1"
ARF_RAW_BASELINE = "ARF"
ARF_S2_BASELINE = "ARF+S2"


def table_ablation(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="dataset", columns="variant")
    if pv.empty:
        return

    body_lines = ["\\begin{tabular}{l" + "c" * len(pv.columns) + "}", "\\toprule"]
    body_lines.append(" & ".join(["Dataset"] + [latex_tables._escape(c) for c in pv.columns]) + " \\\\")
    body_lines.append("\\midrule")
    for ds, row in pv.iterrows():
        arr = np.array(row.values, dtype=float)
        max_idx = -1 if np.all(np.isnan(arr)) else int(np.nanargmax(arr))
        cells = [latex_tables._escape(ds)]
        for j, v in enumerate(row):
            txt = "-" if pd.isna(v) else f"{v:.3f}"
            if j == max_idx:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        body_lines.append(" & ".join(cells) + " \\\\")

    mean_row = pv.mean(axis=0, skipna=True)
    mean_arr = np.array(mean_row.values, dtype=float)
    mean_max_idx = -1 if np.all(np.isnan(mean_arr)) else int(np.nanargmax(mean_arr))
    mean_cells = ["Mean"]
    for j, v in enumerate(mean_row):
        txt = "-" if pd.isna(v) else f"{v:.3f}"
        if j == mean_max_idx:
            txt = r"\textbf{" + txt + "}"
        mean_cells.append(txt)
    body_lines.append("\\midrule")
    body_lines.append(" & ".join(mean_cells) + " \\\\")

    body_lines.append("\\bottomrule")
    body_lines.append("\\end{tabular}")
    latex_tables.write_table(
        "tab_e3_ablation", "\n".join(body_lines),
        caption=(r"E3 DA-SRP / DA-ARF ablation: mean $\kappa$ per (dataset, variant), with a "
                 r"Mean row averaging across the eight datasets. Bold = best per row."),
        label="tab:e3_ablation",
    )

    delta_rows = []
    def add_delta(label: str, col: str, ref_col: str):
        if col not in pv.columns or ref_col not in pv.columns:
            return
        diff = (pv[col] - pv[ref_col]).dropna()
        if diff.empty:
            return
        delta_rows.append({
            "Comparison": label,
            "mean": diff.mean(),
            "median": diff.median(),
            "std": diff.std(),
            "wins": f"{int((diff > 0).sum())}/{len(diff)}",
            "worst": diff.min(),
        })

    add_delta(r"SRP+S1 vs SRP", SRP_S1_BASELINE, SRP_RAW_BASELINE)
    add_delta(r"DA-SRP-A vs SRP", "DA-SRP-A", SRP_RAW_BASELINE)
    add_delta(r"DA-SRP-AB vs SRP", "DA-SRP-AB", SRP_RAW_BASELINE)
    add_delta(r"DA-SRP-ABC vs SRP", "DA-SRP-ABC", SRP_RAW_BASELINE)
    add_delta(r"DA-SRP-A vs SRP+S1", "DA-SRP-A", SRP_S1_BASELINE)
    add_delta(r"DA-SRP-AB vs SRP+S1", "DA-SRP-AB", SRP_S1_BASELINE)
    add_delta(r"DA-SRP-ABC vs SRP+S1", "DA-SRP-ABC", SRP_S1_BASELINE)
    add_delta(r"ARF+S2 vs ARF", ARF_S2_BASELINE, ARF_RAW_BASELINE)
    add_delta(r"DA-ARF-A vs ARF", "DA-ARF-A", ARF_RAW_BASELINE)
    add_delta(r"DA-ARF-AB vs ARF", "DA-ARF-AB", ARF_RAW_BASELINE)
    add_delta(r"DA-ARF-ABC vs ARF", "DA-ARF-ABC", ARF_RAW_BASELINE)
    add_delta(r"DA-ARF-A vs ARF+S2", "DA-ARF-A", ARF_S2_BASELINE)
    add_delta(r"DA-ARF-AB vs ARF+S2", "DA-ARF-AB", ARF_S2_BASELINE)
    add_delta(r"DA-ARF-ABC vs ARF+S2", "DA-ARF-ABC", ARF_S2_BASELINE)
    if delta_rows:
        deltas = pd.DataFrame(delta_rows).set_index("Comparison")
        body = latex_tables.df_to_booktabs(deltas, ndigits=3, index_name="Comparison")
        latex_tables.write_table(
            "tab_e3_ablation_deltas", body,
            caption=(r"E3 ablation summary: $\Delta\kappa$ against the raw and the "
                     r"feature-selection baseline, summarised over the "
                     + str(len(pv.index)) + r" datasets. \emph{wins} counts datasets with "
                     r"$\Delta\kappa>0$ and \emph{worst} is the largest loss on any single "
                     r"dataset. Read \emph{median} and \emph{wins} as the headline: "
                     r"$\Delta\kappa$ is not commensurable across streams of different "
                     r"difficulty, and with 8 datasets a single one can dominate the mean "
                     r"(SEA alone accounts for the whole positive mean of DA-SRP-AB vs SRP)."),
            label="tab:e3_ablation_deltas",
        )

    pv_acc = block_utils.metric_pivot(BLOCK, summary, "accuracy_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e3_accuracy", pv_acc,
        caption="E3 ablation: mean accuracy per (dataset, variant).",
        label="tab:e3_accuracy",
    )

    pv_tk = block_utils.metric_pivot(BLOCK, summary, "kappa_temporal_windowed_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e3_temporal_kappa", pv_tk,
        caption=r"E3 ablation: temporal $\kappa$ averaged over all evaluation windows, per (dataset, variant).",
        label="tab:e3_temporal_kappa",
    )


def table_adaptation_actions(events: pd.DataFrame):
    agg = block_utils.adaptation_summary(events)
    if agg.empty:
        return
    cats = [c for c in ["kept_count", "surgical_count",
                        "full_replacement_count", "no_replacement_count",
                        "ext_keep_count", "ext_full_count"] if c in agg.columns]
    if not cats:
        return
    df = agg.set_index(["dataset", "variant"])[cats].reset_index()
    df.columns = ["Dataset", "Variant"] + [c.replace("_count", "") for c in cats]
    df = df.set_index("Dataset")
    body = latex_tables.df_to_booktabs(df, ndigits=0, index_name="Dataset")
    latex_tables.write_table(
        "tab_e3_adaptation_actions", body,
        caption="E3 adaptation events per (dataset, variant): KEEP / SURGICAL / FULL / NO\\_REPL plus ext counters.",
        label="tab:e3_adaptation_actions",
    )


def plot_ablation_bar(summary: pd.DataFrame):
    block_utils.plot_metric_bar(
        BLOCK, summary, "kappa_mean",
        fname="e3_ablation_bar",
        ylabel=r"$\kappa$ (mean)",
        title="E3: ablation kappa per (dataset, variant)",
    )


def plot_temporal_kappa_bar(summary: pd.DataFrame):
    block_utils.plot_metric_bar(
        BLOCK, summary, "kappa_temporal_windowed_mean",
        fname="e3_temporal_kappa_bar",
        ylabel=r"Temporal $\kappa$ (mean)",
        title="E3: ablation temporal kappa per (dataset, variant)",
    )


def plot_action_proportions(events: pd.DataFrame):
    block_utils.plot_adaptation_stacked(
        BLOCK, events,
        fname="e3_adaptation_actions",
        title="E3: adaptation action proportions per variant",
    )


def plot_action_counts_bar(events: pd.DataFrame):
    """Total event count per (variant, dataset) bar plot."""
    if events is None or len(events) == 0:
        return
    counts = events.groupby(["dataset", "variant"]).size().reset_index(name="events")
    plot_utils.setup_style()
    counts["dataset"] = pd.Categorical(counts.dataset, categories=block_utils.ordered_datasets(counts), ordered=True)
    counts["variant"] = pd.Categorical(counts.variant, categories=block_utils.ordered_variants(BLOCK, counts), ordered=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=counts, x="dataset", y="events", hue="variant", ax=ax, errorbar=None)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Adaptation events (total over all seeds)")
    ax.set_title("E3: adaptation events per (dataset, variant)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plot_utils.short_legend(ax, ncol=2, title="Variant")
    fig.tight_layout()
    plot_utils.save_fig(fig, "e3_adaptation_event_counts")


def plot_importance_evolution(feat_imp: pd.DataFrame):
    """Per dataset, follow the top-5 features' rank over time for one DA."""
    if feat_imp is None or len(feat_imp) == 0:
        return
    plot_utils.setup_style()
    # C4: include the real NYCTaxi stream alongside the synthetic ones.
    interesting = [ds for ds in ["FeatureDrift", "NHTS", "Hyperplane", "NYCTaxi"]
                   if ds in feat_imp.dataset.unique()]
    da_variants = [v for v in block_utils.ordered_variants(BLOCK, feat_imp)
                   if v.startswith("DA-")]
    if not da_variants:
        return
    chosen_v = da_variants[-1]
    has_drift_flag = "is_drifting" in feat_imp.columns
    for ds in interesting:
        sub = feat_imp[(feat_imp.dataset == ds) & (feat_imp.variant == chosen_v)]
        if sub.empty:
            continue
        seed_to_use = sorted(sub.seed.unique())[0]
        sub = sub[sub.seed == seed_to_use]
        top_feats = (sub.groupby("feature_index").importance.mean()
                     .sort_values(ascending=False).head(5).index.tolist())
        sub = sub[sub.feature_index.isin(top_feats)]
        feature_names = plot_utils.arff_attribute_names(ds)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        drift_marked = False
        for i, f in enumerate(top_feats):
            fs = sub[sub.feature_index == f].sort_values("instance_index")
            if feature_names is not None and f < len(feature_names):
                label = feature_names[f]
            else:
                label = f"feat {f}"
            ax.plot(fs.instance_index, fs.importance, label=label,
                    linewidth=1.0, color=f"C{i}")
            if has_drift_flag:
                dr = fs[pd.to_numeric(fs.is_drifting, errors="coerce").fillna(0) > 0]
                if not dr.empty:
                    ax.scatter(dr.instance_index, dr.importance, marker="x",
                               s=28, color=f"C{i}", linewidths=1.1, zorder=5,
                               label="drift-flagged" if not drift_marked else None)
                    drift_marked = True
        drift_pts = config.DRIFT_POINTS_E1E3.get(ds)
        plot_utils.add_drift_lines(ax, drift_pts)
        plot_utils.annotate_continuous(ax, drift_pts)
        ax.set_xlabel("Instance #")
        ax.set_ylabel("Importance")
        note = " (× = feature flagged drifting)" if drift_marked else ""
        ax.set_title(f"E3 ({chosen_v}, seed={seed_to_use}): top-5 feature importance — {ds}{note}")
        plot_utils.short_legend(ax, ncol=6)
        fig.tight_layout()
        plot_utils.save_fig(fig, f"e3_importance_evolution_{ds}", subdir="e3_extra")


def plot_feature_importance(feat_imp: pd.DataFrame):
    block_utils.plot_importance_noise_annotated(
        BLOCK, feat_imp,
        fname_prefix="e3_importance_noise", subdir="e3_importance_noise",
        title_prefix="E3",
    )


def plot_adaptation_timeline(events: pd.DataFrame):
    block_utils.plot_adaptation_timeline(
        BLOCK, events,
        fname_prefix="e3_adaptation_timeline", subdir="e3_timelines",
        drift_points_map=config.DRIFT_POINTS_E1E3, title_prefix="E3",
    )


def plot_selection_timeline(feat_sel: pd.DataFrame):
    block_utils.plot_selection_timeline(
        BLOCK, feat_sel,
        fname_prefix="e3_selection_timeline", subdir="e3_timelines",
        drift_points_map=config.DRIFT_POINTS_E1E3,
        variant_filter=lambda v: v.startswith("DA-"), title_prefix="E3",
    )


def plot_learner_view(events: pd.DataFrame):
    """Per-learner DA-SRP views. Silently skipped on adaptation_events.csv."""
    block_utils.plot_learner_lanes(
        BLOCK, events,
        fname_prefix="e3_learner_lanes", subdir="e3_timelines",
        drift_points_map=config.DRIFT_POINTS_E1E3, title_prefix="E3",
    )
    block_utils.plot_action_vs_overlap(
        BLOCK, events,
        fname="e3_action_vs_overlap",
        title=("E3: what a DA-SRP learner does vs how many just-drifted features "
               "sit in its subspace"),
    )


def plot_causality(data: dict):
    for ds in ["FeatureDrift", "NYCTaxi"]:
        block_utils.plot_causality_overlay(
            BLOCK, data["windows"], data["drift_alarms"], data["adaptation_events"],
            dataset=ds, variant="DA-SRP-ABC",
            drift_points_map=config.DRIFT_POINTS_E1E3,
            fname=f"e3_causality_{ds}", subdir="e3_timelines", title_prefix="E3",
        )


def plot_windows_kappa(window: pd.DataFrame):
    block_utils.plot_window_timeseries(
        BLOCK, window, metric="kappa",
        drift_points_map=config.DRIFT_POINTS_E1E3,
        fname_prefix="e3_kappa_timeseries", subdir="e3_timeseries",
        ylabel=r"$\kappa$ (window)", title_prefix="E3",
    )


def plot_recovery(recovery: pd.DataFrame):
    block_utils.plot_recovery(
        BLOCK, recovery,
        fname="e3_recovery_depth",
        title="E3: Recovery depth per (dataset, variant)",
    )


def plot_drift_alarms(drift_alarms: pd.DataFrame):
    block_utils.plot_alarm_counts(
        BLOCK, drift_alarms,
        fname="e3_drift_alarm_counts",
        title="E3: Mean drift alarms per run",
    )


def write_stat_tables(stat_tests: dict):
    block_utils.friedman_table(BLOCK, stat_tests)
    for metric in config.RANK_TABLE_METRICS:
        block_utils.per_metric_rank_table(BLOCK, stat_tests, metric)
    for metric in ["kappa", "kappa_temporal", "recovery_time"]:
        block_utils.nemenyi_table(BLOCK, stat_tests, metric=metric)
        block_utils.wilcoxon_table(BLOCK, stat_tests, metric=metric)
    block_utils.export_cd_diagrams(BLOCK, stat_tests, subdir="e3_stat_tests")


def run():
    print("\n[E3] DA-SRP / DA-ARF ablation")
    data = loaders.load_e3()
    summary = data["summary"]
    if summary is None or len(summary) == 0:
        print("  E3 summary missing, skipping")
        return
    table_ablation(summary)
    table_adaptation_actions(data["adaptation_events"])
    block_utils.resource_table(
        BLOCK, summary, "tab_e3_resources", "tab:e3_resources",
        "E3 DA-SRP / DA-ARF ablation")
    plot_ablation_bar(summary)
    plot_temporal_kappa_bar(summary)
    plot_action_proportions(data["adaptation_events"])
    plot_action_counts_bar(data["adaptation_events"])
    plot_importance_evolution(data["feature_importance"])
    plot_feature_importance(data["feature_importance"])
    plot_adaptation_timeline(data["adaptation_events"])
    plot_learner_view(data["adaptation_events"])
    plot_selection_timeline(data["feature_selections"])
    plot_causality(data)
    plot_windows_kappa(data["windows"])
    plot_recovery(data["recovery_time"])
    plot_drift_alarms(data["drift_alarms"])
    write_stat_tables(data["stat_tests"])


if __name__ == "__main__":
    run()
