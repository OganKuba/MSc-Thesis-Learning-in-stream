from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, loaders, latex_tables, plot_utils, block_utils

BLOCK = "E4"


def _split_dataset(name: str) -> tuple[str, str]:
    """E4 dataset names are <Generator>-<Dynamics>, e.g. SEA-HiDyn."""
    if name is None or "-" not in name:
        return name, ""
    gen, dyn = name.rsplit("-", 1)
    return gen, dyn


def _augment(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or len(summary) == 0:
        return summary
    df = summary.copy()
    parts = df.dataset.astype(str).str.rsplit("-", n=1, expand=True)
    df["generator"] = parts[0]
    df["dynamics"] = parts[1] if parts.shape[1] > 1 else ""
    return df


def table_high_dynamics(summary: pd.DataFrame):
    pv = block_utils.metric_pivot(BLOCK, summary, "kappa_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e4_kappa", pv,
        caption=r"E4 high-dynamics: mean $\kappa$ per (dataset, variant). Bold = best per row.",
        label="tab:e4_kappa",
    )

    pv_acc = block_utils.metric_pivot(BLOCK, summary, "accuracy_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e4_accuracy", pv_acc,
        caption="E4 high-dynamics: mean accuracy per (dataset, variant).",
        label="tab:e4_accuracy",
    )

    pv_tk = block_utils.metric_pivot(BLOCK, summary, "kappa_temporal_windowed_mean", index="dataset", columns="variant")
    block_utils.write_metric_table(
        "tab_e4_temporal_kappa", pv_tk,
        caption=r"E4 high-dynamics: temporal $\kappa$ averaged over all evaluation windows, per (dataset, variant).",
        label="tab:e4_temporal_kappa",
    )


def table_dynamics_sensitivity(summary: pd.DataFrame):
    df = _augment(summary)
    if "dynamics" not in df.columns or df.dynamics.eq("").all():
        return
    rows = []
    for v in block_utils.ordered_variants(BLOCK, df):
        sub = df[df.variant == v]
        for dyn in config.DYNAMICS_ORDER:
            d = sub[sub.dynamics == dyn]
            rows.append({"Variant": v, dyn: d.kappa_mean.mean() if len(d) else np.nan})
    flat: dict[str, dict] = {}
    for r in rows:
        flat.setdefault(r["Variant"], {})
        for k in config.DYNAMICS_ORDER:
            if k in r:
                flat[r["Variant"]][k] = r[k]
    out = pd.DataFrame(flat).T
    out = out.reindex([v for v in config.E4_VARIANT_ORDER if v in flat])
    out = out.reindex(columns=config.DYNAMICS_ORDER)
    body = latex_tables.df_to_booktabs(out, ndigits=3, index_name="Variant")
    latex_tables.write_table(
        "tab_e4_dynamics_sensitivity", body,
        caption=r"E4: mean $\kappa$ per variant at Low vs HiDyn dynamics.",
        label="tab:e4_dynamics_sensitivity",
    )


def table_adaptation_actions(events: pd.DataFrame):
    agg = block_utils.adaptation_summary(events)
    if agg.empty:
        return
    cats = [c for c in ["kept_count", "surgical_count",
                        "full_replacement_count", "no_replacement_count",
                        "ext_keep_count", "ext_full_count"] if c in agg.columns]
    df = agg.set_index(["dataset", "variant"])[cats].reset_index()
    df.columns = ["Dataset", "Variant"] + [c.replace("_count", "") for c in cats]
    df = df.set_index("Dataset")
    body = latex_tables.df_to_booktabs(df, ndigits=0, index_name="Dataset")
    latex_tables.write_table(
        "tab_e4_adaptation_actions", body,
        caption="E4 adaptation events per (dataset, variant).",
        label="tab:e4_adaptation_actions",
    )


def plot_kappa_by_dynamics(summary: pd.DataFrame):
    df = _augment(summary)
    if df.empty or "dynamics" not in df.columns:
        return
    plot_utils.setup_style()
    gens = sorted(df.generator.unique())
    fig, axes = plt.subplots(1, max(len(gens), 1),
                             figsize=(5 * max(len(gens), 1), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, gen in zip(axes, gens):
        sub = df[df.generator == gen].copy()
        sub["dynamics"] = pd.Categorical(sub.dynamics, categories=config.DYNAMICS_ORDER, ordered=True)
        sub["variant"] = pd.Categorical(sub.variant, categories=block_utils.ordered_variants(BLOCK, sub), ordered=True)
        sns.barplot(data=sub, x="dynamics", y="kappa_mean", hue="variant", ax=ax, errorbar=None)
        ax.set_title(gen)
        ax.set_xlabel("Dynamics")
        ax.set_ylabel(r"$\kappa$ (mean)")
    plot_utils.figure_legend(fig, axes, ncol=3, title="Variant", y=0.91)
    fig.suptitle("E4: kappa per generator / dynamics / variant", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    plot_utils.save_fig(fig, "e4_kappa_by_dynamics")


def plot_windows_kappa(window: pd.DataFrame):
    block_utils.plot_window_timeseries(
        BLOCK, window, metric="kappa",
        drift_points_map=config.DRIFT_POINTS_E4E5,
        fname_prefix="e4_kappa_timeseries", subdir="e4_timeseries",
        ylabel=r"$\kappa$ (window)", title_prefix="E4",
    )


def plot_windows_accuracy(window: pd.DataFrame):
    block_utils.plot_window_timeseries(
        BLOCK, window, metric="accuracy",
        drift_points_map=config.DRIFT_POINTS_E4E5,
        fname_prefix="e4_accuracy_timeseries", subdir="e4_timeseries",
        ylabel="Accuracy (window)", title_prefix="E4",
    )


def plot_drift_alarms(drift_alarms: pd.DataFrame):
    block_utils.plot_alarm_counts(
        BLOCK, drift_alarms,
        fname="e4_drift_alarm_counts",
        title="E4: Mean drift alarms per run",
    )
    block_utils.plot_alarm_timeline(
        BLOCK, drift_alarms,
        fname_prefix="e4_alarms", subdir="e4_alarms",
        drift_points_map=config.DRIFT_POINTS_E4E5,
        title_prefix="E4",
    )


def plot_recovery(recovery: pd.DataFrame):
    block_utils.plot_recovery(
        BLOCK, recovery,
        fname="e4_recovery_depth",
        title="E4: Recovery depth per (dataset, variant)",
    )


def plot_action_proportions(events: pd.DataFrame):
    block_utils.plot_adaptation_stacked(
        BLOCK, events,
        fname="e4_adaptation_actions",
        title="E4: adaptation action proportions per variant",
    )


def plot_adaptation_timeline(events: pd.DataFrame):
    block_utils.plot_adaptation_timeline(
        BLOCK, events,
        fname_prefix="e4_adaptation_timeline", subdir="e4_timelines",
        drift_points_map=config.DRIFT_POINTS_E4E5, title_prefix="E4",
    )


def write_stat_tables(stat_tests: dict):
    block_utils.friedman_table(BLOCK, stat_tests)
    for metric in config.RANK_TABLE_METRICS:
        block_utils.per_metric_rank_table(BLOCK, stat_tests, metric)
    for metric in ["kappa", "recovery_time", "kappa_temporal"]:
        block_utils.nemenyi_table(BLOCK, stat_tests, metric=metric)
        block_utils.wilcoxon_table(BLOCK, stat_tests, metric=metric)
    block_utils.export_cd_diagrams(BLOCK, stat_tests, subdir="e4_stat_tests")


def run():
    print("\n[E4] High-dynamics scenarios")
    data = loaders.load_e4()
    summary = data["summary"]
    if summary is None or len(summary) == 0:
        print("  E4 summary missing, skipping")
        return
    table_high_dynamics(summary)
    table_dynamics_sensitivity(summary)
    table_adaptation_actions(data["adaptation_events"])
    block_utils.resource_table(
        BLOCK, summary, "tab_e4_resources", "tab:e4_resources",
        "E4 drift intensity")
    plot_kappa_by_dynamics(summary)
    plot_windows_kappa(data["windows"])
    plot_windows_accuracy(data["windows"])
    plot_drift_alarms(data["drift_alarms"])
    plot_recovery(data["recovery_time"])
    plot_action_proportions(data["adaptation_events"])
    plot_adaptation_timeline(data["adaptation_events"])
    write_stat_tables(data["stat_tests"])


if __name__ == "__main__":
    run()
