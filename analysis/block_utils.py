"""Shared helpers for per-block (E1..E5) analyses against the unified."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import seaborn as sns

from . import config, latex_tables, loaders, plot_utils


# Ordering

def ordered_datasets(df, col="dataset"):
    if df is None or col not in df.columns:
        return []
    present = [d for d in config.DATASET_ORDER if d in df[col].unique()]
    extras = [d for d in df[col].unique() if d not in present]
    return present + extras


def per_dataset_targets(block: str, df, col="dataset"):
    """Datasets for which a per-dataset figure should be produced."""
    present = ordered_datasets(df, col=col)
    wanted = config.PER_DATASET_FIGURES.get(block)
    if wanted is None:
        return present
    return [d for d in present if d in wanted]


def ordered_variants(block: str, df, col="variant"):
    if df is None or col not in df.columns:
        return []
    order = config.VARIANT_ORDER_BY_BLOCK.get(block, [])
    present = [v for v in order if v in df[col].unique()]
    extras = [v for v in df[col].unique() if v not in present]
    return present + extras


# Summary pivots

def drop_saturated(block: str, df: pd.DataFrame, metric_col: str,
                   dataset_col: str = "dataset") -> pd.DataFrame:
    """Drop datasets that are saturated (kappa/accuracy ~1.0) for this."""
    if df is None or dataset_col not in df.columns:
        return df
    if metric_col not in config.SATURATED_METRICS:
        return df
    drop = config.SATURATED_DATASETS_BY_BLOCK.get(block, set())
    if not drop:
        return df
    return df[~df[dataset_col].isin(drop)]


def metric_pivot(block: str, summary: pd.DataFrame, metric_col: str,
                 index="dataset", columns="variant") -> pd.DataFrame:
    """Pivot a summary CSV (long form) on dataset x variant for any *_mean."""
    summary = drop_saturated(block, summary, metric_col)
    rows = ordered_datasets(summary, index) if index == "dataset" else ordered_variants(block, summary, index)
    cols = ordered_variants(block, summary, columns) if columns == "variant" else ordered_datasets(summary, columns)
    pv = plot_utils.safe_pivot(
        summary, index=index, columns=columns, values=metric_col,
        reindex_rows=rows, reindex_cols=cols,
    )
    return pv


# Window helpers

def aggregate_windows(window: pd.DataFrame, metric: str = "accuracy",
                      group_keys=("dataset", "variant", "window_id"),
                      x_col: str = "end_instance") -> pd.DataFrame:
    """Mean a window-level metric across seeds, keeping start/end instance."""
    if window is None or metric not in window.columns:
        return pd.DataFrame()
    keys = list(group_keys)
    agg_cols = [metric]
    if x_col in window.columns and x_col not in keys:
        agg_cols = [metric, x_col]
    agg = window.groupby(keys, as_index=False)[agg_cols].mean()
    return agg.sort_values(keys)


# LaTeX writers

def write_metric_table(name: str, pv: pd.DataFrame, caption: str, label: str,
                       ndigits: int = 3, bold_max_per_row: bool = True,
                       bold_max_per_col: bool = False):
    if pv is None or pv.empty:
        return
    body = latex_tables.df_to_booktabs(
        pv, ndigits=ndigits, bold_max_per_row=bold_max_per_row,
        bold_max_per_col=bold_max_per_col,
        index_name=pv.index.name or "",
    )
    latex_tables.write_table(name, body, caption=caption, label=label)


def resource_table(block: str, summary: pd.DataFrame, name: str, label: str,
                   block_title: str):
    """Cost side of a block: accuracy against memory and speed, one row."""
    if summary is None or summary.empty or "ram_hours_gb_mean" not in summary.columns:
        return
    df = summary.copy()
    if "instances_mean" not in df.columns:
        return
    inst = pd.to_numeric(df.instances_mean, errors="coerce")
    df["_ramh_norm"] = (pd.to_numeric(df.ram_hours_gb_mean, errors="coerce") / inst
                        * 1e5 * config.RAMH_SCALE)
    variants = ordered_variants(block, df)
    agg = df.groupby("variant").agg(
        kappa=("kappa_mean", "mean"),
        ramh=("_ramh_norm", "mean"),
        peak_mb=("peak_mb_mean", "mean"),
        throughput=("throughput_mean", "mean"),
    ).reindex([v for v in variants if v in df.variant.unique()])
    if agg.empty:
        return
    out = pd.DataFrame({
        "kappa": agg.kappa,
        "RAMh/100k": agg.ramh,
        "peak MB": agg.peak_mb,
        "inst/s": agg.throughput.map(lambda x: "-" if pd.isna(x) else f"{x:.0f}"),
    })
    out.index.name = "Variant"
    body = latex_tables.df_to_booktabs(out, ndigits=3, index_name="Variant")
    latex_tables.write_table(
        name, body,
        caption=(block_title + r": accuracy against cost, averaged over the block's datasets. "
                 r"\emph{RAMh/100k} is RAM-Hours in " + config.RAMH_UNIT_TEX +
                 r" normalised to a 100k-instance stream (the raw metric integrates memory over "
                 r"time, so unnormalised it would mostly reflect how long each stream is); "
                 r"\emph{peak MB} is the deep size of the learner, which needs no normalisation."),
        label=label,
    )


# Common plots

def plot_metric_bar(block: str, summary: pd.DataFrame, metric_col: str,
                    fname: str, ylabel: str, title: str, subdir: str | None = None):
    """Grouped bars per (dataset, variant), with +/- 1 SD over seeds where."""
    if summary is None or metric_col not in summary.columns:
        return
    plot_utils.setup_style()
    df = drop_saturated(block, summary, metric_col).copy()
    datasets = ordered_datasets(df)
    variants = ordered_variants(block, df)
    df["dataset"] = pd.Categorical(df.dataset, categories=datasets, ordered=True)
    df["variant"] = pd.Categorical(df.variant, categories=variants, ordered=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df, x="dataset", y=metric_col, hue="variant", ax=ax, errorbar=None)
    _add_sd_whiskers(ax, df, metric_col, datasets, variants)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    plot_utils.legend_below(ax, ncol=2, title="Variant")
    plot_utils.save_fig(fig, fname, subdir=subdir)


def _add_sd_whiskers(ax, df: pd.DataFrame, metric_col: str, datasets, variants):
    """Draw +/- 1 SD whiskers on an existing grouped barplot, read from."""
    std_col = metric_col.replace("_mean", "_std")
    if not metric_col.endswith("_mean") or std_col not in df.columns:
        return
    std_pv = df.pivot_table(index="variant", columns="dataset", values=std_col, observed=False)
    # One container per hue level, in the order of the variant categories.
    for container, variant in zip(ax.containers, variants):
        if variant not in std_pv.index:
            continue
        for patch, dataset in zip(container.patches, datasets):
            if dataset not in std_pv.columns:
                continue
            sd = std_pv.loc[variant, dataset]
            height = patch.get_height()
            if not np.isfinite(sd) or sd <= 0 or not np.isfinite(height):
                continue
            ax.errorbar(patch.get_x() + patch.get_width() / 2.0, height, yerr=sd,
                        fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6,
                        capthick=0.7, alpha=0.65, zorder=5)


def plot_window_timeseries(block: str, window: pd.DataFrame, metric: str,
                           drift_points_map: dict, fname_prefix: str, subdir: str,
                           ylabel: str | None = None, title_prefix: str | None = None,
                           top_n: int | None = None):
    """One plot per dataset, lines per variant (averaged across seeds)."""
    if window is None or metric not in window.columns:
        return
    if metric == "accuracy" and config.figure_disabled("accuracy_timeseries", block):
        return
    plot_utils.setup_style()
    for ds in per_dataset_targets(block, window):
        sub = window[window.dataset == ds]
        if len(sub) == 0:
            continue
        variants = ordered_variants(block, sub)
        if top_n is not None and len(variants) > top_n:
            best = sub.groupby("variant")[metric].mean().sort_values(ascending=False).head(top_n).index
            variants = [v for v in variants if v in best]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        linestyles = ["-", "--", "-.", ":"]
        for i, v in enumerate(variants):
            vd = sub[sub.variant == v]
            agg = vd.groupby("end_instance", as_index=False)[metric].mean().sort_values("end_instance")
            if agg.empty:
                continue
            ax.plot(agg.end_instance, agg[metric], label=v, linewidth=1.2,
                    linestyle=linestyles[i % len(linestyles)])
        drift_pts = drift_points_map.get(ds)
        plot_utils.add_drift_lines(ax, drift_pts)
        plot_utils.annotate_continuous(ax, drift_pts)
        ax.set_xlabel("Instance #")
        ax.set_ylabel(ylabel or metric)
        ax.set_title(f"{title_prefix or block}: {metric} over time — {ds}")
        plot_utils.short_legend(ax, ncol=2, title="Variant")
        fig.tight_layout()
        plot_utils.save_fig(fig, f"{fname_prefix}_{ds}", subdir=subdir)


# Drift alarms

def alarms_per_run(drift_alarms: pd.DataFrame) -> pd.DataFrame:
    """Mean alarm count per (dataset, variant), averaged across seeds."""
    if drift_alarms is None or len(drift_alarms) == 0:
        return pd.DataFrame()
    counts = (drift_alarms
              .groupby(["dataset", "variant", "seed"])
              .size()
              .reset_index(name="alarm_count"))
    return counts.groupby(["dataset", "variant"], as_index=False).alarm_count.mean()


def plot_alarm_counts(block: str, drift_alarms: pd.DataFrame, fname: str,
                      title: str, subdir: str | None = None):
    counts = alarms_per_run(drift_alarms)
    if counts.empty:
        return
    plot_utils.setup_style()
    counts["dataset"] = pd.Categorical(
        counts.dataset, categories=ordered_datasets(counts), ordered=True,
    )
    counts["variant"] = pd.Categorical(
        counts.variant, categories=ordered_variants(block, counts), ordered=True,
    )
    fig, ax = plt.subplots(figsize=(11, 5.4))
    sns.barplot(data=counts, x="dataset", y="alarm_count", hue="variant", ax=ax, errorbar=None)
    positive = counts.alarm_count[counts.alarm_count > 0]
    if not positive.empty:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(float(positive.min()) * 0.6, 1e-3))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean alarms per run (log scale)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    plot_utils.legend_below(ax, ncol=2, title="Variant")
    plot_utils.save_fig(fig, fname, subdir=subdir)


ALARM_USEFUL_DELTA = 0.01


def alarm_delta(drift_alarms: pd.DataFrame) -> pd.DataFrame:
    """Add `delta_acc` = accuracy one window AFTER the alarm minus."""
    need = {"window_accuracy_before", "window_accuracy_after"}
    if drift_alarms is None or drift_alarms.empty or not need.issubset(drift_alarms.columns):
        return pd.DataFrame()
    df = drift_alarms.copy()
    df["delta_acc"] = (pd.to_numeric(df.window_accuracy_after, errors="coerce")
                       - pd.to_numeric(df.window_accuracy_before, errors="coerce"))
    return df.dropna(subset=["delta_acc"])


def plot_alarm_timeline(block: str, drift_alarms: pd.DataFrame, fname_prefix: str,
                        subdir: str, drift_points_map: dict, title_prefix: str | None = None):
    """One subplot per variant: every alarm as a stem whose height is the."""
    if config.figure_disabled("alarm_timeline", block):
        return
    df = alarm_delta(drift_alarms)
    if df.empty:
        return
    plot_utils.setup_style()
    for ds in per_dataset_targets(block, df):
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        seed = _pick_seed(sub)
        if seed is not None:
            sub = sub[sub.seed == seed]
        variants = ordered_variants(block, sub)
        if not variants:
            continue
        gt = drift_points_map.get(ds)
        lim = float(np.nanmax(np.abs(sub.delta_acc.values))) if len(sub) else 0.0
        lim = max(lim, 0.01) * 1.15
        fig, axes = plt.subplots(len(variants), 1, figsize=(11, 1.05 * len(variants) + 1.0),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes[:, 0]
        for ax, v in zip(axes, variants):
            vd = sub[sub.variant == v].sort_values("instance_index")
            x = vd.instance_index.values
            d = vd.delta_acc.values
            # Literal colours (not C2/C3 cycle indices, which shift meaning whenever
            # config.PALETTE is reordered): green = useful, orange = harmful, grey = neutral.
            colors = np.where(d >= ALARM_USEFUL_DELTA, "#1a9850",
                              np.where(d <= -ALARM_USEFUL_DELTA, "#d55e00", "0.6"))
            ax.vlines(x, 0.0, d, colors=colors, linewidth=1.8, alpha=0.95)
            ax.scatter(x, np.zeros_like(d), s=16, color="#0173b2", zorder=3,
                       edgecolor="white", linewidth=0.4)
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f"{v}\n" + r"$\Delta$acc", fontsize=8)
            plot_utils.add_drift_lines(ax, gt)
        plot_utils.annotate_continuous(axes[0], gt)
        axes[-1].set_xlabel("Instance #")
        fig.suptitle(f"{title_prefix or block}: alarm effectiveness over time — {ds} "
                     f"(seed={seed}, stem = accuracy gained one window after the alarm; "
                     f"dot = alarm, dashed = GT drift)")
        fig.tight_layout()
        plot_utils.save_fig(fig, f"{fname_prefix}_{ds}", subdir=subdir)


def plot_alarm_effectiveness(block: str, drift_alarms: pd.DataFrame, fname: str,
                             title: str, subdir: str | None = None):
    """Aggregate companion to plot_alarm_timeline: distribution of."""
    df = alarm_delta(drift_alarms)
    if df.empty:
        return
    plot_utils.setup_style()
    variants = ordered_variants(block, df)
    df = df[df.variant.isin(variants)].copy()
    df["variant"] = pd.Categorical(df.variant, categories=variants, ordered=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="variant", y="delta_acc", ax=ax, showfliers=False,
                width=0.6, color="0.85", linewidth=0.9)
    sns.stripplot(data=df, x="variant", y="delta_acc", ax=ax, size=2.2,
                  alpha=0.35, color="C0", jitter=0.28)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.axhline(ALARM_USEFUL_DELTA, color="#1a9850", linewidth=0.8, linestyle=":", alpha=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    label_tf = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, v in enumerate(variants):
        d = df[df.variant == v].delta_acc
        if d.empty:
            continue
        share = float((d >= ALARM_USEFUL_DELTA).mean())
        ax.text(i, 0.99, f"{share:.0%}\nn={len(d)}", transform=label_tf,
                ha="center", va="top", fontsize=8, alpha=0.85)
    ax.set_xlabel("Variant")
    ax.set_ylabel(r"$\Delta$accuracy one window after alarm")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    plot_utils.save_fig(fig, fname, subdir=subdir)


def alarm_effectiveness_table(block: str, drift_alarms: pd.DataFrame, name: str,
                              caption: str, label: str):
    """Quantitative anchor for plot_alarm_effectiveness, per (variant."""
    df = alarm_delta(drift_alarms)
    if df.empty:
        return
    keys = ["variant"] + (["detector"] if "detector" in df.columns else [])
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row["alarms"] = f"{len(g)}"
        row["mean_delta"] = float(g.delta_acc.mean())
        row["median_delta"] = float(g.delta_acc.median())
        row["useful_pct"] = 100.0 * float((g.delta_acc >= ALARM_USEFUL_DELTA).mean())
        row["harmful_pct"] = 100.0 * float((g.delta_acc <= -ALARM_USEFUL_DELTA).mean())
        rows.append(row)
    if not rows:
        return
    order = ordered_variants(block, df)
    out = pd.DataFrame(rows)
    out["variant"] = pd.Categorical(out.variant, categories=order, ordered=True)
    out = out.sort_values("useful_pct", ascending=False).set_index("variant")
    body = latex_tables.df_to_booktabs(out, ndigits=3, index_name="Variant")
    latex_tables.write_table(name, body, caption=caption, label=label)


# Recovery time

def recovery_table(block: str, recovery: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-drift recovery stats per (dataset, variant)."""
    if recovery is None or len(recovery) == 0:
        return pd.DataFrame()
    recovery = recovery.copy()
    if "recovery_length" in recovery.columns:
        recovery.loc[recovery["recovery_length"] <= 0, "recovery_length"] = np.nan
    specs = {
        "mean_recovery_length": "recovery_length",
        "mean_max_drop": "max_drop",
        "mean_area": "area_under_recovery_curve",
    }
    named = {out: (src, "mean") for out, src in specs.items() if src in recovery.columns}
    named["drift_count"] = ("drift_id", "count")
    agg = recovery.groupby(["dataset", "variant"], as_index=False).agg(**named)
    return agg


# Recovery depth metrics we prefer over the degenerate
RECOVERY_METRICS = [
    ("mean_max_drop", "Mean max accuracy drop after drift"),
    ("mean_area", "Mean area under recovery curve"),
]


def plot_recovery(block: str, recovery: pd.DataFrame, fname: str, title: str,
                  subdir: str | None = None):
    """Recovery *depth* per (dataset, variant): max accuracy drop + area."""
    agg = recovery_table(block, recovery)
    if agg.empty:
        return
    metrics = [(col, lab) for col, lab in RECOVERY_METRICS if col in agg.columns]
    if not metrics:
        return
    plot_utils.setup_style()
    agg["dataset"] = pd.Categorical(agg.dataset, categories=ordered_datasets(agg), ordered=True)
    agg["variant"] = pd.Categorical(agg.variant, categories=ordered_variants(block, agg), ordered=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5.4), squeeze=False)
    axes = axes[0]
    for ax, (col, label) in zip(axes, metrics):
        sns.barplot(data=agg, x="dataset", y=col, hue="variant", ax=ax, errorbar=None)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(label)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    plot_utils.figure_legend(fig, axes, ncol=5, title="Variant", y=0.99)
    fig.suptitle(title, y=0.86)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    plot_utils.save_fig(fig, fname, subdir=subdir)


# Feature selections

def feature_selection_summary(feat_sel: pd.DataFrame) -> pd.DataFrame:
    if feat_sel is None or len(feat_sel) == 0:
        return pd.DataFrame()
    agg = (feat_sel
           .groupby(["dataset", "variant"], as_index=False)
           .agg(mean_selected_count=("selected_feature_count", "mean"),
                mean_jaccard=("jaccard_to_previous", "mean"),
                mean_stability=("stability_ratio", "mean"),
                num_changes=("trigger_type", "count")))
    return agg


def is_static_selection(feat_sel: pd.DataFrame) -> bool:
    """True when the selection never actually changes (only 'initial'."""
    if feat_sel is None or len(feat_sel) == 0 or "trigger_type" not in feat_sel.columns:
        return True
    return set(feat_sel.trigger_type.unique()) <= {"initial"}


def plot_feature_selection_overview(block: str, feat_sel: pd.DataFrame, fname: str,
                                    title: str, subdir: str | None = None):
    if is_static_selection(feat_sel):
        print(f"  [skip] {fname}: static selection (no non-initial triggers)")
        return
    agg = feature_selection_summary(feat_sel)
    if agg.empty:
        return
    plot_utils.setup_style()
    agg["dataset"] = pd.Categorical(agg.dataset, categories=ordered_datasets(agg), ordered=True)
    agg["variant"] = pd.Categorical(agg.variant, categories=ordered_variants(block, agg), ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=agg, x="dataset", y="mean_selected_count", hue="variant",
                ax=axes[0], errorbar=None)
    axes[0].set_title("Mean selected feature count")
    axes[0].set_ylabel("# features")
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")
    plot_utils.short_legend(axes[0], ncol=2, title="Variant")

    sns.barplot(data=agg, x="dataset", y="mean_stability", hue="variant",
                ax=axes[1], errorbar=None)
    axes[1].set_title("Mean stability ratio")
    axes[1].set_ylabel("stability")
    plt.setp(axes[1].get_xticklabels(), rotation=25, ha="right")
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    fig.suptitle(title)
    fig.tight_layout()
    plot_utils.save_fig(fig, fname, subdir=subdir)


# Feature importance

def importance_top_features(feat_imp: pd.DataFrame, top_k: int = 10,
                             dataset: str | None = None,
                             variant: str | None = None) -> pd.DataFrame:
    if feat_imp is None or len(feat_imp) == 0:
        return pd.DataFrame()
    df = feat_imp
    if dataset is not None:
        df = df[df.dataset == dataset]
    if variant is not None:
        df = df[df.variant == variant]
    if df.empty:
        return pd.DataFrame()
    agg = (df.groupby(["dataset", "variant", "feature_index"], as_index=False)
             .importance.mean())
    out = []
    for (ds, v), g in agg.groupby(["dataset", "variant"]):
        out.append(g.sort_values("importance", ascending=False).head(top_k))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_importance_heatmap(block: str, feat_imp: pd.DataFrame, fname: str,
                            title: str, subdir: str | None = None):
    """Feature importance per (dataset, feature index), with the injected."""
    if feat_imp is None or len(feat_imp) == 0:
        return
    plot_utils.setup_style()
    datasets = ordered_datasets(feat_imp)
    if not datasets:
        return
    rows, noise_spans = {}, {}
    for ds in datasets:
        sub = feat_imp[feat_imp.dataset == ds]
        if sub.empty:
            continue
        width = int(sub.feature_index.max()) + 1
        mean_imp = sub.groupby("feature_index").importance.mean()
        # importance * d: 1.0 = the feature carries exactly a uniform share
        rows[ds] = mean_imp.reindex(range(width)) * width
        idx = noise_feature_indices(ds, width)
        if idx:
            noise_spans[ds] = (min(idx), max(idx))
    if not rows:
        return
    pv = pd.DataFrame(rows).T.reindex([d for d in datasets if d in rows])

    n_col = pv.shape[1]
    fig, ax = plt.subplots(figsize=(min(16, 0.42 * n_col + 4.5), 0.5 * len(pv.index) + 2.4))
    vmax = float(np.nanquantile(pv.values, 0.98))
    vmax = max(vmax, 1.2)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
    sns.heatmap(pv, cmap="vlag", norm=norm, ax=ax,
                annot=n_col <= 20, fmt=".1f",
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "importance relative to uniform (1.0 = 1/d)",
                          "extend": "max"})
    for y, ds in enumerate(pv.index):
        span = noise_spans.get(ds)
        if span is None:
            continue
        lo, hi = span
        ax.add_patch(plt.Rectangle((lo, y), hi - lo + 1, 1, fill=False,
                                   edgecolor="black", linewidth=1.6, zorder=5))
    ax.set_title(title)
    ax.set_xlabel("Feature index (within that dataset)")
    ax.set_ylabel("Dataset")
    ax.text(0.995, -0.14, "black outline = injected noise features",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, alpha=0.8)
    fig.tight_layout()
    plot_utils.save_fig(fig, fname, subdir=subdir)


def noise_feature_indices(dataset: str, total_features: int) -> list[int]:
    """Indices of the injected noise features for a synthetic dataset (B4)."""
    n = config.NOISE_FEATURES.get(dataset, 0)
    if n <= 0 or total_features <= 0:
        return []
    n = min(n, total_features)
    return list(range(total_features - n, total_features))


def plot_importance_noise_annotated(block: str, feat_imp: pd.DataFrame,
                                    fname_prefix: str, subdir: str,
                                    title_prefix: str | None = None):
    """Per-synthetic-dataset importance heatmap (variant x feature) with."""
    if feat_imp is None or len(feat_imp) == 0:
        return
    if config.figure_disabled("importance_noise", block):
        return
    plot_utils.setup_style()
    datasets = [d for d in per_dataset_targets(block, feat_imp)
                if config.NOISE_FEATURES.get(d, 0) > 0 and d in feat_imp.dataset.unique()]
    for ds in datasets:
        sub = feat_imp[feat_imp.dataset == ds]
        if sub.empty:
            continue
        total = int(sub.feature_index.max()) + 1
        noise_idx = noise_feature_indices(ds, total)
        if not noise_idx:
            continue
        pv = sub.pivot_table(index="variant", columns="feature_index",
                             values="importance", aggfunc="mean")
        rows = ordered_variants(block, sub)
        pv = pv.reindex(index=[r for r in rows if r in pv.index])
        pv = pv.reindex(columns=range(total))
        fig, ax = plt.subplots(
            figsize=(min(18, 0.5 * total + 3), 0.45 * len(pv.index) + 2))
        sns.heatmap(pv, annot=False, cmap="viridis", ax=ax,
                    cbar_kws={"label": "mean importance"})
        # Highlight noise columns: shaded band + red x-tick labels.
        for j in noise_idx:
            ax.axvspan(j, j + 1, color="red", alpha=0.12, zorder=3)
        for lbl in ax.get_xticklabels():
            try:
                if int(lbl.get_text()) in noise_idx:
                    lbl.set_color("red")
                    lbl.set_fontweight("bold")
            except ValueError:
                continue
        ax.set_title(
            f"{title_prefix or block}: feature importance — {ds} "
            f"(red = injected noise, last {len(noise_idx)} features)")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Variant")
        fig.tight_layout()
        plot_utils.save_fig(fig, f"{fname_prefix}_{ds}", subdir=subdir)


# Adaptation events

def adaptation_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame()
    cols = ["kept_count", "surgical_count", "full_replacement_count",
            "no_replacement_count", "ext_keep_count", "ext_full_count"]
    cols = [c for c in cols if c in events.columns]
    grouped = events.groupby(["dataset", "variant"], as_index=False)
    agg = grouped[cols].sum()
    agg["n_events"] = grouped.size()["size"].values
    return agg


def plot_adaptation_stacked(block: str, events: pd.DataFrame, fname: str,
                            title: str, subdir: str | None = None):
    agg = adaptation_summary(events)
    if agg.empty:
        return
    plot_utils.setup_style()
    cats = [c for c in ["kept_count", "surgical_count",
                        "full_replacement_count", "no_replacement_count",
                        "ext_keep_count", "ext_full_count"] if c in agg.columns]
    labels = {
        "kept_count": "KEEP",
        "surgical_count": "SURGICAL",
        "full_replacement_count": "FULL",
        "no_replacement_count": "NO_REPL",
        "ext_keep_count": "EXT_KEEP",
        "ext_full_count": "EXT_FULL",
    }
    variants = ordered_variants(block, agg)
    datasets = ordered_datasets(agg)
    fig, axes = plt.subplots(1, max(len(variants), 1),
                             figsize=(4.5 * max(len(variants), 1), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    palette = sns.color_palette(config.PALETTE, n_colors=len(cats))
    for ax, v in zip(axes, variants):
        vd_all = agg[agg.variant == v].set_index("dataset").reindex(datasets)
        n_events = vd_all["n_events"].fillna(0) if "n_events" in vd_all.columns else None
        vd = vd_all[cats].fillna(0)
        totals = vd.sum(axis=1).replace(0, np.nan)
        prop = vd.div(totals, axis=0).fillna(0)
        bottom = np.zeros(len(prop))
        for color, c in zip(palette, cats):
            ax.bar(prop.index.astype(str), prop[c].values, bottom=bottom,
                   label=labels[c], color=color)
            bottom += prop[c].values
        if n_events is not None:
            tf = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            for xi, ds in enumerate(datasets):
                n = int(n_events.get(ds, 0))
                if n == 0:
                    ax.text(xi, 0.5, "no\nevents", ha="center", va="center", fontsize=7,
                            color="0.35", style="italic", transform=tf)
                ax.text(xi, 1.015, f"n={n}", ha="center", va="bottom", fontsize=6.5,
                        color="0.35", transform=tf)
        ax.set_ylim(0, 1.0)
        ax.set_title(v, pad=18)
        ax.set_ylabel("Proportion")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plot_utils.figure_legend(fig, axes, ncol=min(6, len(cats)), y=0.91)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    plot_utils.save_fig(fig, fname, subdir=subdir)


# Time-resolved diagnostics (C1-C4)

# Per-learner adaptation action columns and their display labels /
_ACTION_COLS = ["kept_count", "surgical_count", "full_replacement_count",
                "no_replacement_count", "ext_keep_count", "ext_full_count"]
_ACTION_LABEL = {
    "kept_count": "KEEP", "surgical_count": "SURGICAL",
    "full_replacement_count": "FULL", "no_replacement_count": "NO_REPL",
    "ext_keep_count": "EXT_KEEP", "ext_full_count": "EXT_FULL",
}


def _pick_seed(df):
    """Deterministically pick the smallest available seed (or None)."""
    if df is None or "seed" not in df.columns or df.empty:
        return None
    return sorted(df.seed.unique())[0]


def _da_variants_with_events(block, events):
    order = ordered_variants(block, events)
    present = set(events.variant.unique())
    return [v for v in order if v in present]


def plot_adaptation_timeline(block: str, events: pd.DataFrame, fname_prefix: str,
                             subdir: str, drift_points_map: dict,
                             title_prefix: str | None = None):
    """C1: per-event KEEP/SURGICAL/FULL/... on the instance axis (not."""
    if events is None or len(events) == 0:
        return
    plot_utils.setup_style()
    cats_all = [c for c in _ACTION_COLS if c in events.columns]
    if not cats_all:
        return
    for ds in per_dataset_targets(block, events):
        sub = events[events.dataset == ds]
        if sub.empty:
            continue
        seed = _pick_seed(sub)
        if seed is not None:
            sub = sub[sub.seed == seed]
        variants = _da_variants_with_events(block, sub)
        if not variants:
            continue
        gt = drift_points_map.get(ds)
        fig, axes = plt.subplots(len(variants), 1,
                                 figsize=(11, 1.7 * len(variants) + 1.0),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]
        for ax, v in zip(axes, variants):
            vd = sub[sub.variant == v]
            cats = [c for c in cats_all if pd.to_numeric(vd[c], errors="coerce").fillna(0).abs().sum() > 0]
            for lane, c in enumerate(cats):
                vals = pd.to_numeric(vd[c], errors="coerce").fillna(0)
                mask = vals > 0
                if mask.any():
                    ax.scatter(vd.instance_index[mask], [lane] * int(mask.sum()),
                               s=12 + 6 * vals[mask].clip(upper=10), alpha=0.7,
                               color=f"C{lane}", edgecolor="none")
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels([_ACTION_LABEL[c] for c in cats], fontsize=8)
            ax.set_ylim(-0.5, max(len(cats) - 0.5, 0.5))
            ax.set_ylabel(v, fontsize=9)
            plot_utils.add_drift_lines(ax, gt)
        plot_utils.annotate_continuous(axes[0], gt)
        axes[-1].set_xlabel("Instance #")
        fig.suptitle(f"{title_prefix or block}: adaptation actions over time — {ds} "
                     f"(seed={seed}, dashed=GT drift)")
        fig.tight_layout()
        plot_utils.save_fig(fig, f"{fname_prefix}_{ds}", subdir=subdir)


# Per-learner view of DA-SRP adaptations

# Letters written by RunDetailedRecorder.encodeActions.
_LEARNER_ACTION_LABEL = {"K": "KEEP", "S": "SURGICAL", "F": "FULL", "N": "NO_REPL"}
_LEARNER_ACTION_COLOR = {"K": "0.78", "S": "#0173B2", "F": "#D55E00", "N": "#CC78BC"}
_PER_LEARNER_COLS = ["per_learner_action", "per_learner_overlap", "per_learner_subspace"]


def explode_per_learner(events: pd.DataFrame) -> pd.DataFrame:
    """Long form of the pipe-encoded per-learner columns: one row per."""
    if events is None or events.empty or "per_learner_action" not in events.columns:
        return pd.DataFrame()
    sub = events[events.per_learner_action.notna() & (events.per_learner_action != "")]
    if sub.empty:
        return pd.DataFrame()
    id_cols = [c for c in ["dataset", "variant", "model", "selector", "detector", "seed",
                           "instance_index", "event_type"] if c in sub.columns]
    rows = []
    for r in sub.itertuples(index=False):
        actions = str(getattr(r, "per_learner_action", "")).split("|")
        overlaps = loaders.parse_pipe_int(getattr(r, "per_learner_overlap", ""))
        subspaces = loaders.parse_pipe_int(getattr(r, "per_learner_subspace", ""))
        base = {c: getattr(r, c) for c in id_cols}
        for i, a in enumerate(actions):
            if not a:
                continue
            row = dict(base)
            row["learner"] = i
            row["action"] = a
            row["overlap"] = overlaps[i] if i < len(overlaps) else np.nan
            row["subspace"] = subspaces[i] if i < len(subspaces) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_learner_lanes(block: str, events: pd.DataFrame, fname_prefix: str, subdir: str,
                       drift_points_map: dict, title_prefix: str | None = None,
                       max_variants: int = 3):
    """Per-learner adaptation raster for DA-SRP: which ensemble member did."""
    long = explode_per_learner(events)
    if long.empty:
        return
    plot_utils.setup_style()
    for ds in per_dataset_targets(block, long):
        sub = long[long.dataset == ds]
        if sub.empty:
            continue
        seed = _pick_seed(sub)
        if seed is not None:
            sub = sub[sub.seed == seed]
        variants = [v for v in ordered_variants(block, sub)][:max_variants]
        if not variants:
            continue
        gt = drift_points_map.get(ds)
        fig, axes = plt.subplots(len(variants), 2, sharex=True,
                                 figsize=(12.5, 2.2 * len(variants) + 1.2), squeeze=False)
        n_learners = int(sub.learner.max()) + 1
        overlap_max = float(np.nanmax(sub.overlap.values)) if sub.overlap.notna().any() else 0.0
        scatter_for_bar = None
        for row_i, v in enumerate(variants):
            vd = sub[sub.variant == v]
            ax_a, ax_o = axes[row_i, 0], axes[row_i, 1]
            for letter, group in vd.groupby("action"):
                ax_a.scatter(group.instance_index, group.learner, s=10,
                             color=_LEARNER_ACTION_COLOR.get(letter, "C4"),
                             label=_LEARNER_ACTION_LABEL.get(letter, letter),
                             edgecolor="none", alpha=0.85)
            sc = ax_o.scatter(vd.instance_index, vd.learner, s=10, c=vd.overlap,
                              cmap="viridis", vmin=0, vmax=max(overlap_max, 1),
                              edgecolor="none", alpha=0.9)
            scatter_for_bar = sc
            for ax in (ax_a, ax_o):
                ax.set_ylim(-0.6, n_learners - 0.4)
                ax.set_yticks(range(n_learners))
                ax.set_yticklabels(range(n_learners), fontsize=6)
                plot_utils.add_drift_lines(ax, gt)
            ax_a.set_ylabel(f"{v}\nlearner", fontsize=8)
            if row_i == 0:
                ax_a.set_title("action taken", fontsize=10)
                ax_o.set_title("drifting features in learner's subspace", fontsize=10)
        for ax in axes[-1, :]:
            ax.set_xlabel("Instance #")
        plot_utils.annotate_continuous(axes[0, 0], gt)
        plot_utils.figure_legend(fig, axes[:, 0], ncol=4, title=None, y=0.965)
        if scatter_for_bar is not None:
            fig.colorbar(scatter_for_bar, ax=axes[:, 1].tolist(), label="overlap",
                         fraction=0.03, pad=0.02)
        fig.suptitle(f"{title_prefix or block}: per-learner adaptation — {ds} (seed={seed})",
                     y=0.995)
        plot_utils.save_fig(fig, f"{fname_prefix}_{ds}", subdir=subdir)


def plot_action_vs_overlap(block: str, events: pd.DataFrame, fname: str, title: str,
                           subdir: str | None = None):
    """Aggregate counterpart to plot_learner_lanes: action taken vs."""
    long = explode_per_learner(events)
    if long.empty or long.overlap.isna().all():
        return
    plot_utils.setup_style()
    long = long.copy()
    long["overlap_bucket"] = np.where(long.overlap >= 3, "3+", long.overlap.astype("Int64").astype(str))
    order = [b for b in ["0", "1", "2", "3+"] if b in set(long.overlap_bucket)]
    variants = ordered_variants(block, long)
    fig, axes = plt.subplots(1, len(variants), figsize=(3.4 * len(variants) + 1.2, 5.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    letters = [l for l in ["K", "S", "F", "N"] if l in set(long.action)]
    for ax, v in zip(axes, variants):
        vd = long[long.variant == v]
        bottom = np.zeros(len(order))
        totals = np.array([max((vd.overlap_bucket == b).sum(), 1) for b in order], dtype=float)
        for letter in letters:
            vals = np.array([((vd.overlap_bucket == b) & (vd.action == letter)).sum()
                             for b in order], dtype=float) / totals
            ax.bar(order, vals, bottom=bottom, width=0.72,
                   color=_LEARNER_ACTION_COLOR.get(letter, "C4"),
                   label=_LEARNER_ACTION_LABEL.get(letter, letter))
            bottom += vals
        ax.set_title(v, fontsize=10)
        ax.set_xlabel("drifting features\nin subspace")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of (event, learner) pairs")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.suptitle(title, y=0.985)
    plot_utils.figure_legend(fig, axes, ncol=4, y=0.93)
    plot_utils.save_fig(fig, fname, subdir=subdir)


def plot_selection_timeline(block: str, feat_sel: pd.DataFrame, fname_prefix: str,
                            subdir: str, drift_points_map: dict,
                            variant_filter=None, title_prefix: str | None = None):
    """C2: indices of the currently-selected features over time, per."""
    if feat_sel is None or len(feat_sel) == 0:
        return
    plot_utils.setup_style()
    for ds in per_dataset_targets(block, feat_sel):
        sub = feat_sel[feat_sel.dataset == ds]
        variants = ordered_variants(block, sub)
        if variant_filter is not None:
            variants = [v for v in variants if variant_filter(v)]
        if not variants:
            continue
        variants = variants[:4]
        seed = _pick_seed(sub)
        gt = drift_points_map.get(ds)
        fig, axes = plt.subplots(len(variants), 1, figsize=(11, 1.7 * len(variants) + 1.0),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]
        max_feat = 0
        for v, ax in zip(variants, axes):
            vd = sub[sub.variant == v]
            if seed is not None:
                vd = vd[vd.seed == seed]
            vd = vd.sort_values("instance_index")
            for _, r in vd.iterrows():
                feats = loaders.parse_pipe_int(r.selected_features)
                if feats:
                    ax.plot([r.instance_index] * len(feats), feats, "s",
                            color="C0", markersize=2.5)
                    max_feat = max(max_feat, max(feats))
            plot_utils.add_drift_lines(ax, gt)
            ax.set_ylabel(f"{v}\nfeat idx", fontsize=8)
        # Shade injected-noise feature band (last N indices) on every lane.
        total = max_feat + 1
        noise_idx = noise_feature_indices(ds, total)
        for ax in axes:
            ax.set_ylim(-0.5, max(max_feat + 0.5, 1))
            if noise_idx:
                ax.axhspan(min(noise_idx) - 0.5, max(noise_idx) + 0.5,
                           color="red", alpha=0.08, zorder=0)
        plot_utils.annotate_continuous(axes[0], gt)
        axes[-1].set_xlabel("Instance #")
        note = " (red band = injected noise)" if noise_idx else ""
        fig.suptitle(f"{title_prefix or block}: selection timeline (seed={seed}) — {ds}{note}")
        fig.tight_layout()
        plot_utils.save_fig(fig, f"{fname_prefix}_{ds}", subdir=subdir)


def plot_causality_overlay(block: str, windows: pd.DataFrame,
                           drift_alarms: pd.DataFrame, events: pd.DataFrame,
                           dataset: str, variant: str, drift_points_map: dict,
                           fname: str, subdir: str, title_prefix: str | None = None):
    """C3: alarm -> action -> kappa recovery on a shared instance axis."""
    if windows is None or windows.empty:
        return
    wsub = windows[(windows.dataset == dataset) & (windows.variant == variant)]
    if wsub.empty or "kappa" not in wsub.columns:
        return
    plot_utils.setup_style()
    gt = drift_points_map.get(dataset)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4]})
    ax0, ax1 = axes

    kagg = wsub.groupby("end_instance", as_index=False).kappa.mean().sort_values("end_instance")
    ax0.plot(kagg.end_instance, kagg.kappa, color="C0", linewidth=1.4, label=r"$\kappa$ (window)")
    plot_utils.add_drift_lines(ax0, gt)
    if drift_alarms is not None and not drift_alarms.empty:
        asub = drift_alarms[(drift_alarms.dataset == dataset) & (drift_alarms.variant == variant)]
        seed = _pick_seed(asub)
        if seed is not None:
            asub = asub[asub.seed == seed]
        for i, x in enumerate(asub.instance_index.values):
            ax0.axvline(x, color="C3", alpha=0.35, linewidth=0.7,
                        label="detector alarm" if i == 0 else None)
    ax0.set_ylabel(r"$\kappa$")
    ax0.set_title(f"{title_prefix or block}: alarm → adaptation → recovery — {dataset} ({variant})")
    plot_utils.short_legend(ax0, ncol=2)

    if events is not None and not events.empty:
        esub = events[(events.dataset == dataset) & (events.variant == variant)]
        seed = _pick_seed(esub)
        if seed is not None:
            esub = esub[esub.seed == seed]
        cats = [c for c in _ACTION_COLS
                if c in esub.columns and pd.to_numeric(esub[c], errors="coerce").fillna(0).abs().sum() > 0]
        for lane, c in enumerate(cats):
            vals = pd.to_numeric(esub[c], errors="coerce").fillna(0)
            mask = vals > 0
            if mask.any():
                ax1.scatter(esub.instance_index[mask], [lane] * int(mask.sum()),
                            s=14, color=f"C{lane}", alpha=0.75, edgecolor="none")
        ax1.set_yticks(range(len(cats)))
        ax1.set_yticklabels([_ACTION_LABEL[c] for c in cats], fontsize=8)
        ax1.set_ylim(-0.5, max(len(cats) - 0.5, 0.5))
        plot_utils.add_drift_lines(ax1, gt)
    ax1.set_ylabel("Action")
    ax1.set_xlabel("Instance #")
    fig.tight_layout()
    plot_utils.save_fig(fig, fname, subdir=subdir)


# Statistical tests

def friedman_table(block: str, stat_tests: dict):
    if not stat_tests:
        return
    f = stat_tests.get("friedman")
    if f is None or f.empty:
        return
    cols = [c for c in ["metric", "num_algorithms", "num_datasets",
                        "statistic", "p_value", "significant"] if c in f.columns]
    df = f[cols].copy()
    df["statistic"] = df["statistic"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    df["p_value"] = df["p_value"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    df = df.set_index("metric")
    body = latex_tables.df_to_booktabs(df, ndigits=4, index_name="Metric")
    latex_tables.write_table(
        f"tab_{block.lower()}_friedman", body,
        caption=f"{block} Friedman omnibus across variants (per metric).",
        label=f"tab:{block.lower()}_friedman",
    )


def per_metric_rank_table(block: str, stat_tests: dict, metric: str):
    per = stat_tests.get("per_metric", {}).get(metric, {})
    df = per.get("avg_ranks")
    if df is None or df.empty:
        return
    df = df.sort_values("avg_rank").set_index("method")
    body = latex_tables.df_to_booktabs(df, ndigits=3, index_name="Method")
    latex_tables.write_table(
        f"tab_{block.lower()}_avg_ranks_{metric}", body,
        caption=f"{block} average ranks on {metric}.",
        label=f"tab:{block.lower()}_avg_ranks_{metric}",
    )


def wilcoxon_table(block: str, stat_tests: dict, metric: str = "kappa"):
    w = stat_tests.get("wilcoxon")
    if w is None or w.empty:
        return
    sub = w[w.metric == metric].copy() if "metric" in w.columns else w.copy()
    if sub.empty:
        return
    cols = [c for c in ["variant_a", "variant_b", "n", "statistic",
                        "p_value", "p_adjusted", "significant", "effect_size"]
            if c in sub.columns]
    sub = sub[cols].reset_index(drop=True)
    for c in ["statistic", "p_value", "p_adjusted", "effect_size"]:
        if c in sub.columns:
            sub[c] = sub[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    sub.index = range(1, len(sub) + 1)
    body = latex_tables.df_to_booktabs(sub, ndigits=4, index_name="#")
    latex_tables.write_table(
        f"tab_{block.lower()}_wilcoxon_{metric}", body,
        caption=f"{block} Wilcoxon signed-rank pairs on {metric} (with Holm-adjusted p-values).",
        label=f"tab:{block.lower()}_wilcoxon_{metric}",
    )


def nemenyi_table(block: str, stat_tests: dict, metric: str = "kappa"):
    n = stat_tests.get("nemenyi")
    if n is None or n.empty:
        return
    sub = n[n.metric == metric].copy() if "metric" in n.columns else n.copy()
    if sub.empty:
        return
    cols = [c for c in ["variant_a", "variant_b", "rank_diff",
                        "critical_difference", "significant"]
            if c in sub.columns]
    sub = sub[cols].reset_index(drop=True)
    sub.index = range(1, len(sub) + 1)
    body = latex_tables.df_to_booktabs(sub, ndigits=4, index_name="#")
    latex_tables.write_table(
        f"tab_{block.lower()}_nemenyi_{metric}", body,
        caption=f"{block} Nemenyi post-hoc on {metric}.",
        label=f"tab:{block.lower()}_nemenyi_{metric}",
    )


def export_cd_diagrams(block: str, stat_tests: dict, subdir: str = "stat_tests"):
    """Copy pre-rendered CD diagrams (svg) into the figures dir for the."""
    if not stat_tests:
        return
    import shutil

    out = config.FIGURES_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    for metric, files in stat_tests.get("per_metric", {}).items():
        if metric not in config.CD_DIAGRAM_METRICS:
            continue
        svg = files.get("cd_diagram_svg")
        if svg and svg.exists():
            dest = out / f"{block.lower()}_cd_{metric}.svg"
            shutil.copyfile(svg, dest)
            print(f"  [fig] {dest.relative_to(config.ROOT)}")
