"""Shared helpers for per-block (E1..E5) analyses against the unified runner output."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import seaborn as sns

from . import config, latex_tables, loaders, plot_utils


# --- Ordering ------------------------------------------------------------

def ordered_datasets(df, col="dataset"):
    if df is None or col not in df.columns:
        return []
    present = [d for d in config.DATASET_ORDER if d in df[col].unique()]
    extras = [d for d in df[col].unique() if d not in present]
    return present + extras


def per_dataset_targets(block: str, df, col="dataset"):
    """
    Datasets for which a per-dataset figure should be produced.

    The five generators that emit one file per dataset (window timeseries, alarm timeline,
    adaptation timeline, selection timeline, noise-annotated importance) previously iterated over
    every dataset in the block, which is where most of the unused output came from. They now ask
    here instead; the whitelist lives in config.PER_DATASET_FIGURES and mirrors what the thesis
    cites. A block missing from the config keeps the old behaviour (all datasets).
    """
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


# --- Summary pivots ------------------------------------------------------

def drop_saturated(block: str, df: pd.DataFrame, metric_col: str,
                   dataset_col: str = "dataset") -> pd.DataFrame:
    """Drop datasets that are saturated (kappa/accuracy ~1.0) for this block/metric (B3).

    Only affects kappa/accuracy on E1/E2/E3 (removes generic STAGGER); a no-op for
    every other metric, block, and for temporal_kappa / high-dynamics blocks.
    """
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
    """Pivot a summary CSV (long form) on dataset x variant for any *_mean metric."""
    summary = drop_saturated(block, summary, metric_col)
    rows = ordered_datasets(summary, index) if index == "dataset" else ordered_variants(block, summary, index)
    cols = ordered_variants(block, summary, columns) if columns == "variant" else ordered_datasets(summary, columns)
    pv = plot_utils.safe_pivot(
        summary, index=index, columns=columns, values=metric_col,
        reindex_rows=rows, reindex_cols=cols,
    )
    return pv


# --- Window helpers ------------------------------------------------------

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


# --- LaTeX writers -------------------------------------------------------

def write_metric_table(name: str, pv: pd.DataFrame, caption: str, label: str,
                       ndigits: int = 3, bold_max_per_row: bool = True):
    if pv is None or pv.empty:
        return
    body = latex_tables.df_to_booktabs(
        pv, ndigits=ndigits, bold_max_per_row=bold_max_per_row,
        index_name=pv.index.name or "",
    )
    latex_tables.write_table(name, body, caption=caption, label=label)


# --- Common plots --------------------------------------------------------

def plot_metric_bar(block: str, summary: pd.DataFrame, metric_col: str,
                    fname: str, ylabel: str, title: str, subdir: str | None = None):
    if summary is None or metric_col not in summary.columns:
        return
    plot_utils.setup_style()
    df = drop_saturated(block, summary, metric_col).copy()
    df["dataset"] = pd.Categorical(df.dataset, categories=ordered_datasets(df), ordered=True)
    df["variant"] = pd.Categorical(df.variant, categories=ordered_variants(block, df), ordered=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df, x="dataset", y=metric_col, hue="variant", ax=ax, errorbar=None)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plot_utils.legend_below(ax, ncol=2, title="Variant")
    fig.tight_layout()
    plot_utils.save_fig(fig, fname, subdir=subdir)


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
        for v in variants:
            vd = sub[sub.variant == v]
            agg = vd.groupby("end_instance", as_index=False)[metric].mean().sort_values("end_instance")
            if agg.empty:
                continue
            ax.plot(agg.end_instance, agg[metric], label=v, linewidth=1.2)
        drift_pts = drift_points_map.get(ds)
        plot_utils.add_drift_lines(ax, drift_pts)
        plot_utils.annotate_continuous(ax, drift_pts)
        ax.set_xlabel("Instance #")
        ax.set_ylabel(ylabel or metric)
        ax.set_title(f"{title_prefix or block}: {metric} over time — {ds}")
        plot_utils.short_legend(ax, ncol=2, title="Variant")
        fig.tight_layout()
        plot_utils.save_fig(fig, f"{fname_prefix}_{ds}", subdir=subdir)


# --- Drift alarms --------------------------------------------------------

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
    # Log scale, because the counts span three orders of magnitude: a degenerate baseline can
    # fire >1000 times on a real stream (Majority/NYCTaxi = 1329) while a learner on a
    # synthetic stream fires 5. On a linear axis shared by all datasets the synthetic bars are
    # 0.4% of the plot height — they render as a flat line at zero and the figure reads as
    # "no drift was ever detected on the synthetic data", which is not what happened.
    positive = counts.alarm_count[counts.alarm_count > 0]
    if not positive.empty:
        ax.set_yscale("log")
        # Bars are drawn from 0, which has no place on a log axis; matplotlib clips them at the
        # lower limit, so put that limit just under the smallest real value instead of letting
        # it default to something that swallows the shortest bar.
        ax.set_ylim(bottom=max(float(positive.min()) * 0.6, 1e-3))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean alarms per run (log scale)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    # Further down than the default: the rotated dataset names plus the x-label need the room,
    # and at -0.24 the legend box sat on top of "Dataset".
    plot_utils.legend_below(ax, ncol=2, title="Variant", y=-0.34)
    fig.tight_layout()
    plot_utils.save_fig(fig, fname, subdir=subdir)


# An alarm is called "useful" when the window accuracy one window later is at least this much
# higher than the window accuracy at alarm time. 1pp is above the noise floor of a
# 1000-instance window (~0.3pp s.e. at acc 0.9) without demanding a dramatic recovery.
ALARM_USEFUL_DELTA = 0.01


def alarm_delta(drift_alarms: pd.DataFrame) -> pd.DataFrame:
    """Add `delta_acc` = accuracy one window AFTER the alarm minus accuracy AT the alarm.

    Both columns come from RunDetailedRecorder, which back-fills `window_accuracy_after`
    once `windowSize` further instances have been seen. The difference is what the alarm
    actually bought: ~0 means the detector fired without anything improving (a false alarm,
    or an adaptation that did not pay off), clearly positive means the reset recovered
    accuracy. Returns an empty frame when the columns are absent.
    """
    need = {"window_accuracy_before", "window_accuracy_after"}
    if drift_alarms is None or drift_alarms.empty or not need.issubset(drift_alarms.columns):
        return pd.DataFrame()
    df = drift_alarms.copy()
    df["delta_acc"] = (pd.to_numeric(df.window_accuracy_after, errors="coerce")
                       - pd.to_numeric(df.window_accuracy_before, errors="coerce"))
    return df.dropna(subset=["delta_acc"])


def plot_alarm_timeline(block: str, drift_alarms: pd.DataFrame, fname_prefix: str,
                        subdir: str, drift_points_map: dict, title_prefix: str | None = None):
    """One subplot per variant: every alarm as a stem whose height is the accuracy it recovered.

    The old version drew each alarm as a bare vertical line, which said only "the detector
    fired here" — information the aggregate e*_drift_alarm_counts already carries. Here the
    stem height is `delta_acc` (see alarm_delta), so a detector that fires constantly without
    recovering anything shows up as a flat row of ticks on the zero line, while a detector
    whose alarms precede a real recovery shows tall green stems. Markers sit on the zero line
    too, so alarm timing stays visible whatever the height.
    """
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
        fig, axes = plt.subplots(len(variants), 1, figsize=(11, 1.5 * len(variants) + 1.2),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes[:, 0]
        for ax, v in zip(axes, variants):
            vd = sub[sub.variant == v].sort_values("instance_index")
            x = vd.instance_index.values
            d = vd.delta_acc.values
            colors = np.where(d >= ALARM_USEFUL_DELTA, "C2",
                              np.where(d <= -ALARM_USEFUL_DELTA, "C3", "0.6"))
            ax.vlines(x, 0.0, d, colors=colors, linewidth=1.0, alpha=0.85)
            ax.scatter(x, np.zeros_like(d), s=6, color="C0", zorder=3, edgecolor="none")
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
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
    """Aggregate companion to plot_alarm_timeline: distribution of delta_acc per variant.

    Pools every alarm of every dataset and seed. The box shows how much accuracy an alarm
    typically recovers; the number above it is the share of alarms that recovered at least
    ALARM_USEFUL_DELTA. A detector sitting on 0 with a low percentage is paying the cost of
    an adaptation for nothing.
    """
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
    ax.axhline(ALARM_USEFUL_DELTA, color="C2", linewidth=0.8, linestyle=":", alpha=0.8)
    # Headroom for the per-variant labels, which sit inside the axes so they cannot collide
    # with the title (x in data coords, y in axes coords).
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
    """Quantitative anchor for plot_alarm_effectiveness, per (variant, detector)."""
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


# --- Recovery time -------------------------------------------------------

def recovery_table(block: str, recovery: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-drift recovery stats per (dataset, variant).

    `recovery_length` is kept for backward compatibility but is degenerate
    (≈1 window everywhere: the recovery fits inside one 1000-instance window).
    The informative columns are `mean_max_drop` (how far accuracy fell after a
    drift) and `mean_area` (area under the recovery curve = drop × duration).
    """
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


# Recovery depth metrics we prefer over the degenerate recovery_length (B1).
RECOVERY_METRICS = [
    ("mean_max_drop", "Mean max accuracy drop after drift"),
    ("mean_area", "Mean area under recovery curve"),
]


def plot_recovery(block: str, recovery: pd.DataFrame, fname: str, title: str,
                  subdir: str | None = None):
    """Recovery *depth* per (dataset, variant): max accuracy drop + area under curve.

    Replaces the old single-panel recovery-length bar (which was ~1.0 for every
    variant and carried no signal).
    """
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


# --- Feature selections --------------------------------------------------

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
    """True when the selection never actually changes (only 'initial' triggers).

    In such blocks (E1, and the static baselines in E4/E5) stability≡1.0 and the
    selected-count is constant, so the overview panels carry no signal (B2).
    """
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


# --- Feature importance --------------------------------------------------

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
    """Feature importance per (dataset, feature index), with the injected noise features marked.

    Replaces an earlier version that pivoted variant x feature_index and pooled every dataset
    into one mean. That version was uninterpretable for four independent reasons:

    1. Feature index *i* denotes a different variable in every stream (SEA has 8 features,
       YahooFinance 36), so averaging across datasets adds unlike quantities.
    2. Importance is normalised to sum to 1 per snapshot, so its scale is ~1/d. Low-dimensional
       datasets (SEA: 1/8) automatically outweighed high-dimensional ones (YahooFinance: 1/36).
    3. Snapshots are alarm-triggered, so the pooled mean was weighted by alarm frequency —
       NYCTaxi contributed 1955 snapshots against LED's 33.
    4. The variant axis carried no information: the importance estimator is fed the stream, not
       the model. At a given instance every variant reports bit-identical importance, including
       Majority and NoChange, which do not learn at all. Row-to-row differences were purely an
       artefact of each variant having a different number of alarm-triggered snapshots.

    Cells are therefore reported **relative to the uniform baseline** (importance * d, so 1.0
    means "carries exactly its share"), which is comparable across streams of different width.
    Values are pooled over variants, seeds and snapshots, which is legitimate precisely because
    of point 4. A dataset that has no feature *i* leaves the cell blank.
    """
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
        # importance * d: 1.0 = the feature carries exactly a uniform share of the total.
        rows[ds] = mean_imp.reindex(range(width)) * width
        idx = noise_feature_indices(ds, width)
        if idx:
            noise_spans[ds] = (min(idx), max(idx))
    if not rows:
        return
    pv = pd.DataFrame(rows).T.reindex([d for d in datasets if d in rows])

    n_col = pv.shape[1]
    fig, ax = plt.subplots(figsize=(min(16, 0.42 * n_col + 4.5), 0.5 * len(pv.index) + 2.4))
    # Asymmetric diverging scale pinned at the uniform share: 0 -> blue, 1.0 -> white,
    # top -> red. The upper end is clipped at the 98th percentile because a single feature can
    # reach 4x uniform (YahooFinance/20) and would otherwise wash out every other cell.
    vmax = float(np.nanquantile(pv.values, 0.98))
    vmax = max(vmax, 1.2)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
    sns.heatmap(pv, cmap="vlag", norm=norm, ax=ax,
                annot=n_col <= 20, fmt=".1f",
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "importance relative to uniform (1.0 = 1/d)",
                          "extend": "max"})
    # Outline the injected-noise block of each stream: the claim these figures support is that
    # the ranker pushes importance away from it, and the range differs per dataset.
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
    """Indices of the injected noise features for a synthetic dataset (B4).

    Noise columns are appended last, so they are the final `NOISE_FEATURES[ds]`
    indices of the `total_features`-wide space. Returns [] when the dataset has
    no known noise ground truth (real ARFFs, STAGGER, noise-free variants).
    """
    n = config.NOISE_FEATURES.get(dataset, 0)
    if n <= 0 or total_features <= 0:
        return []
    n = min(n, total_features)
    return list(range(total_features - n, total_features))


def plot_importance_noise_annotated(block: str, feat_imp: pd.DataFrame,
                                    fname_prefix: str, subdir: str,
                                    title_prefix: str | None = None):
    """Per-synthetic-dataset importance heatmap (variant × feature) with the
    injected noise features highlighted (B4).

    One figure per dataset that has a known noise ground truth; noise columns get
    red tick labels and a hatched overlay so it is visible whether the adaptive
    selectors push importance/selection away from the noise features.
    """
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


# --- Adaptation events ---------------------------------------------------

def adaptation_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame()
    cols = ["kept_count", "surgical_count", "full_replacement_count",
            "no_replacement_count", "ext_keep_count", "ext_full_count"]
    cols = [c for c in cols if c in events.columns]
    agg = (events.groupby(["dataset", "variant"], as_index=False)[cols].sum())
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
        vd = agg[agg.variant == v].set_index("dataset").reindex(datasets)
        vd = vd[cats].fillna(0)
        totals = vd.sum(axis=1).replace(0, np.nan)
        prop = vd.div(totals, axis=0).fillna(0)
        bottom = np.zeros(len(prop))
        for color, c in zip(palette, cats):
            ax.bar(prop.index.astype(str), prop[c].values, bottom=bottom,
                   label=labels[c], color=color)
            bottom += prop[c].values
        ax.set_title(v)
        ax.set_ylabel("Proportion")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plot_utils.figure_legend(fig, axes, ncol=min(6, len(cats)), y=0.91)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    plot_utils.save_fig(fig, fname, subdir=subdir)


# --- Time-resolved diagnostics (C1-C4) -----------------------------------

# Per-learner adaptation action columns and their display labels / colours.
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
    """C1: per-event KEEP/SURGICAL/FULL/... on the instance axis (not summed).

    One figure per dataset, one subplot per DA variant. Each adaptation event is a
    marker placed at its `instance_index` in the lane of every action it performed
    (marker area ~ how many learners took that action). GT drifts are dashed lines.
    """
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


# --- Per-learner view of DA-SRP adaptations ------------------------------

# Letters written by RunDetailedRecorder.encodeActions.
_LEARNER_ACTION_LABEL = {"K": "KEEP", "S": "SURGICAL", "F": "FULL", "N": "NO_REPL"}
# Explicit hexes, not the C0..C3 cycle: in the colorblind palette SURGICAL and FULL both came
# out orange, and telling a targeted swap from a full reset is the entire point of the figure.
# Blue / vermillion / pink are separable for the common colour-vision deficiencies.
_LEARNER_ACTION_COLOR = {"K": "0.78", "S": "#0173B2", "F": "#D55E00", "N": "#CC78BC"}
_PER_LEARNER_COLS = ["per_learner_action", "per_learner_overlap", "per_learner_subspace"]


def explode_per_learner(events: pd.DataFrame) -> pd.DataFrame:
    """Long form of the pipe-encoded per-learner columns: one row per (event, learner).

    Columns added: `learner`, `action` (letter), `overlap` (drifting features inside that
    learner's subspace), `subspace` (its size). Returns an empty frame when the columns are
    missing, which is the case for every CSV produced before those columns were added — the
    callers then simply skip the figure instead of failing.
    """
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
    """Per-learner adaptation raster for DA-SRP: which ensemble member did what, and why.

    Left column: one lane per ensemble member, a marker at every adaptation event coloured by
    the action that member took. Right column: the same grid coloured by `overlap` — how many
    of the features that had just drifted were inside that member's random subspace. Reading
    the two side by side answers the question the aggregate counts cannot: a surgical swap
    should land exactly on the members whose subspace contains a drifting feature, and the
    members with overlap 0 should be left alone.
    """
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
    """Aggregate counterpart to plot_learner_lanes: action taken vs subspace overlap.

    Pools every (event, learner) pair over datasets and seeds and shows, per variant, how the
    action distribution changes with the number of drifting features inside the learner's
    subspace. This is the quantitative form of the component-B claim: overlap 0 should be
    dominated by KEEP, and the SURGICAL share should rise with the overlap.
    """
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
    """C2: indices of the currently-selected features over time, per variant.

    Generalises the E2 selection timeline: works for any variant list (e.g.
    DA-SRP-ABC in E3), draws GT drift lines and shades the injected-noise region so
    one can see whether the selector keeps drifting into / out of noise features.
    """
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
    """C3: alarm -> action -> kappa recovery on a shared instance axis.

    Top panel: window kappa for `variant` (mean over seeds) with GT drift lines and
    detector alarms as vertical marks. Bottom panel: adaptation actions raster for
    one seed. Reads windows + drift_alarms + adaptation_events; shows the causal
    chain detector-fires -> ensemble-adapts -> accuracy-recovers.
    """
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


# --- Statistical tests ---------------------------------------------------

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
    """Copy pre-rendered CD diagrams (svg) into the figures dir for the block."""
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
