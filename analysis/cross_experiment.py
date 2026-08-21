from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, loaders, latex_tables, plot_utils, block_utils


def _classify_dataset(name: str) -> str:
    base = name.rsplit("-", 1)[0]
    if base in config.SYNTHETIC_DATASETS or name in config.SYNTHETIC_DATASETS:
        return "synthetic"
    if name in config.REAL_DATASETS:
        return "real"
    return "other"


def _ensure_master(master: pd.DataFrame | None,
                   per_block: dict[str, dict]) -> pd.DataFrame:
    """Use master_summary.csv if available, otherwise concat per-block summaries."""
    if master is not None and len(master):
        return master.copy()
    parts = []
    for block, data in per_block.items():
        s = data.get("summary")
        if s is None or len(s) == 0:
            continue
        df = s.copy()
        if "block" not in df.columns:
            df["block"] = block
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def table_best_methods(master: pd.DataFrame, e1_summary: pd.DataFrame | None):
    if master.empty:
        return
    df = master.copy()
    if "kappa_mean" not in df.columns:
        return

    base = pd.Series(dtype=float)
    if e1_summary is not None and "variant" in e1_summary.columns:
        ht = e1_summary[e1_summary.variant == "HT"]
        base = ht.groupby("dataset").kappa_mean.mean()

    rows = []
    for ds, g in df.groupby("dataset"):
        idx = g.kappa_mean.idxmax()
        if pd.isna(idx):
            continue
        best = g.loc[idx]
        delta = best.kappa_mean - base.get(ds, np.nan) if len(base) else np.nan
        rows.append({
            "Dataset": ds,
            "Best variant": best.variant,
            "Block": best.get("block", "?"),
            "kappa": best.kappa_mean,
            "Delta vs HT": delta,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return
    order = [d for d in config.DATASET_ORDER if d in out.Dataset.unique()] + \
            [d for d in out.Dataset.unique() if d not in config.DATASET_ORDER]
    out["Dataset"] = pd.Categorical(out.Dataset, categories=order, ordered=True)
    out = out.sort_values("Dataset").set_index("Dataset")
    body = latex_tables.df_to_booktabs(out, ndigits=3, index_name="Dataset")
    latex_tables.write_table(
        "tab_cross_best_methods", body,
        caption=r"Best variant per dataset across E1--E5 with $\Delta\kappa$ vs raw HT baseline.",
        label="tab:cross_best_methods",
    )


def table_resource_vs_kappa(master: pd.DataFrame):
    if master.empty or "ram_hours_gb_mean" not in master.columns:
        return
    agg = master.groupby("variant").agg(
        kappa=("kappa_mean", "mean"),
        ram=("ram_hours_gb_mean", "mean"),
        thr=("throughput_mean", "mean"),
        n=("kappa_mean", "size"),
    ).reset_index().sort_values("kappa", ascending=False)
    # Model-size RAM-Hours land around 1e-6 GB-h and would print as 0.0000; rescale
    # so the column carries information (multiplier stated in the caption).
    agg["ram"] = agg["ram"] * config.RAMH_SCALE
    # Pre-formatted so the counts and the throughput do not inherit the 4 decimals that
    # kappa and the (small) RAM-Hours need.
    agg["thr"] = agg["thr"].map(lambda x: "-" if pd.isna(x) else f"{x:.0f}")
    agg["n"] = agg["n"].map(lambda x: f"{int(x)}")
    body = latex_tables.df_to_booktabs(agg.set_index("variant"),
                                       ndigits=4, index_name="Variant")
    latex_tables.write_table(
        "tab_cross_resource_vs_kappa", body,
        caption=(
            r"Cross-experiment: mean $\kappa$, RAM-Hours (in units of " + config.RAMH_UNIT_TEX
            + r", column \emph{ram}) and throughput (instances/sec) per variant "
            r"(n = entries in master\_summary)."
        ),
        label="tab:cross_resource_vs_kappa",
    )


def plot_pareto(master: pd.DataFrame):
    plot_utils.setup_style()
    if master.empty or "ram_hours_gb_mean" not in master.columns:
        return
    agg = master.groupby("variant").agg(
        kappa=("kappa_mean", "mean"),
        ram=("ram_hours_gb_mean", "mean"),
    ).reset_index()
    agg = agg[agg.ram > 0]
    if len(agg) == 0:
        return
    agg["ram"] = agg["ram"] * config.RAMH_SCALE  # same units as tab_cross_resource_vs_kappa
    pareto = []
    sorted_df = agg.sort_values("ram")
    best_kappa = -np.inf
    for _, r in sorted_df.iterrows():
        if r.kappa > best_kappa:
            pareto.append(r.variant)
            best_kappa = r.kappa
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette(config.PALETTE, n_colors=len(agg))
    for color, (_, r) in zip(palette, agg.iterrows()):
        marker = "*" if r.variant in pareto else "o"
        size = 200 if r.variant in pareto else 90
        ax.scatter(r.ram, r.kappa, s=size, marker=marker, color=color,
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(r.variant, (r.ram, r.kappa), fontsize=8, alpha=0.9,
                    xytext=(5, 4), textcoords="offset points")
    front = agg[agg.variant.isin(pareto)].sort_values("ram")
    ax.plot(front.ram, front.kappa, color="grey", linestyle="--", alpha=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"RAM-Hours ($10^{-6}$ GB$\cdot$h, log scale)")
    ax.set_ylabel(r"Mean $\kappa$")
    ax.set_title(r"Cross-experiment Pareto: $\kappa$ vs RAM-Hours (★ = Pareto-optimal)")
    fig.tight_layout()
    plot_utils.save_fig(fig, "cross_pareto_front")


def plot_synthetic_vs_real(master: pd.DataFrame):
    plot_utils.setup_style()
    if master.empty:
        return
    df = master.copy()
    df["type"] = df.dataset.map(_classify_dataset)
    agg = df.groupby(["variant", "type"]).kappa_mean.mean().unstack("type")
    if "synthetic" not in agg.columns or "real" not in agg.columns:
        return
    agg = agg.dropna()
    if len(agg) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette(config.PALETTE, n_colors=len(agg))
    for color, (m, r) in zip(palette, agg.iterrows()):
        ax.scatter(r["synthetic"], r["real"], s=120, color=color,
                   edgecolor="black", linewidth=0.5, label=m)
    lim_lo = min(agg.min().min(), 0)
    lim_hi = max(agg.max().max(), 1)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, label="equal")
    ax.set_xlabel(r"Mean $\kappa$ on synthetic")
    ax.set_ylabel(r"Mean $\kappa$ on real")
    ax.set_title("Cross-experiment generalization: synthetic vs real")
    plot_utils.legend_below(ax, ncol=4, title="Variant")
    fig.tight_layout()
    plot_utils.save_fig(fig, "cross_synthetic_vs_real")


def plot_rq_summary(per_block: dict[str, dict]):
    plot_utils.setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0][0]
    e1 = per_block["E1"].get("summary")
    if e1 is not None and len(e1):
        df = e1.copy()
        df["dataset"] = pd.Categorical(df.dataset,
                                       categories=block_utils.ordered_datasets(df),
                                       ordered=True)
        df["variant"] = pd.Categorical(df.variant,
                                       categories=block_utils.ordered_variants("E1", df),
                                       ordered=True)
        sns.barplot(data=df, x="dataset", y="kappa_mean", hue="variant", ax=ax, errorbar=None)
        ax.set_title("RQ1: baselines (E1)")
        ax.set_ylabel(r"$\kappa$")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plot_utils.legend_below(ax, ncol=2, title="Variant")

    ax = axes[0][1]
    e2 = per_block["E2"].get("summary")
    if e2 is not None and len(e2):
        df = e2.copy()
        s1 = df[df.variant.str.endswith("S1")].groupby("dataset").kappa_mean.max()
        adp = df[df.variant.str.contains(r"\+S[234]$", regex=True)].groupby("dataset").kappa_mean.max()
        common = [d for d in config.DATASET_ORDER if d in s1.index and d in adp.index]
        x = np.arange(len(common))
        ax.bar(x - 0.18, [s1[d] for d in common], width=0.35, label="best S1")
        ax.bar(x + 0.18, [adp[d] for d in common], width=0.35, label="best S2/S3/S4")
        ax.set_xticks(x)
        ax.set_xticklabels(common, rotation=30, ha="right")
        ax.set_ylabel(r"$\kappa$")
        ax.set_title("RQ2: adaptive vs static (E2)")
        plot_utils.legend_below(ax, ncol=2)

    ax = axes[0][2]
    e3 = per_block["E3"].get("summary")
    if e3 is not None and len(e3):
        means = e3.groupby("variant").kappa_mean.mean()
        order = [v for v in config.E3_VARIANT_ORDER if v in means.index]
        ax.bar(order, means[order],
               color=sns.color_palette(config.PALETTE, len(order)))
        ax.set_ylabel(r"Mean $\kappa$ across datasets")
        ax.set_title("RQ3: DA-SRP / DA-ARF ablation (E3)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    ax = axes[1][0]
    e4 = per_block["E4"].get("summary")
    if e4 is not None and len(e4):
        df = e4.copy()
        parts = df.dataset.astype(str).str.rsplit("-", n=1, expand=True)
        df["generator"] = parts[0]
        df["dynamics"] = parts[1] if parts.shape[1] > 1 else ""
        means = (df.groupby(["variant", "dynamics"]).kappa_mean.mean()
                 .reset_index())
        for v in [v for v in config.E4_VARIANT_ORDER if v in means.variant.unique()]:
            sub = means[means.variant == v].copy()
            sub["dyn_i"] = sub.dynamics.map({k: i for i, k in enumerate(config.DYNAMICS_ORDER)})
            sub = sub.sort_values("dyn_i")
            ax.plot(sub.dyn_i, sub.kappa_mean, "o-", label=v, linewidth=1.1)
        ax.set_xticks(range(len(config.DYNAMICS_ORDER)))
        ax.set_xticklabels(config.DYNAMICS_ORDER)
        ax.set_ylabel(r"Mean $\kappa$")
        ax.set_title("RQ4: kappa vs dynamics (E4)")
        plot_utils.legend_below(ax, ncol=2)

    ax = axes[1][1]
    e5 = per_block["E5"].get("summary")
    if e5 is not None and len(e5):
        agg = e5.groupby("variant").kappa_mean.mean().sort_values(ascending=False)
        ax.barh(agg.index, agg.values,
                color=sns.color_palette(config.PALETTE, len(agg)))
        ax.invert_yaxis()
        ax.set_xlabel(r"Mean $\kappa$")
        ax.set_title("RQ5: detector variants (E5)")

    ax = axes[1][2]
    rows = []
    for tag, df in [
        ("E1-HT", e1), ("E1-ARF", e1), ("E1-SRP", e1),
        ("E1-HT+S1", e1), ("E1-ARF+S1", e1), ("E1-SRP+S1", e1),
        ("E2-best", e2),
        ("E3-DA-SRP-A", e3), ("E3-DA-SRP-ABC", e3),
        ("E4-best", e4), ("E5-best", e5),
    ]:
        if df is None or len(df) == 0:
            continue
        if tag.startswith("E1-"):
            v = tag.split("-", 1)[1]
            rows.append({"label": tag, "kappa": df[df.variant == v].kappa_mean.mean()})
        elif tag == "E2-best":
            rows.append({"label": tag, "kappa": df.groupby("dataset").kappa_mean.max().mean()})
        elif tag.startswith("E3-"):
            v = tag.split("-", 1)[1]
            rows.append({"label": tag, "kappa": df[df.variant == v].kappa_mean.mean()})
        elif tag in {"E4-best", "E5-best"}:
            rows.append({"label": tag, "kappa": df.groupby("dataset").kappa_mean.max().mean()})
    if rows:
        s = pd.DataFrame(rows)
        ax.bar(s.label, s.kappa,
               color=sns.color_palette(config.PALETTE, len(s)))
        ax.set_ylabel(r"Mean $\kappa$")
        ax.set_title("Summary across experiments")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    fig.suptitle("Research questions summary", fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.12,
                        hspace=0.95, wspace=0.35)
    plot_utils.save_fig(fig, "cross_rq_summary")


def run():
    print("\n[Cross] Cross-experiment synthesis")
    per_block = {
        "E1": loaders.load_e1(),
        "E2": loaders.load_e2(),
        "E3": loaders.load_e3(),
        "E4": loaders.load_e4(),
        "E5": loaders.load_e5(),
    }
    master = loaders.load_master_summary()
    df = _ensure_master(master, per_block)
    table_best_methods(df, per_block["E1"].get("summary"))
    table_resource_vs_kappa(df)
    plot_pareto(df)
    plot_synthetic_vs_real(df)
    plot_rq_summary(per_block)


if __name__ == "__main__":
    run()
