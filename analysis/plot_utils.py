from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from . import config


def setup_style():
    sns.set_theme(style=config.SNS_STYLE, palette=config.PALETTE, font_scale=config.FONT_SCALE)
    plt.rcParams.update(config.PLOT_RC)


def save_fig(fig, name: str, subdir: str | None = None):
    """Write the figure in each format listed in config.FIGURE_FORMATS (default: pdf only)."""
    out_dir = config.FIGURES_DIR if subdir is None else config.FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in config.FIGURE_FORMATS:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=config.FIG_DPI) if fmt == "png" else fig.savefig(path)
        written.append(path)
    plt.close(fig)
    if written:
        print(f"  [fig] {written[0].relative_to(config.ROOT)}")


def add_drift_lines(ax, drift_points, ymin=None, ymax=None, color="grey", alpha=0.4, ls="--"):
    if drift_points is None or drift_points == "continuous":
        return
    for x in drift_points:
        ax.axvline(x=x, color=color, alpha=alpha, linestyle=ls, linewidth=0.9)


def annotate_continuous(ax, drift_points):
    if drift_points == "continuous":
        ax.text(0.99, 0.02, "continuous drift", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))


def short_legend(ax, ncol=1, loc="best", title=None, bbox_to_anchor=None):
    leg = ax.legend(loc=loc, ncol=ncol, title=title, bbox_to_anchor=bbox_to_anchor,
                    frameon=True, framealpha=0.85)
    if leg is not None:
        leg.get_frame().set_linewidth(0.5)


def legend_below(ax, ncol=2, title=None, y=-0.24):
    """Legend under the axes. `y` is the offset in axes coords — push it further down when the
    axes carry rotated tick labels plus an x-label, which the default offset overlaps."""
    return short_legend(ax, ncol=ncol, loc="upper center", title=title,
                        bbox_to_anchor=(0.5, y))


def figure_legend(fig, axes, ncol=2, title=None, y=0.93):
    axes = list(axes if isinstance(axes, (list, tuple)) else axes.ravel())
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    if not handles:
        return None
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=ncol, title=title, frameon=True, framealpha=0.9)
    leg.get_frame().set_linewidth(0.5)
    return leg


def safe_pivot(df, index, columns, values, agg="mean", reindex_rows=None, reindex_cols=None):
    pv = df.pivot_table(index=index, columns=columns, values=values, aggfunc=agg)
    if reindex_rows is not None:
        keep = [r for r in reindex_rows if r in pv.index]
        pv = pv.reindex(keep)
    if reindex_cols is not None:
        keep = [c for c in reindex_cols if c in pv.columns]
        pv = pv.reindex(columns=keep)
    return pv
