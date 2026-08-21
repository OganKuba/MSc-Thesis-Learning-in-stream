from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from . import config


def friedman_test(kappa_matrix: pd.DataFrame):
    df = kappa_matrix.set_index("dataset") if "dataset" in kappa_matrix.columns else kappa_matrix.copy()
    df = df.dropna(how="any")
    if len(df) < 2 or df.shape[1] < 3:
        return {"chi2": np.nan, "p": np.nan, "n_datasets": len(df), "k": df.shape[1], "ranks": pd.Series(dtype=float)}
    arrays = [df[c].values for c in df.columns]
    chi2, p = stats.friedmanchisquare(*arrays)
    ranks_per_row = df.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks_per_row.mean(axis=0).sort_values()
    n = len(df)
    k = df.shape[1]
    q = _studentized_q(k, alpha=config.ALPHA)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {
        "chi2": chi2, "p": p, "n_datasets": n, "k": k,
        "ranks": avg_ranks, "cd": cd, "q": q,
    }


_STUDENTIZED_Q_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
    20: 3.544,
}


def _studentized_q(k: int, alpha: float = 0.05) -> float:
    if k in _STUDENTIZED_Q_05:
        return _STUDENTIZED_Q_05[k]
    return _STUDENTIZED_Q_05[max(_STUDENTIZED_Q_05.keys())]


def nemenyi_pairs(avg_ranks: pd.Series, cd: float):
    methods = list(avg_ranks.index)
    out = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            diff = abs(avg_ranks[a] - avg_ranks[b])
            out.append({"method_a": a, "method_b": b, "rank_diff": diff, "significant": diff > cd})
    return pd.DataFrame(out)


def wilcoxon_pairs(kappa_matrix: pd.DataFrame, alpha: float = 0.05):
    df = kappa_matrix.set_index("dataset") if "dataset" in kappa_matrix.columns else kappa_matrix.copy()
    methods = list(df.columns)
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            x, y = df[a].values, df[b].values
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 3 or np.allclose(x, y):
                p = np.nan
            else:
                try:
                    _, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    p = np.nan
            wins = int(np.sum(x > y))
            losses = int(np.sum(x < y))
            ties = int(np.sum(x == y))
            rows.append({"method_a": a, "method_b": b, "p": p,
                         "wins_a": wins, "losses_a": losses, "ties": ties,
                         "significant": (p < alpha) if not np.isnan(p) else False})
    return pd.DataFrame(rows)


def cd_diagram(avg_ranks: pd.Series, cd: float, title: str = "Critical Difference Diagram"):
    methods = list(avg_ranks.index)
    ranks = avg_ranks.values
    order = np.argsort(ranks)
    methods = [methods[i] for i in order]
    ranks = ranks[order]
    n = len(methods)
    lo = max(1, int(np.floor(min(ranks))))
    hi = max(int(np.ceil(max(ranks))), lo + 1)
    fig, ax = plt.subplots(figsize=(8.5, 2.0 + 0.25 * n))
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(0, n + 2)
    ax.invert_yaxis()
    ax.axis("off")
    ax.hlines(y=0.5, xmin=lo, xmax=hi, color="black")
    for x in range(lo, hi + 1):
        ax.vlines(x=x, ymin=0.4, ymax=0.6, color="black")
        ax.text(x, 0.25, str(x), ha="center", va="bottom", fontsize=10)
    cd_y = 0.0
    ax.hlines(y=cd_y, xmin=lo, xmax=lo + cd, color="black", lw=2)
    ax.vlines(x=lo, ymin=cd_y - 0.05, ymax=cd_y + 0.05, color="black", lw=2)
    ax.vlines(x=lo + cd, ymin=cd_y - 0.05, ymax=cd_y + 0.05, color="black", lw=2)
    ax.text(lo + cd / 2, cd_y - 0.15, f"CD = {cd:.3f}", ha="center", va="top", fontsize=10)
    half = (n + 1) // 2
    for i in range(half):
        y = 1.5 + i * 0.5
        x = ranks[i]
        ax.plot([x, x], [0.5, y], color="black", lw=1)
        ax.plot([x, lo - 0.1], [y, y], color="black", lw=1)
        ax.text(lo - 0.15, y, f"{methods[i]} ({ranks[i]:.2f})", ha="right", va="center", fontsize=10)
    for j, i in enumerate(range(half, n)):
        y = 1.5 + j * 0.5
        x = ranks[i]
        ax.plot([x, x], [0.5, y], color="black", lw=1)
        ax.plot([x, hi + 0.1], [y, y], color="black", lw=1)
        ax.text(hi + 0.15, y, f"{methods[i]} ({ranks[i]:.2f})", ha="left", va="center", fontsize=10)
    cliques = _find_cliques(ranks, cd)
    line_y = 0.7
    for (i, j) in cliques:
        ax.hlines(y=line_y, xmin=ranks[i] - 0.02, xmax=ranks[j] + 0.02, color="red", lw=3)
        line_y += 0.15
    ax.set_title(title, fontsize=12)
    return fig


def _find_cliques(ranks, cd):
    n = len(ranks)
    cliques = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (ranks[j + 1] - ranks[i]) < cd:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    deduped = []
    for a, b in cliques:
        if not any(a >= x and b <= y for (x, y) in deduped):
            deduped = [(x, y) for (x, y) in deduped if not (x >= a and y <= b)]
            deduped.append((a, b))
    return deduped
