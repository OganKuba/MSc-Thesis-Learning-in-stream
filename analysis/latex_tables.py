from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from . import config


def _escape(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#")
             .replace("_", r"\_"))


def _fmt_num(x, ndigits=3, dash="-"):
    if x is None:
        return dash
    try:
        if isinstance(x, str):
            return _escape(x)
        if np.isnan(x):
            return dash
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if x != 0.0 and abs(x) < 0.5 * 10.0 ** (-ndigits):
            return f"{x:.1e}"
        return f"{x:.{ndigits}f}"
    except Exception:
        return _escape(x)


def _bold(s):
    return r"\textbf{" + s + "}"


def write_table(name: str, body: str, caption: str, label: str, position: str = "ht"):
    path = config.TABLES_DIR / f"{name}.tex"
    full = (
        f"\\begin{{table}}[{position}]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        f"\\end{{table}}\n"
    )
    path.write_text(full)
    print(f"  [tab] {path.relative_to(config.ROOT)}")


def df_to_booktabs(df: pd.DataFrame, ndigits=3, bold_max_per_row=False, index_name=None,
                   bold_max_per_col=False) -> str:
    """Booktabs table. Bolding marks the winner along one axis - pick the."""
    df = df.copy()
    if index_name is not None:
        df.index.name = index_name
    if bold_max_per_row and bold_max_per_col:
        raise ValueError("bold_max_per_row and bold_max_per_col are mutually exclusive")
    cols = list(df.columns)
    col_max = {}
    if bold_max_per_col:
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            col_max[c] = None if vals.isna().all() else vals.idxmax()
    align = "l" + "c" * len(cols)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    header_cells = [_escape(df.index.name or "")] + [_escape(c) for c in cols]
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")
    for idx, row in df.iterrows():
        cells = [_escape(idx)]
        if bold_max_per_row:
            numeric_vals = []
            for v in row:
                try:
                    numeric_vals.append(float(v))
                except Exception:
                    numeric_vals.append(np.nan)
            arr = np.array(numeric_vals, dtype=float)
            if np.all(np.isnan(arr)):
                max_idx = -1
            else:
                max_idx = int(np.nanargmax(arr))
        else:
            max_idx = -1
        for j, v in enumerate(row):
            cell = _fmt_num(v, ndigits=ndigits)
            if j == max_idx and bold_max_per_row:
                cell = _bold(cell)
            elif bold_max_per_col and col_max.get(cols[j]) == idx:
                cell = _bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def matrix_table_with_baselines(
    df: pd.DataFrame, baseline_rows: dict | None = None,
    ndigits=3, bold_max_per_row=True, index_name=None,
) -> str:
    df = df.copy()
    if index_name is not None:
        df.index.name = index_name
    cols = list(df.columns)
    align = "l" + "c" * len(cols)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join([_escape(df.index.name or "")] + [_escape(c) for c in cols]) + " \\\\")
    lines.append("\\midrule")
    for idx, row in df.iterrows():
        cells = [_escape(idx)]
        arr = np.array([float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan for v in row.values], dtype=float)
        max_idx = -1 if np.all(np.isnan(arr)) else int(np.nanargmax(arr))
        for j, v in enumerate(row):
            txt = _fmt_num(v, ndigits=ndigits)
            if bold_max_per_row and j == max_idx:
                txt = _bold(txt)
            cells.append(txt)
        lines.append(" & ".join(cells) + " \\\\")
    if baseline_rows:
        lines.append("\\midrule")
        for label, row in baseline_rows.items():
            cells = [_escape(label)]
            for c in cols:
                v = row.get(c, np.nan)
                cells.append(_fmt_num(v, ndigits=ndigits))
            lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
