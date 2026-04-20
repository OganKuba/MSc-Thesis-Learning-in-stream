import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary public function (simplified interface)
# ---------------------------------------------------------------------------

def dataframe_to_arff(
    df: pd.DataFrame,
    relation_name: str,
    target_col: str,
    output_path: str,
    missing_value: str = "?",
) -> None:

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found in DataFrame columns."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    feature_cols = [c for c in df.columns if c != target_col]
    ordered_cols = feature_cols + [target_col]
    df = df[ordered_cols]

    nominal_cols: dict[str, list[str]] = {}
    for col in ordered_cols:
        series = df[col]
        if col == target_col or _should_be_nominal(series):
            vals = sorted(
                str(v) for v in series.dropna().unique()
            )
            nominal_cols[col] = vals

    logger.info(
        "Writing ARFF: %s  (%d rows × %d attrs, target=%s)",
        output_path, len(df), len(ordered_cols), target_col,
    )
    _write_arff_core(
        df=df,
        path=output_path,
        relation=relation_name,
        cols=ordered_cols,
        nominal_cols=nominal_cols,
        missing_value=missing_value,
    )
    logger.info("ARFF written → %s", output_path)



def write_arff(
    df: pd.DataFrame,
    path: str,
    relation: str,
    nominal_cols: Optional[dict[str, list[str]]] = None,
    class_col: Optional[str] = None,
    missing_value: str = "?",
) -> None:
    if nominal_cols is None:
        nominal_cols = {}

    if class_col is not None and class_col not in df.columns:
        raise ValueError(f"class_col '{class_col}' not found in DataFrame columns.")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Build ordered column list
    cols = [c for c in df.columns if c != class_col]
    if class_col is not None:
        cols.append(class_col)
    df = df[cols]

    logger.info("Writing ARFF: %s  (%d rows × %d attrs)", path, len(df), len(cols))
    _write_arff_core(
        df=df,
        path=path,
        relation=relation,
        cols=cols,
        nominal_cols=nominal_cols,
        missing_value=missing_value,
    )
    logger.info("ARFF written successfully → %s", path)


def _should_be_nominal(series: pd.Series) -> bool:
    """Return True when the series should be written as a NOMINAL attribute."""
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    if hasattr(series, "cat"):  # CategoricalDtype
        return True
    return False


def _infer_arff_type(series: pd.Series) -> str:
    """Return the ARFF type string for a pandas Series (fallback for write_arff)."""
    if _should_be_nominal(series):
        return "STRING"
    return "NUMERIC"


def _write_arff_core(
    df: pd.DataFrame,
    path: str,
    relation: str,
    cols: list[str],
    nominal_cols: dict[str, list[str]],
    missing_value: str,
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"@RELATION {relation}\n\n")

        for col in cols:
            safe_name = _quote_if_needed(col)
            if col in nominal_cols:
                values = ",".join(str(v) for v in nominal_cols[col])
                arff_type = f"{{{values}}}"
            else:
                arff_type = _infer_arff_type(df[col])
            fh.write(f"@ATTRIBUTE {safe_name} {arff_type}\n")

        fh.write("\n@DATA\n")
        for row in df.itertuples(index=False, name=None):
            parts = [
                _format_value(val, col, nominal_cols, missing_value)
                for val, col in zip(row, cols)
            ]
            fh.write(",".join(parts) + "\n")


def _quote_if_needed(name: str) -> str:
    """Wrap attribute name in single quotes if it contains special characters."""
    if any(ch in name for ch in (" ", ",", "'", "%", "\\", "{", "}", "@")):
        escaped = name.replace("'", "\\'")
        return f"'{escaped}'"
    return name


def _format_value(
    val,
    col: str,
    nominal_cols: dict[str, list[str]],
    missing_value: str,
) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return missing_value

    if col in nominal_cols:
        # Always render as string (int floats → drop decimal point)
        if isinstance(val, (float, np.floating)) and not np.isnan(val):
            return str(int(val))
        return str(val)

    if isinstance(val, (bool, np.bool_)):
        return "1" if val else "0"

    if isinstance(val, (int, np.integer)):
        return str(int(val))

    if isinstance(val, (float, np.floating)):
        return f"{val:.10g}"

    s = str(val)
    if "," in s or "'" in s or '"' in s or "\n" in s:
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s
