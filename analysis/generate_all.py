from __future__ import annotations
import shutil
import sys
import time
import warnings

from analysis import (
    config,
    loaders,
    e1_analysis,
    e2_analysis,
    e3_analysis,
    e4_analysis,
    e5_analysis,
    cross_experiment,
)

# Extensions this pipeline produces. Anything else found under figures/ or tables/ is left
# alone, so a hand-added note or README survives a regeneration.
_GENERATED_SUFFIXES = {".pdf", ".png", ".svg", ".tex"}


def clean_outputs() -> int:
    """
    Delete figures/ and tables/ artefacts from a previous run before regenerating.

    Plot and table writers overwrite by filename but never remove outputs that stopped being
    produced. After a metric is renamed or a block's dataset list changes, the obsolete files
    stay behind under names almost identical to the current ones — `tab_e4_avg_ranks_kappa_per`
    next to `tab_e4_avg_ranks_kappa_temporal`, or E4 figures for a stream that block no longer
    uses. Only the mtime distinguishes them, which makes them very easy to paste into the thesis
    by accident.
    """
    removed = 0
    for root in (config.FIGURES_DIR, config.TABLES_DIR):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_file() and path.suffix.lower() in _GENERATED_SUFFIXES:
                path.unlink()
                removed += 1
            elif path.is_dir() and not any(path.iterdir()):
                shutil.rmtree(path, ignore_errors=True)
    return removed


def main():
    warnings.filterwarnings("default")
    t0 = time.time()
    print(">>> Thesis results pipeline starting")

    removed = clean_outputs()
    print(f">>> Cleaned {removed} stale figure/table artefact(s)")

    loaders.data_inventory()
    e1_analysis.run()
    e2_analysis.run()
    e3_analysis.run()
    e4_analysis.run()
    e5_analysis.run()
    cross_experiment.run()
    print(f"\n>>> Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
