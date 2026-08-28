"""
Select the best model (by mean of a chosen metric) from compare-models outputs.

For a given SV variant, this reads the per-sample ``model_metrics_comparison.csv``
files produced by ``vartrustml compare-models`` (one per ``<sample>_<variant>``
subdataset), averages the chosen metric across samples for each model, and prints
the name of the best model to **stdout**. The full ranking is written to stderr
so it shows up in logs without polluting the captured name.

It generalizes to any set of samples / variant / metric, so it can be reused as
the model roster or sample set changes.

Example
-------
    python scripts/select_best_model.py --results-dir results --variant DEL \\
        --samples HG002 NA12878 REACH --metric "Matthews Corr. Coef."

Typical use in a shell pipeline:
    BEST=$(python scripts/select_best_model.py --results-dir results \\
        --variant DEL --samples HG002 NA12878 REACH)
    vartrustml cross-dataset ... --models "$BEST"
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


def select_best(
    results_dir: str,
    variant: str,
    samples: List[str],
    metric: str = "Matthews Corr. Coef.",
) -> str:
    """Return the model with the highest mean ``metric`` across ``samples``."""
    col = f"{metric}_mean"
    per_model_scores: dict = {}  # model -> list of values, one per sample
    used: List[str] = []

    for sample in samples:
        csv = Path(results_dir) / f"{sample}_{variant}" / "model_metrics_comparison.csv"
        if not csv.exists():
            print(f"[warn] missing: {csv}", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        if "model" not in df.columns or col not in df.columns:
            print(
                f"[warn] {csv}: missing 'model' or '{col}' column "
                f"(have: {list(df.columns)})",
                file=sys.stderr,
            )
            continue
        used.append(sample)
        for _, row in df.iterrows():
            per_model_scores.setdefault(row["model"], []).append(float(row[col]))

    if not per_model_scores:
        print(
            f"[error] no usable model_metrics_comparison.csv for variant "
            f"'{variant}' under '{results_dir}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Mean across the samples that actually had a value for each model.
    mean_scores = {m: sum(v) / len(v) for m, v in per_model_scores.items()}
    ranking = sorted(mean_scores.items(), key=lambda kv: kv[1], reverse=True)

    print(f"[{variant}] mean {metric} over {used}:", file=sys.stderr)
    for model_name, score in ranking:
        print(f"    {score:.4f}  {model_name}", file=sys.stderr)

    best = ranking[0][0]
    print(best)  # stdout: ONLY the model name (so it can be captured with $(...))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the best model by mean metric from compare-models outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory with the per-subdataset results.",
    )
    parser.add_argument(
        "--variant", required=True, help="SV type, e.g. DEL / DUP / INS / INV."
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="Sample names, e.g. HG002 NA12878 REACH.",
    )
    parser.add_argument(
        "--metric",
        default="Matthews Corr. Coef.",
        help="Metric base name; the column '<metric>_mean' is used.",
    )
    args = parser.parse_args()
    select_best(args.results_dir, args.variant, args.samples, args.metric)


if __name__ == "__main__":
    main()
