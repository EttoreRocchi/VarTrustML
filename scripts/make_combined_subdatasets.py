"""
Build *combined* (pooled) SV subdatasets by concatenating, for each SV type,
the corresponding rows from several SV datasets.

This generalizes the per-sample extraction (see extract_sv_subdatasets.py).
Given N full datasets that each contain SVTYPE_CALLER_* one-hot columns, it
extracts the per-SV-type subdatasets from each dataset and pools them across
datasets into a single file per SV type, e.g.:

    combined_DEL.csv = HG002_DEL + NA12878_DEL + REACH_DEL

It works for any number of input datasets and any SV types present, so the same
script can be reused as more samples are added in the future.

Example
-------
    python scripts/make_combined_subdatasets.py \\
        data/HG002.csv data/NA12878.csv data/REACH.csv \\
        --output-dir data --prefix combined --target-column state
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Reuse the extraction logic from the sibling script (same scripts/ directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_sv_subdatasets import extract_sv_subdatasets, load_sv_dataset


def build_combined(
    input_files: List[str],
    output_dir: str,
    prefix: str = "combined",
    target_column: str = "state",
) -> List[Path]:
    """
    Extract per-SV-type subdatasets from each input dataset and pool them across
    datasets into one combined file per SV type.

    Parameters
    ----------
    input_files : List[str]
        Full SV dataset files (each with SVTYPE_CALLER_* columns).
    output_dir : str
        Directory where ``<prefix>_<SV>.csv`` files are written.
    prefix : str
        Filename prefix for the combined subdatasets.
    target_column : str
        Target column name, used only for the class-balance report.

    Returns
    -------
    List[Path]
        Paths of the combined files written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # sv_type -> list of per-dataset subdataframes
    pools: dict = {}
    print("Extracting per-SV-type subdatasets from each dataset...")
    for filepath in input_files:
        name = Path(filepath).stem
        df = load_sv_dataset(filepath)
        subs = extract_sv_subdatasets(df)
        summary = ", ".join(f"{k}={len(v)}" for k, v in sorted(subs.items()))
        print(f"  {name}: {summary}")
        for sv_type, sub in subs.items():
            pools.setdefault(sv_type, []).append(sub)

    written: List[Path] = []
    print("\nWriting combined (pooled) subdatasets...")
    for sv_type in sorted(pools):
        combined = pd.concat(pools[sv_type], ignore_index=True)
        out_file = output_path / f"{prefix}_{sv_type}.csv"
        combined.to_csv(out_file, index=False)

        if target_column in combined.columns:
            vc = combined[target_column].value_counts().to_dict()
            balance = ", ".join(f"{k}={v}" for k, v in sorted(vc.items()))
        else:
            balance = "n/a"
        print(
            f"  {out_file.name}: {len(combined):,} rows  [{target_column}: {balance}]"
        )
        written.append(out_file)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined (pooled) SV subdatasets across multiple datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Full SV dataset files (each with SVTYPE_CALLER_* columns).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory to write the combined_<SV>.csv files.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="combined",
        help="Filename prefix for the combined subdatasets.",
    )
    parser.add_argument(
        "-t",
        "--target-column",
        default="state",
        help="Target column name (used for the class-balance report).",
    )
    args = parser.parse_args()

    valid_files = [f for f in args.input_files if Path(f).exists()]
    for f in args.input_files:
        if not Path(f).exists():
            print(f"Warning: file not found: {f}")
    if not valid_files:
        print("Error: no valid input files.")
        sys.exit(1)

    written = build_combined(
        valid_files, args.output_dir, args.prefix, args.target_column
    )
    print(f"\nDone. {len(written)} combined subdatasets written to {args.output_dir}/")


if __name__ == "__main__":
    main()
