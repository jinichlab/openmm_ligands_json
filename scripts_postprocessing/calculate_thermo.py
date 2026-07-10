#!/usr/bin/env python3
"""
Parse the OpenMM StateDataReporter CSV output and produce summary statistics
plus a cleaned CSV with computed rolling averages.
"""
import argparse
import json

import numpy as np
import pandas as pd


def create_ag_parser():
    parser = argparse.ArgumentParser(
        description="Parse OpenMM StateDataReporter CSV and compute thermodynamic summary"
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the OpenMM CSV reporter file")
    parser.add_argument("-o", "--output", type=str, default="thermo_summary.csv",
                        help="Output summary CSV (default: thermo_summary.csv)")
    parser.add_argument("--rolling_window", type=int, default=100,
                        help="Window size for rolling averages (default: 100 rows)")
    parser.add_argument("--rolling_out", type=str, default=None,
                        help="Optional path for CSV with rolling-average columns appended")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON result file")
    return parser


def main(input_csv, output="thermo_summary.csv", rolling_window=100,
         rolling_out=None, json_output=None):
    print(f"Loading thermodynamic data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Identify numeric columns (skip step / time columns for stats)
    skip_cols = {"#\"Step\"", "Step", "Time (ps)"}
    numeric_cols = [c for c in df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Per-column summary statistics
    summary_records = []
    for col in numeric_cols:
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        summary_records.append({
            "quantity": col,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
        })

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(output, index=False)
    print(f"Summary saved to {output}")

    # Optional rolling averages
    if rolling_out is not None:
        df_rolling = df.copy()
        for col in numeric_cols:
            df_rolling[f"{col}_rolling{rolling_window}"] = (
                df[col].rolling(window=rolling_window, center=True, min_periods=1).mean()
            )
        df_rolling.to_csv(rolling_out, index=False)
        print(f"Rolling averages saved to {rolling_out}")

    # Build JSON-friendly summary dict
    summary_dict = {r["quantity"]: {k: v for k, v in r.items() if k != "quantity"}
                    for r in summary_records}

    result = {
        "step": "thermo",
        "status": "completed",
        "input_csv": input_csv,
        "output": output,
        "rolling_out": rolling_out,
        "n_rows": int(len(df)),
        "summary": summary_dict,
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.input, args.output, args.rolling_window, args.rolling_out, args.json)
