#!/usr/bin/env python3
"""
Analyze equilibration convergence from an OpenMM StateDataReporter CSV.

Generates a 2x3 figure (5 panels, last slot hidden):
  NVT: Temperature (K) | Potential Energy | Total Energy | Density | RMSD
  NPT: Temperature (K) | Potential Energy | Box Volume   | Density | RMSD

RMSD is computed from the trajectory (backbone CA, relative to first frame).
Density comes from the CSV reporter.

Convergence heuristic uses only the last 50% of frames (ignores warmup):
  - drift = |mean(first half of tail) - mean(second half of tail)| / |mean(tail)| * 100  [%]
  - cv    = std(tail) / |mean(tail)| * 100                                                [%]
Flagged as stable when drift < 5% AND cv < 5%.
Overall converged = all CSV quantities stable (RMSD is plotted but not included
in the pass/fail decision since it need not plateau for equilibration).
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# CSV column names as written by OpenMM StateDataReporter
# ------------------------------------------------------------------ #
COL_TEMP    = "Temperature (K)"
COL_EPOT    = "Potential Energy (kJ/mole)"
COL_ETOT    = "Total Energy (kJ/mole)"
COL_VOL     = "Box Volume (nm^3)"
COL_DENSITY = "Density (g/mL)"

NVT_QUANTITIES = [
    (COL_TEMP,    "Temperature (K)",         "tab:red"),
    (COL_EPOT,    "Potential Energy\n(kJ/mol)", "tab:blue"),
    (COL_ETOT,    "Total Energy\n(kJ/mol)",  "tab:purple"),
    (COL_DENSITY, "Density (g/mL)",          "tab:brown"),
]

NPT_QUANTITIES = [
    (COL_TEMP,    "Temperature (K)",         "tab:red"),
    (COL_EPOT,    "Potential Energy\n(kJ/mol)", "tab:blue"),
    (COL_VOL,     "Box Volume (nm³)",        "tab:orange"),
    (COL_DENSITY, "Density (g/mL)",          "tab:brown"),
]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"').lstrip("#").strip() for c in df.columns]
    return df


def _convergence_metrics(values):
    """
    Uses last 50% of frames only (ignores warmup/heating).
    drift = |mean(first half of tail) - mean(second half of tail)| / |mean(tail)| * 100
    cv    = std(tail) / |mean(tail)| * 100
    """
    n = len(values)
    half = max(1, n // 2)
    tail = values[-half:]

    q = max(1, len(tail) // 2)
    tail_first = tail[:q]
    tail_last  = tail[q:]

    mean_tail = np.mean(tail)
    std_tail  = np.std(tail)

    if abs(mean_tail) < 1e-12:
        drift = 0.0
        cv    = 0.0
    else:
        drift = abs(np.mean(tail_first) - np.mean(tail_last)) / abs(mean_tail) * 100.0
        cv    = std_tail / abs(mean_tail) * 100.0

    stable = bool(drift < 5.0) and bool(cv < 5.0)
    return {"drift_pct": float(round(drift, 3)), "cv_last_pct": float(round(cv, 3)), "stable": stable}


def _plot_panel(ax, x, y, label, color, window=50, x_label="Step"):
    """Plot raw data (faded) + rolling mean (solid)."""
    ax.plot(x, y, color=color, alpha=0.6, linewidth=0.8)
    rm = pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().values
    ax.plot(x, rm, color="black", linewidth=2.0)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xlabel(x_label, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)


def _compute_rmsd(trajectory, topology):
    """
    Load trajectory with MDTraj and compute backbone RMSD relative to frame 0.
    Returns (time_ps_or_frames, rmsd_nm).
    """
    import mdtraj as md
    print(f"  Loading trajectory for RMSD: {trajectory}")
    traj = md.load(trajectory, top=topology)
    backbone = traj.topology.select("protein and name CA")
    rmsd = md.rmsd(traj, traj, frame=0, atom_indices=backbone)
    time = traj.time if traj.time is not None else np.arange(traj.n_frames)
    return time, rmsd


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def create_ag_parser():
    parser = argparse.ArgumentParser(
        description="Plot equilibration convergence from an OpenMM CSV reporter file"
    )
    parser.add_argument("-c", "--csv_file", required=True,
                        help="Path to the OpenMM StateDataReporter CSV")
    parser.add_argument("-t", "--trajectory", type=str, default=None,
                        help="Path to the trajectory (.dcd) for RMSD panel")
    parser.add_argument("-to", "--topology", type=str, default=None,
                        help="Path to the topology (.pdb/.cif) for RMSD panel")
    parser.add_argument("-o", "--output", default="equilibration_check.svg",
                        help="Output plot path (default: equilibration_check.svg)")
    parser.add_argument("--step_type", choices=["nvt", "npt"], required=True,
                        help="Simulation type: 'nvt' or 'npt'")
    parser.add_argument("--rolling_window", type=int, default=50,
                        help="Window for rolling mean overlay (default: 50 rows)")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to write JSON result")
    return parser


def main(csv_file, trajectory=None, topology=None,
         output="equilibration_check.svg", step_type="nvt",
         rolling_window=50, json_output=None):

    print(f"Loading CSV: {csv_file}")
    df = _load_csv(csv_file)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    quantities = NVT_QUANTITIES if step_type == "nvt" else NPT_QUANTITIES

    for col, _, _ in quantities:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Available: {list(df.columns)}")

    step_col = "Step" if "Step" in df.columns else df.columns[0]
    x = df[step_col].values

    # 2x3 grid — 4 CSV panels + 1 RMSD panel, last slot hidden
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Equilibration convergence check — {step_type.upper()}", fontsize=12)
    flat = axs.flat

    metrics = {}
    all_stable = True

    # --- CSV panels (4) ---
    for ax, (col, label, color) in zip(flat, quantities):
        y = df[col].values
        _plot_panel(ax, x, y, label, color, window=rolling_window)
        m = _convergence_metrics(y)
        metrics[col] = m
        status = "STABLE" if m["stable"] else "NOT STABLE"
        if not m["stable"]:
            all_stable = False
        ax.set_title(
            f"{status}  |  drift={m['drift_pct']:.1f}%  cv={m['cv_last_pct']:.1f}%",
            fontsize=8,
            color="green" if m["stable"] else "red",
        )

    # --- RMSD panel (5th) ---
    ax_rmsd = axs[1, 1]
    if trajectory is not None and topology is not None:
        try:
            t_rmsd, rmsd = _compute_rmsd(trajectory, topology)
            _plot_panel(ax_rmsd, t_rmsd, rmsd, "RMSD (nm)\nvs frame 0", "tab:green",
                        window=rolling_window, x_label="Time (ps)")
            ax_rmsd.set_title("RMSD (info only)", fontsize=8, color="black")
            metrics["RMSD (nm)"] = _convergence_metrics(rmsd)
        except Exception as e:
            ax_rmsd.text(0.5, 0.5, f"RMSD failed:\n{e}", transform=ax_rmsd.transAxes,
                         ha="center", va="center", fontsize=8, color="red")
            ax_rmsd.set_title("RMSD (failed)", fontsize=8, color="red")
    else:
        ax_rmsd.text(0.5, 0.5, "No trajectory provided", transform=ax_rmsd.transAxes,
                     ha="center", va="center", fontsize=9, color="gray")
        ax_rmsd.set_title("RMSD", fontsize=8, color="gray")

    # --- Hide last slot ---
    axs[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output}")

    # Print summary
    print(f"\nConvergence summary ({step_type.upper()}):")
    for col, label, _ in quantities:
        m = metrics[col]
        flag = "OK" if m["stable"] else "REVIEW"
        print(f"  [{flag}] {label.replace(chr(10), ' ')}: "
              f"drift={m['drift_pct']:.1f}%  cv={m['cv_last_pct']:.1f}%")
    if "RMSD (nm)" in metrics:
        m = metrics["RMSD (nm)"]
        print(f"  [INFO] RMSD: drift={m['drift_pct']:.1f}%  cv={m['cv_last_pct']:.1f}%")
    print(f"  Overall converged: {all_stable}")

    result = {
        "step": f"check_{step_type}",
        "status": "completed",
        "plot": output,
        "converged": bool(all_stable),
        "metrics": metrics,
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.csv_file, args.trajectory, args.topology,
         args.output, args.step_type, args.rolling_window, args.json)
