#!/usr/bin/env python3
"""
Pipeline Performance Analyzer
==============================
Reads JSON config files and reports per-step elapsed times, protein sequence
length, and total solvated system size.

Usage:
  python analyze_performance.py -c protein.json
  python analyze_performance.py -d /runs/jsons/
  python analyze_performance.py -d /runs/jsons/ --csv performance.csv
  python analyze_performance.py -d /runs/jsons/ --tsv performance.tsv
  python analyze_performance.py -d /runs/jsons/ --plot performance.png
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Step definitions (mirrors check_pipeline.py)
# ---------------------------------------------------------------------------

SHARED_STEPS = [
    "pdb_fixer",
    "minimization_vac",
    "system_creation",
    "minimization_sol",
]

REPLICA_MD_STEPS = [
    "nvt",
    "check_nvt",
    "npt",
    "check_npt",
    "production",
]

PP_STEPS = [
    "unwrap",
    "cvs",
    "rmsf",
    "dssp",
    "sasa",
    "thermo",
    "tica",
    "order_parameter",
    "network_analysis",
]

ALL_STEPS = SHARED_STEPS + REPLICA_MD_STEPS + PP_STEPS

# ANSI colours
_USE_COLOR = sys.stdout.isatty()
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
BOLD = lambda t: _c("1",  t)
DIM  = lambda t: _c("2",  t)
CYAN = lambda t: _c("36", t)
YEL  = lambda t: _c("33", t)


# ---------------------------------------------------------------------------
# Sequence length from fixed structure (MDTraj)
# ---------------------------------------------------------------------------

def _get_sequence_length(fixed_cif_path):
    """Return number of protein residues from the fixed CIF file."""
    if not fixed_cif_path or not os.path.exists(fixed_cif_path):
        return None
    try:
        import mdtraj as md
        top = md.load_topology(fixed_cif_path)
        protein_atoms = top.select("protein")
        if len(protein_atoms) == 0:
            return None
        residues = {top.atom(i).residue.index for i in protein_atoms}
        return len(residues)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Extract timing and system info from a single JSON
# ---------------------------------------------------------------------------

def _elapsed(result_dict, step):
    """Return elapsed_seconds for a step, or None if not run."""
    res = result_dict.get(step, {})
    if res.get("status") == "completed":
        return res.get("elapsed_seconds")
    return None


def analyze_config(config_path):
    """
    Returns a dict:
      {
        "name":          str,
        "n_residues":    int | None,
        "n_atoms":       int | None,   # total solvated system
        "shared": {step: seconds | None, ...},
        "replicas": [
          {"id": int, "seed": int, step: seconds | None, ...},
          ...
        ],
      }
    """
    with open(config_path) as f:
        config = json.load(f)

    name = config.get("prefix", os.path.splitext(os.path.basename(config_path))[0])

    # ── System info ─────────────────────────────────────────────────────────
    shared_results = config.get("results", {})

    # Total solvated atoms — stored directly in system_creation result
    n_atoms = shared_results.get("system_creation", {}).get("n_atoms")

    # Sequence length — prefer values already stored in postprocessing results,
    # fall back to reading the fixed CIF file if available locally.
    n_residues = None
    replicas_raw = config.get("replicas", [])
    for _rep in replicas_raw:
        for _step in ("rmsf", "dssp", "sasa", "order_parameter"):
            _v = _rep.get("results", {}).get(_step, {}).get("n_residues")
            if _v:
                n_residues = _v
                break
        if n_residues:
            break
    if not n_residues:
        fixed_path = shared_results.get("pdb_fixer", {}).get("output")
        n_residues = _get_sequence_length(fixed_path)

    # ── Simulation speed: ns/hour for production ────────────────────────────
    prod_cfg     = config.get("production", {})
    prod_steps   = prod_cfg.get("steps", 0)
    prod_dt_ps   = prod_cfg.get("time_step", 0.002)   # ps
    prod_sim_ns  = prod_steps * prod_dt_ps / 1000.0   # ns

    # ── Shared step timings ─────────────────────────────────────────────────
    shared_timings = {step: _elapsed(shared_results, step) for step in SHARED_STEPS}

    # ── Replica timings ─────────────────────────────────────────────────────
    replicas_data = []
    replicas = config.get("replicas", [])

    def _ns_per_hour(results_dict):
        t = _elapsed(results_dict, "production")
        if t and prod_sim_ns > 0:
            return prod_sim_ns / (t / 3600.0)
        return None

    if replicas:
        for rep in replicas:
            rep_results = rep.get("results", {})
            row = {
                "id":          rep["id"],
                "seed":        rep.get("seed", 0),
                "ns_per_hour": _ns_per_hour(rep_results),
            }
            for step in REPLICA_MD_STEPS + PP_STEPS:
                row[step] = _elapsed(rep_results, step)
            replicas_data.append(row)
    else:
        # Legacy flat format
        row = {
            "id":          0,
            "seed":        0,
            "ns_per_hour": _ns_per_hour(shared_results),
        }
        for step in REPLICA_MD_STEPS + PP_STEPS:
            row[step] = _elapsed(shared_results, step)
        replicas_data.append(row)

    return {
        "name":        name,
        "n_residues":  n_residues,
        "n_atoms":     n_atoms,
        "prod_sim_ns": prod_sim_ns,
        "shared":      shared_timings,
        "replicas":    replicas_data,
    }


# ---------------------------------------------------------------------------
# Pretty-print table for one protein
# ---------------------------------------------------------------------------

def _fmt_time(seconds):
    """Format seconds → human-readable string."""
    if seconds is None:
        return DIM("  --   ")
    if seconds < 60:
        return f"{seconds:6.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m):3d}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    m = int(rem // 60)
    return f"{int(h):2d}h{m:02d}m"


def _fmt_time_raw(seconds):
    """Return seconds as plain float string for CSV."""
    return "" if seconds is None else f"{seconds:.2f}"


def print_protein_report(data):
    name       = data["name"]
    n_res      = data["n_residues"]
    n_atoms    = data["n_atoms"]
    n_rep      = len(data["replicas"])

    res_str   = f"{n_res} aa" if n_res else "n/a"
    atoms_str = f"{n_atoms:,} atoms" if n_atoms else "n/a"
    nsh_vals  = [r["ns_per_hour"] for r in data["replicas"] if r.get("ns_per_hour")]
    nsh_str   = f"{sum(nsh_vals)/len(nsh_vals):.2f} ns/hr (avg)" if nsh_vals else "n/a"

    print(BOLD(f"\n{'─'*58}"))
    print(BOLD(f"  {name}"))
    print(f"  Sequence length : {CYAN(res_str)}")
    print(f"  System size     : {CYAN(atoms_str)}")
    print(f"  Sim. speed      : {CYAN(nsh_str)}")
    print(f"  Replicas        : {n_rep}")
    print(f"{'─'*58}")

    # ── Shared steps ────────────────────────────────────────────────────────
    print(DIM("  Shared steps"))
    shared_total = 0.0
    for step in SHARED_STEPS:
        t = data["shared"].get(step)
        shared_total += t or 0.0
        print(f"    {step:<22s}  {_fmt_time(t)}")
    print(f"    {'Total':<22s}  {YEL(_fmt_time(shared_total))}")

    # ── Per-replica steps ────────────────────────────────────────────────────
    for rep in data["replicas"]:
        rep_id = rep["id"]
        seed   = rep["seed"]
        seed_s = f"seed={seed}" if seed else "seed=random"
        print(DIM(f"\n  Replica {rep_id}  ({seed_s})"))

        md_total = 0.0
        print(DIM("    MD steps"))
        for step in REPLICA_MD_STEPS:
            t = rep.get(step)
            md_total += t or 0.0
            print(f"      {step:<22s}  {_fmt_time(t)}")

        pp_total = 0.0
        print(DIM("    Postprocessing"))
        for step in PP_STEPS:
            t = rep.get(step)
            pp_total += t or 0.0
            print(f"      {step:<22s}  {_fmt_time(t)}")

        rep_total = md_total + pp_total
        print(f"      {'Replica total':<22s}  {YEL(_fmt_time(rep_total))}")

    # ── Grand total ──────────────────────────────────────────────────────────
    grand = shared_total
    for rep in data["replicas"]:
        grand += sum(rep.get(s) or 0 for s in REPLICA_MD_STEPS + PP_STEPS)
    print(f"\n  {'Grand total':<24s}  {BOLD(_fmt_time(grand))}")


# ---------------------------------------------------------------------------
# Summary table across all proteins
# ---------------------------------------------------------------------------

def print_summary_table(all_data):
    print(BOLD(f"\n{'='*80}"))
    print(BOLD("  PERFORMANCE SUMMARY"))
    print(f"{'='*80}")

    col_protein = 20
    col_time    = 9
    step_cols   = ["min_vac", "min_sol", "nvt", "npt", "production"]
    header_labels = {
        "min_vac":    "vac_min",
        "min_sol":    "sol_min",
        "nvt":        "NVT",
        "npt":        "NPT",
        "production": "prod",
    }

    hdr  = f"  {'protein':<{col_protein}}  {'aa':>5}  {'atoms':>8}"
    hdr += f"  {'pdb_fix':>{col_time}}  {'sys_cre':>{col_time}}"
    for s in step_cols:
        hdr += f"  {header_labels[s]:>{col_time}}"
    hdr += f"  {'total':>{col_time}}  {'ns/hour':>8}"
    print(DIM(hdr))
    print(DIM(f"  {'-'*76}"))

    for data in all_data:
        name     = data["name"][:col_protein]
        n_res    = f"{data['n_residues']}" if data["n_residues"] else "n/a"
        n_atoms  = f"{data['n_atoms']:,}"  if data["n_atoms"]    else "n/a"

        shared   = data["shared"]
        # Average across replicas for per-replica steps
        n_rep    = len(data["replicas"])
        def avg(step):
            vals = [r.get(step) for r in data["replicas"] if r.get(step) is not None]
            return sum(vals) / len(vals) if vals else None

        row  = f"  {name:<{col_protein}}  {n_res:>5}  {n_atoms:>8}"
        row += f"  {_fmt_time(shared.get('pdb_fixer')):>{col_time}}"
        row += f"  {_fmt_time(shared.get('system_creation')):>{col_time}}"
        for s in step_cols:
            row += f"  {_fmt_time(avg(s)):>{col_time}}"

        # Grand total (shared + average of per-replica steps)
        grand = sum(v or 0 for v in shared.values())
        if n_rep > 0:
            rep_avg = sum(
                sum(rep.get(s) or 0 for s in REPLICA_MD_STEPS + PP_STEPS)
                for rep in data["replicas"]
            ) / n_rep
            grand += rep_avg
        row += f"  {_fmt_time(grand):>{col_time}}"

        # ns/hour averaged across replicas
        nsh_vals = [r["ns_per_hour"] for r in data["replicas"] if r.get("ns_per_hour")]
        nsh = f"{sum(nsh_vals)/len(nsh_vals):6.2f}" if nsh_vals else "   n/a"
        row += f"  {nsh:>8}"
        print(row)


# ---------------------------------------------------------------------------
# CSV / TSV export
# ---------------------------------------------------------------------------

def write_table(all_data, out_path, sep=","):
    """
    One row per (protein × replica).
    Columns: protein, replica_id, seed, n_residues, n_atoms, [step_seconds...], total_seconds
    """
    import csv

    fieldnames = (
        ["protein", "replica_id", "seed", "n_residues", "n_atoms", "ns_per_hour"]
        + [f"{s}_s" for s in ALL_STEPS]
        + ["shared_total_s", "replica_total_s", "grand_total_s"]
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep,
                                extrasaction="ignore")
        writer.writeheader()

        for data in all_data:
            shared = data["shared"]
            shared_total = sum(v or 0 for v in shared.values())

            for rep in data["replicas"]:
                nsh = rep.get("ns_per_hour")
                row = {
                    "protein":     data["name"],
                    "replica_id":  rep["id"],
                    "seed":        rep["seed"],
                    "n_residues":  data["n_residues"] or "",
                    "n_atoms":     data["n_atoms"]    or "",
                    "ns_per_hour": f"{nsh:.4f}" if nsh else "",
                }
                rep_total = 0.0
                for step in ALL_STEPS:
                    if step in shared:
                        t = shared.get(step)
                    else:
                        t = rep.get(step)
                    row[f"{step}_s"] = _fmt_time_raw(t)
                    rep_total += t or 0.0

                replica_only = sum(rep.get(s) or 0 for s in REPLICA_MD_STEPS + PP_STEPS)
                row["shared_total_s"]  = _fmt_time_raw(shared_total)
                row["replica_total_s"] = _fmt_time_raw(replica_only)
                row["grand_total_s"]   = _fmt_time_raw(shared_total + replica_only)
                writer.writerow(row)

    print(f"\nTable written to: {out_path}")



def write_binned_table(all_data, out_path, bin_size=100):
    """
    Bins proteins by sequence length and reports average time per step per bin.
    One row per bin. Times in minutes.
    Columns: seq_min, seq_max, n_proteins,
             pdb_fixer_min, minimization_vac_min, system_creation_min, minimization_sol_min,
             nvt_min, npt_min, production_min, postprocessing_total_min,
             grand_total_min, ns_per_hour
    """
    import csv
    import math

    def _avg_rep(data, step):
        vals = [rep.get(step) for rep in data["replicas"] if rep.get(step) is not None]
        return sum(vals) / len(vals) if vals else None

    def _pp_avg(data):
        vals = []
        for rep in data["replicas"]:
            t = sum(rep.get(s) or 0 for s in PP_STEPS)
            if t > 0:
                vals.append(t)
        return sum(vals) / len(vals) if vals else None

    def _nsh_avg(data):
        vals = [r["ns_per_hour"] for r in data["replicas"] if r.get("ns_per_hour")]
        return sum(vals) / len(vals) if vals else None

    # Assign each protein to a bin
    proteins_with_len = [(d, d["n_residues"]) for d in all_data if d["n_residues"]]
    if not proteins_with_len:
        print("No sequence length data available for binning.")
        return

    max_len = max(n for _, n in proteins_with_len)
    bins = range(0, int(math.ceil(max_len / bin_size)) * bin_size + 1, bin_size)
    bin_edges = list(bins)

    # Group proteins into bins
    groups = {}  # (lo, hi) → [data]
    for data, n in proteins_with_len:
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo <= n < hi or (n == hi and i == len(bin_edges) - 2):
                groups.setdefault((lo, hi), []).append(data)
                break

    step_keys = SHARED_STEPS + ["nvt", "npt", "production", "postprocessing_total"]
    fieldnames = (
        ["seq_min", "seq_max", "n_proteins"]
        + [f"{s}_min" for s in step_keys]
        + ["grand_total_min", "ns_per_hour"]
    )

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _fmt(seconds):
        return f"{seconds / 60:.2f}" if seconds is not None else ""

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for lo, hi in sorted(groups):
            group = groups[(lo, hi)]
            row = {
                "seq_min":    lo,
                "seq_max":    hi,
                "n_proteins": len(group),
            }
            for s in SHARED_STEPS:
                row[f"{s}_min"] = _fmt(_mean([d["shared"].get(s) for d in group]))
            for s in ["nvt", "npt", "production"]:
                row[f"{s}_min"] = _fmt(_mean([_avg_rep(d, s) for d in group]))
            row["postprocessing_total_min"] = _fmt(_mean([_pp_avg(d) for d in group]))

            grand_vals = []
            for d in group:
                sh = sum(v or 0 for v in d["shared"].values())
                rep = sum(_avg_rep(d, s) or 0 for s in ["nvt", "npt", "production"])
                pp  = _pp_avg(d) or 0
                grand_vals.append(sh + rep + pp)
            row["grand_total_min"] = _fmt(_mean(grand_vals))
            row["ns_per_hour"]     = f"{_mean([_nsh_avg(d) for d in group]):.4f}" if _mean([_nsh_avg(d) for d in group]) else ""

            writer.writerow(row)

    print(f"\nBinned table written to: {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Steps to include in scatter panels (label, step_key, shared_or_replica)
_PLOT_STEPS = [
    ("PDB Fixer",        "pdb_fixer",        "shared"),
    ("Vac. Minimization","minimization_vac",  "shared"),
    ("System Creation",  "system_creation",   "shared"),
    ("Sol. Minimization","minimization_sol",  "shared"),
    ("NVT",              "nvt",               "replica"),
    ("NPT",              "npt",               "replica"),
    ("Production",       "production",        "replica"),
    ("Postprocessing",   "_pp_total",         "replica"),   # sum of all PP steps
]


def _rep_avg(data, step):
    """Average elapsed time across replicas for a given step (or None if all missing)."""
    if step == "_pp_total":
        vals = []
        for rep in data["replicas"]:
            total = sum(rep.get(s) or 0 for s in PP_STEPS)
            if total > 0:
                vals.append(total)
        return sum(vals) / len(vals) if vals else None
    vals = [rep.get(step) for rep in data["replicas"] if rep.get(step) is not None]
    return sum(vals) / len(vals) if vals else None


def plot_performance(all_data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # ── Collect data points ─────────────────────────────────────────────────
    seq_lens  = [d["n_residues"] for d in all_data if d["n_residues"]]
    n_atoms   = [d["n_atoms"]    for d in all_data if d["n_atoms"]]

    # Per-step time arrays aligned to seq_lens
    step_times = {}
    seq_for_steps = []  # sequence lengths for proteins with at least some timing
    for d in all_data:
        if not d["n_residues"]:
            continue
        seq_for_steps.append(d["n_residues"])
        for label, step, kind in _PLOT_STEPS:
            if kind == "shared":
                t = d["shared"].get(step)
            else:
                t = _rep_avg(d, step)
            step_times.setdefault(label, []).append(t)

    seq_arr = np.array(seq_for_steps)

    # Total time per protein (shared + avg replica MD + avg postprocessing)
    total_times = []
    nsh_vals_plot = []
    for d in all_data:
        if not d["n_residues"]:
            continue
        sh_total = sum(v or 0 for v in d["shared"].values())
        rep_avg = (
            sum(
                sum(rep.get(s) or 0 for s in REPLICA_MD_STEPS + PP_STEPS)
                for rep in d["replicas"]
            ) / len(d["replicas"])
            if d["replicas"] else 0
        )
        total_times.append((sh_total + rep_avg) / 60)  # minutes
        nsh = [r["ns_per_hour"] for r in d["replicas"] if r.get("ns_per_hour")]
        nsh_vals_plot.append(sum(nsh) / len(nsh) if nsh else None)

    # ── Layout: 4 rows ───────────────────────────────────────────────────────
    # Row 0: two histograms (sequence length, system size)
    # Row 1: scatter grid — shared steps (4 panels)
    # Row 2: scatter grid — replica steps (4 panels)
    # Row 3: total time + ns/hour (wide panels)
    n_scatter_cols = 4
    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(
        4, n_scatter_cols,
        figure=fig,
        hspace=0.55, wspace=0.40,
        height_ratios=[1, 1, 1, 1],
    )

    # ── Row 0: histograms ────────────────────────────────────────────────────
    ax_hist_seq = fig.add_subplot(gs[0, :2])
    ax_hist_sys = fig.add_subplot(gs[0, 2:])

    # Sequence length histogram
    bins_seq = np.linspace(min(seq_lens) - 5, max(seq_lens) + 5, 12) if len(seq_lens) > 1 else 10
    ax_hist_seq.hist(seq_lens, bins=bins_seq, color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax_hist_seq.set_xlabel("Sequence length (aa)", fontsize=11)
    ax_hist_seq.set_ylabel("Count", fontsize=11)
    ax_hist_seq.set_title("Distribution of Sequence Lengths", fontsize=12, fontweight="bold")
    ax_hist_seq.spines[["top", "right"]].set_visible(False)

    # System size histogram
    bins_sys = np.linspace(min(n_atoms) - 500, max(n_atoms) + 500, 12) if len(n_atoms) > 1 else 10
    ax_hist_sys.hist(n_atoms, bins=bins_sys, color="#DD8452", edgecolor="white", linewidth=0.8)
    ax_hist_sys.set_xlabel("Total atoms (solvated)", fontsize=11)
    ax_hist_sys.set_ylabel("Count", fontsize=11)
    ax_hist_sys.set_title("Distribution of System Sizes", fontsize=12, fontweight="bold")
    ax_hist_sys.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax_hist_sys.spines[["top", "right"]].set_visible(False)

    # ── Rows 1-2: scatter plots per step ─────────────────────────────────────
    colors_shared  = "#4C72B0"
    colors_replica = "#55A868"
    colors_pp      = "#C44E52"

    step_color = {
        "shared":  colors_shared,
        "replica": colors_replica,
    }
    # override postprocessing
    step_color_map = {label: step_color[kind] for label, _, kind in _PLOT_STEPS}
    step_color_map["Postprocessing"] = colors_pp

    shared_labels  = [l for l, _, k in _PLOT_STEPS if k == "shared"]
    replica_labels = [l for l, _, k in _PLOT_STEPS if k == "replica"]

    def _draw_scatter(ax, label, row_color):
        times = np.array([t / 60 if t is not None else np.nan
                          for t in step_times.get(label, [])])
        mask  = ~np.isnan(times)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
        else:
            ax.scatter(seq_arr[mask], times[mask],
                       color=row_color, alpha=0.85, s=60, edgecolors="white", linewidth=0.5)
            # trend line if ≥ 3 points
            if mask.sum() >= 3:
                z = np.polyfit(seq_arr[mask], times[mask], 1)
                x_line = np.linspace(seq_arr[mask].min(), seq_arr[mask].max(), 100)
                ax.plot(x_line, np.polyval(z, x_line),
                        color=row_color, linewidth=1.5, linestyle="--", alpha=0.6)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Sequence length (aa)", fontsize=9)
        ax.set_ylabel("Time (min)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    # Row 1 — shared steps
    for col, label in enumerate(shared_labels):
        ax = fig.add_subplot(gs[1, col])
        _draw_scatter(ax, label, step_color_map[label])

    # Row 2 — replica steps
    for col, label in enumerate(replica_labels):
        ax = fig.add_subplot(gs[2, col])
        _draw_scatter(ax, label, step_color_map[label])

    # ── Row 3: total time + ns/hour ──────────────────────────────────────────
    colors_total = "#8172B2"

    ax_total = fig.add_subplot(gs[3, :2])
    tot_arr = np.array([t if t is not None else np.nan for t in total_times])
    mask_tot = ~np.isnan(tot_arr)
    if mask_tot.sum() == 0:
        ax_total.text(0.5, 0.5, "no data", ha="center", va="center",
                      transform=ax_total.transAxes, color="grey")
    else:
        ax_total.scatter(seq_arr[mask_tot], tot_arr[mask_tot],
                         color=colors_total, alpha=0.85, s=70,
                         edgecolors="white", linewidth=0.5)
        if mask_tot.sum() >= 3:
            z = np.polyfit(seq_arr[mask_tot], tot_arr[mask_tot], 1)
            x_line = np.linspace(seq_arr[mask_tot].min(), seq_arr[mask_tot].max(), 100)
            ax_total.plot(x_line, np.polyval(z, x_line),
                          color=colors_total, linewidth=1.5, linestyle="--", alpha=0.6)
    ax_total.set_title("Total Pipeline Time", fontsize=10, fontweight="bold")
    ax_total.set_xlabel("Sequence length (aa)", fontsize=9)
    ax_total.set_ylabel("Time (min)", fontsize=9)
    ax_total.spines[["top", "right"]].set_visible(False)
    ax_total.tick_params(labelsize=8)

    ax_nsh = fig.add_subplot(gs[3, 2:])
    nsh_arr = np.array([v if v is not None else np.nan for v in nsh_vals_plot])
    mask_nsh = ~np.isnan(nsh_arr)
    if mask_nsh.sum() == 0:
        ax_nsh.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax_nsh.transAxes, color="grey")
    else:
        ax_nsh.scatter(seq_arr[mask_nsh], nsh_arr[mask_nsh],
                       color="#DD8452", alpha=0.85, s=70,
                       edgecolors="white", linewidth=0.5)
        if mask_nsh.sum() >= 3:
            z = np.polyfit(seq_arr[mask_nsh], nsh_arr[mask_nsh], 1)
            x_line = np.linspace(seq_arr[mask_nsh].min(), seq_arr[mask_nsh].max(), 100)
            ax_nsh.plot(x_line, np.polyval(z, x_line),
                        color="#DD8452", linewidth=1.5, linestyle="--", alpha=0.6)
    ax_nsh.set_title("Simulation Speed", fontsize=10, fontweight="bold")
    ax_nsh.set_xlabel("Sequence length (aa)", fontsize=9)
    ax_nsh.set_ylabel("ns / hour", fontsize=9)
    ax_nsh.spines[["top", "right"]].set_visible(False)
    ax_nsh.tick_params(labelsize=8)

    fig.suptitle("MD Pipeline Performance vs Sequence Length", fontsize=14, fontweight="bold", y=1.01)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_shared,  label="Shared steps"),
        Patch(facecolor=colors_replica, label="Replica steps (avg)"),
        Patch(facecolor=colors_pp,      label="Postprocessing (avg)"),
        Patch(facecolor=colors_total,   label="Total pipeline time"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON discovery (mirrors check_pipeline._find_jsons)
# ---------------------------------------------------------------------------

def _find_jsons(directory):
    found = []
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        # Layout 1: subdir/subdir.json
        if os.path.isdir(full):
            candidate = os.path.join(full, f"{entry}.json")
            if os.path.isfile(candidate):
                found.append(candidate)
        # Layout 2: flat — JSON directly in the directory
        elif entry.endswith(".json"):
            found.append(full)
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-step timing and system info from MD pipeline JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", nargs="+", metavar="JSON",
                       help="One or more JSON config files")
    group.add_argument("-d", "--dir", metavar="DIR",
                       help="Directory containing per-protein subdirs (each with a .json)")
    parser.add_argument("--csv", metavar="FILE",
                        help="Export results to a CSV file")
    parser.add_argument("--tsv", metavar="FILE",
                        help="Export results to a TSV file")
    parser.add_argument("--plot", metavar="FILE",
                        help="Save performance plots to a PNG/PDF file")
    parser.add_argument("--grouped", metavar="FILE",
                        help="Export binned table (avg time per step per sequence-length range) to CSV")
    parser.add_argument("--bin_size", type=int, default=100, metavar="N",
                        help="Sequence length bin width for --grouped (default: 100)")
    parser.add_argument("--no_detail", action="store_true",
                        help="Skip per-protein detailed tables, show summary only")
    args = parser.parse_args()

    if args.config:
        json_files = [os.path.abspath(p) for p in args.config]
    else:
        json_files = _find_jsons(os.path.abspath(args.dir))

    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    all_data = []
    for path in json_files:
        if not os.path.isfile(path):
            print(f"Not found: {path}")
            continue
        data = analyze_config(path)
        if not args.no_detail:
            print_protein_report(data)
        all_data.append(data)

    if all_data:
        print_summary_table(all_data)

    if args.csv:
        write_table(all_data, args.csv, sep=",")
    if args.tsv:
        write_table(all_data, args.tsv, sep="\t")
    if args.plot:
        plot_performance(all_data, args.plot)
    if args.grouped:
        write_binned_table(all_data, args.grouped, bin_size=args.bin_size)


if __name__ == "__main__":
    main()
