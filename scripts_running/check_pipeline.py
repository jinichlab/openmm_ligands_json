#!/usr/bin/env python3
"""
Pipeline status checker.

Given one or more JSON config files (or a directory containing them), reports
the completion status of every MD and postprocessing step, flags convergence
warnings, and verifies that key output files are present on disk.

Usage:
  # Single protein
  python check_pipeline.py -c protein.json

  # Multiple proteins
  python check_pipeline.py -c a.json b.json c.json

  # Entire directory of protein subdirs (each containing one .json)
  python check_pipeline.py -d /runs/jsons/
"""

import argparse
import csv
import json
import os
import sys

# ---------------------------------------------------------------------------
# Pipeline definition
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

# Keep for backward-compat (legacy single flat results dict)
MD_STEPS = SHARED_STEPS + REPLICA_MD_STEPS
ALL_STEPS = MD_STEPS + PP_STEPS

# Key output files to verify on disk per step (field names inside the result dict)
OUTPUT_FIELDS = {
    "pdb_fixer":        ["output"],
    "minimization_vac": ["output"],
    "system_creation":  ["output"],
    "minimization_sol": ["output"],
    "nvt":              ["trajectory", "final_structure", "state_file"],
    "npt":              ["trajectory", "final_structure", "state_file"],
    "production":       ["trajectory", "final_structure", "state_file"],
    "unwrap":           ["trajectory", "topology"],
    "cvs":              [],   # outputs is a nested dict — handled separately
    "rmsf":             ["output"],
    "dssp":             ["output"],
    "sasa":             ["output"],
    "thermo":           ["output"],
    "tica":             ["output"],
    "order_parameter":  ["output", "global_output"],
    "network_analysis": [],   # dynetan saves its own files; prefix not a single path
}

# Steps where a "converged" boolean is meaningful
CONVERGENCE_STEPS = {"check_nvt", "check_npt"}

# ANSI colours (disabled automatically on non-TTY)
_USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

OK    = lambda t: _c("32", t)   # green
WARN  = lambda t: _c("33", t)   # yellow
ERR   = lambda t: _c("31", t)   # red
DIM   = lambda t: _c("2",  t)   # grey
BOLD  = lambda t: _c("1",  t)   # bold


# ---------------------------------------------------------------------------
# Per-step check
# ---------------------------------------------------------------------------

def _check_step(step, result):
    """
    Returns (tag, detail_lines) where tag is one of:
      'ok', 'warn', 'fail', 'missing'
    """
    if result is None:
        return "missing", []

    status = result.get("status", "unknown")
    if status != "completed":
        return "fail", [f"status = {status!r}"]

    details = []
    issues  = []

    # Convergence check
    if step in CONVERGENCE_STEPS:
        converged = result.get("converged")
        if converged is False:
            issues.append("NOT converged")
            for metric, info in result.get("metrics", {}).items():
                if not info.get("stable", True):
                    issues.append(f"  unstable: {metric}")
        elif converged is True:
            details.append("converged")

    # File existence check
    for field in OUTPUT_FIELDS.get(step, []):
        path = result.get(field)
        if path and not os.path.exists(path):
            issues.append(f"missing file: {os.path.basename(path)}")

    # CVs: check nested outputs dict
    if step == "cvs":
        for name, path in result.get("outputs", {}).items():
            if path and not os.path.exists(path):
                issues.append(f"missing file: {os.path.basename(path)}")

    # Elapsed time
    elapsed = result.get("elapsed_seconds")
    if elapsed is not None:
        details.append(f"{elapsed:.1f}s")

    if issues:
        return "warn", issues + details
    return "ok", details


# ---------------------------------------------------------------------------
# Single-JSON report
# ---------------------------------------------------------------------------

def check_config(config_path):
    """
    Returns a dict:
      {
        "shared": { step: {"tag": ..., "details": [...]} },
        "replicas": [
          { step: {"tag": ..., "details": [...]} },
          ...
        ],
        "n_replicas": int,
        "seeds": [int, ...],
      }
    """
    with open(config_path) as f:
        config = json.load(f)

    replicas = config.get("replicas", [])

    time_step_ps = config.get("production", {}).get("time_step", 0.002)

    if replicas:
        # New multi-replica format
        shared_results = config.get("results", {})
        shared = {}
        for step in SHARED_STEPS:
            res = shared_results.get(step)
            tag, details = _check_step(step, res)
            shared[step] = {"tag": tag, "details": details}

        rep_reports = []
        total_ns = 0.0
        for rep in replicas:
            rep_results = rep.get("results", {})
            rep_report = {}
            for step in REPLICA_MD_STEPS + PP_STEPS:
                res = rep_results.get(step)
                tag, details = _check_step(step, res)
                rep_report[step] = {"tag": tag, "details": details}
            rep_reports.append(rep_report)
            prod = rep_results.get("production", {})
            if prod.get("status") == "completed":
                total_ns += prod.get("steps_run", 0) * time_step_ps / 1000.0

        return {
            "shared": shared,
            "replicas": rep_reports,
            "n_replicas": len(replicas),
            "seeds": [r.get("seed", 0) for r in replicas],
            "total_ns": total_ns,
        }
    else:
        # Legacy flat format
        results = config.get("results", {})
        flat = {}
        for step in ALL_STEPS:
            res = results.get(step)
            tag, details = _check_step(step, res)
            flat[step] = {"tag": tag, "details": details}
        prod = results.get("production", {})
        total_ns = 0.0
        if prod.get("status") == "completed":
            total_ns = prod.get("steps_run", 0) * time_step_ps / 1000.0
        return {"shared": flat, "replicas": [], "n_replicas": 1, "seeds": [0],
                "total_ns": total_ns}


def _tag_str(tag):
    labels = {
        "ok":      OK("  OK  "),
        "warn":    WARN(" WARN "),
        "fail":    ERR(" FAIL "),
        "missing": DIM("  --  "),
    }
    return labels.get(tag, f"[{tag}]")


def _print_step_row(info_dict, step):
    info = info_dict.get(step, {"tag": "missing", "details": []})
    tag        = info["tag"]
    details    = info["details"]
    detail_str = "  " + "  |  ".join(details) if details else ""
    print(f"    [{_tag_str(tag)}]  {step:<22s}{DIM(detail_str)}")


def print_report(config_path, report):
    name = os.path.splitext(os.path.basename(config_path))[0]
    n_rep = report["n_replicas"]
    print(BOLD(f"\n=== {name}  ({n_rep} replica{'s' if n_rep > 1 else ''}) ==="))
    print(f"    {DIM(config_path)}")

    # ── Shared steps ─────────────────────────────────────────────────────
    print(DIM("    --- shared steps (pdb_fixer → minimization_sol) ---"))
    for step in SHARED_STEPS:
        _print_step_row(report["shared"], step)

    # ── Per-replica steps ─────────────────────────────────────────────────
    for i, rep_report in enumerate(report["replicas"]):
        seed = report["seeds"][i]
        seed_str = f"  seed={seed}" if seed else "  seed=random"
        print(DIM(f"    --- replica {i}{seed_str} ---"))
        for step in REPLICA_MD_STEPS:
            _print_step_row(rep_report, step)
        print(DIM("    --- postprocessing ---"))
        for step in PP_STEPS:
            _print_step_row(rep_report, step)

    # Legacy flat format (no replicas list)
    if not report["replicas"]:
        print(DIM("    --- MD pipeline ---"))
        for step in MD_STEPS:
            _print_step_row(report["shared"], step)
        print(DIM("    --- postprocessing ---"))
        for step in PP_STEPS:
            _print_step_row(report["shared"], step)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _count_tags(info_dict):
    counts = {"ok": 0, "warn": 0, "fail": 0, "missing": 0}
    for info in info_dict.values():
        counts[info["tag"]] += 1
    return counts


def _add_counts(a, b):
    return {k: a[k] + b[k] for k in a}


def _is_complete_through(report, through):
    """
    Return True if the protein has completed all steps up to (and including)
    the specified stage.  'ok' and 'warn' both count as completed.

    through='production'     — shared steps + all replica MD steps
    through='postprocessing' — all of the above + all PP steps
    """
    def _done(tag):
        return tag in ("ok", "warn")

    for info in report["shared"].values():
        if not _done(info["tag"]):
            return False

    md_check   = REPLICA_MD_STEPS
    pp_check   = REPLICA_MD_STEPS + PP_STEPS if through == "postprocessing" else []
    steps_check = md_check if through == "production" else pp_check

    for rep_report in report["replicas"]:
        for step in steps_check:
            tag = rep_report.get(step, {"tag": "missing"})["tag"]
            if not _done(tag):
                return False

    return True


def _get_incomplete_steps(report):
    """Return list of label strings for steps with tag fail or missing."""
    items = []
    for step, info in report["shared"].items():
        if info["tag"] in ("fail", "missing"):
            items.append(step)
    for i, rep_report in enumerate(report["replicas"]):
        for step, info in rep_report.items():
            if info["tag"] in ("fail", "missing"):
                items.append(f"rep{i}:{step}")
    return items


def print_summary(all_results, csv_path=None):
    """all_results: list of (config_path, report)"""
    totals  = {"ok": 0, "warn": 0, "fail": 0, "missing": 0}
    rows    = []

    total_grand_ns = 0.0
    for path, report in all_results:
        pc = _count_tags(report["shared"])
        for rep_report in report["replicas"]:
            pc = _add_counts(pc, _count_tags(rep_report))
        totals = _add_counts(totals, pc)
        incomplete = _get_incomplete_steps(report)
        ns = report.get("total_ns", 0.0)
        total_grand_ns += ns
        rows.append((path, pc, incomplete, ns))

    n_proteins = len(all_results)
    print(BOLD(f"\n{'='*60}"))
    print(BOLD(f"SUMMARY  —  {n_proteins} protein(s)"))
    print(f"{'='*60}")

    col = 30
    hdr = f"  {'protein':<{col}}  {'ok':>4}  {'warn':>4}  {'fail':>4}  {'n/a':>4}  {'ns':>7}  incomplete steps"
    print(DIM(hdr))
    for path, pc, incomplete, ns in rows:
        name   = os.path.splitext(os.path.basename(path))[0]
        ok_s   = OK(f"{pc['ok']:>4}")     if pc["ok"]   else DIM(f"{'0':>4}")
        warn_s = WARN(f"{pc['warn']:>4}") if pc["warn"] else DIM(f"{'0':>4}")
        fail_s = ERR(f"{pc['fail']:>4}")  if pc["fail"] else DIM(f"{'0':>4}")
        miss_s = DIM(f"{pc['missing']:>4}")
        ns_s   = OK(f"{ns:>7.2f}") if ns > 0 else DIM(f"{'0.00':>7}")
        inc_s  = DIM(", ".join(incomplete)) if incomplete else OK("none")
        print(f"  {name:<{col}}  {ok_s}  {warn_s}  {fail_s}  {miss_s}  {ns_s}  {inc_s}")

    print(DIM(f"  {'-'*60}"))
    tot_ok   = OK(f"{totals['ok']:>4}")   if totals['ok']   else DIM('   0')
    tot_warn = WARN(f"{totals['warn']:>4}") if totals['warn'] else DIM('   0')
    tot_fail = ERR(f"{totals['fail']:>4}") if totals['fail'] else DIM('   0')
    tot_miss = DIM(f"{totals['missing']:>4}")
    tot_ns   = OK(f"{total_grand_ns:>7.2f}") if total_grand_ns > 0 else DIM(f"{'0.00':>7}")
    print(f"  {'TOTAL':<{col}}  {tot_ok}  {tot_warn}  {tot_fail}  {tot_miss}  {tot_ns}")
    print()

    # Flag anything needing attention
    def _all_infos(report):
        yield from report["shared"].values()
        for rep in report["replicas"]:
            yield from rep.values()

    attention = [(p, r) for p, r in all_results
                 if any(i["tag"] in ("fail", "warn") for i in _all_infos(r))]
    if attention:
        print(WARN("Steps needing attention:"))
        for path, report in attention:
            name = os.path.splitext(os.path.basename(path))[0]
            for step, info in report["shared"].items():
                if info["tag"] in ("fail", "warn"):
                    tag_s = ERR("FAIL") if info["tag"] == "fail" else WARN("WARN")
                    print(f"  {tag_s}  {name}  →  {step}  {DIM(' | '.join(info['details']))}")
            for i, rep_report in enumerate(report["replicas"]):
                for step, info in rep_report.items():
                    if info["tag"] in ("fail", "warn"):
                        tag_s = ERR("FAIL") if info["tag"] == "fail" else WARN("WARN")
                        print(f"  {tag_s}  {name} rep{i}  →  {step}  {DIM(' | '.join(info['details']))}")
        print()

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["protein", "ok", "warn", "fail", "missing", "total_ns", "incomplete_steps"])
            for path, pc, incomplete, ns in rows:
                name = os.path.splitext(os.path.basename(path))[0]
                writer.writerow([name, pc["ok"], pc["warn"], pc["fail"], pc["missing"],
                                  f"{ns:.3f}", "; ".join(incomplete)])
        print(f"Summary saved to: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_jsons(directory):
    found = []
    for entry in sorted(os.listdir(directory)):
        sub = os.path.join(directory, entry)
        if os.path.isdir(sub):
            candidate = os.path.join(sub, f"{entry}.json")
            if os.path.isfile(candidate):
                found.append(candidate)
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Check completion status of the MD pipeline for one or more proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", nargs="+", metavar="JSON",
                       help="One or more JSON config files")
    group.add_argument("-d", "--dir", metavar="DIR",
                       help="Directory containing per-protein subdirs (each with a .json)")
    parser.add_argument("--csv", metavar="FILE",
                        help="Save summary table to a CSV file")
    parser.add_argument("--filter", metavar="STAGE",
                        choices=["production", "postprocessing"],
                        help="Only show proteins that have completed up to "
                             "'production' (shared + MD steps) or "
                             "'postprocessing' (all steps)")
    args = parser.parse_args()

    if args.config:
        json_files = [os.path.abspath(p) for p in args.config]
    else:
        json_files = _find_jsons(os.path.abspath(args.dir))

    if not json_files:
        print(ERR("No JSON files found."))
        sys.exit(1)

    all_results = []
    for path in json_files:
        if not os.path.isfile(path):
            print(ERR(f"Not found: {path}"))
            continue
        step_results = check_config(path)
        all_results.append((path, step_results))

    if args.filter:
        all_results = [(p, r) for p, r in all_results
                       if _is_complete_through(r, args.filter)]
        n_total = len(json_files)
        print(BOLD(f"\nFilter: completed through '{args.filter}'  "
                   f"({len(all_results)}/{n_total} proteins match)"))

    for path, step_results in all_results:
        print_report(path, step_results)

    if len(all_results) > 1 or True:   # always show summary
        print_summary(all_results, csv_path=args.csv)


if __name__ == "__main__":
    main()
