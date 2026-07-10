#!/usr/bin/env python3
"""
Reconcile JSON
==============
Scans output folders for files that were produced by pipeline steps but are
not recorded (or are recorded as incomplete) in the JSON config.

For every step whose primary output file(s) exist on disk, the script patches
the JSON with a minimal result entry so downstream steps can proceed and
check_pipeline/analyze_performance report the correct status.

All patched entries carry  "reconciled": true  so they are distinguishable
from entries written by actual runs.

Usage:
  python reconcile_json.py -c protein.json
  python reconcile_json.py -d /runs/jsons/
  python reconcile_json.py -d /runs/jsons/ --dry-run   # show only, no writes
"""

import argparse
import json
import os
import sys
import tempfile


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

SHARED_STEPS  = ["pdb_fixer", "minimization_vac", "system_creation", "minimization_sol"]
REPLICA_STEPS = ["nvt", "check_nvt", "npt", "check_npt", "production"]
PP_STEPS      = ["unwrap", "cvs", "rmsf", "dssp", "sasa", "thermo",
                 "tica", "order_parameter", "network_analysis"]


# ---------------------------------------------------------------------------
# Per-step probe: what files to look for and how to build a minimal result
# ---------------------------------------------------------------------------

def _exists(*paths):
    return all(os.path.isfile(p) for p in paths if p)


def _probe_shared(step, out_dir, prefix):
    """
    Returns (primary_file_exists: bool, minimal_result: dict | None).
    """
    p = lambda s: os.path.join(out_dir, f"{prefix}_{s}")

    if step == "pdb_fixer":
        f = p("fixed.cif")
        if not _exists(f):
            return False, None
        return True, {
            "step": "pdb_fixer", "status": "completed",
            "output": f, "reconciled": True,
        }

    if step == "minimization_vac":
        f = p("min_vac.cif")
        if not _exists(f):
            return False, None
        return True, {
            "step": "minimization_vac", "status": "completed",
            "output": f, "reconciled": True,
        }

    if step == "system_creation":
        f = p("solvated.cif")
        if not _exists(f):
            return False, None
        n_atoms = _read_n_atoms(f)
        result = {
            "step": "system_creation", "status": "completed",
            "output": f, "reconciled": True,
        }
        if n_atoms is not None:
            result["n_atoms"] = n_atoms
        return True, result

    if step == "minimization_sol":
        f = p("min_sol.cif")
        if not _exists(f):
            return False, None
        return True, {
            "step": "minimization_sol", "status": "completed",
            "output": f, "reconciled": True,
        }

    return False, None


def _probe_replica(step, out_dir, prefix):
    """
    Returns (primary_file_exists: bool, minimal_result: dict | None).
    """
    p = lambda s: os.path.join(out_dir, f"{prefix}_{s}")

    if step == "nvt":
        pdb = p("nvt.pdb")
        if not _exists(pdb):
            return False, None
        result = {
            "step": "nvt", "status": "completed", "reconciled": True,
            "final_structure": pdb,
            "output_prefix": os.path.join(out_dir, f"{prefix}_nvt"),
        }
        for key, suf in [("state_file", "nvt.xml"), ("trajectory", "nvt.dcd"),
                         ("csv_file", "nvt.csv"), ("checkpoint", "nvt.chk")]:
            f = p(suf)
            if os.path.isfile(f):
                result[key] = f
        return True, result

    if step == "check_nvt":
        svg = p("check_nvt.svg")
        if not _exists(svg):
            return False, None
        return True, {
            "step": "check_nvt", "status": "completed",
            "output": svg, "reconciled": True,
        }

    if step == "npt":
        pdb = p("npt.pdb")
        if not _exists(pdb):
            return False, None
        result = {
            "step": "npt", "status": "completed", "reconciled": True,
            "final_structure": pdb,
            "output_prefix": os.path.join(out_dir, f"{prefix}_npt"),
        }
        for key, suf in [("state_file", "npt.xml"), ("trajectory", "npt.dcd"),
                         ("csv_file", "npt.csv"), ("checkpoint", "npt.chk")]:
            f = p(suf)
            if os.path.isfile(f):
                result[key] = f
        return True, result

    if step == "check_npt":
        svg = p("check_npt.svg")
        if not _exists(svg):
            return False, None
        return True, {
            "step": "check_npt", "status": "completed",
            "output": svg, "reconciled": True,
        }

    if step == "production":
        pdb = p("production.pdb")
        if not _exists(pdb):
            return False, None
        result = {
            "step": "production", "status": "completed", "reconciled": True,
            "final_structure": pdb,
            "output_prefix": os.path.join(out_dir, f"{prefix}_production"),
        }
        for key, suf in [("state_file", "production.xml"), ("trajectory", "production.dcd"),
                         ("csv_file", "production.csv"), ("checkpoint", "production.chk")]:
            f = p(suf)
            if os.path.isfile(f):
                result[key] = f
        return True, result

    return False, None


def _probe_pp(step, out_dir, prefix):
    """
    Returns (primary_file_exists: bool, minimal_result: dict | None).
    """
    p = lambda s: os.path.join(out_dir, f"{prefix}_{s}")

    probes = {
        "unwrap": ("unwrapped.dcd", lambda f: {
            "step": "unwrap", "status": "completed", "reconciled": True,
            "trajectory": f,
            "topology": p("unwrapped_topology.pdb") if os.path.isfile(p("unwrapped_topology.pdb")) else None,
        }),
        "cvs": ("q.dat", lambda f: {
            "step": "cvs", "status": "completed", "reconciled": True,
            "outputs": {
                k: p(suf) for k, suf in [
                    ("q",            "q.dat"),
                    ("rmsd",         "rmsd.dat"),
                    ("ca_distances", "ca_distances.npy"),
                    ("ca_pairs",     "ca_distances_pairs.npy"),
                    ("phi_psi",      "phi_psi.dat"),
                    ("rg",           "rg.dat"),
                ] if os.path.isfile(p(suf))
            },
        }),
        "rmsf": ("rmsf.csv", lambda f: {
            "step": "rmsf", "status": "completed", "reconciled": True,
            "output": f,
        }),
        "dssp": ("dssp.csv", lambda f: {
            "step": "dssp", "status": "completed", "reconciled": True,
            "output": f,
            "summary": p("dssp_summary.csv") if os.path.isfile(p("dssp_summary.csv")) else None,
        }),
        "sasa": ("sasa.csv", lambda f: {
            "step": "sasa", "status": "completed", "reconciled": True,
            "output": f,
        }),
        "thermo": ("thermo_rolling.csv", lambda f: {
            "step": "thermo", "status": "completed", "reconciled": True,
            "output": f,
            "summary": p("thermo_summary.csv") if os.path.isfile(p("thermo_summary.csv")) else None,
        }),
        "tica": ("tica.npy", lambda f: {
            "step": "tica", "status": "completed", "reconciled": True,
            "output": f,
        }),
        "order_parameter": ("order_parameter.csv", lambda f: {
            "step": "order_parameter", "status": "completed", "reconciled": True,
            "output": f,
        }),
        "network_analysis": ("nxGraphs.pickle", lambda f: {
            "step": "network_analysis", "status": "completed", "reconciled": True,
            "output": f,
        }),
    }

    if step not in probes:
        return False, None

    suffix, builder = probes[step]
    f = p(suffix)
    if not os.path.isfile(f):
        return False, None
    result = builder(f)
    # Remove None values from result
    result = {k: v for k, v in result.items() if v is not None}
    if isinstance(result.get("outputs"), dict):
        result["outputs"] = {k: v for k, v in result["outputs"].items() if v}
    return True, result


# ---------------------------------------------------------------------------
# Read n_atoms from solvated CIF using OpenMM (optional)
# ---------------------------------------------------------------------------

def _read_n_atoms(cif_path):
    try:
        from openmm.app import PDBxFile
        pdb = PDBxFile(cif_path)
        return pdb.topology.getNumAtoms()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core reconcile logic for one JSON file
# ---------------------------------------------------------------------------

def _is_completed(results_dict, step):
    return results_dict.get(step, {}).get("status") == "completed"


def reconcile(config_path, dry_run=False):
    with open(config_path) as f:
        config = json.load(f)

    shared_results = config.setdefault("results", {})
    out_dir_top    = config.get("output_dir", os.path.dirname(config_path))
    prefix_top     = config.get("prefix", os.path.splitext(os.path.basename(config_path))[0])
    replicas       = config.get("replicas", [])

    patches = []   # list of human-readable strings describing what was patched

    # ── Shared steps ─────────────────────────────────────────────────────────
    for step in SHARED_STEPS:
        if _is_completed(shared_results, step):
            continue
        found, result = _probe_shared(step, out_dir_top, prefix_top)
        if found:
            if not dry_run:
                shared_results[step] = result
            patches.append(f"  [shared]    {step}")

    # ── Replica steps ─────────────────────────────────────────────────────────
    for rep in replicas:
        rep_id     = rep["id"]
        rep_dir    = rep.get("output_dir", out_dir_top)
        rep_prefix = rep.get("prefix", prefix_top)
        rep_results = rep.setdefault("results", {})

        for step in REPLICA_STEPS:
            if _is_completed(rep_results, step):
                continue
            found, result = _probe_replica(step, rep_dir, rep_prefix)
            if found:
                if not dry_run:
                    rep_results[step] = result
                patches.append(f"  [replica {rep_id}] {step}")

        for step in PP_STEPS:
            if _is_completed(rep_results, step):
                continue
            found, result = _probe_pp(step, rep_dir, rep_prefix)
            if found:
                if not dry_run:
                    rep_results[step] = result
                patches.append(f"  [replica {rep_id}] {step} (postprocessing)")

    # ── Save if anything changed ──────────────────────────────────────────────
    if patches and not dry_run:
        dir_ = os.path.dirname(os.path.abspath(config_path))
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tf:
            json.dump(config, tf, indent=2)
            tmp = tf.name
        os.replace(tmp, config_path)

    return patches


# ---------------------------------------------------------------------------
# JSON discovery
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconcile JSON config with files on disk — patch missing completed-step entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", nargs="+", metavar="JSON",
                       help="One or more JSON config files")
    group.add_argument("-d", "--dir", metavar="DIR",
                       help="Directory containing per-protein subdirs (each with a .json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be patched without modifying any files")
    args = parser.parse_args()

    if args.config:
        json_files = [os.path.abspath(p) for p in args.config]
    else:
        json_files = _find_jsons(os.path.abspath(args.dir))

    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    if args.dry_run:
        print("[dry-run] no files will be modified\n")

    total_patched = 0
    for path in json_files:
        if not os.path.isfile(path):
            print(f"Not found: {path}")
            continue
        patches = reconcile(path, dry_run=args.dry_run)
        name = os.path.basename(path)
        if patches:
            action = "would patch" if args.dry_run else "patched"
            print(f"{name}  — {action} {len(patches)} step(s):")
            for p in patches:
                print(p)
            total_patched += len(patches)
        else:
            print(f"{name}  — nothing to reconcile")

    print(f"\nDone. {total_patched} entries {'would be ' if args.dry_run else ''}added across {len(json_files)} file(s).")


if __name__ == "__main__":
    main()
