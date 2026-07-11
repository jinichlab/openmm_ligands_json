#!/usr/bin/env python3
"""
Postprocessing Runner
=====================
Runs a single postprocessing step by reading inputs from the JSON config/state
file and writing the result back into the same file.

Mirrors the pattern of scripts_running/step_runner.py.

Usage (one call per SLURM job or sequential run):
  python postprocessing_runner.py --config protein_001.json --step unwrap
  python postprocessing_runner.py --config protein_001.json --step cvs
  python postprocessing_runner.py --config protein_001.json --step rmsf
  python postprocessing_runner.py --config protein_001.json --step dssp
  python postprocessing_runner.py --config protein_001.json --step sasa
  python postprocessing_runner.py --config protein_001.json --step thermo
  python postprocessing_runner.py --config protein_001.json --step order_parameter

Pipeline order and what each step reads from the JSON
------------------------------------------------------
  unwrap           <- results.production.trajectory
                      results.production.final_structure  (topology)
                      [outputs: unwrapped.dcd, unwrapped_topology.pdb]
  cvs              <- results.unwrap.trajectory
                      results.unwrap.topology
  rmsf             <- results.unwrap.trajectory
                      results.unwrap.topology
  dssp             <- results.unwrap.trajectory
                      results.unwrap.topology
  sasa             <- results.unwrap.trajectory
                      results.unwrap.topology
  thermo           <- results.production.csv_file
  order_parameter  <- results.unwrap.trajectory
                      results.unwrap.topology
  tica             <- results.cvs.outputs.ca_distances
  network_analysis <- results.unwrap.trajectory
                      results.unwrap.topology
"""

import argparse
import json
import os
import sys
import tempfile
import time

# Make sure this directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unwrapp as _unwrap
import calculate_cvs as _cvs
import calculate_rmsf as _rmsf
import calculate_dssp as _dssp
import calculate_sasa as _sasa
import calculate_thermo as _thermo
import calculate_order_parameter as _order
import ligand_rmsd as _ligrmsd
import export_amber as _export_amber
import prepare_mmpbsa as _prepare_mmpbsa

VALID_STEPS = [
    "unwrap",
    "cvs",
    "ligand_rmsd",
    "rmsf",
    "dssp",
    "sasa",
    "thermo",
    "order_parameter",
    "tica",
    "network_analysis",
    "export_amber",
    "prepare_mmpbsa",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(config_path):
    with open(config_path) as f:
        return json.load(f)


def _save(config_path, config):
    dir_ = os.path.dirname(os.path.abspath(config_path))
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
        json.dump(config, f, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, config_path)  # atomic rename — original never partially overwritten


def _locked_update(config_path, step, result, replica_id):
    """
    Acquire an exclusive file lock, reload the JSON, write the result for
    this step, then save — all while holding the lock.  This prevents two
    replicas finishing at the same time from overwriting each other's results.
    """
    import fcntl
    lock_path = config_path + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            config = _load(config_path)
            if config.get("replicas"):
                config["replicas"][replica_id].setdefault("results", {})[step] = result
            else:
                config.setdefault("results", {})[step] = result
            _save(config_path, config)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _get(config, step_key, field):
    """Fetch a field from a completed step's result, with a descriptive error."""
    results = config.get("results", {})
    step = results.get(step_key, {})
    if field not in step:
        raise RuntimeError(
            f"Missing '{field}' in results.{step_key}. "
            f"Did the '{step_key}' step complete successfully? "
            f"Check the JSON file for its status."
        )
    return step[field]


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------

def run_step(config_path, step, replica_id=None):
    config = _load(config_path)

    # ── Resolve output context from replica or top-level ───────────────────
    replicas = config.get("replicas", [])
    if replicas:
        if replica_id is None:
            raise ValueError(f"This config has replicas — pass --replica N")
        if replica_id >= len(replicas):
            raise ValueError(f"--replica {replica_id} out of range ({len(replicas)} replicas)")
        rep     = replicas[replica_id]
        out_dir = rep["output_dir"]
        prefix  = rep["prefix"]
        results = rep.get("results", {})
    else:
        # Legacy single-replica JSON (no replicas list)
        out_dir = config.get("output_dir", ".")
        prefix  = config.get("prefix", "sim")
        results = config.get("results", {})

    os.makedirs(out_dir, exist_ok=True)

    def p(suffix):
        return os.path.join(out_dir, f"{prefix}_{suffix}")

    def g(step_key, field):
        step_res = results.get(step_key, {})
        if field not in step_res:
            raise RuntimeError(
                f"Missing '{field}' in results.{step_key} (replica={replica_id}). "
                f"Did '{step_key}' complete?"
            )
        return step_res[field]

    # Shared (non-replica) results — e.g. system_creation, needed by export_amber
    shared_results = config.get("results", {})

    def gs(step_key, field):
        step_res = shared_results.get(step_key, {})
        if field not in step_res:
            raise RuntimeError(
                f"Missing '{field}' in shared results.{step_key}. Did '{step_key}' complete?"
            )
        return step_res[field]

    pp_cfg = config.get("postprocessing", {})

    t0 = time.time()

    # ------------------------------------------------------------------ #
    if step == "unwrap":
        cfg = pp_cfg.get("unwrap", {})
        result = _unwrap.main(
            trajectory=g("production", "trajectory"),
            topology=g("production", "final_structure"),
            output=p("unwrapped.dcd"),
            remove_water=cfg.get("remove_water", True),
        )

    # ------------------------------------------------------------------ #
    elif step == "cvs":
        cfg = pp_cfg.get("cvs", {})
        result = _cvs.main(
            trajectory=g("unwrap", "trajectory"),
            topology=g("unwrap", "topology"),
            q_out=p("q.dat") if cfg.get("q", True) else None,
            rmsd_out=p("rmsd.dat") if cfg.get("rmsd", True) else None,
            ca_dist_out=p("ca_distances.npy") if cfg.get("ca_distances", True) else None,
            phi_psi_out=p("phi_psi.dat") if cfg.get("phi_psi", False) else None,
            rg_out=p("rg.dat") if cfg.get("rg", True) else None,
            chainid=cfg.get("chainid", 0),
            native_cutoff=cfg.get("native_cutoff", 0.45),
            beta=cfg.get("beta", 50.0),
            lam=cfg.get("lam", 1.8),
            min_seq_sep=cfg.get("min_seq_sep", 3),
            rmsd_sel=cfg.get("rmsd_sel", None),
            ca_min_seq_sep=cfg.get("ca_min_seq_sep", 3),
            phi_psi_deg=cfg.get("phi_psi_deg", False),
            rg_sel=cfg.get("rg_sel", None),
        )

    # ------------------------------------------------------------------ #
    elif step == "ligand_rmsd":
        cfg = pp_cfg.get("ligand_rmsd", {})
        result = _ligrmsd.main(
            trajectory=g("unwrap", "trajectory"),
            topology=g("unwrap", "topology"),
            output=p("ligand_rmsd.csv"),
            align_selection=cfg.get("align_selection", "protein and name CA"),
            ref_frame=cfg.get("ref_frame", 0),
            heavy_only=cfg.get("heavy_only", True),
            ligand_resnames=cfg.get("ligand_resnames") or None,
        )

    # ------------------------------------------------------------------ #
    elif step == "export_amber":
        cfg = pp_cfg.get("ligand_rmsd", {})   # reuse ligand_resnames for the mask
        # export_amber rebuilds an *unconstrained* System from the FF + ligands
        # (the frozen dynamics System has constrained, type-less H/water bonds).
        result = _export_amber.main(
            system_xml=gs("system_creation", "system"),
            topology_pkl=gs("system_creation", "output"),
            output_prefix=p("mmpbsa"),
            ligand_resnames=cfg.get("ligand_resnames") or None,
            ligand_specs=config.get("ligands", []),
            forcefield_name=config.get("force_field", "amber/ff14SB.xml"),
            water_ff=config.get("force_field_water", "amber/tip3p_standard.xml"),
            ligand_ff=config.get("ligand_force_field", "gaff-2.11"),
        )

    # ------------------------------------------------------------------ #
    elif step == "prepare_mmpbsa":
        # Build the dry, pbc-removed trajectory + mmpbsa.in + run_mmpbsa.sh, and
        # run AmberTools MM/PB(GB)SA if it is on PATH (else emit the script).
        cfg = pp_cfg.get("prepare_mmpbsa", {})
        lig_cfg = pp_cfg.get("ligand_rmsd", {})
        resnames = (cfg.get("ligand_resnames") or lig_cfg.get("ligand_resnames")
                    or [l.get("resname") for l in config.get("ligands", []) if l.get("resname")])
        result = _prepare_mmpbsa.main(
            trajectory=g("unwrap", "trajectory"),
            topology=g("unwrap", "topology"),
            dry_prmtop=p("mmpbsa_complex_dry.prmtop"),
            output_dir=os.path.join(out_dir, "mmpbsa"),
            ligand_resnames=resnames,
            align_selection=cfg.get("align_selection", "protein and name CA"),
            igb=cfg.get("igb", 5),
            saltcon=cfg.get("saltcon", 0.15),
            startframe=cfg.get("startframe", 1),
            interval=cfg.get("interval", 1),
            run=cfg.get("run", "auto"),
        )

    # ------------------------------------------------------------------ #
    elif step == "rmsf":
        cfg = pp_cfg.get("rmsf", {})
        result = _rmsf.main(
            trajectory=g("unwrap", "trajectory"),
            topology=g("unwrap", "topology"),
            output=p("rmsf.csv"),
            selection=cfg.get("selection", "protein and name CA"),
            ref_frame=cfg.get("ref_frame", 0),
        )

    # ------------------------------------------------------------------ #
    elif step == "dssp":
        cfg = pp_cfg.get("dssp", {})
        result = _dssp.main(
            trajectory=g("unwrap", "trajectory"),
            topology=g("unwrap", "topology"),
            output=p("dssp.csv"),
            simplified=cfg.get("simplified", True),
            summary_out=p("dssp_summary.csv") if cfg.get("summary", True) else None,
        )

    # ------------------------------------------------------------------ #
    elif step == "sasa":
        cfg = pp_cfg.get("sasa", {})
        result = _sasa.main(
            trajectory=g("unwrap", "trajectory"),
            topology=g("unwrap", "topology"),
            output=p("sasa.csv"),
            mode=cfg.get("mode", "residue"),
            probe_radius=cfg.get("probe_radius", 0.14),
            n_sphere_points=cfg.get("n_sphere_points", 960),
        )

    # ------------------------------------------------------------------ #
    elif step == "thermo":
        cfg = pp_cfg.get("thermo", {})
        rolling_window = cfg.get("rolling_window", 100)
        result = _thermo.main(
            input_csv=g("production", "csv_file"),
            output=p("thermo_summary.csv"),
            rolling_window=rolling_window,
            rolling_out=p("thermo_rolling.csv") if cfg.get("rolling", True) else None,
        )

    # ------------------------------------------------------------------ #
    elif step == "order_parameter":
        cfg = pp_cfg.get("order_parameter", {})
        result = _order.main(
            trajectory=g("unwrap", "trajectory"),
            topology=g("unwrap", "topology"),
            output=p("order_parameter.csv"),
            global_out=p("order_parameter_global.dat"),
            window=cfg.get("window", 5),
            chain=cfg.get("chain", 0),
        )

    # ------------------------------------------------------------------ #
    elif step == "tica":
        import calculate_tica as _tica
        cfg = pp_cfg.get("tica", {})
        ca_distances = g("cvs", "outputs")["ca_distances"]
        result = _tica.main(
            ca_distances=ca_distances,
            output=p("tica.npy"),
            output_dat=p("tica.dat") if cfg.get("save_dat", True) else None,
            lag=cfg.get("lag", 10),
            dim=cfg.get("dim", 2),
            stride=cfg.get("stride", 1),
            nan_policy=cfg.get("nan_policy", "error"),
            model_out=p("tica_model.pkl") if cfg.get("save_model", False) else None,
        )

    # ------------------------------------------------------------------ #
    elif step == "network_analysis":
        import network_analysis as _network
        cfg = pp_cfg.get("network_analysis", {})
        result = _network.make_network_analysis(
            pdb_path=g("unwrap", "topology"),
            trajectory_path=g("unwrap", "trajectory"),
            seg_ids=cfg.get("seg_ids", ["A"]),
            job_name=prefix,
            n_windows=cfg.get("n_windows", 4),
            sampled_frames=cfg.get("sampled_frames", 10),
            cutoff=cfg.get("cutoff", 4.5),
            n_cores=cfg.get("n_jobs", 4),
            new_dcd=cfg.get("new_dcd", False),
            dcd_stride=cfg.get("dcd_stride", 1),
            out=out_dir,
        )

    # ------------------------------------------------------------------ #
    else:
        raise ValueError(f"Unknown step '{step}'. Valid steps: {VALID_STEPS}")

    result["elapsed_seconds"] = round(time.time() - t0, 2)

    _locked_update(config_path, step, result, replica_id)

    print(f"\nResult for '{step}' written to: {config_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run one postprocessing step, reading inputs from and writing results to a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Steps (in order):",
            *[f"  {s}" for s in VALID_STEPS],
        ]),
    )
    parser.add_argument("--config", required=True,
                        help="Path to the JSON config/state file for this protein")
    parser.add_argument("--step", required=True, choices=VALID_STEPS,
                        help="Which postprocessing step to run")
    parser.add_argument("--replica", type=int, default=None,
                        help="Replica index (required when the JSON contains a replicas list)")
    parser.add_argument("--skip_if_done", action="store_true",
                        help="Do nothing if this step already has status=completed in the JSON")
    args = parser.parse_args()

    if args.skip_if_done:
        cfg = _load(args.config)
        if cfg.get("replicas"):
            rep_id = args.replica or 0
            done = (cfg["replicas"][rep_id].get("results", {})
                       .get(args.step, {}).get("status") == "completed")
        else:
            done = cfg.get("results", {}).get(args.step, {}).get("status") == "completed"
        if done:
            print(f"[skip] '{args.step}' already completed for {args.config} replica={args.replica}")
            return

    run_step(args.config, args.step, replica_id=args.replica)


if __name__ == "__main__":
    main()
