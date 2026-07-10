#!/usr/bin/env python3
"""
Step Runner
===========
Runs a single pipeline step by reading inputs from a JSON config/state file
and writing the result back into the same file.

The JSON serves as both configuration and progress tracker across SLURM jobs.
Each step looks up the output files of its upstream steps from the JSON,
so there is no need to hard-code paths or count files between jobs.

Usage (one call per SLURM job):
  python step_runner.py --config protein_001.json --step pdb_fixer
  python step_runner.py --config protein_001.json --step minimization_vac
  python step_runner.py --config protein_001.json --step system_creation
  python step_runner.py --config protein_001.json --step minimization_sol
  python step_runner.py --config protein_001.json --step nvt
  python step_runner.py --config protein_001.json --step npt
  python step_runner.py --config protein_001.json --step production

Restart an interrupted NPT or production run:
  python step_runner.py --config protein_001.json --step npt        --restart
  python step_runner.py --config protein_001.json --step production --restart

Pipeline order and what each step reads from the JSON
------------------------------------------------------
  pdb_fixer        <- config.pdb
  minimization_vac <- results.pdb_fixer.output
  system_creation  <- results.minimization_vac.output
  minimization_sol <- results.system_creation.output
  nvt              <- results.minimization_sol.output
  check_nvt        <- results.nvt.csv_file  (convergence plot, review before npt)
  npt              <- results.system_creation.output  (topology)
                      results.nvt.state_file          (positions/velocities/box)
  check_npt        <- results.npt.csv_file  (convergence plot, review before production)
  production       <- results.system_creation.output  (topology)
                      results.npt.state_file           (positions/velocities/box)
"""

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts_postprocessing"))

import pdb_fixer as _pdb_fixer
import system_creation as _system_creation
import minimization as _minimization
import equilibration_nvt_steps as _nvt
import equilibration_npt as _npt
import analyze_equilibrations as _check

VALID_STEPS = [
    "pdb_fixer",
    "minimization_vac",
    "system_creation",
    "minimization_sol",
    "nvt",
    "check_nvt",
    "npt",
    "check_npt",
    "production",
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


def _locked_update(config_path, step, result, shared, replica_id=None):
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
            if shared:
                config.setdefault("results", {})[step] = result
            else:
                config["replicas"][replica_id].setdefault("results", {})[step] = result
            _save(config_path, config)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _get(result_dicts, step_key, field):
    """
    Fetch a field from a completed step's result.
    result_dicts: list of result dicts to search in order (replica first, then shared).
    """
    for results in result_dicts:
        step = results.get(step_key, {})
        if field in step:
            return step[field]
    raise RuntimeError(
        f"Missing '{field}' in results.{step_key}. "
        f"Did the '{step_key}' step complete successfully? "
        f"Check the JSON file for its status."
    )


# Steps that belong to the shared portion of the pipeline
SHARED_STEPS = {"pdb_fixer", "minimization_vac", "system_creation", "minimization_sol"}
# Steps that belong to a specific replica
REPLICA_STEPS = {"nvt", "check_nvt", "npt", "check_npt", "production"}


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------

def run_step(config_path, step, restart=False, replica_id=None):
    config = _load(config_path)

    ff      = config.get("force_field", "amber/ff14SB.xml")
    ff_w    = config.get("force_field_water", "amber/tip3p_standard.xml")
    lig_ff  = config.get("ligand_force_field", "gaff-2.11")
    ligands = config.get("ligands", [])           # list of {"sdf": ..., "resname": ...}; [] = apo
    plat    = config.get("platform", "CUDA")
    device  = config.get("device", 0)
    hmass   = config.get("hydrogen_mass", None)   # None = no HMR; 4.0 = 4 amu HMR

    shared_results = config.get("results", {})

    # ── Resolve context: shared vs replica ─────────────────────────────────
    if step in SHARED_STEPS:
        out_dir = config.get("output_dir", ".")
        prefix  = config.get("prefix", "sim")
        seed    = 0
        # Results lookup: shared only
        result_dicts = [shared_results]

    elif step in REPLICA_STEPS:
        if replica_id is None:
            raise ValueError(f"Step '{step}' requires --replica N")
        replicas = config.get("replicas", [])
        if replica_id >= len(replicas):
            raise ValueError(f"--replica {replica_id} out of range ({len(replicas)} replicas)")
        rep      = replicas[replica_id]
        out_dir  = rep["output_dir"]
        prefix   = rep["prefix"]
        seed     = rep.get("seed", 0)
        rep_results = rep.get("results", {})
        # Results lookup: replica first, then shared (e.g. npt needs system_creation from shared)
        result_dicts = [rep_results, shared_results]

    else:
        raise ValueError(f"Unknown step '{step}'. Valid steps: {VALID_STEPS}")

    os.makedirs(out_dir, exist_ok=True)

    def p(suffix):
        return os.path.join(out_dir, f"{prefix}_{suffix}")

    def g(step_key, field):
        return _get(result_dicts, step_key, field)

    t0 = time.time()

    # ------------------------------------------------------------------ #
    if step == "pdb_fixer":
        result = _pdb_fixer.main(
            path=config["pdb"],
            ph=config.get("ph", 7.0),
            output=p("fixed"),
            remove_heterogens=config.get("remove_heterogens", False),
            keep_water=config.get("keep_water", False),
            force_field=ff,
            force_field_water=ff_w,
        )

    # ------------------------------------------------------------------ #
    elif step == "minimization_vac":
        cfg = config.get("minimization_vac", config.get("minimization", {}))
        result = _minimization.minimize_vacuum(
            protein_pdbx=g("pdb_fixer", "output"),
            ligand_specs=ligands,
            force_field=ff,
            water_ff=ff_w,
            ligand_ff=lig_ff,
            steps=cfg.get("steps", 5000),
            platform_=plat,
            cores=device,
            output=p("min_vac"),
            cache=p("ligand_cache.json"),
        )

    # ------------------------------------------------------------------ #
    elif step == "system_creation":
        result = _system_creation.main(
            complex_pkl=g("minimization_vac", "output"),
            ligand_specs=ligands,
            forcefield_name=ff,
            water_ff=ff_w,
            ligand_ff=lig_ff,
            box_size=config.get("box_size", 8.0),
            output=p("solvated"),
            hydrogen_mass=hmass,
            cache=p("ligand_cache.json"),
        )

    # ------------------------------------------------------------------ #
    elif step == "minimization_sol":
        cfg = config.get("minimization_sol", config.get("minimization", {}))
        result = _minimization.minimize_solvated(
            topology_pkl=g("system_creation", "output"),
            system_xml=g("system_creation", "system"),
            steps=cfg.get("steps", 5000),
            platform_=plat,
            cores=device,
            output=p("min_sol"),
            hydrogen_mass=hmass,
        )

    # ------------------------------------------------------------------ #
    elif step == "nvt":
        cfg = config.get("nvt", {})
        result = _nvt.main(
            topology_pkl=g("minimization_sol", "output"),
            system_xml=g("system_creation", "system"),
            simulation_steps=cfg.get("steps", 200000),
            recorder_steps=cfg.get("recorder", 500),
            platform_=plat,
            cores=device,
            output=p("nvt"),
            seed=seed,
            time_step=cfg.get("time_step", 0.001),
            restrain_ligand=cfg.get("restrain_ligand", False),
            hydrogen_mass=hmass,
        )

    # ------------------------------------------------------------------ #
    elif step == "npt":
        cfg = config.get("npt", {})
        chk = cfg.get("checkpoint", p("npt.chk"))
        result = _npt.main(
            topology_pkl=g("system_creation", "output"),
            system_xml=g("system_creation", "system"),
            trajectory=g("nvt", "state_file"),
            simulation_steps=cfg.get("steps", 500000),
            time_step=cfg.get("time_step", 0.002),
            recorder_steps=cfg.get("recorder", 5000),
            platform_=plat,
            cores=device,
            output=p("npt"),
            apply_restraints=True,
            restart=restart,
            checkpoint_path=chk,
            seed=seed,
            hydrogen_mass=hmass,
        )

    # ------------------------------------------------------------------ #
    elif step == "check_nvt":
        cfg = config.get("check_nvt", {})
        result = _check.main(
            csv_file=g("nvt", "csv_file"),
            trajectory=g("nvt", "trajectory"),
            topology=g("nvt", "final_structure"),
            output=p("check_nvt.svg"),
            step_type="nvt",
            rolling_window=cfg.get("rolling_window", 50),
        )

    # ------------------------------------------------------------------ #
    elif step == "check_npt":
        cfg = config.get("check_npt", {})
        result = _check.main(
            csv_file=g("npt", "csv_file"),
            trajectory=g("npt", "trajectory"),
            topology=g("npt", "final_structure"),
            output=p("check_npt.svg"),
            step_type="npt",
            rolling_window=cfg.get("rolling_window", 50),
        )

    # ------------------------------------------------------------------ #
    elif step == "production":
        cfg = config.get("production", {})
        chk = cfg.get("checkpoint", p("production.chk"))
        result = _npt.main(
            topology_pkl=g("system_creation", "output"),
            system_xml=g("system_creation", "system"),
            trajectory=g("npt", "state_file"),
            simulation_steps=cfg.get("steps", 2500000),
            time_step=cfg.get("time_step", 0.002),
            recorder_steps=cfg.get("recorder", 5000),
            platform_=plat,
            cores=device,
            output=p("production"),
            apply_restraints=False,
            restart=restart,
            checkpoint_path=chk,
            seed=seed,
            hydrogen_mass=hmass,
        )

    result["elapsed_seconds"] = round(time.time() - t0, 2)

    _locked_update(config_path, step, result,
                   shared=(step in SHARED_STEPS), replica_id=replica_id)

    print(f"\nResult for '{step}' written to: {config_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run one pipeline step, reading inputs from and writing results to a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Steps (in order):",
            *[f"  {s}" for s in VALID_STEPS],
        ]),
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config/state file for this protein")
    parser.add_argument("--step",   required=True, choices=VALID_STEPS, help="Which pipeline step to run")
    parser.add_argument("--replica", type=int, default=None,
                        help="Replica index (required for nvt/check_nvt/npt/check_npt/production)")
    parser.add_argument("--restart", action="store_true",
                        help="For npt/production: restart from checkpoint instead of loading the upstream state file")
    parser.add_argument("--skip_if_done", action="store_true",
                        help="Do nothing if this step already has status=completed in the JSON")
    args = parser.parse_args()

    if args.skip_if_done:
        cfg = _load(args.config)
        if args.step in SHARED_STEPS:
            done = cfg.get("results", {}).get(args.step, {}).get("status") == "completed"
        else:
            rep_id = args.replica or 0
            done = (cfg.get("replicas", [{}])[rep_id]
                       .get("results", {}).get(args.step, {}).get("status") == "completed")
        if done:
            print(f"[skip] '{args.step}' already completed for {args.config} replica={args.replica}")
            return

    run_step(args.config, args.step, restart=args.restart, replica_id=args.replica)


if __name__ == "__main__":
    main()
