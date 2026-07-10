#!/usr/bin/env python3
"""
MD Simulation Pipeline Runner
==============================
Reads a JSON config file and runs the full OpenMM MD pipeline:

  1. pdb_fixer          - Fix missing residues/atoms, add hydrogens
  2. minimization_vac   - Vacuum energy minimization (backbone restrained)
  3. system_creation    - Solvate in water box with ions
  4. minimization_sol   - Solvated energy minimization (backbone restrained, PME)
  5. nvt                - NVT equilibration with temperature ramp (backbone restrained)
  6. npt                - NPT equilibration with backbone restraints
  7. production         - NPT without restraints (loads state from restrained NPT)

Outputs a JSON file with status, timing, and results from every step.

Usage:
  python run_pipeline.py config.json
  python run_pipeline.py config.json -o results.json
  python run_pipeline.py config.json --skip-to nvt      # all files before must exist
  python run_pipeline.py config.json --skip-to production --restart-production

Example config: see config_example.json
"""

import argparse
import json
import os
import sys
import time
import traceback

# Ensure the scripts directory is on the path when running from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pdb_fixer as _pdb_fixer
import system_creation as _system_creation
import minimization as _minimization
import equilibration_nvt_steps as _nvt
import equilibration_npt as _npt


STEPS_ORDER = [
    "pdb_fixer",
    "minimization_vac",
    "system_creation",
    "minimization_sol",
    "nvt",
    "npt",
    "production",
]


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run the full OpenMM MD simulation pipeline from a JSON config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path to the JSON configuration file")
    parser.add_argument(
        "-o", "--output", type=str, default="pipeline_result.json",
        help="Path to write the JSON results file (default: pipeline_result.json)",
    )
    parser.add_argument(
        "--skip-to", type=str, default=None,
        choices=STEPS_ORDER[1:],
        help="Skip all steps before this one (all intermediate files must already exist)",
    )
    parser.add_argument(
        "--restart-production", action="store_true",
        help="When running the production step, restart from its checkpoint instead of loading NPT state",
    )
    return parser


def _run_step(name, fn, results):
    """Run one pipeline step, record timing, catch and store errors."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name.upper()}")
    print(f"{'='*60}\n")
    t0 = time.time()
    try:
        result = fn()
        elapsed = round(time.time() - t0, 2)
        result["elapsed_seconds"] = elapsed
        results[name] = result
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        results[name] = {
            "step": name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
        }
        print(f"\n[FAILED] {name}: {e}")
        raise


def main():
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    out_dir = config.get("output_dir", ".")
    os.makedirs(out_dir, exist_ok=True)

    prefix = config.get("prefix", "sim")
    ff = config.get("force_field", "amber/ff14SB.xml")
    ff_water = config.get("force_field_water", "amber/tip3p_standard.xml")
    lig_ff = config.get("ligand_force_field", "gaff-2.11")
    ligands = config.get("ligands", [])
    hmass = config.get("hydrogen_mass", None)
    platform = config.get("platform", "CUDA")
    device = config.get("device", 0)

    def p(suffix):
        return os.path.join(out_dir, f"{prefix}_{suffix}")

    # Determine which steps to run
    skip_to = args.skip_to
    if skip_to:
        idx = STEPS_ORDER.index(skip_to)
        active_steps = set(STEPS_ORDER[idx:])
        print(f"Skipping steps before '{skip_to}'. Intermediate files must already exist.")
    else:
        active_steps = set(STEPS_ORDER)

    results = {}
    status = "completed"
    pipeline_start = time.time()

    try:
        # ------------------------------------------------------------------
        # Step 1: PDB Fixer
        # ------------------------------------------------------------------
        if "pdb_fixer" in active_steps:
            _run_step("pdb_fixer", lambda: _pdb_fixer.main(
                path=config["pdb"],
                ph=config.get("ph", 7.0),
                output=p("fixed"),
                remove_heterogens=config.get("remove_heterogens", False),
                keep_water=config.get("keep_water", False),
                force_field=ff,
                force_field_water=ff_water,
            ), results)
        fixed_cif = p("fixed.cif")
        cache = p("ligand_cache.json")

        # ------------------------------------------------------------------
        # Step 2: Vacuum minimization (protein + ligand combined here)
        # ------------------------------------------------------------------
        if "minimization_vac" in active_steps:
            min_cfg = config.get("minimization_vac", config.get("minimization", {}))
            _run_step("minimization_vac", lambda: _minimization.minimize_vacuum(
                protein_pdbx=fixed_cif,
                ligand_specs=ligands,
                force_field=ff,
                water_ff=ff_water,
                ligand_ff=lig_ff,
                steps=min_cfg.get("steps", 5000),
                platform_=platform,
                cores=device,
                output=p("min_vac"),
                cache=cache,
            ), results)
        min_vac_pkl = p("min_vac.pkl")

        # ------------------------------------------------------------------
        # Step 3: Solvation + parameterization (serializes system.xml)
        # ------------------------------------------------------------------
        if "system_creation" in active_steps:
            _run_step("system_creation", lambda: _system_creation.main(
                complex_pkl=min_vac_pkl,
                ligand_specs=ligands,
                forcefield_name=ff,
                water_ff=ff_water,
                ligand_ff=lig_ff,
                box_size=config.get("box_size", 8.0),
                output=p("solvated"),
                hydrogen_mass=hmass,
                cache=cache,
            ), results)
        solvated_pkl = p("solvated.pkl")
        system_xml = p("solvated_system.xml")

        # ------------------------------------------------------------------
        # Step 4: Solvated minimization
        # ------------------------------------------------------------------
        if "minimization_sol" in active_steps:
            min_cfg = config.get("minimization_sol", config.get("minimization", {}))
            _run_step("minimization_sol", lambda: _minimization.minimize_solvated(
                topology_pkl=solvated_pkl,
                system_xml=system_xml,
                steps=min_cfg.get("steps", 5000),
                platform_=platform,
                cores=device,
                output=p("min_sol"),
                hydrogen_mass=hmass,
            ), results)
        min_sol_pkl = p("min_sol.pkl")

        # ------------------------------------------------------------------
        # Step 5: NVT equilibration
        # ------------------------------------------------------------------
        if "nvt" in active_steps:
            nvt_cfg = config.get("nvt", {})
            _run_step("nvt", lambda: _nvt.main(
                topology_pkl=min_sol_pkl,
                system_xml=system_xml,
                simulation_steps=nvt_cfg.get("steps", 200000),
                recorder_steps=nvt_cfg.get("recorder", 500),
                platform_=platform,
                cores=device,
                output=p("nvt"),
                time_step=nvt_cfg.get("time_step", 0.001),
                hydrogen_mass=hmass,
            ), results)
        nvt_xml = p("nvt.xml")

        # ------------------------------------------------------------------
        # Step 6: NPT equilibration with backbone restraints
        # ------------------------------------------------------------------
        if "npt" in active_steps:
            npt_cfg = config.get("npt", {})
            _run_step("npt", lambda: _npt.main(
                topology_pkl=solvated_pkl,
                system_xml=system_xml,
                trajectory=nvt_xml,
                simulation_steps=npt_cfg.get("steps", 500000),
                time_step=npt_cfg.get("time_step", 0.002),
                recorder_steps=npt_cfg.get("recorder", 5000),
                platform_=platform,
                cores=device,
                output=p("npt"),
                apply_restraints=True,
                restart=npt_cfg.get("restart", False),
                checkpoint_path=npt_cfg.get("checkpoint", p("npt.chk")),
                hydrogen_mass=hmass,
            ), results)
        npt_xml = p("npt.xml")

        # ------------------------------------------------------------------
        # Step 7: Production — NPT without restraints
        # ------------------------------------------------------------------
        if "production" in active_steps:
            prod_cfg = config.get("production", {})
            restart_prod = args.restart_production or prod_cfg.get("restart", False)
            _run_step("production", lambda: _npt.main(
                topology_pkl=solvated_pkl,
                system_xml=system_xml,
                trajectory=npt_xml,
                simulation_steps=prod_cfg.get("steps", 2500000),
                time_step=prod_cfg.get("time_step", 0.002),
                recorder_steps=prod_cfg.get("recorder", 5000),
                platform_=platform,
                cores=device,
                output=p("production"),
                apply_restraints=False,
                restart=restart_prod,
                checkpoint_path=prod_cfg.get("checkpoint", p("production.chk")),
                hydrogen_mass=hmass,
            ), results)

    except Exception:
        status = "failed"

    total_elapsed = round(time.time() - pipeline_start, 2)
    pipeline_result = {
        "status": status,
        "config_file": os.path.abspath(args.config),
        "config": config,
        "steps": results,
        "total_elapsed_seconds": total_elapsed,
    }

    with open(args.output, "w") as f:
        json.dump(pipeline_result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Pipeline {status.upper()}  —  {total_elapsed:.1f}s total")
    print(f"  Results written to: {args.output}")
    print(f"{'='*60}\n")

    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
