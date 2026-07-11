#!/usr/bin/env python3
"""
Create a JSON config file for a protein MD pipeline run.

Usage:
  python create_config.py -p protein.pdb -o /runs/protein_001/
  python create_config.py -p protein.pdb -o /runs/protein_001/ --prefix prot1 --device 1
  python create_config.py -p protein.pdb -o /runs/protein_001/ --remove_heterogens
"""

import argparse
import json
import os
import random


def create_parser():
    parser = argparse.ArgumentParser(description="Generate a JSON config file for the MD pipeline")

    # Required
    parser.add_argument("-p", "--pdb", type=str, required=True,
                        help="Path to the input PDB file")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory where all output files for this run will be written")
    parser.add_argument("--n_replicas", type=int, default=1,
                        help="Number of independent MD replicas to create (default: 1)")

    # Naming
    parser.add_argument("--prefix", type=str, default=None,
                        help="Prefix for output files (default: PDB filename without extension)")
    parser.add_argument("--config_out", type=str, default=None,
                        help="Where to write the JSON config (default: {output_dir}/{prefix}.json)")

    # PDB fixer
    parser.add_argument("--ph", type=float, default=7.0,
                        help="pH for adding hydrogens (default: 7.0)")
    parser.add_argument("--remove_heterogens", action="store_true",
                        help="Remove ligands/cofactors during pdb_fixer step")
    parser.add_argument("--keep_water", action="store_true",
                        help="Keep crystallographic waters when --remove_heterogens is set")

    # System
    parser.add_argument("--force_field", type=str, default="amber/ff14SB.xml",
                        help="Protein force field ffxml (default: amber/ff14SB.xml; e.g. charmm36.xml)")
    parser.add_argument("--force_field_water", type=str, default="amber/tip3p_standard.xml",
                        help="Water force field ffxml (default: amber/tip3p_standard.xml)")
    parser.add_argument("--ligand_force_field", type=str, default="gaff-2.11",
                        help="Small-molecule force field for ligands, any openmmforcefields "
                             "name (default: gaff-2.11; e.g. openff-2.1.0, espaloma-0.3.2)")
    parser.add_argument("--box_size", type=float, default=8.0,
                        help="Solvation box padding in nm (default: 8.0)")

    # Ligands (protein–ligand systems). Omit all for an apo run.
    parser.add_argument("--ligands_json", type=str, default=None,
                        help="JSON emitted by split_ligand.py listing the ligand SDF(s). "
                             "Its 'ligands' list is copied verbatim into the config.")
    parser.add_argument("--ligand", action="append", default=None, metavar="RESNAME:PATH.sdf",
                        help="Add one ligand as RESNAME:path.sdf (repeatable). "
                             "Alternative to --ligands_json for hand-specified ligands.")

    # HMR
    parser.add_argument("--hmr", action="store_true",
                        help="Enable Hydrogen Mass Repartitioning (4 amu H mass, 4 fs timestep)")
    parser.add_argument("--hydrogen_mass", type=float, default=None,
                        help="Custom hydrogen mass in amu for HMR (overrides --hmr default of 4.0)")

    # Platform
    parser.add_argument("--platform", type=str, default="CUDA", choices=["CUDA", "CPU", "OpenCL"],
                        help="Compute platform (default: CUDA)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index or CPU thread count (default: 0)")

    # MD step parameters
    parser.add_argument("--min_steps", type=int, default=5000,
                        help="Max iterations for both minimization steps (default: 5000)")
    parser.add_argument("--nvt_steps", type=int, default=200000,
                        help="NVT production steps at 300K, 1 fs timestep (default: 200000 = 200 ps)")
    parser.add_argument("--nvt_recorder", type=int, default=500,
                        help="NVT recording interval in steps (default: 500)")
    parser.add_argument("--npt_steps", type=int, default=500000,
                        help="NPT restrained steps, 2 fs timestep (default: 500000 = 1 ns)")
    parser.add_argument("--npt_recorder", type=int, default=5000,
                        help="NPT recording interval in steps (default: 5000)")
    parser.add_argument("--prod_steps", type=int, default=2500000,
                        help="Production steps, 2 fs timestep (default: 2500000 = 5 ns)")
    parser.add_argument("--prod_recorder", type=int, default=5000,
                        help="Production recording interval in steps (default: 5000)")

    # Convergence check
    parser.add_argument("--check_rolling_window", type=int, default=50,
                        help="Rolling mean window for NVT/NPT convergence plots (default: 50)")

    # Postprocessing — unwrap
    parser.add_argument("--pp_keep_water", action="store_true",
                        help="Keep water molecules in unwrapped trajectory (default: remove water)")

    # Postprocessing — CVs
    parser.add_argument("--cv_chainid", type=int, default=0,
                        help="Chain index for Q/RMSD/Rg/CA-distances (default: 0)")
    parser.add_argument("--cv_native_cutoff", type=float, default=0.45,
                        help="Native contact cutoff in nm for Best-Hummer Q (default: 0.45)")
    parser.add_argument("--cv_beta", type=float, default=50.0,
                        help="Best-Hummer beta parameter (default: 50.0)")
    parser.add_argument("--cv_lam", type=float, default=1.8,
                        help="Best-Hummer lambda parameter (default: 1.8)")
    parser.add_argument("--cv_min_seq_sep", type=int, default=3,
                        help="Min residue separation for Q native contacts (default: 3)")
    parser.add_argument("--cv_ca_min_seq_sep", type=int, default=3,
                        help="Min residue separation for CA distance pairs (default: 3)")
    parser.add_argument("--cv_no_q", action="store_true",
                        help="Disable Best-Hummer Q calculation")
    parser.add_argument("--cv_no_rmsd", action="store_true",
                        help="Disable RMSD calculation")
    parser.add_argument("--cv_no_ca_dist", action="store_true",
                        help="Disable pairwise CA distance calculation")
    parser.add_argument("--cv_no_rg", action="store_true",
                        help="Disable radius of gyration calculation")
    parser.add_argument("--cv_phi_psi", action="store_true",
                        help="Enable phi/psi dihedral calculation (default: disabled)")

    # Postprocessing — ligand RMSD
    parser.add_argument("--ligrmsd_align", type=str, default="protein and name CA",
                        help="Atom selection the trajectory is aligned on before measuring "
                             "ligand RMSD (default: 'protein and name CA')")
    parser.add_argument("--ligrmsd_ref_frame", type=int, default=0,
                        help="Reference frame for ligand RMSD (default: 0)")
    parser.add_argument("--ligrmsd_heavy_only", action="store_true", default=True,
                        help="Use ligand heavy atoms only for RMSD (default: True)")

    # Postprocessing — RMSF
    parser.add_argument("--rmsf_selection", type=str, default="protein and name CA",
                        help="MDTraj atom selection for RMSF (default: 'protein and name CA')")
    parser.add_argument("--rmsf_ref_frame", type=int, default=0,
                        help="Reference frame index for RMSF alignment (default: 0)")

    # Postprocessing — DSSP
    parser.add_argument("--dssp_full", action="store_true",
                        help="Use full 8-state DSSP instead of simplified H/E/C (default: simplified)")
    parser.add_argument("--dssp_no_summary", action="store_true",
                        help="Disable per-residue DSSP summary CSV (default: enabled)")

    # Postprocessing — SASA
    parser.add_argument("--sasa_mode", type=str, default="residue",
                        choices=["total", "residue", "atom"],
                        help="SASA output granularity (default: residue)")
    parser.add_argument("--sasa_probe_radius", type=float, default=0.14,
                        help="SASA probe radius in nm (default: 0.14 = 1.4 Å)")
    parser.add_argument("--sasa_n_sphere_points", type=int, default=960,
                        help="Sphere points for SASA numerical integration (default: 960)")

    # Postprocessing — thermodynamics
    parser.add_argument("--thermo_rolling_window", type=int, default=100,
                        help="Rolling average window for thermodynamic quantities (default: 100)")
    parser.add_argument("--thermo_no_rolling", action="store_true",
                        help="Disable thermodynamic rolling-average CSV output (default: enabled)")

    # Postprocessing — tICA
    parser.add_argument("--tica_lag", type=int, default=10,
                        help="tICA lag time in frames (default: 10)")
    parser.add_argument("--tica_dim", type=int, default=2,
                        help="Number of tIC dimensions to keep (default: 2)")
    parser.add_argument("--tica_stride", type=int, default=1,
                        help="Frame stride for tICA input (default: 1)")
    parser.add_argument("--tica_save_dat", action="store_true",
                        help="Also save tICA projection as .dat text file (default: disabled)")
    parser.add_argument("--tica_save_model", action="store_true",
                        help="Save the tICA model as a pickle file (default: disabled)")

    # Postprocessing — order parameter
    parser.add_argument("--op_window", type=int, default=5,
                        help="Half-window size in residues for per-residue nematic S2 (default: 5)")
    parser.add_argument("--op_chain", type=int, default=0,
                        help="Chain index for order parameter calculation (default: 0)")

    # Postprocessing — network analysis
    parser.add_argument("--net_seg_ids", nargs="+", default=["A"],
                        help="Segment IDs for dynetan network analysis (default: A)")
    parser.add_argument("--net_n_windows", type=int, default=4,
                        help="Number of windows for correlation calculation (default: 4)")
    parser.add_argument("--net_sampled_frames", type=int, default=10,
                        help="Frames sampled per window (default: 10)")
    parser.add_argument("--net_cutoff", type=float, default=4.5,
                        help="Node contact cutoff distance in Angstroms (default: 4.5)")
    parser.add_argument("--net_n_jobs", type=int, default=4,
                        help="Threads for dynetan correlation/path calculation (default: 4)")
    parser.add_argument("--net_new_dcd", action="store_true",
                        help="Save a stride-reduced DCD from dynetan (default: disabled)")
    parser.add_argument("--net_dcd_stride", type=int, default=1,
                        help="Stride for dynetan reduced DCD output (default: 1)")

    return parser


def _resolve_ligands(args):
    """Merge ligands from --ligands_json and repeated --ligand into one list."""
    ligands = []
    if args.ligands_json:
        with open(args.ligands_json) as f:
            payload = json.load(f)
        ligands.extend(payload.get("ligands", []))
    for spec in (args.ligand or []):
        if ":" not in spec:
            raise ValueError(f"--ligand must be RESNAME:path.sdf, got: {spec}")
        resname, path = spec.split(":", 1)
        ligands.append({"sdf": os.path.abspath(path), "resname": resname})
    # Absolutize any relative SDF paths coming from the JSON
    for lig in ligands:
        if "sdf" in lig:
            lig["sdf"] = os.path.abspath(lig["sdf"])
    return ligands


def _resolve_hmr(args):
    """Return (hydrogen_mass, nvt_dt, npt_dt) based on --hmr / --hydrogen_mass flags."""
    if args.hydrogen_mass is not None:
        hmass = args.hydrogen_mass
    elif args.hmr:
        hmass = 4.0
    else:
        hmass = None
    dt = 0.004 if hmass is not None else 0.002
    nvt_dt = 0.002 if hmass is not None else 0.001  # NVT stays at 2 fs even with HMR (heating phases are fragile)
    return hmass, nvt_dt, dt


def _build_config(args, pdb_path, base_prefix, out_dir, n_replicas, seeds, ligands):
    # Per-replica subdirs: rep0/, rep1/, ... (or top-level dir if single replica)
    replicas = []
    for i in range(n_replicas):
        if n_replicas == 1:
            rep_prefix  = base_prefix
            rep_out_dir = out_dir
        else:
            rep_prefix  = f"{base_prefix}_rep{i}"
            rep_out_dir = os.path.join(out_dir, f"rep{i}")
        os.makedirs(rep_out_dir, exist_ok=True)
        replicas.append({
            "id":        i,
            "seed":      seeds[i],
            "prefix":    rep_prefix,
            "output_dir": rep_out_dir,
            "results":   {},
        })

    config = {
        # ── shared system settings ──────────────────────────────────────────
        "pdb":                pdb_path,
        "prefix":             base_prefix,
        "output_dir":         out_dir,
        "ph":                 args.ph,
        "remove_heterogens":  args.remove_heterogens,
        "keep_water":         args.keep_water,

        "force_field":        args.force_field,
        "force_field_water":  args.force_field_water,
        "ligand_force_field": args.ligand_force_field,
        "ligands":            ligands,          # [] = apo run
        "box_size":           args.box_size,

        "platform":           args.platform,
        "device":             args.device,
        "hydrogen_mass":      _resolve_hmr(args)[0],

        # ── step parameter blocks (shared across replicas) ──────────────────
        "minimization_vac": {"steps": args.min_steps},
        "minimization_sol": {"steps": args.min_steps},
        "nvt": {
            "steps":     args.nvt_steps,
            "recorder":  args.nvt_recorder,
            "time_step": _resolve_hmr(args)[1],
        },
        "npt": {
            "steps":     args.npt_steps,
            "time_step": _resolve_hmr(args)[2],
            "recorder":  args.npt_recorder,
        },
        "production": {
            "steps":     args.prod_steps,
            "time_step": _resolve_hmr(args)[2],
            "recorder":  args.prod_recorder,
        },
        "check_nvt": {"rolling_window": args.check_rolling_window},
        "check_npt": {"rolling_window": args.check_rolling_window},

        "postprocessing": {
            "unwrap": {"remove_water": not args.pp_keep_water},
            "cvs": {
                "q":              not args.cv_no_q,
                "rmsd":           not args.cv_no_rmsd,
                "ca_distances":   not args.cv_no_ca_dist,
                "rg":             not args.cv_no_rg,
                "phi_psi":        args.cv_phi_psi,
                "chainid":        args.cv_chainid,
                "native_cutoff":  args.cv_native_cutoff,
                "beta":           args.cv_beta,
                "lam":            args.cv_lam,
                "min_seq_sep":    args.cv_min_seq_sep,
                "ca_min_seq_sep": args.cv_ca_min_seq_sep,
            },
            "ligand_rmsd": {
                "align_selection": args.ligrmsd_align,
                "ref_frame":       args.ligrmsd_ref_frame,
                "heavy_only":      args.ligrmsd_heavy_only,
                "ligand_resnames": sorted({lig.get("resname") for lig in ligands
                                           if lig.get("resname")}),
            },
            "rmsf": {
                "selection": args.rmsf_selection,
                "ref_frame": args.rmsf_ref_frame,
            },
            "dssp": {
                "simplified": not args.dssp_full,
                "summary":    not args.dssp_no_summary,
            },
            "sasa": {
                "mode":           args.sasa_mode,
                "probe_radius":   args.sasa_probe_radius,
                "n_sphere_points": args.sasa_n_sphere_points,
            },
            "thermo": {
                "rolling_window": args.thermo_rolling_window,
                "rolling":        not args.thermo_no_rolling,
            },
            "tica": {
                "lag":        args.tica_lag,
                "dim":        args.tica_dim,
                "stride":     args.tica_stride,
                "save_dat":   args.tica_save_dat,
                "save_model": args.tica_save_model,
            },
            "order_parameter": {
                "window": args.op_window,
                "chain":  args.op_chain,
            },
            "network_analysis": {
                "seg_ids":        args.net_seg_ids,
                "n_windows":      args.net_n_windows,
                "sampled_frames": args.net_sampled_frames,
                "cutoff":         args.net_cutoff,
                "n_jobs":         args.net_n_jobs,
                "new_dcd":        args.net_new_dcd,
                "dcd_stride":     args.net_dcd_stride,
            },
        },

        # ── shared step results (pdb_fixer → minimization_sol) ─────────────
        "results": {},

        # ── per-replica entries (nvt → production + postprocessing) ─────────
        "replicas": replicas,
    }
    return config


def main():
    parser = create_parser()
    args = parser.parse_args()

    pdb_path = os.path.abspath(args.pdb)
    if not os.path.exists(pdb_path):
        parser.error(f"PDB file not found: {pdb_path}")

    base_prefix = args.prefix or os.path.splitext(os.path.basename(pdb_path))[0]
    n_replicas  = args.n_replicas
    out_dir     = os.path.abspath(os.path.join(args.output_dir, base_prefix))
    os.makedirs(out_dir, exist_ok=True)

    # Generate unique seeds for each replica (0 = OpenMM picks randomly)
    seeds = [0] if n_replicas == 1 else [random.randint(1, 2**31 - 1) for _ in range(n_replicas)]

    ligands = _resolve_ligands(args)

    config_path = args.config_out or os.path.join(out_dir, f"{base_prefix}.json")
    config = _build_config(args, pdb_path, base_prefix, out_dir, n_replicas, seeds, ligands)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    rep_info = ", ".join(f"rep{i} seed={seeds[i]}" for i in range(n_replicas))
    lig_info = (", ".join(f"{l.get('resname','?')}:{os.path.basename(l['sdf'])}" for l in ligands)
                if ligands else "apo (no ligands)")
    print(f"Config written to: {config_path}  ({rep_info})")
    print(f"Ligands: {lig_info}  |  ligand FF: {args.ligand_force_field}")

    shared_steps = ["pdb_fixer", "minimization_vac", "system_creation", "minimization_sol"]
    replica_steps = ["nvt", "check_nvt", "npt", "check_npt", "production"]
    pp_steps = ["unwrap", "cvs", "ligand_rmsd", "rmsf", "dssp", "sasa", "thermo",
                "tica", "order_parameter", "network_analysis", "export_amber",
                "prepare_mmpbsa"]

    print(f"\nCommands for {os.path.basename(config_path)}:")
    print("  # Shared steps (run once):")
    for step in shared_steps:
        print(f"  python step_runner.py --config {config_path} --step {step}")
    print("  # Per-replica steps:")
    for i in range(n_replicas):
        for step in replica_steps:
            print(f"  python step_runner.py --config {config_path} --step {step} --replica {i}")
    print("  # Postprocessing (per replica):")
    for i in range(n_replicas):
        for step in pp_steps:
            print(f"  python postprocessing_runner.py --config {config_path} --step {step} --replica {i}")


if __name__ == "__main__":
    main()
