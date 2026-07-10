#!/usr/bin/env python3
import argparse
import json

import mdtraj as md
import numpy as np
import pandas as pd


def create_ag_parser():
    parser = argparse.ArgumentParser(description="Calculate SASA (solvent accessible surface area) per frame")
    parser.add_argument("-t", "--trajectory", type=str, required=True, help="Path to the trajectory (.dcd)")
    parser.add_argument("-to", "--topology", type=str, required=True, help="Path to the topology (.pdb/.cif)")
    parser.add_argument("-o", "--output", type=str, default="sasa.csv",
                        help="Output CSV filename (default: sasa.csv)")
    parser.add_argument("--mode", choices=["total", "residue", "atom"], default="residue",
                        help="Output granularity: 'total' (one value per frame), 'residue', or 'atom' (default: residue)")
    parser.add_argument("--probe_radius", type=float, default=0.14,
                        help="Probe radius in nm (default: 0.14 nm = 1.4 Å, water molecule)")
    parser.add_argument("--n_sphere_points", type=int, default=960,
                        help="Points on sphere for numerical integration (default: 960)")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON result file")
    return parser


def main(trajectory, topology, output="sasa.csv", mode="residue",
         probe_radius=0.14, n_sphere_points=960, json_output=None):
    print(f"Loading trajectory: {trajectory}")
    traj = md.load(trajectory, top=topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    print(f"Computing SASA (mode={mode}, probe_radius={probe_radius} nm)...")
    # sasa shape: (n_frames, n_atoms) in nm²
    sasa = md.shrake_rupley(traj, probe_radius=probe_radius, n_sphere_points=n_sphere_points,
                            mode="atom")

    if mode == "total":
        total_per_frame = sasa.sum(axis=1)  # (n_frames,)
        df = pd.DataFrame({"time_ps": traj.time, "sasa_nm2": total_per_frame})
        df.to_csv(output, index=False)

    elif mode == "residue":
        # Sum over atoms belonging to each residue
        residues = list(traj.topology.residues)
        res_sasa = np.zeros((traj.n_frames, len(residues)))
        for j, res in enumerate(residues):
            atom_ids = [a.index for a in res.atoms]
            res_sasa[:, j] = sasa[:, atom_ids].sum(axis=1)

        col_names = ["time_ps"] + [f"{r.name}{r.resSeq}_{r.chain.index}" for r in residues]
        data = np.concatenate([traj.time.reshape(-1, 1), res_sasa], axis=1)
        df = pd.DataFrame(data, columns=col_names)
        df.to_csv(output, index=False)

    else:  # atom
        atoms = list(traj.topology.atoms)
        col_names = ["time_ps"] + [f"{a.residue.name}{a.residue.resSeq}_{a.name}" for a in atoms]
        data = np.concatenate([traj.time.reshape(-1, 1), sasa], axis=1)
        df = pd.DataFrame(data, columns=col_names)
        df.to_csv(output, index=False)

    print(f"SASA saved to {output}")

    total = sasa.sum(axis=1)
    result = {
        "step": "sasa",
        "status": "completed",
        "output": output,
        "mode": mode,
        "probe_radius_nm": probe_radius,
        "n_frames": int(traj.n_frames),
        "mean_total_sasa_nm2": float(total.mean()),
        "std_total_sasa_nm2": float(total.std()),
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.trajectory, args.topology, args.output, args.mode,
         args.probe_radius, args.n_sphere_points, args.json)
