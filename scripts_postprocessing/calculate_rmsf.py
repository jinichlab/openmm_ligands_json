#!/usr/bin/env python3
import argparse
import json

import mdtraj as md
import numpy as np
import pandas as pd


def create_ag_parser():
    parser = argparse.ArgumentParser(description="Calculate per-residue RMSF from an MD trajectory")
    parser.add_argument("-t", "--trajectory", type=str, required=True, help="Path to the trajectory (.dcd)")
    parser.add_argument("-to", "--topology", type=str, required=True, help="Path to the topology (.pdb/.cif)")
    parser.add_argument("-o", "--output", type=str, default="rmsf.csv", help="Output CSV filename (default: rmsf.csv)")
    parser.add_argument("--selection", type=str, default="protein and name CA",
                        help="MDTraj atom selection for RMSF (default: 'protein and name CA')")
    parser.add_argument("--ref_frame", type=int, default=0,
                        help="Reference frame index for alignment before RMSF (default: 0)")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON result file")
    return parser


def main(trajectory, topology, output="rmsf.csv", selection="protein and name CA",
         ref_frame=0, json_output=None):
    print(f"Loading trajectory: {trajectory}")
    traj = md.load(trajectory, top=topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    atom_indices = traj.topology.select(selection)
    if len(atom_indices) == 0:
        raise ValueError(f"No atoms selected with '{selection}'")
    print(f"Computing RMSF on {len(atom_indices)} atoms (selection: '{selection}')")

    traj_sel = traj.atom_slice(atom_indices)
    traj_sel.superpose(traj_sel, frame=ref_frame)

    # RMSF: sqrt(mean over frames of squared deviation from mean position)
    mean_xyz = traj_sel.xyz.mean(axis=0)           # (n_atoms, 3)
    diff = traj_sel.xyz - mean_xyz[np.newaxis, :, :]  # (n_frames, n_atoms, 3)
    rmsf_nm = np.sqrt((diff ** 2).sum(axis=2).mean(axis=0))  # (n_atoms,) in nm

    # Build residue-level information
    atoms = [traj.topology.atom(i) for i in atom_indices]
    records = []
    for atom, rmsf_val in zip(atoms, rmsf_nm):
        records.append({
            "residue_index": atom.residue.index,
            "residue_name": atom.residue.name,
            "residue_seq": atom.residue.resSeq,
            "chain": atom.residue.chain.index,
            "atom_name": atom.name,
            "rmsf_nm": float(rmsf_val),
            "rmsf_angstrom": float(rmsf_val * 10.0),
        })

    df = pd.DataFrame(records)
    df.to_csv(output, index=False)
    print(f"RMSF saved to {output}")

    result = {
        "step": "rmsf",
        "status": "completed",
        "output": output,
        "n_residues": int(df["residue_index"].nunique()),
        "n_atoms": len(atom_indices),
        "mean_rmsf_nm": float(rmsf_nm.mean()),
        "max_rmsf_nm": float(rmsf_nm.max()),
        "selection": selection,
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.trajectory, args.topology, args.output, args.selection, args.ref_frame, args.json)
