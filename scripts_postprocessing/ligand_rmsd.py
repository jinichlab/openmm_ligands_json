#!/usr/bin/env python3
"""
ligand_rmsd.py
==============
Per-frame ligand RMSD after aligning the trajectory on the protein — the
standard measure of how far each ligand drifts/reorients in its binding site.

The trajectory is superposed on `align_selection` (default protein Cα), then for
each ligand residue the heavy-atom RMSD to a reference frame is computed.  One
column per ligand copy (labelled RESNAME_chain_resSeq), plus a mean column.

Ligands are located by residue: every residue that is not protein, water, or a
monatomic ion.  If `ligand_resnames` is given, only those resnames are used.
Apo trajectories (no ligand residues) return status "skipped".
"""

import argparse
import json

import mdtraj as md
import numpy as np
import pandas as pd

# Residues that are never ligands (kept in sync with scripts_running/ligand_utils.py)
NON_LIGAND = {
    "HOH", "WAT", "TIP3", "SOL", "NA", "SOD", "CL", "CLA", "K", "POT", "LI",
    "RB", "CS", "F", "BR", "I", "MG", "CA", "ZN", "MN", "FE", "FE2", "FE3",
    "CO", "CU", "CU1", "NI", "CD", "HG", "SR", "BA",
}
AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE",
    "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN",
}


def _ligand_residues(top, ligand_resnames=None):
    resset = {r.upper() for r in ligand_resnames} if ligand_resnames else None
    residues = []
    for res in top.residues:
        name = res.name.upper()
        if resset is not None:
            if name in resset:
                residues.append(res)
        elif name not in AA and name not in NON_LIGAND and not res.is_water:
            residues.append(res)
    return residues


def main(trajectory, topology, output="ligand_rmsd.csv",
         align_selection="protein and name CA", ref_frame=0,
         heavy_only=True, ligand_resnames=None, json_output=None):
    print(f"Loading trajectory: {trajectory}")
    traj = md.load(trajectory, top=topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    ligands = _ligand_residues(traj.topology, ligand_resnames)
    if not ligands:
        print("No ligand residues found — skipping ligand RMSD (apo trajectory).")
        result = {"step": "ligand_rmsd", "status": "skipped",
                  "reason": "no ligand residues in trajectory"}
        if json_output:
            with open(json_output, "w") as f:
                json.dump(result, f, indent=2)
        return result

    # Align whole trajectory on the protein selection
    align_idx = traj.topology.select(align_selection)
    if len(align_idx) == 0:
        raise ValueError(f"Alignment selection matched no atoms: '{align_selection}'")
    traj.superpose(traj, frame=ref_frame, atom_indices=align_idx)

    df = pd.DataFrame({"frame": np.arange(traj.n_frames)})
    per_ligand_mean = {}
    for res in ligands:
        atoms = [a.index for a in res.atoms
                 if (not heavy_only) or a.element.symbol != "H"]
        label = f"{res.name}_{res.chain.index}_{res.resSeq}"
        ref = traj.xyz[ref_frame, atoms, :]
        diff = traj.xyz[:, atoms, :] - ref[np.newaxis, :, :]
        rmsd_nm = np.sqrt((diff ** 2).sum(axis=2).mean(axis=1))  # (n_frames,)
        df[f"{label}_rmsd_nm"] = rmsd_nm
        per_ligand_mean[label] = float(rmsd_nm.mean())
        print(f"  {label}: mean RMSD {rmsd_nm.mean()*10:.2f} Å ({len(atoms)} atoms)")

    rmsd_cols = [c for c in df.columns if c.endswith("_rmsd_nm")]
    df["mean_rmsd_nm"] = df[rmsd_cols].mean(axis=1)
    df.to_csv(output, index=False)
    print(f"Ligand RMSD saved to: {output}")

    result = {
        "step": "ligand_rmsd",
        "status": "completed",
        "output": output,
        "n_ligands": len(ligands),
        "align_selection": align_selection,
        "heavy_only": heavy_only,
        "mean_rmsd_nm_per_ligand": per_ligand_mean,
        "overall_mean_rmsd_nm": float(df["mean_rmsd_nm"].mean()),
    }
    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-frame ligand RMSD after protein alignment")
    parser.add_argument("-t", "--trajectory", required=True)
    parser.add_argument("-to", "--topology", required=True)
    parser.add_argument("-o", "--output", default="ligand_rmsd.csv")
    parser.add_argument("--align_selection", default="protein and name CA")
    parser.add_argument("--ref_frame", type=int, default=0)
    parser.add_argument("--no_heavy_only", action="store_true", help="Include hydrogens in RMSD")
    parser.add_argument("--ligand_resnames", nargs="+", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    main(args.trajectory, args.topology, args.output, args.align_selection,
         args.ref_frame, not args.no_heavy_only, args.ligand_resnames, args.json)
