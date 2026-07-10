import argparse
import json
import mdtraj as md


def create_ag_parser():
    parser = argparse.ArgumentParser(description="Unwrap a periodic MD trajectory and optionally strip solvent")
    parser.add_argument("-t", "--trajectory", type=str, required=True, help="Path to the trajectory (.dcd)")
    parser.add_argument("-to", "--topology", type=str, required=True, help="Path to the topology (.pdb/.cif)")
    parser.add_argument("-o", "--output", type=str, default="unwrapped.dcd", help="Output trajectory filename (default: unwrapped.dcd)")
    parser.add_argument("--remove_water", action="store_true", help="Remove water and ions from the trajectory")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON result file")
    return parser


def main(trajectory, topology, output, remove_water=False, json_output=None):
    print(f"Loading trajectory: {trajectory}")
    traj = md.load(trajectory, top=topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    # Anchor imaging on the protein so the ligand is kept in the same periodic
    # image as its binding site (essential for ligand RMSD / MM-PBSA).
    print("Imaging molecules (unwrapping PBC, protein-anchored)...")
    protein_atoms = traj.topology.select("protein")
    if len(protein_atoms) > 0:
        anchor = {traj.topology.atom(i) for i in protein_atoms}
        try:
            traj.image_molecules(inplace=True, anchor_molecules=[anchor])
        except Exception as e:
            print(f"Protein-anchored imaging failed ({e}); falling back to default.")
            traj.image_molecules(inplace=True)
    else:
        traj.image_molecules(inplace=True)

    if remove_water:
        # Strip water + monatomic ions only — the ligand (any other HETATM) is kept.
        remove_resnames = {"HOH", "WAT", "SOL", "TIP3", "NA", "SOD", "CL", "CLA",
                           "K", "POT", "LI", "RB", "CS", "MG", "CA", "ZN", "MN",
                           "FE", "FE2", "FE3", "CO", "CU", "CU1", "NI", "CD"}
        keep_atoms = [atom.index for atom in traj.topology.atoms
                      if atom.residue.name.upper() not in remove_resnames]
        traj = traj.atom_slice(keep_atoms)
        print(f"Water/ions removed: {traj.n_atoms} atoms remaining (protein + ligand)")

    print(f"Saving unwrapped trajectory to: {output}")
    traj.save(output)

    # Save a topology PDB from the first frame — needed by all downstream analysis scripts
    topo_out = output.rsplit(".", 1)[0] + "_topology.pdb"
    traj[0].save_pdb(topo_out)
    print(f"Topology PDB saved to: {topo_out}")

    result = {
        "step": "unwrap",
        "status": "completed",
        "trajectory": output,
        "topology": topo_out,
        "water_removed": remove_water,
        "n_frames": int(traj.n_frames),
        "n_atoms": int(traj.n_atoms),
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.trajectory, args.topology, args.output, args.remove_water, args.json)
