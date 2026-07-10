import argparse
import json
from pdbfixer import PDBFixer
from openmm.app import PDBxFile, ForceField, Modeller
from openmm.app.element import hydrogen


def create_ag_parser():
    parser = argparse.ArgumentParser(description="Fix PDB structure: missing residues/atoms and add hydrogens")
    parser.add_argument("-p", "--pdb", type=str, required=True, help="Path to the input PDB file")
    parser.add_argument("-ph", "--ph", type=float, default=7.0, help="pH for adding hydrogen atoms (default: 7.0)")
    parser.add_argument("-o", "--output", type=str, default="fixed", help="Output name without extension (default: fixed)")
    parser.add_argument("--remove_heterogens", action="store_true",
                        help="Remove heterogens (ligands, cofactors, crystallographic waters)")
    parser.add_argument("--keep_water", action="store_true",
                        help="When --remove_heterogens is set, keep crystallographic water molecules")
    parser.add_argument("--ff", type=str, default=None,
                        help="Force field name (e.g. charmm36) — used to add hydrogens with correct terminal naming")
    parser.add_argument("--ff_water", type=str, default=None,
                        help="Water force field name (e.g. charmm36/water)")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON result file")
    return parser


def _xml(name):
    return name if name.endswith(".xml") else name + ".xml"


def main(path, ph, output, remove_heterogens=False, keep_water=False,
         force_field=None, force_field_water=None, json_output=None):

    fixer = PDBFixer(filename=path)

    # Replace non-standard residues (e.g. MSE→MET, HYP→PRO) before anything else
    fixer.findNonstandardResidues()
    nonstandard = [r[0].name for r in fixer.nonstandardResidues] if fixer.nonstandardResidues else []
    if nonstandard:
        print(f"Replacing non-standard residues: {nonstandard}")
    fixer.replaceNonstandardResidues()

    # Optionally strip heterogens (ligands, cofactors, HETATM waters)
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=keep_water)
        print(f"Heterogens removed (keep_water={keep_water})")

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Strip all existing hydrogens so we start from clean heavy atoms.
    # This avoids duplicate / mis-named N-terminal H atoms already present
    # in the input PDB (e.g. H instead of H1, or duplicate H).
    modeller = Modeller(fixer.topology, fixer.positions)
    h_atoms = [a for a in modeller.topology.atoms() if a.element == hydrogen]
    modeller.delete(h_atoms)

    if force_field:
        # Add hydrogens using the actual downstream FF so terminal H names
        # (HT1/HT2/HT3 for CHARMM36, H1/H2/H3 for AMBER) match the templates.
        ff_files = [_xml(force_field)]
        if force_field_water:
            ff_files.append(_xml(force_field_water))
        ff_obj = ForceField(*ff_files)
        modeller.addHydrogens(ff_obj, pH=ph)
        print(f"Hydrogens added using forcefield: {ff_files}")
    else:
        # Fallback: use PDBFixer's generic hydrogen addition
        fixer.topology, fixer.positions = modeller.topology, modeller.positions
        fixer.addMissingHydrogens(pH=ph)
        modeller = Modeller(fixer.topology, fixer.positions)
        print("Hydrogens added using PDBFixer (no forcefield specified)")

    output_file = output if output.endswith(".cif") else f"{output}.cif"
    with open(output_file, "w") as f:
        PDBxFile.writeFile(modeller.topology, modeller.positions, f)

    print(f"Fixed structure written to: {output_file}")

    result = {
        "step": "pdb_fixer",
        "status": "completed",
        "output": output_file,
        "ph": ph,
        "nonstandard_residues_replaced": nonstandard,
        "heterogens_removed": remove_heterogens,
        "force_field_used": force_field or "pdbfixer_default",
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.pdb, args.ph, args.output, args.remove_heterogens, args.keep_water,
         args.ff, args.ff_water, args.json)
