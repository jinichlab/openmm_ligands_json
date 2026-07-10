"""
system_creation.py
==================
Solvate the (vacuum-minimized) protein–ligand complex and parameterize it ONCE.

This is the heart of the protein–ligand adaptation:
  1. Load the minimized complex topology+positions (protein + ligand, bonds intact).
  2. Build a SystemGenerator that knows the protein/water ffxml AND the ligand
     small-molecule force field (gaff-2.11 / openff-2.1.0 / ...).
  3. Solvate with Modeller.addSolvent() using the generator's ForceField (which
     carries the ligand template, so water/ions are added around the ligand).
  4. create_system() to get the fully-parameterized PME System.
  5. Serialize the System to system.xml and pickle the solvated topology.

Downstream steps (min_sol, nvt, npt, production) deserialize system.xml instead
of re-parameterizing — so antechamber/AM1-BCC runs only here, once per complex.

Apo case: an empty ligand list solvates the bare protein, same code path.
"""

import json

from openmm.app import PDBxFile
from openmm.unit import nanometers, molar

import ligand_utils as lu


def main(complex_pkl, ligand_specs, forcefield_name, water_ff, ligand_ff,
         box_size, output, water_model="tip3p", ionic_strength=0.15,
         hydrogen_mass=None, cache=None, json_output=None):
    topology, positions = lu.load_topology(complex_pkl)
    molecules, _ = lu.load_ligand_molecules(ligand_specs)
    print(f"Solvating complex with {len(molecules)} ligand copy(ies).")

    generator = lu.build_system_generator(
        forcefield_name, water_ff, ligand_ff, molecules,
        periodic=True, hydrogen_mass=hydrogen_mass, cache=cache,
    )

    from openmm.app import Modeller
    modeller = Modeller(topology, positions)
    # Strip any crystallographic waters carried in, then add a fresh solvent box.
    modeller.delete([r for r in modeller.topology.residues()
                     if r.name in ("HOH", "WAT", "SOL")])
    modeller.addSolvent(
        generator.forcefield,
        model=water_model,
        padding=box_size * nanometers,
        ionicStrength=ionic_strength * molar,
        neutralize=True,
    )

    system = generator.create_system(modeller.topology)

    out = output[:-4] if output.endswith(".cif") else output
    system_xml = f"{out}_system.xml"
    topo_pkl = f"{out}.pkl"
    cif_out = f"{out}.cif"

    lu.save_system(system_xml, system)
    lu.save_topology(topo_pkl, modeller.topology, modeller.positions)
    with open(cif_out, "w") as f:
        PDBxFile.writeFile(modeller.topology, modeller.positions, f)

    n_atoms = modeller.topology.getNumAtoms()
    print(f"Solvated system: {n_atoms} atoms. System -> {system_xml}, topology -> {topo_pkl}")

    result = {
        "step": "system_creation",
        "status": "completed",
        "output": topo_pkl,          # pickled topology+positions (downstream input)
        "system": system_xml,        # serialized parameterized System
        "cif": cif_out,
        "box_padding_nm": box_size,
        "n_atoms": n_atoms,
        "n_ligands": len(molecules),
        "hydrogen_mass_amu": hydrogen_mass,
    }
    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Solvate the vacuum-minimized complex and parameterize it once (serialize System).")
    p.add_argument("-i", "--complex_pkl", required=True,
                   help="Pickled (topology, positions) of the minimized complex (from minimization vacuum)")
    p.add_argument("--ligand", action="append", metavar="RESNAME:PATH.sdf",
                   help="Ligand SDF (repeatable); omit for apo")
    p.add_argument("--ligands_json", help="split_ligand.py ligand-list JSON")
    p.add_argument("--force_field", default="amber/ff14SB.xml")
    p.add_argument("--force_field_water", default="amber/tip3p_standard.xml")
    p.add_argument("--ligand_force_field", default="gaff-2.11")
    p.add_argument("--water_model", default="tip3p", help="Solvent model for addSolvent (default tip3p)")
    p.add_argument("--box_size", type=float, default=1.0, help="Solvent padding in nm")
    p.add_argument("--ionic_strength", type=float, default=0.15, help="Ionic strength in molar")
    p.add_argument("--hydrogen_mass", type=float, default=None, help="HMR hydrogen mass in amu")
    p.add_argument("--cache", default=None, help="Ligand parameter cache JSON")
    p.add_argument("-o", "--output", required=True, help="Output prefix (_system.xml/.pkl/.cif written)")
    p.add_argument("--json_output", default=None, help="Write the result dict to this JSON path")
    args = p.parse_args()

    main(
        complex_pkl=args.complex_pkl,
        ligand_specs=lu.ligands_from_cli(args.ligand, args.ligands_json),
        forcefield_name=args.force_field, water_ff=args.force_field_water,
        ligand_ff=args.ligand_force_field, box_size=args.box_size,
        output=args.output, water_model=args.water_model,
        ionic_strength=args.ionic_strength, hydrogen_mass=args.hydrogen_mass,
        cache=args.cache, json_output=args.json_output,
    )


if __name__ == "__main__":
    _cli()
