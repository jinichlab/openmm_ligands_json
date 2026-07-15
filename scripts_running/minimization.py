"""
Energy minimization for protein–ligand (or apo) systems.

Two entry points, matching the two pipeline stages:

  minimize_vacuum   — builds the protein+ligand complex from the fixed protein
                      (PDBx) plus the ligand SDF(s), parameterizes it in vacuum
                      (NoCutoff) via SystemGenerator, and minimizes with the
                      protein backbone restrained.  This is where the ligand
                      first joins the protein.  Output: min_vac.pkl (topology +
                      positions) and min_vac.cif.

  minimize_solvated — loads the already-serialized solvated System (ligand
                      parameters baked in) plus the pickled topology, applies
                      backbone restraints, and minimizes under PME.

Apo case: an empty ligand list makes minimize_vacuum a plain protein minimization.
"""

import json
import time

from openmm import CustomIntegrator, Platform
from openmm.app import Simulation, PDBxFile
from openmm.unit import kilojoules_per_mole, nanometers, picoseconds

import ligand_utils as lu


def _integrator():
    # CustomIntegrator is only needed to build the Simulation; minimizeEnergy()
    # uses L-BFGS internally and ignores it.
    integrator = CustomIntegrator(0.001 * picoseconds)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "0")
    integrator.addComputePerDof("x", "x - 0.01 * f / max(1e-5, sqrt(f*f))")
    integrator.addConstrainPositions()
    return integrator


def _platform(platform_, cores):
    platform = Platform.getPlatformByName(platform_)
    if platform_ == "CUDA":
        return platform, {"CudaPrecision": "mixed", "CudaDeviceIndex": str(cores)}
    if platform_ == "CPU":
        return platform, {"Threads": str(cores)}
    return platform, {}


def _minimize(simulation, steps):
    state = simulation.context.getState(getEnergy=True)
    e0 = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    print(f"Starting energy: {e0:.2f} kJ/mol")
    simulation.minimizeEnergy(tolerance=100.0 * kilojoules_per_mole / nanometers, maxIterations=steps)
    state = simulation.context.getState(getEnergy=True)
    e1 = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    print(f"Minimized energy: {e1:.2f} kJ/mol")
    return e0, e1


def minimize_vacuum(protein_pdbx, ligand_specs, force_field, water_ff, ligand_ff,
                    steps, platform_, cores, output, cache=None, json_output=None):
    """Combine protein + ligand(s), parameterize in vacuum, minimize (backbone restrained)."""
    molecules, resnames = lu.load_ligand_molecules(ligand_specs)
    print(f"Loaded {len(molecules)} ligand copy(ies): {resnames or '(apo)'}")

    modeller = lu.build_complex_modeller(protein_pdbx, molecules, resnames)
    generator = lu.build_system_generator(
        force_field, water_ff, ligand_ff, molecules,
        periodic=False, hydrogen_mass=None, cache=cache,
    )
    lu.log.info("Parameterizing complex with ligand FF '%s' — the AM1-BCC charge "
                "fit (antechamber/sqm) runs here, once per unique ligand%s...",
                ligand_ff, " (cached)" if cache else "")
    _t0 = time.time()
    system = generator.create_system(modeller.topology)
    lu.log.info("System parameterized in %.1f s (%d atoms, %d unique ligand(s))",
                time.time() - _t0, modeller.topology.getNumAtoms(),
                len(lu.unique_by_chemistry(molecules)))
    n_restr = lu.add_backbone_restraints(system, modeller.topology, modeller.positions, periodic=False)
    print(f"Backbone restraints on {n_restr} protein atoms.")

    platform, props = _platform(platform_, cores)
    simulation = Simulation(modeller.topology, system, _integrator(), platform, props)
    simulation.context.setPositions(modeller.positions)
    e0, e1 = _minimize(simulation, steps)

    positions = simulation.context.getState(getPositions=True).getPositions()

    pkl_out = output if output.endswith(".pkl") else f"{output}.pkl"
    lu.save_topology(pkl_out, modeller.topology, positions)
    cif_out = pkl_out[:-4] + ".cif"
    with open(cif_out, "w") as f:
        PDBxFile.writeFile(modeller.topology, positions, f)
    print(f"Minimized complex written to: {pkl_out} / {cif_out}")

    result = {
        "step": "minimization_vac",
        "status": "completed",
        "output": pkl_out,
        "cif": cif_out,
        "n_ligands": len(molecules),
        "ligand_resnames": resnames,
        "initial_energy_kJ_mol": round(e0, 4),
        "final_energy_kJ_mol": round(e1, 4),
    }
    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _cli():
    import argparse

    lu.configure_logging()
    p = argparse.ArgumentParser(description="Energy minimization for protein–ligand (or apo) systems.")
    sub = p.add_subparsers(dest="stage", required=True)

    v = sub.add_parser("vacuum", help="Build protein+ligand complex, parameterize in vacuum, minimize.")
    v.add_argument("-i", "--protein_pdbx", required=True, help="Fixed protein (PDBx/CIF from pdb_fixer)")
    v.add_argument("--ligand", action="append", metavar="RESNAME:PATH.sdf",
                   help="Ligand SDF (repeatable); omit for apo")
    v.add_argument("--ligands_json", help="split_ligand.py ligand-list JSON")
    v.add_argument("--force_field", default="amber/ff14SB.xml")
    v.add_argument("--force_field_water", default="amber/tip3p_standard.xml")
    v.add_argument("--ligand_force_field", default="gaff-2.11")
    v.add_argument("--cache", default=None, help="Ligand parameter cache JSON")

    s = sub.add_parser("solvated", help="Load serialized solvated System, minimize under PME.")
    s.add_argument("-t", "--topology_pkl", required=True, help="Pickled (topology, positions) from system_creation")
    s.add_argument("-x", "--system_xml", required=True, help="Serialized System XML from system_creation")
    s.add_argument("--hydrogen_mass", type=float, default=None)

    for sp in (v, s):
        sp.add_argument("-o", "--output", required=True, help="Output prefix (.pkl/.cif written)")
        sp.add_argument("--steps", type=int, default=5000, help="Max minimization iterations")
        sp.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL"])
        sp.add_argument("--device", type=int, default=0, help="CUDA device index / CPU thread count")
        sp.add_argument("--json_output", default=None, help="Write the result dict to this JSON path")

    args = p.parse_args()
    if args.stage == "vacuum":
        minimize_vacuum(
            protein_pdbx=args.protein_pdbx,
            ligand_specs=lu.ligands_from_cli(args.ligand, args.ligands_json),
            force_field=args.force_field, water_ff=args.force_field_water,
            ligand_ff=args.ligand_force_field, steps=args.steps,
            platform_=args.platform, cores=args.device, output=args.output,
            cache=args.cache, json_output=args.json_output,
        )
    else:
        minimize_solvated(
            topology_pkl=args.topology_pkl, system_xml=args.system_xml,
            steps=args.steps, platform_=args.platform, cores=args.device,
            output=args.output, hydrogen_mass=args.hydrogen_mass,
            json_output=args.json_output,
        )


def minimize_solvated(topology_pkl, system_xml, steps, platform_, cores, output,
                      hydrogen_mass=None, json_output=None):
    """Load the serialized solvated System, restrain backbone, minimize under PME."""
    topology, positions = lu.load_topology(topology_pkl)
    system = lu.load_system(system_xml)
    n_restr = lu.add_backbone_restraints(system, topology, positions, periodic=True)
    print(f"Backbone restraints on {n_restr} protein atoms.")

    platform, props = _platform(platform_, cores)
    simulation = Simulation(topology, system, _integrator(), platform, props)
    simulation.context.setPositions(positions)
    e0, e1 = _minimize(simulation, steps)

    new_positions = simulation.context.getState(getPositions=True).getPositions()
    pkl_out = output if output.endswith(".pkl") else f"{output}.pkl"
    lu.save_topology(pkl_out, topology, new_positions)
    cif_out = pkl_out[:-4] + ".cif"
    with open(cif_out, "w") as f:
        PDBxFile.writeFile(topology, new_positions, f)
    print(f"Minimized solvated system written to: {pkl_out} / {cif_out}")

    result = {
        "step": "minimization_sol",
        "status": "completed",
        "output": pkl_out,
        "cif": cif_out,
        "initial_energy_kJ_mol": round(e0, 4),
        "final_energy_kJ_mol": round(e1, 4),
        "hydrogen_mass_amu": hydrogen_mass,
    }
    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    _cli()
