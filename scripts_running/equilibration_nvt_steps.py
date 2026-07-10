"""
NVT equilibration with a gradual temperature ramp (5K -> 50K -> 150K -> 300K).

Protein–ligand adaptation: instead of rebuilding the System from a ForceField,
this loads the serialized solvated System (ligand parameters baked in) and the
pickled topology.  Backbone restraints (protein only) are added in-memory; the
ligand is free to relax with the protein.
"""

import json
import math

from openmm import LangevinIntegrator, Platform
from openmm.app import (
    PDBFile, Simulation, StateDataReporter, DCDReporter, CheckpointReporter,
)
from openmm.unit import nanometers, kelvin, picosecond, picoseconds

import ligand_utils as lu


def validate_positions(positions):
    for i, pos in enumerate(positions):
        if any(math.isnan(x.value_in_unit(nanometers)) for x in pos):
            raise ValueError(f"NaN found in position of atom {i}")


def main(topology_pkl, system_xml, simulation_steps, recorder_steps, platform_, cores, output,
         seed=0, time_step=0.001, restrain_ligand=False, hydrogen_mass=None, json_output=None):
    topology, positions = lu.load_topology(topology_pkl)
    system = lu.load_system(system_xml)
    platform = Platform.getPlatformByName(platform_)

    print("Applying backbone position restraints...")
    n_restr = lu.add_backbone_restraints(system, topology, positions, periodic=True)
    print(f"Restrained {n_restr} protein backbone atoms.")
    if restrain_ligand:
        n_lig = lu.add_ligand_restraints(system, topology, positions, periodic=True)
        print(f"Restrained {n_lig} ligand heavy atoms (restrain_ligand=True).")

    if platform_ == "CUDA":
        properties = {"CudaPrecision": "mixed", "CudaDeviceIndex": str(cores)}
    elif platform_ == "CPU":
        properties = {"Threads": str(cores)}
    else:
        properties = {}

    # Start integrator at 5K; temperature is updated per phase
    integrator = LangevinIntegrator(5 * kelvin, 1 / picosecond, time_step * picoseconds)
    if seed != 0:
        integrator.setRandomNumberSeed(seed)
    simulation = Simulation(topology, system, integrator, platform, properties)

    validate_positions(positions)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(5 * kelvin, seed if seed != 0 else 0)

    print("Running energy minimization before NVT...")
    simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True)
    print(f"Minimization complete. Potential energy: {state.getPotentialEnergy()}")

    # 3 warmup phases (2000 steps each) + production phase
    warmup_steps = 6000
    total_steps = warmup_steps + simulation_steps

    simulation.reporters.append(StateDataReporter(
        f"{output}.csv", recorder_steps, step=True, potentialEnergy=True,
        totalEnergy=True, temperature=True, progress=True, remainingTime=True,
        speed=True, volume=True, density=True, totalSteps=total_steps, separator=",",
    ))
    simulation.reporters.append(DCDReporter(f"{output}.dcd", recorder_steps))
    simulation.reporters.append(CheckpointReporter(f"{output}.chk", recorder_steps * 10))

    try:
        for temp, nsteps, label in [
            (5, 2000, "Phase 1: 5K equilibration"),
            (50, 2000, "Phase 2: 50K equilibration"),
            (150, 2000, "Phase 3: 150K equilibration"),
            (300, simulation_steps, f"Phase 4: 300K production ({simulation_steps} steps)"),
        ]:
            print(label)
            simulation.context.setVelocitiesToTemperature(temp * kelvin)
            integrator.setTemperature(temp * kelvin)
            simulation.step(nsteps)
        print("NVT equilibration complete.")

    except Exception as e:
        print(f"Simulation crashed: {e}")
        state = simulation.context.getState(getPositions=True)
        for i, pos in enumerate(state.getPositions()):
            if any(math.isnan(x.value_in_unit(nanometers)) for x in pos):
                print(f"NaN detected in atom {i}: {pos}")
        raise

    final_positions = simulation.context.getState(getPositions=True).getPositions()
    with open(f"{output}.pdb", "w") as f:
        PDBFile.writeFile(simulation.topology, final_positions, f)
    simulation.saveState(f"{output}.xml")
    print(f"NVT state saved to: {output}.xml")

    result = {
        "step": "nvt",
        "status": "completed",
        "output_prefix": output,
        "state_file": f"{output}.xml",
        "checkpoint": f"{output}.chk",
        "trajectory": f"{output}.dcd",
        "csv_file": f"{output}.csv",
        "final_structure": f"{output}.pdb",
        "time_step_ps": time_step,
        "hydrogen_mass_amu": hydrogen_mass,
        "phases": {
            "warmup_5K_steps": 2000,
            "warmup_50K_steps": 2000,
            "warmup_150K_steps": 2000,
            "production_300K_steps": simulation_steps,
        },
    }
    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _cli():
    import argparse

    p = argparse.ArgumentParser(description="NVT equilibration with a 5K->300K temperature ramp.")
    p.add_argument("-t", "--topology_pkl", required=True, help="Pickled (topology, positions) from minimization_sol")
    p.add_argument("-x", "--system_xml", required=True, help="Serialized System XML from system_creation")
    p.add_argument("-o", "--output", required=True, help="Output prefix (.xml/.dcd/.chk/.pdb/.csv written)")
    p.add_argument("--steps", type=int, default=200000, help="Production-phase NVT steps at 300K")
    p.add_argument("--recorder", type=int, default=500, help="Reporter interval (steps)")
    p.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL"])
    p.add_argument("--device", type=int, default=0, help="CUDA device index / CPU thread count")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = unset)")
    p.add_argument("--time_step", type=float, default=0.001, help="Integrator timestep in ps")
    p.add_argument("--restrain_ligand", action="store_true", help="Also restrain ligand heavy atoms")
    p.add_argument("--hydrogen_mass", type=float, default=None)
    p.add_argument("--json_output", default=None, help="Write the result dict to this JSON path")
    args = p.parse_args()

    main(
        topology_pkl=args.topology_pkl, system_xml=args.system_xml,
        simulation_steps=args.steps, recorder_steps=args.recorder,
        platform_=args.platform, cores=args.device, output=args.output,
        seed=args.seed, time_step=args.time_step,
        restrain_ligand=args.restrain_ligand, hydrogen_mass=args.hydrogen_mass,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    _cli()
