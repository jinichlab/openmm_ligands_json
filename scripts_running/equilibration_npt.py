"""
NPT equilibration / production for protein–ligand (or apo) systems.

Loads the serialized solvated System (ligand parameters baked in) plus the
pickled topology, adds a MonteCarlo barostat (and, for equilibration, protein
backbone restraints) in-memory, then either:
  * loads the full upstream State (restrained NPT <- NVT state), or
  * manually sets positions/velocities/box (production, no restraints — avoids
    a global-parameter mismatch when the upstream state carries the restraint 'k'), or
  * restarts from a binary checkpoint.
"""

import json
import os

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, XmlSerializer
from openmm.app import PDBFile, Simulation, StateDataReporter, DCDReporter, CheckpointReporter
from openmm.unit import kelvin, picosecond, picoseconds, bar

import ligand_utils as lu


def main(topology_pkl, system_xml, trajectory,
         simulation_steps, time_step, recorder_steps,
         platform_, cores, output,
         apply_restraints=False, restart=False, checkpoint_path=None,
         seed=0, hydrogen_mass=None, json_output=None):
    topology, positions = lu.load_topology(topology_pkl)
    system = lu.load_system(system_xml)
    platform = Platform.getPlatformByName(platform_)

    if apply_restraints:
        n_restr = lu.add_backbone_restraints(system, topology, positions, periodic=True)
        print(f"Applied backbone restraints on {n_restr} protein atoms.")

    # Barostat must be added before creating the Simulation/Context
    system.addForce(MonteCarloBarostat(1.0 * bar, 300 * kelvin, 25))

    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, time_step * picoseconds)
    if seed != 0 and not restart:
        integrator.setRandomNumberSeed(seed)

    if platform_ == "CUDA":
        properties = {"CudaPrecision": "mixed", "CudaDeviceIndex": str(cores)}
    elif platform_ == "CPU":
        properties = {"Threads": str(cores)}
    else:
        properties = {}

    simulation = Simulation(topology, system, integrator, platform, properties)

    if restart:
        chk_file = checkpoint_path or f"{output}.chk"
        if not os.path.exists(chk_file):
            raise FileNotFoundError(
                f"Checkpoint file not found: {chk_file}\nRun without --restart first to generate it."
            )
        print(f"Restarting NPT from checkpoint: {chk_file}")
        simulation.loadCheckpoint(chk_file)
        append_output = True

    elif apply_restraints:
        # Fresh restrained NPT: load full NVT state (restraint 'k' present in both)
        print(f"Loading NVT state (with restraints) from: {trajectory}")
        simulation.loadState(trajectory)
        append_output = False

    else:
        # Fresh production (no restraints): manually set positions/velocities/box
        # from the upstream XML state — avoids the 'k' global-parameter mismatch.
        print(f"Loading upstream state from: {trajectory}")
        with open(trajectory) as f:
            state = XmlSerializer.deserialize(f.read())
        simulation.context.setPositions(state.getPositions())
        simulation.context.setVelocities(state.getVelocities())
        simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        append_output = False

    simulation.reporters.append(StateDataReporter(
        f"{output}.csv", recorder_steps, step=True, potentialEnergy=True,
        totalEnergy=True, temperature=True, progress=True, remainingTime=True,
        speed=True, volume=True, density=True, totalSteps=simulation_steps,
        separator=",", append=append_output,
    ))
    simulation.reporters.append(DCDReporter(f"{output}.dcd", recorder_steps, append=append_output))
    simulation.reporters.append(CheckpointReporter(f"{output}.chk", recorder_steps * 2))

    print(f"Running NPT for {simulation_steps} steps (timestep {time_step} ps)...")
    simulation.step(simulation_steps)

    final_positions = simulation.context.getState(getPositions=True).getPositions()
    with open(f"{output}.pdb", "w") as f:
        PDBFile.writeFile(simulation.topology, final_positions, f)
    simulation.saveState(f"{output}.xml")
    print(f"NPT complete. State saved to: {output}.xml")

    result = {
        "step": "npt",
        "status": "completed",
        "output_prefix": output,
        "state_file": f"{output}.xml",
        "checkpoint": f"{output}.chk",
        "trajectory": f"{output}.dcd",
        "csv_file": f"{output}.csv",
        "final_structure": f"{output}.pdb",
        "steps_run": simulation_steps,
        "restarted": restart,
        "restrained": apply_restraints,
        "time_step_ps": time_step,
        "hydrogen_mass_amu": hydrogen_mass,
    }
    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _cli():
    import argparse

    p = argparse.ArgumentParser(
        description="NPT equilibration (--apply_restraints) or production (default) for protein–ligand systems.")
    p.add_argument("-t", "--topology_pkl", required=True, help="Pickled (topology, positions) from system_creation")
    p.add_argument("-x", "--system_xml", required=True, help="Serialized System XML from system_creation")
    p.add_argument("-r", "--trajectory", required=True,
                   help="Upstream state XML: NVT state for equil, NPT state for production")
    p.add_argument("-o", "--output", required=True, help="Output prefix (.xml/.dcd/.chk/.pdb/.csv written)")
    p.add_argument("--steps", type=int, default=500000, help="NPT/production steps")
    p.add_argument("--time_step", type=float, default=0.002, help="Integrator timestep in ps")
    p.add_argument("--recorder", type=int, default=5000, help="Reporter interval (steps)")
    p.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL"])
    p.add_argument("--device", type=int, default=0, help="CUDA device index / CPU thread count")
    p.add_argument("--apply_restraints", action="store_true",
                   help="Restrain protein backbone (NPT equilibration); omit for production")
    p.add_argument("--restart", action="store_true", help="Restart from checkpoint instead of upstream state")
    p.add_argument("--checkpoint_path", default=None, help="Checkpoint file (default <output>.chk)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = unset)")
    p.add_argument("--hydrogen_mass", type=float, default=None)
    p.add_argument("--json_output", default=None, help="Write the result dict to this JSON path")
    args = p.parse_args()

    main(
        topology_pkl=args.topology_pkl, system_xml=args.system_xml,
        trajectory=args.trajectory, simulation_steps=args.steps,
        time_step=args.time_step, recorder_steps=args.recorder,
        platform_=args.platform, cores=args.device, output=args.output,
        apply_restraints=args.apply_restraints, restart=args.restart,
        checkpoint_path=args.checkpoint_path, seed=args.seed,
        hydrogen_mass=args.hydrogen_mass, json_output=args.json_output,
    )


if __name__ == "__main__":
    _cli()
