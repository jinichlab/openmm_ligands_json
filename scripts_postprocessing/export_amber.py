#!/usr/bin/env python3
"""
export_amber.py
===============
Export Amber topologies for MM/PB(GB)SA (e.g. gmx_MMPBSA / AmberTools MMPBSA.py).

Because the pipeline defaults to an Amber-family force field (ff14SB + GAFF/OpenFF
+ TIP3P), the serialized OpenMM System converts cleanly to Amber prmtop/inpcrd via
ParmEd.  This writes:

  {prefix}_complex.prmtop / .inpcrd   solvated complex; atom order matches
                                      production.dcd  (use as -cp with the raw
                                      solvated trajectory)
  {prefix}_complex_dry.prmtop         protein+ligand only (water/ions stripped);
                                      atom order matches the water-stripped
                                      unwrapped.dcd

and reports the ligand mask (e.g. ":LIG") so gmx_MMPBSA / ante-MMPBSA.py can split
receptor and ligand.

Typical downstream use (documented, not run here):
  ante-MMPBSA.py -p {prefix}_complex_dry.prmtop -c com.prmtop -r rec.prmtop \\
                 -l lig.prmtop -s ':WAT,Na+,Cl-' -n ':LIG'
  gmx_MMPBSA -O -i mmpbsa.in -cp {prefix}_complex_dry.prmtop \\
             -ct {prefix}_unwrapped.dcd -lm ':LIG'

NOTE: MM/PB(GB)SA is only physically meaningful for a matched Amber-family run.
If the protein FF was overridden to CHARMM36, these files still write but the
resulting energies are not valid.
"""

import argparse
import json
import os

# scripts_running holds the serialization + ligand helpers
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts_running"))
import ligand_utils as lu

WATER_IONS = "HOH,WAT,SOL,Na+,Cl-,K+,NA,CL,K,MG,ZN,Mg2+,Zn2+"


def main(system_xml, topology_pkl, output_prefix, ligand_resnames=None, json_output=None):
    import parmed

    topology, positions = lu.load_topology(topology_pkl)
    system = lu.load_system(system_xml)

    print("Converting OpenMM System -> ParmEd Structure...")
    structure = parmed.openmm.load_topology(topology, system, xyz=positions)

    complex_prmtop = f"{output_prefix}_complex.prmtop"
    complex_inpcrd = f"{output_prefix}_complex.inpcrd"
    structure.save(complex_prmtop, overwrite=True)
    structure.save(complex_inpcrd, format="rst7", overwrite=True)
    print(f"Solvated complex -> {complex_prmtop} / {complex_inpcrd}")

    # Dry complex: keep everything that is NOT water/ion (protein + ligand)
    keep_mask = "!:" + WATER_IONS
    dry = structure[keep_mask]
    complex_dry = f"{output_prefix}_complex_dry.prmtop"
    dry.save(complex_dry, overwrite=True)
    print(f"Dry complex (protein+ligand) -> {complex_dry}")

    ligand_mask = None
    if ligand_resnames:
        ligand_mask = ":" + ",".join(sorted(set(ligand_resnames)))

    result = {
        "step": "export_amber",
        "status": "completed",
        "complex_prmtop": complex_prmtop,
        "complex_inpcrd": complex_inpcrd,
        "complex_dry_prmtop": complex_dry,
        "ligand_mask": ligand_mask,
        "strip_mask": ":" + WATER_IONS,
    }
    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Amber prmtop/inpcrd for MM/PBSA from a serialized System")
    parser.add_argument("-s", "--system_xml", required=True, help="Serialized System XML (from system_creation)")
    parser.add_argument("-t", "--topology_pkl", required=True, help="Pickled topology+positions (from system_creation)")
    parser.add_argument("-o", "--output_prefix", required=True)
    parser.add_argument("--ligand_resnames", nargs="+", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    main(args.system_xml, args.topology_pkl, args.output_prefix, args.ligand_resnames, args.json)
