#!/usr/bin/env python3
"""
export_amber.py
===============
Export Amber topologies for MM/PB(GB)SA (e.g. gmx_MMPBSA / AmberTools MMPBSA.py).

Because the pipeline defaults to an Amber-family force field (ff14SB + GAFF/OpenFF
+ TIP3P), the OpenMM System converts to Amber prmtop/inpcrd via ParmEd.

The dynamics System (solvated_system.xml) is built with constraints=HBonds and
rigid water, so all X-H / water bonds are *constraints* with no HarmonicBondForce
term — ParmEd then leaves those bonds type-less and prmtop writing fails.  So this
step rebuilds an equivalent **unconstrained** System from the same solvated
topology (constraints=None, rigidWater=False), reusing the ligand parameter cache
(no re-parameterization).  Atom order is identical to the topology, hence to
production.dcd.  The removed bond constraints don't change MM/PB(GB)SA energies.

The dry complex additionally has its periodic box stripped (GB/PB need IFBOX=0)
and mbondi2 GB radii assigned (OpenMM carries none -> EGB would be NaN).

This writes:

  {prefix}_complex.prmtop / .inpcrd   solvated complex; atom order matches
                                      production.dcd  (use as -cp with the raw
                                      solvated trajectory)
  {prefix}_complex_dry.prmtop         protein+ligand only (water/ions stripped,
                                      no box, mbondi2 radii); atom order matches
                                      a water-stripped, pbc-removed trajectory

and reports the ligand mask (e.g. ":LIG") so ante-MMPBSA.py can split receptor
and ligand.

Typical downstream use with AmberTools MMPBSA.py (verified end-to-end):
  ante-MMPBSA.py -p {prefix}_complex_dry.prmtop -c com.prmtop -r rec.prmtop \\
                 -l lig.prmtop -s ':WAT,HOH,Na+,Cl-,NA,CL' -n ':LIG'
  MMPBSA.py -O -i mmpbsa.in -cp {prefix}_complex_dry.prmtop \\
            -rp rec.prmtop -lp lig.prmtop -y dry.dcd -o FINAL_RESULTS.dat

(gmx_MMPBSA 1.6.x is NOT a drop-in: it only takes GROMACS .top/.tpr/.xtc, so it
would need a prmtop -> .top conversion first.)

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


def main(system_xml, topology_pkl, output_prefix, ligand_resnames=None, json_output=None,
         ligand_specs=None, forcefield_name="amber/ff14SB.xml",
         water_ff="amber/tip3p_standard.xml", ligand_ff="gaff-2.11", cache=None):
    import parmed

    topology, positions = lu.load_topology(topology_pkl)

    # Rebuild an *unconstrained* System from the same topology so every bond
    # carries a real HarmonicBondForce term (constrained bonds have none, which
    # breaks the prmtop writer).  Reuses the ligand cache -> no re-parameterization.
    print("Rebuilding unconstrained System for a clean Amber export...")
    molecules, _ = lu.load_ligand_molecules(ligand_specs or [])
    generator = lu.build_system_generator(
        forcefield_name, water_ff, ligand_ff, molecules,
        periodic=True, cache=cache, constraints=None, rigid_water=False,
    )
    system = generator.create_system(topology)

    print("Converting OpenMM System -> ParmEd Structure...")
    structure = parmed.openmm.load_topology(topology, system, xyz=positions)

    # OpenMM Systems carry no implicit-solvent (GB) radii, so ParmEd writes them
    # as zero -> MM/GBSA returns EGB = NaN.  Assign the mbondi2 radius set (the
    # one recommended for igb=2/5, the common MM-GBSA choices) so the exported
    # prmtops are usable as-is.  Applied to the full structure before the dry /
    # receptor / ligand subsets are derived, so all inherit it.
    from parmed.tools import changeRadii
    changeRadii(structure, "mbondi2").execute()

    complex_prmtop = f"{output_prefix}_complex.prmtop"
    complex_inpcrd = f"{output_prefix}_complex.inpcrd"
    structure.save(complex_prmtop, overwrite=True)
    structure.save(complex_inpcrd, format="rst7", overwrite=True)
    print(f"Solvated complex -> {complex_prmtop} / {complex_inpcrd}")

    # Dry complex: keep everything that is NOT water/ion (protein + ligand).
    # Drop the periodic box: MM/PB(GB)SA runs implicit-solvent (gb>0 / PB) and
    # rejects a prmtop with IFBOX!=0 ("gb>0 is incompatible with periodic
    # boundary conditions"). The dry topology + a pbc-removed trajectory is what
    # the downstream MMPBSA.py / ante-MMPBSA.py split expects.
    keep_mask = "!:" + WATER_IONS
    dry = structure[keep_mask]
    dry.box = None
    complex_dry = f"{output_prefix}_complex_dry.prmtop"
    dry.save(complex_dry, overwrite=True)
    print(f"Dry complex (protein+ligand, no box) -> {complex_dry}")

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
    lu.configure_logging()
    parser = argparse.ArgumentParser(description="Export Amber prmtop/inpcrd for MM/PBSA (rebuilds an unconstrained System)")
    parser.add_argument("-s", "--system_xml", default=None,
                        help="Serialized System XML (kept for interface compatibility; not used — "
                             "the export rebuilds an unconstrained System from the FF + topology)")
    parser.add_argument("-t", "--topology_pkl", required=True, help="Pickled topology+positions (from system_creation)")
    parser.add_argument("-o", "--output_prefix", required=True)
    parser.add_argument("--ligand_resnames", nargs="+", default=None)
    parser.add_argument("--ligand", action="append", metavar="RESNAME:PATH.sdf",
                        help="Ligand SDF (repeatable); omit for apo")
    parser.add_argument("--ligands_json", help="split_ligand.py ligand-list JSON")
    parser.add_argument("--force_field", default="amber/ff14SB.xml")
    parser.add_argument("--force_field_water", default="amber/tip3p_standard.xml")
    parser.add_argument("--ligand_force_field", default="gaff-2.11")
    parser.add_argument("--cache", default=None, help="Ligand parameter cache JSON (reuse system_creation's)")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    main(
        args.system_xml, args.topology_pkl, args.output_prefix, args.ligand_resnames, args.json,
        ligand_specs=lu.ligands_from_cli(args.ligand, args.ligands_json),
        forcefield_name=args.force_field, water_ff=args.force_field_water,
        ligand_ff=args.ligand_force_field, cache=args.cache,
    )
