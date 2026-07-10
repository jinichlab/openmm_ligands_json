"""
ligand_utils.py
===============
Shared helpers for protein–ligand MD.  Everything that needs to know about the
ligand(s) goes through here, so the rest of the pipeline stays close to the
protein-only version.

The ligand chemistry (bond orders + protonation) comes from SDF/MOL2 files —
one file per physical copy — produced upstream (e.g. the PyMOL split in
split_ligand.py).  Coordinates come from those same files.

Force fields are fully user-selectable:
    protein/water  -> ordinary OpenMM ffxml  (amber/ff14SB.xml, charmm36.xml, ...)
    ligand         -> a small_molecule_forcefield string understood by
                      openmmforcefields (gaff-2.11, openff-2.1.0, espaloma-0.3.2, ...)

All three are wired into a single System by openmmforcefields.SystemGenerator.
The SystemGenerator's internal ForceField also carries the ligand template, so
Modeller.addSolvent() works transparently around the ligand.
"""

import json
import os
import pickle

from openff.toolkit import Molecule
from openmm import unit, XmlSerializer
from openmm.app import Modeller, PDBxFile, PME, NoCutoff, HBonds


# Residue names that are NOT ligands even though they are HETATMs: crystallographic
# water and monatomic ions (handled by the protein/water force field + neutralize).
NON_LIGAND_RESNAMES = {
    "HOH", "WAT", "TIP3", "TIP", "SOL", "T3P",
    "NA", "SOD", "CL", "CLA", "K", "POT", "LI", "RB", "CS", "F", "BR", "I",
    "MG", "CA", "ZN", "MN", "FE", "FE2", "FE3", "CO", "CU", "CU1", "NI", "CD", "HG", "SR", "BA",
}


def load_ligand_molecules(ligand_specs):
    """
    Load ligand SDF/MOL2 files into OpenFF Molecule objects.

    Parameters
    ----------
    ligand_specs : list
        Each entry is either a path string, or a dict with at least a "sdf"
        (or "file") key and optional "resname" metadata.

    Returns
    -------
    molecules : list[openff.toolkit.Molecule]
        One Molecule per physical copy, carrying its 3D conformer.
    resnames : list[str]
        The residue name used for each molecule in the combined topology.
    """
    molecules, resnames = [], []
    for i, spec in enumerate(ligand_specs):
        if isinstance(spec, str):
            path, resname = spec, None
        else:
            path = spec.get("sdf") or spec.get("file") or spec.get("path")
            resname = spec.get("resname")
        if not path:
            raise ValueError(f"Ligand spec #{i} has no SDF/MOL2 path: {spec!r}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ligand file not found: {path}")

        loaded = Molecule.from_file(path, allow_undefined_stereo=True)
        # from_file may return a single Molecule or a list (multi-record SDF)
        mols = loaded if isinstance(loaded, list) else [loaded]
        for mol in mols:
            if mol.n_conformers == 0:
                raise ValueError(
                    f"Ligand {path} has no 3D conformer — the SDF must carry "
                    f"coordinates (PyMOL 'save file.sdf' preserves them)."
                )
            molecules.append(mol)
            # Fall back to a generated 3-letter code if none was supplied
            resnames.append((resname or f"L{i:02d}")[:4])
    return molecules, resnames


def unique_by_chemistry(molecules):
    """
    De-duplicate a list of Molecules by chemical identity (graph isomorphism via
    canonical isomeric SMILES).  SystemGenerator only needs one template per
    unique species, even when many copies appear in the topology.
    """
    seen, unique = {}, []
    for mol in molecules:
        key = mol.to_smiles(isomeric=True, mapped=False)
        if key not in seen:
            seen[key] = True
            unique.append(mol)
    return unique


def build_system_generator(protein_ff, water_ff, ligand_ff, molecules,
                           periodic, hydrogen_mass=None, cache=None,
                           nonbonded_cutoff=1.2):
    """
    Construct an openmmforcefields SystemGenerator that knows about the protein,
    water, and every unique ligand chemistry.

    periodic=True  -> PME system (solvated steps)
    periodic=False -> NoCutoff system (vacuum minimization)
    """
    from openmmforcefields.generators import SystemGenerator

    forcefield_kwargs = {
        "constraints": HBonds,
        "rigidWater": True,
        "removeCMMotion": False,
    }
    if hydrogen_mass is not None:
        forcefield_kwargs["hydrogenMass"] = hydrogen_mass * unit.amu

    kwargs = dict(
        forcefields=[_ffxml(protein_ff), _ffxml(water_ff)],
        small_molecule_forcefield=ligand_ff,
        molecules=unique_by_chemistry(molecules),
        forcefield_kwargs=forcefield_kwargs,
    )
    if periodic:
        kwargs["periodic_forcefield_kwargs"] = {
            "nonbondedMethod": PME,
            "nonbondedCutoff": nonbonded_cutoff * unit.nanometers,
        }
    else:
        kwargs["nonperiodic_forcefield_kwargs"] = {"nonbondedMethod": NoCutoff}
    if cache:
        kwargs["cache"] = cache

    return SystemGenerator(**kwargs)


def _ffxml(name):
    """Normalize a force-field name to an .xml filename OpenMM's ForceField expects."""
    return name if name.endswith(".xml") else name + ".xml"


def build_complex_modeller(protein_pdbx, molecules, resnames):
    """
    Combine a fixed protein (PDBx/mmCIF) with one or more ligand Molecules into a
    single Modeller, tagging each ligand residue with its resname.

    Returns a Modeller holding protein + all ligand copies (no solvent yet).
    """
    protein = PDBxFile(protein_pdbx)
    modeller = Modeller(protein.topology, protein.positions)

    for mol, resname in zip(molecules, resnames):
        off_top = mol.to_topology()
        omm_top = off_top.to_openmm()
        # Name the single ligand residue so downstream selections can find it
        for res in omm_top.residues():
            res.name = resname
        positions = mol.conformers[0].to_openmm()
        modeller.add(omm_top, positions)

    return modeller


def is_ligand_resname(resname):
    """True if a HETATM residue name should be treated as a parameterized ligand."""
    return resname.strip().upper() not in NON_LIGAND_RESNAMES


def ligands_from_cli(ligand=None, ligands_json=None):
    """Build a ligand_specs list from standalone-CLI arguments.

    Shared by the per-step scripts so `--ligand RESNAME:PATH.sdf` (repeatable)
    and `--ligands_json <split output>` behave identically everywhere.  Returns
    a list of {"sdf", "resname"} dicts (empty list == apo).
    """
    specs = []
    if ligands_json:
        with open(ligands_json) as f:
            payload = json.load(f)
        # Accept either the split_ligand.py payload ({"ligands":[...]}) or a bare list.
        specs.extend(payload["ligands"] if isinstance(payload, dict) else payload)
    for item in (ligand or []):
        if ":" not in item:
            raise ValueError(f"--ligand expects RESNAME:PATH.sdf, got '{item}'")
        resname, path = item.split(":", 1)
        specs.append({"resname": resname, "sdf": path})
    return specs


# ---------------------------------------------------------------------------
# Serialization helpers
#
# The ligand is parameterized once (in system_creation) and the resulting
# System is serialized to XML.  Every downstream step deserializes that System
# instead of rebuilding it from a ForceField — so ligand parameters flow through
# minimization/nvt/npt/production unchanged, and antechamber/AM1-BCC runs only
# once.  The Topology (which carries ligand bonds and residue names that a
# CIF round-trip can lose) is pickled alongside it, losslessly.
# ---------------------------------------------------------------------------

def save_system(path, system):
    with open(path, "w") as f:
        f.write(XmlSerializer.serialize(system))


def load_system(path):
    with open(path) as f:
        return XmlSerializer.deserialize(f.read())


def save_topology(path, topology, positions):
    """Pickle an OpenMM Topology + positions together (bonds/resnames preserved)."""
    with open(path, "wb") as f:
        pickle.dump({"topology": topology, "positions": positions}, f)


def load_topology(path):
    """Return (topology, positions) from a pickle written by save_topology."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["topology"], data["positions"]


# ---------------------------------------------------------------------------
# Position restraints
# ---------------------------------------------------------------------------

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # common protonation/terminal variants
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN",
}


def add_backbone_restraints(system, topology, positions, k=100.0, periodic=True):
    """
    Restrain PROTEIN backbone atoms (CA, N, C) to their current positions.
    Ligand, ions and water are left free.  Uses the periodic-aware form for
    solvated (PME) systems and a plain harmonic form for vacuum.

    Returns the number of atoms restrained.
    """
    from openmm import CustomExternalForce

    if periodic:
        expr = "k*periodicdistance(x, y, z, x0, y0, z0)^2"
    else:
        expr = "0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"

    force = CustomExternalForce(expr)
    force.addGlobalParameter("k", k * unit.kilojoules_per_mole / unit.nanometer**2)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)

    pos = positions.value_in_unit(unit.nanometer)
    n = 0
    for atom in topology.atoms():
        if atom.residue.name.upper() in STANDARD_AA and atom.name in ("CA", "N", "C"):
            p = pos[atom.index]
            force.addParticle(atom.index, [p[0], p[1], p[2]])
            n += 1
    system.addForce(force)
    return n


def add_ligand_restraints(system, topology, positions, k=100.0, periodic=True):
    """
    Optionally restrain ligand heavy atoms during equilibration (keeps the pose
    fixed while water relaxes).  A ligand residue is any residue that is not a
    standard amino acid, water, or monatomic ion.  Returns the atom count.
    """
    from openmm import CustomExternalForce

    # Distinct parameter name (k_lig) so this force can coexist with the
    # backbone restraint's global parameter "k" in the same System.
    expr = ("k_lig*periodicdistance(x, y, z, x0, y0, z0)^2" if periodic
            else "0.5*k_lig*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force = CustomExternalForce(expr)
    force.addGlobalParameter("k_lig", k * unit.kilojoules_per_mole / unit.nanometer**2)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)

    pos = positions.value_in_unit(unit.nanometer)
    n = 0
    for atom in topology.atoms():
        rname = atom.residue.name.upper()
        is_ligand = (rname not in STANDARD_AA and rname not in NON_LIGAND_RESNAMES
                     and not atom.residue.name.upper().startswith("HOH"))
        if is_ligand and atom.element is not None and atom.element.symbol != "H":
            p = pos[atom.index]
            force.addParticle(atom.index, [p[0], p[1], p[2]])
            n += 1
    if n > 0:
        system.addForce(force)
    return n
