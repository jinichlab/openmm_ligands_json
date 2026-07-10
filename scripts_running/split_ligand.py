#!/usr/bin/env python3
"""
split_ligand.py
===============
Split a protein–ligand complex PDB into:
  * a protein-only PDB   (protein atoms, HETATM ligands removed)
  * one SDF per ligand   (organic HETATM residues, hydrogens added, bond orders)

This reproduces the manual PyMOL workflow — select `organic`, `h_add`, save an
SDF per ligand instance — so it is scriptable across many complexes.  `organic`
excludes protein, water, and monatomic ions by construction, so ions/waters stay
with the protein path (the force field + neutralization handles them).

The emitted `--ligand-list` JSON is exactly the `ligands` block create_config.py
expects, so the number of ligands is discovered here (where the HETATMs are
visible) and frozen into the config.

Requires PyMOL (`pymol-open-source`).  For an apo structure this simply writes
the protein PDB and an empty ligand list.

Ligand protonation
------------------
PyMOL's `h_add` uses a valence/geometry model — it fills open valences but is
NOT pH-aware, so a titratable group keeps whatever state the input heavy atoms
imply.  Pass `--ligand-ph <pH>` to reassign the ligand protonation state at that
pH with OpenBabel (`obabel -p`), an empirical pKa model.  Protein hydrogens are
handled separately (pH-aware) in the pdb_fixer step.

Usage:
  python split_ligand.py -p complex.pdb -o out_dir/
  python split_ligand.py -p complex.pdb -o out_dir/ --exclude HOH GOL --ligand-list out_dir/ligands.json
  python split_ligand.py -p complex.pdb -o out_dir/ --ligand-ph 7.4
"""

import argparse
import json
import os
import shutil
import subprocess

from ligand_utils import NON_LIGAND_RESNAMES, is_ligand_resname


def _protonate_at_ph(in_sdf, out_sdf, ph):
    """Reassign ligand protonation at the given pH with OpenBabel (`obabel -p`).

    OpenBabel's `-p` applies an empirical pKa model, (de)protonating titratable
    groups for the target pH while preserving 3D coordinates.  Raises if obabel
    is unavailable or fails, so a requested pH is never silently ignored.
    """
    obabel = shutil.which("obabel")
    if obabel is None:
        raise RuntimeError(
            "--ligand-ph requires OpenBabel, but `obabel` was not found on PATH. "
            "Install it (conda install -c conda-forge openbabel) or drop --ligand-ph "
            "to keep PyMOL's geometry-based protonation."
        )
    cmd_line = [obabel, in_sdf, "-O", out_sdf, "-p", str(ph)]
    proc = subprocess.run(cmd_line, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_sdf) or os.path.getsize(out_sdf) == 0:
        raise RuntimeError(
            f"obabel pH protonation failed for {in_sdf} at pH {ph}:\n"
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def split(pdb_path, out_dir, prefix=None, extra_exclude=None, ph=None):
    """Return (protein_pdb_path, [ligand_spec, ...]) using PyMOL."""
    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-qc"])  # quiet, no GUI

    os.makedirs(out_dir, exist_ok=True)
    prefix = prefix or os.path.splitext(os.path.basename(pdb_path))[0]
    exclude = {r.upper() for r in (extra_exclude or [])} | NON_LIGAND_RESNAMES

    cmd.load(pdb_path, "complex")

    # ── Protein path: everything that is polymer.protein, minus any HETATM ───
    protein_pdb = os.path.join(out_dir, f"{prefix}_protein.pdb")
    cmd.create("prot", "complex and polymer.protein")
    cmd.save(protein_pdb, "prot")
    print(f"[split] protein -> {protein_pdb}")

    # ── Ligand path: organic HETATMs, grouped per residue instance ───────────
    ligand_specs = []
    cmd.select("ligs", "complex and organic")
    # Enumerate unique (chain, resi, resn) residues in the ligand selection
    residues = set()
    cmd.iterate("ligs", "residues.add((chain, resi, resn))", space={"residues": residues})

    for chain, resi, resn in sorted(residues):
        if not is_ligand_resname(resn) or resn.upper() in exclude:
            print(f"[split] skipping {resn} {chain}/{resi} (excluded)")
            continue
        sel = f"complex and organic and chain {chain} and resi {resi}"
        obj = f"lig_{resn}_{chain}_{resi}"
        cmd.create(obj, sel)
        cmd.remove(f"{obj} and hydro")   # start from heavy atoms
        cmd.h_add(obj)                    # add hydrogens (PyMOL valence model)
        sdf = os.path.join(out_dir, f"{prefix}_lig_{resn}_{chain}_{resi}.sdf")
        if ph is not None:
            # PyMOL geometry protonation first, then reassign at target pH.
            geom_sdf = os.path.join(out_dir, f"{prefix}_lig_{resn}_{chain}_{resi}_geom.sdf")
            cmd.save(geom_sdf, obj)
            _protonate_at_ph(geom_sdf, sdf, ph)
            print(f"[split] ligand {resn} {chain}/{resi} -> {sdf} (protonated at pH {ph})")
        else:
            cmd.save(sdf, obj)
            print(f"[split] ligand {resn} {chain}/{resi} -> {sdf}")
        ligand_specs.append({"sdf": sdf, "resname": resn, "chain": chain, "resid": resi})

    cmd.delete("all")
    return protein_pdb, ligand_specs


def main():
    parser = argparse.ArgumentParser(description="Split a complex PDB into protein PDB + ligand SDFs (PyMOL)")
    parser.add_argument("-p", "--pdb", required=True, help="Input protein–ligand complex PDB")
    parser.add_argument("-o", "--out_dir", required=True, help="Directory for the protein PDB and ligand SDFs")
    parser.add_argument("--prefix", default=None, help="Output name prefix (default: PDB basename)")
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="Extra residue names to treat as non-ligand (e.g. crystallization additives GOL EDO)")
    parser.add_argument("--ligand-list", default=None,
                        help="Write the discovered ligand list as JSON (feeds create_config.py --ligands_json)")
    parser.add_argument("--ligand-ph", type=float, default=None,
                        help="Reassign ligand protonation at this pH via OpenBabel (obabel -p). "
                             "Omit to keep PyMOL's geometry-based protonation. Match the pdb_fixer --ph.")
    args = parser.parse_args()

    protein_pdb, ligand_specs = split(args.pdb, args.out_dir, args.prefix, args.exclude, ph=args.ligand_ph)

    print(f"\n[split] {len(ligand_specs)} ligand(s) discovered.")
    payload = {"protein_pdb": protein_pdb, "ligands": ligand_specs}
    list_path = args.ligand_list or os.path.join(args.out_dir, f"{args.prefix or 'complex'}_ligands.json")
    with open(list_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[split] ligand list written to: {list_path}")


if __name__ == "__main__":
    main()
