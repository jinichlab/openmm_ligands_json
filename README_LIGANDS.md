# Protein–ligand MD pipeline (OpenMM, at scale)

Adapted from the protein-only `openmm_scripts_json` pipeline to run **unbiased MD
of protein–ligand complexes** — one or many ligands — at scale, keeping the same
JSON-as-config-and-state-tracker design and the same SLURM step orchestration.

The same pipeline also runs **apo** proteins (empty ligand list) — it is a strict
superset of the original.

## What changed vs. the protein-only pipeline

| Concern | Protein-only | Protein–ligand |
|---|---|---|
| Ligand handling | `removeHeterogens()` deleted it | Parameterized and kept |
| Force field | CHARMM36, rebuilt every step | **Independently selectable** protein / water / ligand FFs, wired by `openmmforcefields.SystemGenerator` |
| System build | `ForceField.createSystem` in every step | **Parameterized once** in `system_creation`, serialized to `*_system.xml`; every downstream step deserializes it |
| Topology passing | `.cif` between steps | pickled `(topology, positions)` (lossless — keeps ligand bonds/resnames) |
| Restraints | backbone `CA,N,C` | backbone (protein only); optional ligand-heavy-atom restraint in NVT |
| Postprocessing | protein CVs | **+ ligand RMSD**, **+ Amber export for gmx_MMPBSA** |

## Force fields — all three are user-selectable

Set in the JSON (or via `create_config.py` flags). Defaults are the matched
AMBER family:

```json
"force_field":        "amber/ff14SB.xml",
"force_field_water":  "amber/tip3p_standard.xml",
"ligand_force_field": "gaff-2.11"
```

`ligand_force_field` accepts any `openmmforcefields` small-molecule FF:
`gaff-2.11`, `gaff-1.81`, `openff-2.1.0` (Sage), `espaloma-0.3.2`, …

> Mixing families (e.g. `charmm36.xml` protein + `gaff-2.11` ligand) works
> mechanically but is not ideal (LJ combining rules / CHARMM NBFIX differ).
> For MM/PBSA, stay in the AMBER family.

## Ligand input

Ligand **chemistry** (bond orders + protonation) comes from **SDF/MOL2** files —
one per physical copy — not from the PDB. The `split_ligand.py` helper reproduces
the PyMOL workflow (split → `h_add` → save SDF) and discovers how many ligands
there are:

```bash
python scripts_running/split_ligand.py -p complex.pdb -o split/ --ligand-list split/ligands.json
```

`ligands.json` lists the protein PDB and one entry per ligand SDF — exactly the
`ligands` block the config expects.

## How the ligand count is known

The config's `ligands` list is the single source of truth (count = its length).
`split_ligand.py` populates it from the complex PDB (PyMOL `organic` selection,
which excludes protein/water/ions), one SDF per residue instance. Multiple copies
of the same molecule → many SDFs in the topology but **one** GAFF/OpenFF template
(de-duplicated by graph isomorphism).

## Running

```bash
# apo or holo, single machine:
python scripts_running/create_config.py -p protein.pdb -o runs/ \
       --ligands_json split/ligands.json          # omit for apo
python scripts_running/run_pipeline.py runs/<name>/<name>.json

# at scale (SLURM), one complex per array task:
sbatch sh_scripts/ligand_submit.sh
```

Pipeline steps (unchanged order): `pdb_fixer → minimization_vac → system_creation
→ minimization_sol → nvt → npt → production`, then postprocessing
`unwrap → cvs → ligand_rmsd → rmsf → dssp → sasa → thermo → … → export_amber`.

### Running a single step standalone

`step_runner.py --step <name>` is the normal way to run one step (it pulls all
args from the JSON). Every step script is **also** individually runnable for
debugging — same as the protein-only pipeline:

```bash
python scripts_running/minimization.py vacuum -i fixed.cif --ligand LIG:lig.sdf -o min_vac
python scripts_running/system_creation.py -i min_vac.pkl --ligands_json ligands.json -o solvated
python scripts_running/equilibration_nvt_steps.py -t min_sol.pkl -x solvated_system.xml -o nvt
python scripts_running/equilibration_npt.py -t solvated.pkl -x solvated_system.xml \
       -r nvt.xml -o npt --apply_restraints          # omit --apply_restraints for production
```

Pass `--help` to any of them for the full option list. (`ligand_utils.py` is a
shared library, not a CLI.)

The ligand joins the protein at **`minimization_vac`**; it is parameterized and
serialized at **`system_creation`**, so antechamber/AM1-BCC runs only once per
complex (cached in `*_ligand_cache.json`).

## Ligand RMSD

`ligand_rmsd` aligns each frame on the protein (`protein and name CA` by default)
and reports per-ligand heavy-atom RMSD to a reference frame — the standard
binding-pose-stability metric. One CSV column per ligand copy + a mean.

## gmx_MMPBSA compatibility

`export_amber` converts the serialized System to Amber topologies via ParmEd:

- `*_mmpbsa_complex.prmtop` / `.inpcrd` — solvated; atom order matches `production.dcd`
- `*_mmpbsa_complex_dry.prmtop` — protein+ligand only; matches the water-stripped `unwrapped.dcd`
- ligand mask (e.g. `:LIG`) recorded in the JSON for receptor/ligand splitting

Then (in your `gmxmmpbsa` env):

```bash
ante-MMPBSA.py -p <prefix>_mmpbsa_complex_dry.prmtop -c com.prmtop -r rec.prmtop \
               -l lig.prmtop -s ':WAT,Na+,Cl-' -n ':LIG'
gmx_MMPBSA -O -i mmpbsa.in -cp <prefix>_mmpbsa_complex_dry.prmtop \
           -ct <prefix>_unwrapped.dcd -lm ':LIG'
```

Only valid for a matched AMBER-family run.

## Limitations / edge cases

- **Structural metals & non-organic cofactors.** `split_ligand.py` keeps only
  `polymer.protein` in the protein path and organic HETATMs as ligands, so a
  catalytic Zn²⁺/Mg²⁺ or an inorganic cofactor in a binding site is **dropped**
  (you'll see a `skipping … (excluded)` line). For metalloproteins, add the metal
  back explicitly (as a ligand with its own parameters, or via an ion ffxml) — it
  is not handled automatically.
- **Covalent ligands** are treated as a separate molecule (no protein–ligand bond).
- **Ligand protonation.** PyMOL `h_add` is a geometry/valence model, **not**
  pH-aware. Pass `split_ligand.py --ligand-ph <pH>` (matched to the protein
  `--ph`) to reassign titratable groups at that pH via OpenBabel (`obabel -p`,
  an empirical pKa model); `ligand_submit.sh` does this automatically via the
  `PH` variable. Without it, the ligand keeps whatever state the input heavy
  atoms imply. For difficult chemistries (tautomers, phosphates, metals nearby),
  still verify — or pre-protonate with a dedicated tool (Epik/Dimorphite-DL) and
  skip `--ligand-ph`. Protein hydrogens are always pH-aware (pdb_fixer step).

## Dependencies

`environment.yml` adds `openmmforcefields`, `openff-toolkit`, `rdkit`
(parameterization) and `pymol-open-source` (splitting) to the original stack.

```bash
conda env create -f environment.yml     # creates env "openmm-ligands"
```
