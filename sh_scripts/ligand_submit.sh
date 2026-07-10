#!/bin/bash
#SBATCH --job-name=ligand_md
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/ligand_%a.log
#
# Full protein–ligand MD pipeline, one complex per array task.
# Per complex:  split (PyMOL) -> create_config -> shared steps -> replicas
#               -> ligand-aware postprocessing (incl. ligand_rmsd + export_amber).
#
# Each complex is one PDB containing protein + ligand(s).  split_ligand.py carves
# the ligand(s) out as SDF(s) and discovers how many there are; create_config
# freezes that list into the JSON.  An apo PDB (no organic HETATMs) just runs the
# protein-only path.
#
# Force fields (override per run): protein amber/ff14SB.xml, water tip3p, ligand gaff-2.11.
# HMR: sbatch --export=ALL,HMR=1 ligand_submit.sh
#
# Submit:  sbatch ligand_submit.sh

set -euo pipefail

HMR=${HMR:-0}
PH=${PH:-7.4}                  # protein AND ligand protonation pH (keep them matched)

# One PDB per complex (protein + ligand together)
COMPLEXES=(cplx01 cplx02 cplx03 cplx04 cplx05 cplx06 cplx07 cplx08 cplx09 cplx10)

INPUT_DIR="/data/ralmadamonter/ligand_pdbs"          # <basename>.pdb complexes
OUTPUT_DIR="/data/ralmadamonter/ligand_md/jsons"
SPLIT_DIR="/data/ralmadamonter/ligand_md/split"
SCRIPT_DIR="/data/ralmadamonter/openmm_scripts_json_ligands/scripts_running"
PP_DIR="/data/ralmadamonter/openmm_scripts_json_ligands/scripts_postprocessing"
LOG_DIR="/data/ralmadamonter/ligand_md/slurm_logs"

mkdir -p "$LOG_DIR" "$SPLIT_DIR"

BASENAME=${COMPLEXES[$SLURM_ARRAY_TASK_ID]}
PDB="${INPUT_DIR}/${BASENAME}.pdb"
CONFIG="${OUTPUT_DIR}/${BASENAME}/${BASENAME}.json"
LIGLIST="${SPLIT_DIR}/${BASENAME}/${BASENAME}_ligands.json"

exec > "${LOG_DIR}/ligand_${BASENAME}${HMR:+_hmr}.log" 2>&1
echo "=== $(date) ===  complex=$BASENAME  task=$SLURM_ARRAY_TASK_ID  HMR=$HMR"

# ── 1. Split complex -> protein PDB + ligand SDF(s) ───────────────────────────
if [ ! -f "$LIGLIST" ]; then
    echo "--- split_ligand ---"
    python "$SCRIPT_DIR/split_ligand.py" \
        -p "$PDB" -o "${SPLIT_DIR}/${BASENAME}" \
        --prefix "$BASENAME" --ligand-list "$LIGLIST" \
        --ligand-ph "$PH"
fi
PROTEIN_PDB=$(python -c "import json;print(json.load(open('$LIGLIST'))['protein_pdb'])")

# ── 2. Create config (freezes the discovered ligand list) ─────────────────────
if [ ! -f "$CONFIG" ]; then
    echo "--- create_config ---"
    python "$SCRIPT_DIR/create_config.py" \
        -p "$PROTEIN_PDB" -o "$OUTPUT_DIR" --prefix "$BASENAME" \
        --ligands_json "$LIGLIST" \
        --n_replicas 3 --ph "$PH" \
        --force_field amber/ff14SB.xml \
        --force_field_water amber/tip3p_standard.xml \
        --ligand_force_field gaff-2.11 \
        --box_size 1.0 --platform CUDA --device 0 \
        --min_steps 10000 --nvt_steps 200000 \
        --npt_steps 500000 --prod_steps 5000000 \
        --nvt_recorder 500 --npt_recorder 5000 --prod_recorder 5000 \
        --ligrmsd_align "protein and name CA" --ligrmsd_ref_frame 0 \
        --rmsf_selection "protein and name CA" \
        --sasa_mode residue \
        $([ "$HMR" -eq 1 ] && echo "--hmr")
fi

RUNNER="python $SCRIPT_DIR/step_runner.py --config $CONFIG"
PP_RUNNER="python $PP_DIR/postprocessing_runner.py --config $CONFIG"
N_REP=$(python -c "import json;print(len(json.load(open('$CONFIG')).get('replicas',[None])))")

# ── 3. Shared steps (parameterize ligand once, serialize system.xml) ──────────
for STEP in pdb_fixer minimization_vac system_creation minimization_sol; do
    echo "--- $STEP ---"; $RUNNER --step $STEP --skip_if_done
done

# ── 4. Replica steps ──────────────────────────────────────────────────────────
for r in $(seq 0 $((N_REP - 1))); do
    echo "=== Replica $r ==="
    for STEP in nvt check_nvt npt check_npt production; do
        echo "--- $STEP ---"; $RUNNER --step $STEP --replica $r --skip_if_done
    done
done

# ── 5. Postprocessing (incl. ligand_rmsd + Amber export for gmx_MMPBSA) ───────
for r in $(seq 0 $((N_REP - 1))); do
    echo "=== Postprocessing replica $r ==="
    for STEP in unwrap cvs ligand_rmsd rmsf dssp sasa thermo order_parameter network_analysis export_amber; do
        echo "--- $STEP ---"; $PP_RUNNER --step $STEP --replica $r --skip_if_done
    done
done

echo "=== Done: $(date) ==="
