#!/bin/bash
#SBATCH --job-name=atlas_md
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/atlas_%a.log
#
# Full MD pipeline for 10 proteins (3 replicates each).
# Each array task runs one protein end-to-end: config → production → postprocessing.
#
# HMR control:
#   Set HMR=1 to enable Hydrogen Mass Repartitioning (4 amu H, 4 fs timestep).
#   Set HMR=0 (default) to use standard masses (2 fs timestep).
#   Override at submission: sbatch --export=ALL,HMR=1 atlas_submit.sh
#
# Selected proteins (38 – 2128 aa):
#   3tvj_I   38 aa     5i8j_A   64 aa     2bfw_A  200 aa
#   3q0i_A  318 aa     1esw_A  500 aa     1ikp_A  613 aa
#   3wol_A  698 aa     3eh1_A  751 aa     2qmj_A  870 aa
#   6sup_A 2128 aa
#
# Submit with:
#   sbatch atlas_submit.sh               # no HMR
#   sbatch --export=ALL,HMR=1 atlas_submit.sh  # with HMR

set -euo pipefail

HMR=${HMR:-0}   # default: no HMR; set HMR=1 to enable

PDBS=(3tvj_I 5i8j_A 2bfw_A 3q0i_A 1esw_A 1ikp_A 3wol_A 3eh1_A 2qmj_A 6sup_A)

INPUT_DIR="/data/ralmadamonter/atlas_pdbs"
if [ "$HMR" -eq 1 ]; then
    OUTPUT_DIR="/data/ralmadamonter/openmm_scripts_json/tests/jsons_hmr"
else
    OUTPUT_DIR="/data/ralmadamonter/openmm_scripts_json/tests/jsons"
fi
SCRIPT_DIR="/data/ralmadamonter/openmm_scripts_json/scripts_running"
PP_DIR="/data/ralmadamonter/openmm_scripts_json/scripts_postprocessing"
LOG_DIR="/data/ralmadamonter/openmm_scripts_json/tests/slurm_logs"

mkdir -p "$LOG_DIR"

BASENAME=${PDBS[$SLURM_ARRAY_TASK_ID]}
PDB="${INPUT_DIR}/${BASENAME}.pdb"
CONFIG="${OUTPUT_DIR}/${BASENAME}/${BASENAME}.json"

exec > "${LOG_DIR}/atlas_${BASENAME}${HMR:+_hmr}.log" 2>&1
echo "=== $(date) ==="
echo "Protein: $BASENAME  |  Task: $SLURM_ARRAY_TASK_ID  |  HMR: $HMR"

# ── Create config ─────────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
    echo "--- create_config ---"
    python "$SCRIPT_DIR/create_config.py" \
        -p "$PDB" -o "$OUTPUT_DIR" \
        --n_replicas 3 --ph 7.4 --remove_heterogens \
        --force_field charmm36 --force_field_water charmm36/water \
        --box_size 1.5 --platform CUDA --device 0 \
        --min_steps 10000 --nvt_steps 200000 \
        --npt_steps 500000 --prod_steps 500000 \
        --nvt_recorder 500 --npt_recorder 5000 --prod_recorder 5000 \
        --cv_chainid 0 --cv_native_cutoff 0.45 --cv_beta 50.0 --cv_lam 1.8 \
        --cv_min_seq_sep 3 --cv_ca_min_seq_sep 3 \
        --rmsf_selection "protein and name CA" --rmsf_ref_frame 0 \
        --sasa_mode residue --sasa_probe_radius 0.14 --sasa_n_sphere_points 960 \
        --thermo_rolling_window 100 \
        --tica_lag 10 --tica_dim 2 --tica_stride 1 \
        --op_window 5 --op_chain 0 \
        --net_seg_ids A --net_n_windows 4 --net_sampled_frames 10 \
        --net_cutoff 4.5 --net_n_jobs 4 \
        $([ "$HMR" -eq 1 ] && echo "--hmr")
else
    echo "Config exists, skipping create_config."
fi

RUNNER="python $SCRIPT_DIR/step_runner.py --config $CONFIG"
PP_RUNNER="python $PP_DIR/postprocessing_runner.py --config $CONFIG"
N_REP=$(python -c "import json; c=json.load(open('$CONFIG')); print(len(c.get('replicas', [None])))")

# ── Shared steps (run once) ───────────────────────────────────────────────────
for STEP in pdb_fixer minimization_vac system_creation minimization_sol; do
    echo "--- $STEP ---"
    $RUNNER --step $STEP --skip_if_done
done

# ── Replica steps ─────────────────────────────────────────────────────────────
for r in $(seq 0 $((N_REP - 1))); do
    echo "=== Replica $r ==="
    for STEP in nvt check_nvt npt check_npt production; do
        echo "--- $STEP ---"
        $RUNNER --step $STEP --replica $r --skip_if_done
    done
done

# ── Postprocessing ────────────────────────────────────────────────────────────
for r in $(seq 0 $((N_REP - 1))); do
    echo "=== Postprocessing replica $r ==="
    for STEP in unwrap cvs rmsf dssp sasa thermo order_parameter network_analysis; do
        echo "--- $STEP ---"
        $PP_RUNNER --step $STEP --replica $r --skip_if_done
    done
done

# ── Performance summary (runs on last task only) ──────────────────────────────
if [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
    echo "--- analyze_performance ---"
    python "$SCRIPT_DIR/analyze_performance.py" \
        -d "$OUTPUT_DIR" \
        --csv "${OUTPUT_DIR}/performance.csv" \
        --no_detail
fi

echo "=== Done: $(date) ==="
