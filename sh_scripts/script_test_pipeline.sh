#!/bin/bash
#SBATCH --job-name=config_md
#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --output=slurm_logs/config_md_%a.log
#SBATCH --cpus-per-task=5

INPUT_DIR="/data/ralmadamonter/atlas_pdbs"
OUTPUT_DIR="/data/ralmadamonter/openmm_scripts_json/tests/jsons"
SCRIPT="/data/ralmadamonter/openmm_scripts_json/scripts_running/create_config.py"
LOG_DIR="slurm_logs"

# Create an array of all .pdb files
FILES_PDB=($(ls ${INPUT_DIR}/*.pdb | sort -V))

# Get the file for this task
PDB_FILE=${FILES_PDB[$SLURM_ARRAY_TASK_ID]}

# Get basename (without extension) for output
BASENAME=$(basename "$PDB_FILE" .pdb)

# The config JSON lives inside the protein's own subfolder
CONFIG_FILE="${OUTPUT_DIR}/${BASENAME}/${BASENAME}.json"
# Note: with --n_replicas N, replica outputs go to {OUTPUT_DIR}/{BASENAME}/rep0/, rep1/, etc.

mkdir -p "$LOG_DIR"
echo "Logging to: ${LOG_DIR}/config_md_${BASENAME}.log"
exec > "${LOG_DIR}/config_md_${BASENAME}.log" 2>&1

echo "Running task ID: $SLURM_ARRAY_TASK_ID"
echo "PDB:    $PDB_FILE"
echo "Config: $CONFIG_FILE"

if [ -f "$CONFIG_FILE" ]; then
    echo "Skipping $BASENAME — config already exists."
else
    echo "Making config for $BASENAME"
    python "$SCRIPT" -p "$PDB_FILE" -o "$OUTPUT_DIR" --ph 7.4 --remove_heterogens \
        --force_field "charmm36" --force_field_water "charmm36/water" --box_size 1.5 \
        --platform CUDA --device 0 --n_replicas 3 --min_steps 10000 --nvt_steps 200000 \
        --npt_steps 500000 --prod_steps 5000000 \
        --cv_chainid 0 --cv_native_cutoff 0.45 --cv_beta 50.0 --cv_lam 1.8 \
        --cv_min_seq_sep 3 --cv_ca_min_seq_sep 3 \
        --rmsf_selection "protein and name CA" --rmsf_ref_frame 0 \
        --sasa_mode residue --sasa_probe_radius 0.14 --sasa_n_sphere_points 960 \
        --thermo_rolling_window 100 \
        --tica_lag 10 --tica_dim 2 --tica_stride 1 \
        --op_window 5 --op_chain 0 \
        --net_seg_ids A --net_n_windows 4 --net_sampled_frames 10 \
        --net_cutoff 4.5 --net_n_jobs 4
fi
