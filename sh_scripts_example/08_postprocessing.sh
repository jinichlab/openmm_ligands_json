#!/bin/bash
#SBATCH --job-name=postprocessing
#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --output=slurm_logs/postprocessing_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

JSON_DIR="/data/ralmadamonter/openmm_scripts_json/tests/jsons"
SCRIPT="/data/ralmadamonter/openmm_scripts_json/scripts_postprocessing/postprocessing_runner.py"
LOG_DIR="slurm_logs"

FILES=($(ls ${JSON_DIR}/*/*.json | sort -V))
JSON_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$JSON_FILE" .json)

mkdir -p "$LOG_DIR"
echo "Logging to: ${LOG_DIR}/postprocessing_${BASENAME}.log"
exec > "${LOG_DIR}/postprocessing_${BASENAME}.log" 2>&1

echo "Running task ID: $SLURM_ARRAY_TASK_ID"
echo "JSON:   $JSON_FILE"

# Steps run sequentially for each protein (all CPU-only)
for STEP in unwrap cvs rmsf dssp sasa thermo order_parameter tica network_analysis; do
    echo "--- Step: $STEP ---"
    python "$SCRIPT" --config "$JSON_FILE" --step "$STEP" --skip_if_done
done
