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

N_REPLICAS=$(python -c "import json; c=json.load(open('$JSON_FILE')); print(len(c.get('replicas', [None])))")

# Steps run sequentially per replica (all CPU-only)
for i in $(seq 0 $((N_REPLICAS - 1))); do
    echo "=== Replica $i ==="
    for STEP in unwrap cvs ligand_rmsd rmsf dssp sasa thermo order_parameter network_analysis export_amber; do
        echo "--- Step: $STEP ---"
        python "$SCRIPT" --config "$JSON_FILE" --step "$STEP" --replica "$i" --skip_if_done
    done
done
