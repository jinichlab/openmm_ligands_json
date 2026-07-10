#!/bin/bash
#SBATCH --job-name=npt
#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --output=slurm_logs/npt_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

JSON_DIR="/data/ralmadamonter/openmm_scripts_json/tests/jsons"
SCRIPT="/data/ralmadamonter/openmm_scripts_json/scripts_running/step_runner.py"
LOG_DIR="slurm_logs"

FILES=($(ls ${JSON_DIR}/*/*.json | sort -V))
JSON_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$JSON_FILE" .json)

mkdir -p "$LOG_DIR"
echo "Logging to: ${LOG_DIR}/npt_${BASENAME}.log"
exec > "${LOG_DIR}/npt_${BASENAME}.log" 2>&1

echo "Running task ID: $SLURM_ARRAY_TASK_ID"
echo "JSON:   $JSON_FILE"

N_REPLICAS=$(python -c "import json; c=json.load(open('$JSON_FILE')); print(len(c.get('replicas', [None])))")

for i in $(seq 0 $((N_REPLICAS - 1))); do
    echo "--- Replica $i ---"
    python "$SCRIPT" --config "$JSON_FILE" --step npt --replica "$i" --skip_if_done
done
