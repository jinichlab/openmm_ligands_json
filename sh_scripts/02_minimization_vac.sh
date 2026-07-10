#!/bin/bash
#SBATCH --job-name=min_vac
#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --output=slurm_logs/min_vac_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

JSON_DIR="/data/ralmadamonter/openmm_scripts_json/tests/jsons"
SCRIPT="/data/ralmadamonter/openmm_scripts_json/scripts_running/step_runner.py"
LOG_DIR="slurm_logs"

FILES=($(ls ${JSON_DIR}/*/*.json | sort -V))
JSON_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$JSON_FILE" .json)

mkdir -p "$LOG_DIR"
echo "Logging to: ${LOG_DIR}/min_vac_${BASENAME}.log"
exec > "${LOG_DIR}/min_vac_${BASENAME}.log" 2>&1

echo "Running task ID: $SLURM_ARRAY_TASK_ID"
echo "JSON:   $JSON_FILE"

python "$SCRIPT" --config "$JSON_FILE" --step minimization_vac --skip_if_done
