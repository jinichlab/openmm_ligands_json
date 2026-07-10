#!/bin/bash
#SBATCH --job-name=check_npt
#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --output=slurm_logs/check_npt_%a.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

JSON_DIR="/data/ralmadamonter/openmm_scripts_json/tests/jsons"
SCRIPT="/data/ralmadamonter/openmm_scripts_json/scripts_running/step_runner.py"
LOG_DIR="slurm_logs"

FILES=($(ls ${JSON_DIR}/*/*.json | sort -V))
JSON_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$JSON_FILE" .json)

mkdir -p "$LOG_DIR"
echo "Logging to: ${LOG_DIR}/check_npt_${BASENAME}.log"
exec > "${LOG_DIR}/check_npt_${BASENAME}.log" 2>&1

echo "Running task ID: $SLURM_ARRAY_TASK_ID"
echo "JSON:   $JSON_FILE"

python "$SCRIPT" --config "$JSON_FILE" --step check_npt --skip_if_done

echo ""
echo "Review the convergence plot before submitting 07_production.sh"
