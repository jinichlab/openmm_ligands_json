#!/bin/bash
#SBATCH --job-name=atlas_configs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/atlas_create_configs.log

# Selected proteins: 10 across a broad sequence length range
# 38, 64, 200, 318, 500, 613, 698, 751, 870, 2128 aa
PDBS=(
    3tvj_I
    5i8j_A
    2bfw_A
    3q0i_A
    1esw_A
    1ikp_A
    3wol_A
    3eh1_A
    2qmj_A
    6sup_A
)

INPUT_DIR="/data/ralmadamonter/atlas_pdbs"
OUTPUT_DIR="/data/ralmadamonter/atlas_md/jsons"
SCRIPT="/data/ralmadamonter/openmm_scripts_json/scripts_running/create_config.py"
LOG_DIR="slurm_logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

for BASENAME in "${PDBS[@]}"; do
    PDB="${INPUT_DIR}/${BASENAME}.pdb"
    CONFIG="${OUTPUT_DIR}/${BASENAME}/${BASENAME}.json"

    if [ ! -f "$PDB" ]; then
        echo "WARNING: PDB not found: $PDB — skipping"
        continue
    fi

    if [ -f "$CONFIG" ]; then
        echo "Skipping $BASENAME — config already exists"
        continue
    fi

    echo "Creating config for $BASENAME"
    python "$SCRIPT" \
        -p "$PDB" \
        -o "$OUTPUT_DIR" \
        --n_replicas 3 \
        --ph 7.4 \
        --remove_heterogens \
        --force_field "charmm36" \
        --force_field_water "charmm36/water" \
        --box_size 1.5 \
        --platform CUDA \
        --device 0 \
        --min_steps 10000 \
        --nvt_steps 200000 \
        --npt_steps 500000 \
        --prod_steps 5000000 \
        --nvt_recorder 500 \
        --npt_recorder 5000 \
        --prod_recorder 5000 \
        --cv_chainid 0 --cv_native_cutoff 0.45 --cv_beta 50.0 --cv_lam 1.8 \
        --cv_min_seq_sep 3 --cv_ca_min_seq_sep 3 \
        --rmsf_selection "protein and name CA" --rmsf_ref_frame 0 \
        --sasa_mode residue --sasa_probe_radius 0.14 --sasa_n_sphere_points 960 \
        --thermo_rolling_window 100 \
        --tica_lag 10 --tica_dim 2 --tica_stride 1 \
        --op_window 5 --op_chain 0 \
        --net_seg_ids A --net_n_windows 4 --net_sampled_frames 10 \
        --net_cutoff 4.5 --net_n_jobs 4
done

echo "Done. JSON files in: $OUTPUT_DIR"
