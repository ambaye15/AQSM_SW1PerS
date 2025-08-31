#!/bin/bash
#SBATCH --job-name=getFeatures
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=5:00:00
#SBATCH --mem=60G
#SBATCH --partition=short
#SBATCH --array=0-26  

module load anaconda3/2022.05
source activate pytorch_env

# Get all bottom-level folders inside data/Study*
folder_list=($(find data/Study* -mindepth 1 -maxdepth 1 -type d | sort))

# Bounds check
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#folder_list[@]}" ]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of bounds"
    exit 1
fi

# Get the folder for this task
FOLDER="${folder_list[$SLURM_ARRAY_TASK_ID]}"
echo "Processing folder: $FOLDER"

python video_pose_sw1pers.py --input "$FOLDER" 
