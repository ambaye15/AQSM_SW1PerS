#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=video_processing
#SBATCH --array=0-27
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1

module load anaconda3/2022.05
source activate pytorch_env

FILENAME=$(ls Vids/*.avi | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Processing file: $FILENAME"

python process_video_pose.py --input "$FILENAME"
