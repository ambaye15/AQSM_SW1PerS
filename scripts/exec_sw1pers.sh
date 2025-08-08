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

FILENAME=$(ls Vids/*.avi | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Processing file: $FILENAME"

python feature_processing.py --input "$FILENAME" 
