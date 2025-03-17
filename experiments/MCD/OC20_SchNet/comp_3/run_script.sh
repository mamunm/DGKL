#!/bin/bash
#SBATCH --job-name=mcd_eagle_67057
#SBATCH --output=mcd_eagle_67057.out
#SBATCH --error=mcd_eagle_67057.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=1
#SBATCH --mem=48GB
#SBATCH --partition=gpu-shared
#SBATCH --account=mit197

module purge
module load gpu/0.15.4
module load cuda11.7/toolkit/11.7.1

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Check if script exists
if [ ! -f "script.py" ]; then
    echo "Error: script.py not found"
    exit 1
fi

# Run the training script
echo "Starting Evidential training..."
poetry run python script.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Error: Training failed."
    exit 1
fi

echo "Job finished on $(date)"
