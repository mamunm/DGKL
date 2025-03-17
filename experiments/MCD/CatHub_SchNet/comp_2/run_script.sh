#!/bin/bash
#SBATCH --job-name=mcd_cat
#SBATCH --output=mcd_cat.out
#SBATCH --error=mcd_cat.err
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48GB

module purge
module load cuda/12.2

# Print job start details
echo "Starting job $SLURM_JOB_ID at $(date) on $(hostname)"
echo "Home directory is $HOME"

# Define scratch directory
MYTMP=/tmp/$USER/$SLURM_JOB_ID
mkdir -p $MYTMP || { echo "Error: Failed to create scratch directory"; exit 1; }
echo "Scratch directory created: $MYTMP"

# Check if script exists
SCRIPT_NAME="script.py"
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: $SCRIPT_NAME not found"
    exit 1
fi

# Copy input files to scratch directory
echo "Copying input files to scratch directory..."
cp -rp $SLURM_SUBMIT_DIR/* $MYTMP || { echo "Error: Failed to copy input files"; exit 1; }
echo "Input files copied: $(ls -lh $MYTMP)"

# Navigate to the scratch directory
cd $MYTMP || { echo "Error: Failed to change directory to $MYTMP"; exit 1; }

# Define Python interpreter from virtual environment
PYTHON="/home/fs01/om235/CatUncertainty/.venv/bin/python"

# Run the training script
echo "Starting MCD training..."
source /home/fs01/om235/CatUncertainty/.venv/bin/activate
echo "Python interpreter: $(which python)"
$PYTHON $SCRIPT_NAME || { echo "Error: Training script failed"; exit 1; }

# Copy all files back to the submission directory
echo "Copying all files back to submission directory..."
rsync -av --exclude="ensemble_training_*.out" --exclude="ensemble_training_*.err" \
    $MYTMP/ $SLURM_SUBMIT_DIR || { echo "Error: Failed to copy files back"; exit 1; }
echo "Files copied back to: $SLURM_SUBMIT_DIR"

# Clean up scratch directory
echo "Cleaning up scratch directory..."
rm -rf $MYTMP || { echo "Error: Failed to remove scratch directory"; exit 1; }
echo "Scratch directory removed."

# Print job completion status
echo "Training completed successfully."
echo "Job finished on $(date)"

