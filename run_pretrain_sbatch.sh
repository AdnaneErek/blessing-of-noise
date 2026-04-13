#!/bin/bash
#SBATCH --job-name=rmgpt_pretrain
#SBATCH --partition=gpua100  # A100 GPU partition (allows 24 hours)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G  # Increased memory for loading multiple large datasets
#SBATCH --time=1-00:00:00  # 1 day (format: D-HH:MM:SS)
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

# Change to project directory
cd /gpfs/workdir/erekrakead/RmGPT

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment if needed (adjust path/name as needed)
# Uncomment and modify if using conda/virtualenv:
# source activate moment_env
# or
# module load python/3.x  # Load Python module if needed

# Run pretraining (resume from checkpoint)
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt

echo "Training complete at $(date)"
