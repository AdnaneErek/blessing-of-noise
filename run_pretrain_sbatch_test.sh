#!/bin/bash
#SBATCH --job-name=rmgpt_pretrain_test
#SBATCH --partition=gpu_test  # Test partition (1 hour limit - good for quick tests)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00  # 1 hour maximum for gpu_test
#SBATCH --output=logs/pretrain_test_%j.out
#SBATCH --error=logs/pretrain_test_%j.err

# Change to project directory
cd /gpfs/workdir/erekrakead/RmGPT

# Create logs directory if it doesn't exist
mkdir -p logs

# Run pretraining (resume from checkpoint)
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt

echo "Training complete at $(date)"
