#!/bin/bash
#SBATCH --job-name=finetune_new_pretrain
#SBATCH --output=logs/finetune_new_%j.out
#SBATCH --error=logs/finetune_new_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge
source ~/.bashrc

# Use direct python path to bypass conda activate issues
PYTHON_EXEC=/gpfs/workdir/fernandeda/conda_envs/research/bin/python
PROJECT_ROOT=/gpfs/workdir/fernandeda/projects/moment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "[DEBUG SLURM] Hostname: $(hostname)"

# Finetune MOMENT using the newly generated pretraining weights
$PYTHON_EXEC -u scripts/finetune_robot_moment_dropout.py \
    --pretrained_path checkpoints/pretrain_moment_sim_best.pt \
    --epochs 200 \
    --lr 1e-3 \
    --patience 30 \
    --batch_size 16 \
    --window_size 512 \
    --head_dropout 0.4
