#!/bin/bash
#SBATCH --job-name=pretrain_rmgpt
#SBATCH --output=logs/pretrain_rmgpt_%j.out
#SBATCH --error=logs/pretrain_rmgpt_%j.err
#SBATCH --time=4:00:00
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
echo "[DEBUG SLURM] Which python: $(which python)"
echo "[DEBUG SLURM] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

$PYTHON_EXEC -u - << 'EOF'
import os, torch
print("[DEBUG PY] torch file:", torch.__file__)
print("[DEBUG PY] torch version:", torch.__version__)
print("[DEBUG PY] cuda available:", torch.cuda.is_available())
print("[DEBUG PY] cuda device count:", torch.cuda.device_count())
print("[DEBUG PY] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
EOF

# Pretrain using the friend's script and config
echo "[INFO] Running RmGPT pretraining pipeline..."
$PYTHON_EXEC -u pretrain_robot_sim.py --config pretrain_robot_sim.yaml
