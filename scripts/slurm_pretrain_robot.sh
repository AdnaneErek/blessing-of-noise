#!/bin/bash
#SBATCH --job-name=pretrain_robot
#SBATCH --output=logs/pretrain_robot_%j.out
#SBATCH --error=logs/pretrain_robot_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclude=ruche-gpu13

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

# Pretrain MOMENT
# $PYTHON_EXEC -u scripts/pretrain_robot_moment.py --epochs 40 --batch_size 32

# Pretrain MOMENT (Supervised)
$PYTHON_EXEC -u scripts/pretrain_moment_supervised.py --config pretrain_robot_sim_supervised_v2.yaml
