#!/bin/bash
#SBATCH --job-name=eval_robot
#SBATCH --output=logs/eval_robot_%j.out
#SBATCH --error=logs/eval_robot_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge
source ~/.bashrc

PYTHON_EXEC=/gpfs/workdir/fernandeda/conda_envs/research/bin/python
PROJECT_ROOT=/gpfs/workdir/fernandeda/projects/moment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "[DEBUG SLURM] Hostname: $(hostname)"
echo "[DEBUG SLURM] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

$PYTHON_EXEC -u scripts/evaluate_robot_moment_slicing.py \
    --data_dir data/raw/dataset/testDatasets/20241016 \
    --window_size 512 \
    --batch_size 16
