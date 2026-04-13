#!/bin/bash
#SBATCH --job-name=pretrain_deep
#SBATCH --output=logs/pretrain_deep_%j.out
#SBATCH --error=logs/pretrain_deep_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge
source ~/.bashrc

PYTHON_EXEC=/gpfs/workdir/fernandeda/conda_envs/research/bin/python
PROJECT_ROOT=/gpfs/workdir/fernandeda/projects/moment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "[DEBUG SLURM] Hostname: $(hostname)"

# Pretrain MOMENT using smaller batch size to simulate the old 0.15 loss length
$PYTHON_EXEC -u pretrain_moment_sim.py --config pretrain_moment_sim_deep.yaml
