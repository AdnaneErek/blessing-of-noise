#!/bin/bash
# Supervised pretraining: Train RmGPT (backbone + head) on robot simulation data

# Default checkpoint (optional, for resuming)
RESUME_CHECKPOINT=${1:-""}

# Config file
CONFIG="configs/pretrain_robot_sim_supervised.yaml"

echo "=========================================="
echo "Supervised Pretraining on Robot Simulation Data"
echo "=========================================="
echo "Config: $CONFIG"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
    python pretrain_robot_sim_supervised.py --config "$CONFIG" --resume "$RESUME_CHECKPOINT"
else
    echo "Starting from scratch"
    python pretrain_robot_sim_supervised.py --config "$CONFIG"
fi
