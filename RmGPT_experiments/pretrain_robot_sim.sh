#!/bin/bash
# Pretrain RmGPT on robot simulation data using next-token prediction

# Default config
CONFIG="configs/pretrain_robot_sim.yaml"
RESUME=""

# Parse arguments
if [ $# -ge 1 ]; then
    RESUME="--resume $1"
fi
if [ $# -ge 2 ]; then
    CONFIG="$2"
fi

echo "=========================================="
echo "Pretraining RmGPT on Robot Simulation Data"
echo "=========================================="
echo "Config: $CONFIG"
if [ -n "$RESUME" ]; then
    echo "Resume: $RESUME"
fi
echo ""

# Run pretraining
python pretrain_robot_sim.py \
    --config "$CONFIG" \
    $RESUME \
    2>&1 | tee -a logs/pretrain_robot_sim.log

echo ""
echo "Pretraining complete!"
echo "Best model: checkpoints/pretrain_robot_sim_best.pt"
echo "Final model: checkpoints/pretrain_robot_sim_final.pt"
