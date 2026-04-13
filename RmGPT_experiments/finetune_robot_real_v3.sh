#!/bin/bash
# Fine-tune RmGPT on Real Robot Data (IMPROVED V3)
# Strategy: Lower LRs + Progressive Unfreezing (50% layers) to minimize forgetting

CONFIG="configs/finetune_robot_real_v3.yaml"
CHECKPOINT="${1:-checkpoints/pretrain_robot_sim_supervised_final.pt}"  # Use supervised pretraining checkpoint
RESUME="${2:-}"  # Optional: pass fine-tuning checkpoint path as second argument

echo "=========================================="
echo "Fine-tuning RmGPT on Real Robot Data (V3 - IMPROVED)"
echo "=========================================="
echo "Config: $CONFIG"
echo "Simulation Checkpoint: $CHECKPOINT"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
else
    echo "Starting fine-tuning from simulation checkpoint"
fi
echo ""
echo "IMPROVEMENTS IN V3:"
echo "  - Lower LRs: Backbone 5.0e-7, Head 5.0e-6 (20x/10x lower)"
echo "  - Progressive Unfreezing: Last 50% layers (2/4) instead of 25% (1/4)"
echo "  - Fewer epochs: 20 instead of 30 (prevent overfitting)"
echo "  - Cosine decay: Gradual adaptation"
echo ""

if [ -n "$RESUME" ]; then
    echo "Resuming fine-tuning from checkpoint..."
    python finetune_robot_real.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --resume "$RESUME" \
        2>&1 | tee -a "logs/finetune_robot_real_v3.log"
else
    echo "Starting fine-tuning from simulation checkpoint..."
    python finetune_robot_real.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        2>&1 | tee "logs/finetune_robot_real_v3.log"
fi
