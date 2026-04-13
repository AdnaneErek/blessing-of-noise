#!/bin/bash
# Fine-tune ONLY the classification head on real robot data (frozen backbone)
# This script loads a pretrained backbone and trains only the head

# Default checkpoint (pretrained on simulation)
PRETRAINED_CHECKPOINT="checkpoints/pretrain_robot_sim_final.pt"
CONFIG="configs/finetune_robot_head_only.yaml"
RESUME=""

# Parse arguments
if [ $# -ge 1 ]; then
    PRETRAINED_CHECKPOINT="$1"
fi
if [ $# -ge 2 ]; then
    CONFIG="$2"
fi
if [ $# -ge 3 ]; then
    RESUME="--resume $3"
fi

echo "=========================================="
echo "Fine-tuning Head Only (Frozen Backbone)"
echo "=========================================="
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Config: $CONFIG"
if [ -n "$RESUME" ]; then
    echo "Resume: $RESUME"
fi
echo ""

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CHECKPOINT"
    echo "Please run pretraining first: bash pretrain_robot_sim.sh"
    exit 1
fi

# Run fine-tuning
python finetune_robot_real.py \
    --config "$CONFIG" \
    --checkpoint "$PRETRAINED_CHECKPOINT" \
    $RESUME \
    2>&1 | tee -a logs/finetune_robot_head_only.log

echo ""
echo "Fine-tuning complete!"
echo "Best model: checkpoints/final_model_finetune_real.pt"
