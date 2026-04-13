#!/bin/bash
# Evaluate Robot-B inverse checkpoints on:
# 1) Robot-B source folder
# 2) Robot-A finetuning folders

CHECKPOINT="${1:-checkpoints/robot_b_inverse/final_model_robot_b_inverse.pt}"
CONFIG="${2:-configs/finetune_robot_b_inverse.yaml}"

echo "=========================================="
echo "Evaluating Robot-B Inverse Model"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo ""

python evaluate_robot_b_inverse.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  2>&1 | tee "logs/robot_b_inverse/eval_robot_b_inverse_$(basename $CHECKPOINT .pt).log"
