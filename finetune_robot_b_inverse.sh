#!/bin/bash
# Fine-tune in inverse setup:
# Train on Robot-B from testDatasets/20241016
# Evaluate separately on Robot-A finetuningDatasets via evaluate script

CONFIG="configs/finetune_robot_b_inverse.yaml"
CHECKPOINT="${1:-checkpoints/pretrain_robot_sim_supervised_epoch_30.pt}"
RESUME="${2:-}"

echo "=========================================="
echo "Fine-tuning RmGPT: Robot-B Inverse Setup"
echo "=========================================="
echo "Config: $CONFIG"
echo "Base checkpoint: $CHECKPOINT"
if [ -n "$RESUME" ]; then
  echo "Resuming from: $RESUME"
fi
echo "Train source: testDatasets/20241016 (Robot-B few-shot)"
echo "Eval target: finetuningDatasets/* (Robot-A)"
echo ""

if [ -n "$RESUME" ]; then
  python finetune_robot_b_inverse.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --resume "$RESUME" \
    2>&1 | tee -a "logs/robot_b_inverse/finetune_robot_b_inverse.log"
else
  python finetune_robot_b_inverse.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    2>&1 | tee "logs/robot_b_inverse/finetune_robot_b_inverse.log"
fi
