#!/bin/bash
# Fine-tuning RmGPT on Robot Dataset - FROM SCRATCH
# No pretrained weights - train from scratch since domain mismatch is too severe

CONFIG="configs/finetune_robot_from_scratch.yaml"
DATASET="ROBOT"
TASK="diagnosis"
RESUME="${1:-}"  # Optional: pass checkpoint path as first argument

echo "=========================================="
echo "Fine-tuning RmGPT on Robot Dataset - FROM SCRATCH"
echo "=========================================="
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "Task: $TASK"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
else
    echo "Starting from scratch"
fi
echo ""
echo "TRAINING FROM SCRATCH:"
echo "  - No pretrained checkpoint (domain mismatch too severe)"
echo "  - Higher LR: 1.0e-4 (all parameters)"
echo "  - Label Smoothing: 0.1"
echo "  - Focal Loss: Enabled (alpha=0.25, gamma=2.0)"
echo "  - Epochs: 100 (initial test)"
echo "  - Improved Diagnosis Head: Enabled"
echo ""
echo "FEATURES:"
echo "  - DesiredTrajectory xyz (3)"
echo "  - RealizedTrajectory xyz (3)"
echo "  - Error xyz (e_x, e_y, e_z) (3)"
echo "  - Total: 9 features"
echo ""

if [ -n "$RESUME" ]; then
    echo "Resuming training from checkpoint..."
    python train_rmgpt.py --config "$CONFIG" --task "$TASK" --dataset "$DATASET" --resume "$RESUME" 2>&1 | tee -a "logs/finetune_robot_from_scratch.log"
else
    echo "Starting training from scratch..."
    python train_rmgpt.py --config "$CONFIG" --task "$TASK" --dataset "$DATASET" 2>&1 | tee "logs/finetune_robot_from_scratch.log"
fi
