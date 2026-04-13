#!/bin/bash
# Train RmGPT on Mixed Data (Simulation + Real)
# Combines simulation data from trainingDatasets with real data from finetuningDatasets
# Real test dataset (testDatasets/) is NOT used in training

CONFIG="configs/train_robot_mixed.yaml"
DATASET="ROBOT_MIXED"
TASK="diagnosis"
RESUME="${1:-}"  # Optional: pass checkpoint path as first argument

echo "=========================================="
echo "Training RmGPT on Mixed Data (Sim + Real)"
echo "=========================================="
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "Task: $TASK"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
else
    echo "Starting training from scratch"
fi
echo ""
echo "MIXED TRAINING STRATEGY:"
echo "  - Simulation data: trainingDatasets/ (all folders)"
echo "  - Real data: finetuningDatasets/ (all 3 folders)"
echo "  - Mix ratio: 80% simulation, 20% real (configurable)"
echo "  - Test set: Simulation only (20% split)"
echo "  - Real test dataset (testDatasets/) is NOT used"
echo ""
echo "TRAINING SETTINGS:"
echo "  - LR: 1.0e-4 (standard for training from scratch)"
echo "  - Epochs: 100"
echo "  - Batch Size: 256"
echo "  - Label Smoothing: 0.1"
echo "  - Focal Loss: Enabled (alpha=0.25, gamma=2.0)"
echo ""

if [ -n "$RESUME" ]; then
    echo "Resuming training from checkpoint..."
    python train_rmgpt.py --config "$CONFIG" --task "$TASK" --dataset "$DATASET" --resume "$RESUME" 2>&1 | tee -a "logs/train_robot_mixed.log"
else
    echo "Starting training from scratch..."
    python train_rmgpt.py --config "$CONFIG" --task "$TASK" --dataset "$DATASET" 2>&1 | tee "logs/train_robot_mixed.log"
fi
