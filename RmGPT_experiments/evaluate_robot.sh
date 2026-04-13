#!/bin/bash
# Evaluate fine-tuned RmGPT model on robot dataset
# Evaluates on both simulation test data and real robot test data

# Default to fine-tuned model, but allow override
CHECKPOINT="${1:-checkpoints/final_model_finetune_real.pt}"
CONFIG="${2:-configs/finetune_robot_from_scratch.yaml}"
TEST_FOLDER="${3:-20241016}"  # Real robot test data folder

echo "=========================================="
echo "Evaluating Robot Dataset Model"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Test Folder: $TEST_FOLDER"
echo ""
echo "This will evaluate on:"
echo "  1. Simulation test data (from training split - all folders)"
echo "  2. Real robot test data (from testDatasets/$TEST_FOLDER)"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -1 checkpoints/final_model*.pt checkpoints/best_finetune*.pt 2>/dev/null | head -5
    echo ""
    echo "Usage:"
    echo "  bash evaluate_robot.sh [checkpoint] [config] [test_folder]"
    echo ""
    echo "Examples:"
    echo "  bash evaluate_robot.sh checkpoints/final_model_finetune_real.pt"
    echo "  bash evaluate_robot.sh checkpoints/best_finetune_real_epoch_3.pt"
    echo "  bash evaluate_robot.sh checkpoints/final_model_diagnosis.pt configs/finetune_robot_from_scratch.yaml"
    exit 1
fi

python evaluate_robot.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --test_folder "$TEST_FOLDER" \
    2>&1 | tee "logs/eval_robot_$(basename $CHECKPOINT .pt).log"

echo ""
echo "Evaluation complete! Check logs/eval_robot_$(basename $CHECKPOINT .pt).log for details."
