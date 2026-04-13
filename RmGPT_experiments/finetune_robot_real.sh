#!/bin/bash
# Fine-tune RmGPT on Real Robot Data
# Fine-tunes a model trained on simulation data using real robot data from finetuningDatasets

CONFIG="configs/finetune_robot_real.yaml"
CHECKPOINT="${1:-checkpoints/final_model_diagnosis.pt}"  # Default to simulation-trained model
RESUME="${2:-}"  # Optional: pass fine-tuning checkpoint path as second argument

echo "=========================================="
echo "Fine-tuning RmGPT on Real Robot Data"
echo "=========================================="
echo "Config: $CONFIG"
echo "Simulation Checkpoint: $CHECKPOINT"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
else
    echo "Starting fine-tuning from simulation checkpoint"
fi
echo ""
echo "FINE-TUNING STRATEGY:"
echo "  - Load model trained on simulation data"
echo "  - Fine-tune on real robot data (finetuningDatasets/)"
echo "  - Backbone LR: 1.0e-5 (low, preserve sim features)"
echo "  - Head LR: 5.0e-4 (high, adapt to real data)"
echo "  - Epochs: 20 (monitor for overfitting)"
echo "  - Batch Size: 32 (limited real data: ~190 samples)"
echo "  - Label Smoothing: 0.1"
echo "  - Focal Loss: Enabled (alpha=0.25, gamma=2.0)"
echo ""
echo "REAL DATA:"
echo "  - Location: data/raw/dataset/finetuningDatasets/"
echo "  - Auto-discovers all folders"
echo "  - Expected: ~190 samples (30 Healthy, 20 per other class)"
echo "  - Train/Val split: 90/10"
echo ""

if [ -n "$RESUME" ]; then
    echo "Resuming fine-tuning from checkpoint..."
    python finetune_robot_real.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --resume "$RESUME" \
        2>&1 | tee -a "logs/finetune_robot_real.log"
else
    echo "Starting fine-tuning from simulation checkpoint..."
    python finetune_robot_real.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        2>&1 | tee "logs/finetune_robot_real.log"
fi
