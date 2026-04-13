#!/bin/bash
# AGGRESSIVE Fine-tuning for 90%+ Accuracy
# Uses: Very high head LR (5e-3) + Label Smoothing + Focal Loss

PRETRAINED_CKPT="checkpoints/final_model_pretrain.pt"
CONFIG="configs/finetune_aggressive.yaml"
DATASET="CWRU"
TASK="diagnosis"

echo "=========================================="
echo "AGGRESSIVE Fine-tuning for 90%+ Accuracy"
echo "=========================================="
echo "Config: $CONFIG"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "Dataset: $DATASET"
echo "Task: $TASK"
echo ""
echo "AGGRESSIVE SETTINGS:"
echo "  - Head LR: 5.0e-3 (5x previous, 333x backbone)"
echo "  - Label Smoothing: 0.1"
echo "  - Focal Loss: Enabled (alpha=0.25, gamma=2.0)"
echo "  - Epochs: 200"
echo ""

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    exit 1
fi

# Run fine-tuning
python train_rmgpt.py \
    --config $CONFIG \
    --task $TASK \
    --dataset $DATASET \
    --resume $PRETRAINED_CKPT

echo ""
echo "=========================================="
echo "AGGRESSIVE Fine-tuning completed!"
echo "=========================================="
