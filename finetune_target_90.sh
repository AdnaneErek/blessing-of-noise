#!/bin/bash
# Fine-tune pretrained model targeting 90%+ accuracy
# Uses different learning rates for backbone vs head

PRETRAINED_CKPT="checkpoints/final_model_pretrain.pt"
CONFIG="configs/finetune_target_90.yaml"
DATASET="CWRU"
TASK="diagnosis"

echo "=========================================="
echo "Fine-tuning for 90%+ Accuracy"
echo "=========================================="
echo "Config: $CONFIG"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "Dataset: $DATASET"
echo "Task: $TASK"
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
echo "Fine-tuning completed!"
echo "=========================================="
