#!/bin/bash
# Fine-tune pretrained model on a single dataset
# Usage: ./finetune_single.sh <DATASET> [TASK]

DATASET=${1:-"CWRU"}
TASK=${2:-"diagnosis"}  # diagnosis or prognosis

PRETRAINED_CKPT="checkpoints/final_model_pretrain.pt"
CONFIG="configs/finetune_improved.yaml"  # Use improved config with better hyperparameters

echo "Fine-tuning on dataset: $DATASET"
echo "Task: $TASK"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo ""

python train_rmgpt.py \
    --config $CONFIG \
    --task $TASK \
    --dataset $DATASET \
    --resume $PRETRAINED_CKPT

echo ""
echo "Fine-tuning complete!"
echo "Check checkpoint: checkpoints/final_model_${TASK}.pt"
