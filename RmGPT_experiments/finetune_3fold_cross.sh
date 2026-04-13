#!/bin/bash
# 3-Fold Cross-Experiment Fine-tuning
# Usage:
#   bash finetune_3fold_cross.sh [checkpoint] [exp_num]
#   bash finetune_3fold_cross.sh                          # run all 3 with default ckpt
#   bash finetune_3fold_cross.sh my_ckpt.pt               # run all 3 with custom ckpt
#   bash finetune_3fold_cross.sh my_ckpt.pt 2             # run only experiment 2

CHECKPOINT="${1:-checkpoints/pretrain_robot_sim_supervised_best.pt}"
EXP="${2:-}"

CMD="python finetune_3fold_cross.py --config configs/finetune_3fold_cross.yaml --checkpoint $CHECKPOINT"

if [ -n "$EXP" ]; then
    CMD="$CMD --exp $EXP"
fi

echo "Running: $CMD"
$CMD
