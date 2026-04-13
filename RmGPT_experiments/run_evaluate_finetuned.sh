#!/bin/bash
# Evaluate fine-tuned model on test set
# Usage: ./run_evaluate_finetuned.sh <CHECKPOINT> [DATASET] [TASK]

CHECKPOINT=${1:-"checkpoints/final_model_diagnosis.pt"}
DATASET=${2:-"CWRU"}
TASK=${3:-"diagnosis"}

CONFIG="configs/paper_exact_config.yaml"

echo "Evaluating fine-tuned model"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "Task: $TASK"
echo ""

python evaluate_finetuned.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --dataset $DATASET \
    --task $TASK

echo ""
echo "Evaluation complete! Check results/ directory for outputs."
