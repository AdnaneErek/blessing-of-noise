#!/bin/bash
# Evaluate pretrained model on all datasets and create accuracy summary

python evaluate_all_datasets.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt \
    --output results/pretrained_accuracy_summary.json

echo ""
echo "Check results/pretrained_accuracy_summary.json for detailed results"
echo "Check results/pretrained_accuracy_summary.txt for CSV format"
