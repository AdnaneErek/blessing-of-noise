#!/bin/bash
# Quick script to evaluate pretrained model

python evaluate_pretrained.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt \
    --linear-probe

echo "Evaluation complete! Check results/ directory for outputs."
