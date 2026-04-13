#!/bin/bash
# Evaluation script with linear probing (for accuracy)

python evaluate_pretrained.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt \
    --linear-probe

echo "Evaluation complete! Check results/ directory for outputs with accuracy metrics."
