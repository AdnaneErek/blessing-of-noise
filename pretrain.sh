#!/bin/bash
# Pretraining script for RmGPT
# Uses exact paper configuration

# Activate your environment (adjust if needed)
# source activate moment_env  # or however you activate moment_env

# Run pretraining with paper exact config
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain

# The script will:
# 1. Load CWRU dataset with paper-compliant 80/20 split
# 2. Use 80% train for pretraining (unlabeled)
# 3. Use 10% of train80 for validation
# 4. Train for 20 epochs with batch size 256, lr 3.0e-7
# 5. Save checkpoints to checkpoints/
