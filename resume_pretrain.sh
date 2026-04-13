#!/bin/bash
# Resume pretraining from checkpoint
# 
# OPTION 1: Continue from epoch 10 (RECOMMENDED)
# This will re-train epochs 10-20, ensuring consistency
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt

# OPTION 2: Jump to epoch 19 (NOT RECOMMENDED - loses progress from 11-18)
# Uncomment below to skip epochs 11-18:
# python train_rmgpt.py \
#     --config configs/paper_exact_config.yaml \
#     --task pretrain \
#     --resume checkpoints/checkpoint_epoch_10.pt \
#     --start-epoch 19
