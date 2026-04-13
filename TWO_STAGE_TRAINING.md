# Two-Stage Training Approach

## Overview

This document describes the two-stage training approach for RmGPT on the robot dataset:

1. **Stage 1: Pretraining** - Self-supervised next-token prediction on simulation data
2. **Stage 2: Head-Only Fine-tuning** - Supervised classification head training on real data (frozen backbone)

## Motivation

The goal is to:
- Learn general signal patterns from large simulation datasets (self-supervised)
- Adapt the classification head to real-world data without catastrophic forgetting
- Preserve the learned representations in the backbone

## Stage 1: Pretraining on Simulation Data

### Objective
Self-supervised learning via next-token prediction. The model learns to predict the next signal token embedding from previous tokens.

### Script
```bash
bash pretrain_robot_sim.sh
```

### Configuration
- **Config**: `configs/pretrain_robot_sim.yaml`
- **Data**: Robot simulation data from `trainingDatasets/`
- **Task**: Next-token prediction (self-supervised)
- **Output**: `checkpoints/pretrain_robot_sim_final.pt`

### Key Settings
- `pretrain_epochs: 50`
- `lr: 1.0e-4` (standard LR for pretraining)
- `lr_schedule: "cosine"` (with warmup)
- No noise augmentation (clean signals)
- No classification head (only backbone)

## Stage 2: Head-Only Fine-tuning on Real Data

### Objective
Train only the classification head on real robot data while keeping the backbone frozen.

### Script
```bash
bash finetune_robot_head_only.sh checkpoints/pretrain_robot_sim_final.pt
```

### Configuration
- **Config**: `configs/finetune_robot_head_only.yaml`
- **Data**: Real robot data from `finetuningDatasets/`
- **Task**: Classification (supervised)
- **Output**: `checkpoints/final_model_finetune_real.pt`

### Key Settings
- `freeze_backbone: true` (CRITICAL)
- `finetune_epochs: 50`
- `lr: 1.0e-4` (for head only)
- `lr_schedule: "cosine"`
- Only diagnosis head parameters are trainable

## Workflow

```bash
# Step 1: Pretrain on simulation data
bash pretrain_robot_sim.sh

# Step 2: Fine-tune head only on real data
bash finetune_robot_head_only.sh checkpoints/pretrain_robot_sim_final.pt

# Step 3: Evaluate
bash evaluate_robot.sh checkpoints/final_model_finetune_real.pt configs/finetune_robot_head_only.yaml
```

## Advantages

1. **No Catastrophic Forgetting**: Backbone remains frozen, preserving simulation-learned features
2. **Efficient Training**: Only head parameters are updated (much faster)
3. **Better Generalization**: Backbone learns general signal patterns, head adapts to real data
4. **Small Real Dataset**: Works well even with limited real data (190 samples)

## Files Created

- `pretrain_robot_sim.py` - Pretraining script
- `pretrain_robot_sim.sh` - Launch script for pretraining
- `configs/pretrain_robot_sim.yaml` - Pretraining configuration
- `configs/finetune_robot_head_only.yaml` - Head-only fine-tuning configuration
- `finetune_robot_head_only.sh` - Launch script for head-only fine-tuning

## Notes

- The pretrained checkpoint does NOT include a diagnosis head (it's added during fine-tuning)
- The backbone learns general signal representations via next-token prediction
- The head learns task-specific classification on real data
- This approach is similar to transfer learning in vision (ImageNet pretraining → task-specific head)
