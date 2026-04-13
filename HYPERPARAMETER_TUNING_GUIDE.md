# Hyperparameter Tuning Guide for Fine-Tuning

## Problem Analysis

Your current training shows:
- **Learning rate**: 0.0000 (scheduler issue)
- **Loss**: Stuck around 1.3-1.4 (decreasing very slowly)
- **Accuracy**: ~25-30% during training, 70% final (not satisfactory)
- **Epochs**: 80 (way too many - should converge faster)

## Root Causes

1. **Learning rate too low**: `1.0e-5` is too low for fine-tuning
   - Paper's `3.00 × 10^-7` is for **pretraining** (self-supervised)
   - Fine-tuning typically needs **10-100x higher** learning rate

2. **No learning rate decay**: Scheduler only does warmup, then stays constant
   - Should decay learning rate over time for better convergence

3. **Too many warmup steps**: 1000 steps is excessive for fine-tuning
   - Fine-tuning needs less warmup (50-100 steps)

## Recommended Hyperparameter Changes

### 1. Learning Rate: **5.0e-4** (50x increase)
- **Current**: `1.0e-5` (too low)
- **Recommended**: `5.0e-4` to `1.0e-3`
- **Rationale**: Fine-tuning needs higher LR to adapt pretrained weights

### 2. Learning Rate Schedule: **Cosine Annealing**
- **Current**: Constant after warmup
- **Recommended**: Cosine annealing with warmup
- **Benefits**: 
  - Starts high, gradually decreases
  - Better convergence
  - Prevents overfitting

### 3. Warmup Steps: **50** (20x reduction)
- **Current**: 1000 steps
- **Recommended**: 50-100 steps
- **Rationale**: Fine-tuning needs minimal warmup

### 4. Epochs: **10** (8x reduction)
- **Current**: 80 epochs
- **Recommended**: 10-15 epochs
- **Rationale**: With proper LR, should converge much faster

## Implementation

### Option 1: Use Improved Config (Recommended)

```bash
python train_rmgpt.py \
    --config configs/finetune_improved.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt
```

### Option 2: Modify Existing Config

Edit `configs/paper_exact_config.yaml`:

```yaml
training:
  lr: 5.0e-4  # Changed from 1.0e-5
  warmup_steps: 50  # Changed from 1000
  finetune_epochs: 10  # Changed from 80
  lr_schedule: "cosine"  # Add this
  min_lr: 1.0e-6  # Add this
```

## Expected Improvements

With these changes, you should see:
- **Faster convergence**: Loss should drop below 0.5 within 5-10 epochs
- **Higher accuracy**: Should reach 85-95% (depending on dataset)
- **Better learning rate**: Should start at 5e-4, decay smoothly
- **Fewer epochs needed**: 10-15 epochs instead of 80

## Learning Rate Schedule Comparison

### Current (Constant):
```
Step 0-1000: LR increases linearly 0 → 1e-5
Step 1000+: LR stays at 1e-5 (constant)
```

### Improved (Cosine):
```
Step 0-50: LR increases linearly 0 → 5e-4
Step 50+: LR decays cosine 5e-4 → 1e-6
```

## Additional Tips

### If Still Not Converging:

1. **Try even higher LR**: `1.0e-3` (1000x paper's pretraining LR)
2. **Different LR for head vs backbone**:
   - Backbone (pretrained): `1.0e-4`
   - Head (new): `1.0e-3`
3. **Reduce batch size**: If memory allows, try 128 for more gradient updates
4. **Check data quality**: Ensure labels are correct and balanced

### If Overfitting:

1. **Increase weight decay**: `0.01` → `0.1`
2. **Add dropout**: Increase from `0.1` to `0.2`
3. **Early stopping**: Stop when val loss stops decreasing
4. **Reduce learning rate**: Try `1.0e-4` instead of `5.0e-4`

## Monitoring

Watch for:
- **Learning rate**: Should start high, decay smoothly
- **Loss**: Should drop quickly in first few epochs
- **Accuracy**: Should improve steadily
- **Gradient norm**: Should be stable (not exploding/vanishing)

## Quick Start

```bash
# Use improved config
python train_rmgpt.py \
    --config configs/finetune_improved.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt
```

This should give you much better results!
