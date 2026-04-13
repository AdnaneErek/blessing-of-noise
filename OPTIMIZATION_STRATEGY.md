# Optimization Strategy: From 70% to 85-95%

## Current Status

**Best Result**: `paper_exact_config.yaml` → **70.36% accuracy**
- Loss: 0.6916
- Much better than `finetune_improved.yaml` (61.21%)

## Key Insights

### What Worked Better (paper_exact_config)
1. **Lower Learning Rate**: `1.0e-5` vs `1.0e-3` (100x difference!)
   - Lower LR preserves pretrained features better
   - Allows gradual adaptation without destroying pretrained knowledge

2. **Lower Dropout**: `0.1` vs `0.2`
   - Less regularization = more capacity to learn
   - Model needs all its capacity for fine-tuning

3. **Constant LR Schedule**: No decay vs cosine decay
   - Constant LR keeps learning throughout
   - Cosine decay may have been too aggressive

4. **Longer Warmup**: `1000` steps vs `100`
   - More stable gradient initialization

## Strategy to Reach 85-95%

### Option 1: Incremental Improvements (Recommended)
Start from `paper_exact_config.yaml` and make small tweaks:

```yaml
training:
  lr: 2.0e-5  # Slightly higher (2x) than paper
  warmup_steps: 500  # Faster start but still stable
  finetune_epochs: 100  # More epochs
  lr_schedule: "linear"  # Slow linear decay
  min_lr: 5.0e-6  # Very low minimum
```

### Option 2: Different LR for Backbone vs Head
- **Backbone (pretrained)**: `1.0e-5` (very low, preserve features)
- **Head (new)**: `1.0e-4` (higher, learn faster)

This allows the head to learn quickly while keeping backbone stable.

### Option 3: Learning Rate Restart
- Train 80 epochs with `1.0e-5` (current best)
- Restart from epoch 60 checkpoint with `2.0e-5`
- Train 20-30 more epochs

### Option 4: Evaluate on Test Set First
Before optimizing further, check:
- **Train accuracy**: 70.36%
- **Test accuracy**: ??? (might be different)
- **Overfitting?**: If test << train, need more regularization
- **Underfitting?**: If test ≈ train, need more capacity/epochs

## Recommended Next Steps

1. **Evaluate current model on test set**:
   ```bash
   python evaluate_finetuned.py \
       --checkpoint checkpoints/final_model_diagnosis.pt \
       --dataset CWRU \
       --task diagnosis
   ```

2. **Try Option 1** (incremental improvements):
   ```bash
   python train_rmgpt.py \
       --config configs/finetune_optimized.yaml \
       --task diagnosis \
       --dataset CWRU \
       --resume checkpoints/final_model_pretrain.pt
   ```

3. **If still not enough, try Option 2** (different LRs):
   - Implement separate optimizers for backbone and head
   - Backbone: 1e-5, Head: 1e-4

4. **Monitor for overfitting**:
   - If train accuracy >> test accuracy, reduce LR or increase dropout
   - If both are similar and low, increase epochs or LR slightly

## Expected Results

With incremental improvements:
- **Target**: 80-85% within 100 epochs
- **Stretch goal**: 85-95% with learning rate restart or different LRs

## Why Lower LR Works Better

For fine-tuning pretrained models:
- **High LR** (1e-3): Destroys pretrained features, model has to relearn
- **Low LR** (1e-5): Preserves pretrained features, allows gradual adaptation
- **Sweet spot**: 1e-5 to 5e-5 for fine-tuning

The paper's very low LR (3e-7) is for pretraining, but for fine-tuning we need slightly higher (1e-5 to 1e-4).
