# Hyperparameter Updates for Better Accuracy

## Problem
- **Accuracy too low**: Only 42% after 30 epochs (target: 85-95%)
- **Loss decreasing slowly**: 1.375 → 1.325 over 10 epochs
- **Need more training**: Model needs more epochs to converge

## Changes Made

### 1. Increased Epochs: **30 → 80**
- More time for the model to learn
- Should allow loss to drop below 0.5
- Target accuracy: 85-95%

### 2. Increased Learning Rate: **5e-4 → 1e-3** (2x)
- Faster learning
- Better gradient updates
- Should see faster loss reduction

### 3. Adjusted Learning Rate Schedule
- **Min LR**: 1e-6 → **1e-5** (10x higher)
- Keeps learning rate higher for longer
- Slower decay = more learning

### 4. Reduced Regularization
- **Weight decay**: 0.05 → **0.01**
- Less regularization = more capacity to learn
- Can always increase later if overfitting

### 5. Increased Warmup: **50 → 100 steps**
- More stable start
- Better gradient initialization

## New Configuration

```yaml
training:
  lr: 1.0e-3  # 2x higher than before
  weight_decay: 0.01  # Less regularization
  warmup_steps: 100
  finetune_epochs: 80  # 2.7x more epochs
  lr_schedule: "cosine"
  min_lr: 1.0e-5  # Higher minimum
```

## Expected Results

With these changes, you should see:
- **Faster loss reduction**: Should drop below 0.5 within 20-30 epochs
- **Higher accuracy**: Should reach 80-95% by epoch 50-60
- **Better convergence**: Model will have more time to learn
- **Learning rate**: Will start at 1e-3, decay smoothly to 1e-5

## Usage

```bash
python train_rmgpt.py \
    --config configs/finetune_improved.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt
```

## Monitoring

Watch for:
- **Loss**: Should drop below 0.5 by epoch 30
- **Accuracy**: Should reach 80%+ by epoch 50
- **Learning rate**: Check progress bar - should decay smoothly
- **Overfitting**: If val loss increases while train loss decreases, reduce epochs or increase weight_decay

## If Still Not Converging

1. **Try even higher LR**: `2.0e-3` or `3.0e-3`
2. **Different LR for head vs backbone**:
   - Backbone: `1.0e-4`
   - Head: `1.0e-3`
3. **Reduce batch size**: Try 128 for more gradient updates per epoch
4. **Check data**: Ensure labels are correct and balanced
