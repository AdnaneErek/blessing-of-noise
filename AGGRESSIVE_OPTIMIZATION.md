# Aggressive Optimization Strategy for 90%+ Accuracy

## Current Status
- **Previous Best**: 77.60% (Epoch 200, head_lr=1e-3)
- **Target**: 90%+
- **Gap**: 12.4%

## Aggressive Changes Implemented

### 1. Very High Head Learning Rate ✅
- **Previous**: `head_lr: 1.0e-3`
- **New**: `head_lr: 5.0e-3` (**5x increase!**)
- **Ratio**: 333x backbone LR (5e-3 vs 1.5e-5)
- **Rationale**: Head needs to learn VERY fast to reach 90%

### 2. Label Smoothing ✅
- **Value**: `0.1`
- **Benefit**: Prevents overconfidence, improves generalization
- **Effect**: Softens hard labels, helps model learn better decision boundaries

### 3. Focal Loss ✅
- **Enabled**: `use_focal_loss: true`
- **Alpha**: `0.25` (class weighting)
- **Gamma**: `2.0` (focusing parameter)
- **Benefit**: 
  - Handles class imbalance better
  - Focuses learning on hard examples
  - Reduces impact of easy examples

### 4. Full 200 Epochs ✅
- **Epochs**: 200 (full training)
- **Rationale**: With aggressive LR, model needs time to converge

## Configuration: `configs/finetune_aggressive.yaml`

```yaml
training:
  lr: 1.5e-5  # Backbone (low, preserve features)
  head_lr: 5.0e-3  # Head (VERY HIGH - 5x previous!)
  label_smoothing: 0.1
  use_focal_loss: true
  focal_alpha: 0.25
  focal_gamma: 2.0
  finetune_epochs: 200
```

## Expected Improvements

### Why This Should Work

1. **5x Higher Head LR**: Head learns 5x faster, should reach higher accuracy quicker
2. **Label Smoothing**: Prevents overfitting, improves generalization to test set
3. **Focal Loss**: Better handles hard examples and class imbalance
4. **Combined Effect**: All three work together for maximum learning

### Expected Accuracy Progression

- **Epochs 1-50**: Very rapid learning → 60-70%
- **Epochs 51-100**: Continued rapid improvement → 75-85%
- **Epochs 101-150**: Fine-tuning → 85-90%
- **Epochs 151-200**: Final convergence → **90%+** 🎯

## Risk Assessment

### Potential Issues

1. **Instability**: Very high LR (5e-3) might cause training instability
   - **Mitigation**: Gradient clipping (max_grad_norm=1.0) should help
   - **Monitor**: Watch for NaN losses or exploding gradients

2. **Overfitting**: Aggressive learning might overfit
   - **Mitigation**: Label smoothing (0.1) helps prevent this
   - **Monitor**: Check validation accuracy

3. **Too Fast Learning**: Model might overshoot optimal solution
   - **Mitigation**: 200 epochs allows time to settle
   - **Monitor**: If accuracy plateaus early, might need LR decay

## Monitoring During Training

Watch for:
- ✅ **Training loss**: Should decrease rapidly, then stabilize
- ✅ **Training accuracy**: Should increase steadily to 90%+
- ✅ **No NaN/Inf**: Check for numerical stability
- ✅ **Gradient norms**: Should stay reasonable (< 10)
- ⚠️ **Validation accuracy**: Should track training (no large gap)

## If 90% Still Not Reached

### Next Aggressive Steps

1. **Even Higher Head LR**: Try `1.0e-2` (10x previous)
2. **Learning Rate Schedule for Head**: Start high, decay over time
3. **Deeper Head**: 4-5 layers instead of 3
4. **Data Augmentation**: Add noise, time shifts, frequency domain
5. **Ensemble**: Train 3-5 models, average predictions

## Running the Training

```bash
# Option 1: Use the script
./finetune_aggressive.sh

# Option 2: Direct command
python train_rmgpt.py \
    --config configs/finetune_aggressive.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt
```

## Success Criteria

- ✅ Training loss decreases rapidly then stabilizes
- ✅ Training accuracy reaches **90%+** by epoch 200
- ✅ No numerical instability (NaN/Inf)
- ✅ Validation accuracy tracks training (no overfitting)
- ✅ Test accuracy ≥ 90%

## Summary

This is the **most aggressive configuration** we've tried:
- **5x higher head LR** (5e-3 vs 1e-3)
- **Label smoothing** (0.1) for generalization
- **Focal loss** for better class handling
- **200 epochs** for full convergence

This should push us from **77.60% → 90%+**! 🚀
