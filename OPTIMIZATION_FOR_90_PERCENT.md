# Optimization Strategy for 90%+ Accuracy

## Current Status
- **Current Accuracy**: 73.26% (Epoch 100)
- **Target Accuracy**: 90%+
- **Gap**: ~17-20%

## Key Changes Implemented

### 1. Different Learning Rates for Backbone vs Head ✅

**Problem**: The pretrained backbone needs low LR to preserve learned features, while the new DiagnosisHead needs high LR to learn quickly.

**Solution**: Implemented separate parameter groups with different learning rates:
- **Backbone (pretrained)**: `1.5e-5` (low, preserves features)
- **Head (new)**: `1.0e-3` (high, learns fast)

**Implementation**:
- Modified `train/trainer.py` to accept `head_lr` and `head_params`
- Created separate parameter groups in optimizer
- Both groups use the same LR schedule (constant), but with different base LRs

### 2. Improved Diagnosis Head ✅

**Problem**: The original 2-layer head may not have enough capacity.

**Solution**: Using the improved 3-layer head:
- Original: `512 → 256 → 4`
- Improved: `512 → 512 → 256 → 4`

**Benefits**:
- More capacity to learn complex decision boundaries
- Better feature transformation

### 3. More Training Epochs ✅

**Problem**: 100 epochs may not be enough for convergence.

**Solution**: Increased to **200 epochs** in `configs/finetune_target_90.yaml`

**Rationale**:
- With different LRs, the head learns faster but needs more time to fully converge
- More epochs allow the model to fine-tune the decision boundaries

### 4. Constant Learning Rate Schedule ✅

**Problem**: Cosine/linear decay was too aggressive and reduced learning too quickly.

**Solution**: Using **constant LR** (proven to work best in previous experiments)

**Evidence**: 
- `finetune_optimized.yaml` with constant LR: 73.26%
- `finetune_optimized.yaml` with linear decay: 26.65% (failed)

## Configuration: `configs/finetune_target_90.yaml`

```yaml
training:
  lr: 1.5e-5  # Backbone (pretrained)
  head_lr: 1.0e-3  # Head (new) - 66x higher!
  lr_schedule: "constant"
  finetune_epochs: 200  # More epochs
  warmup_steps: 500
```

## Expected Improvements

### Why This Should Work

1. **Head Learns Fast**: High LR (1e-3) allows the head to quickly learn the classification mapping
2. **Backbone Preserves Features**: Low LR (1.5e-5) keeps pretrained features intact
3. **More Capacity**: Improved head has more layers to learn complex patterns
4. **More Time**: 200 epochs gives plenty of time for convergence

### Expected Accuracy Progression

- **Epochs 1-50**: Rapid learning (head adapting) → 75-80%
- **Epochs 51-100**: Continued improvement → 80-85%
- **Epochs 101-150**: Fine-tuning → 85-90%
- **Epochs 151-200**: Final convergence → 90%+

## Monitoring

Watch for:
- **Training loss**: Should decrease steadily
- **Training accuracy**: Should increase steadily
- **Validation accuracy**: Should track training accuracy
- **Learning rates**: Both should remain constant (no decay)

## If 90% Not Reached

### Additional Strategies (if needed)

1. **Label Smoothing**: Add label smoothing to loss function (prevents overconfidence)
2. **Data Augmentation**: Add noise, time shifts, or frequency domain augmentation
3. **Ensemble**: Train multiple models and ensemble predictions
4. **Focal Loss**: Use focal loss instead of cross-entropy (handles class imbalance)
5. **Different Optimizer**: Try Adam with different betas or SGD with momentum
6. **Learning Rate Finder**: Use LR range test to find optimal LRs

## Files Modified

1. `train/trainer.py`: Added support for different LRs per parameter group
2. `train_rmgpt.py`: Pass head parameters and head_lr to trainer
3. `configs/finetune_target_90.yaml`: New optimized configuration
4. `finetune_target_90.sh`: Training script

## Running the Training

```bash
# Option 1: Use the script
./finetune_target_90.sh

# Option 2: Direct command
python train_rmgpt.py \
    --config configs/finetune_target_90.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt
```

## Evaluation

After training completes, evaluate with:

```bash
python evaluate_finetuned.py \
    --checkpoint checkpoints/finetuned/CWRU_final_model_diagnosis.pt \
    --dataset CWRU \
    --task diagnosis
```

## Success Criteria

- ✅ Training loss decreases steadily
- ✅ Training accuracy increases steadily
- ✅ Final accuracy ≥ 90%
- ✅ No overfitting (train/val accuracy close)
