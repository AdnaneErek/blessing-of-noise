# Robot Dataset Training Issue Analysis

## Problem
After 200 epochs of fine-tuning on the robot dataset, the model accuracy is stuck at **~10-17%** (random guessing for 9 classes = 11.1%). The model is **not learning**.

## Training Results
- **Final Train Accuracy**: 16.37% (epoch 200)
- **Loss**: Barely decreased from 0.4344 to 0.4270
- **Learning Rate Display**: Shows 0.0000 (likely display issue - backbone LR is 1.5e-5)

## Configuration
- **Backbone LR**: 1.5e-5 (pretrained weights)
- **Head LR**: 5.0e-3 (diagnosis head)
- **Channel Projection LR**: 5.0e-3 (9 channels -> 2 channels)
- **Epochs**: 200
- **Batch Size**: 256
- **Warmup Steps**: 500
- **Label Smoothing**: 0.1
- **Focal Loss**: Enabled (alpha=0.25, gamma=2.0)

## Potential Issues

### 1. Channel Projection Dimensionality Reduction
- **Input**: 9 channels (DesiredTraj xyz, RealizedTraj xyz, Error xyz)
- **Output**: 2 channels (pretrained model expects 2)
- **Issue**: Reducing from 9 to 2 channels might be losing critical information
- **Solution**: Consider using a different approach (e.g., learnable projection with more output channels, or adapt the pretrained model)

### 2. Learning Rate Display Issue
- LR shows as 0.0000 in progress bar
- This is likely a display rounding issue (1.5e-5 rounds to 0.0000)
- **Fix Applied**: Changed LR display to show max LR (head LR = 5.0e-3) instead of first parameter group

### 3. Gradient Flow
- Need to verify gradients are flowing through:
  - Channel projection layer
  - Model backbone
  - Diagnosis head
- **Action**: Add gradient checking/debugging

### 4. Data Normalization
- Data is normalized using StandardScaler
- Need to verify normalization is correct for robot dataset
- **Action**: Check normalized data statistics

### 5. Model Architecture Mismatch
- Pretrained model was trained on rotating machinery data (vibration signals)
- Robot dataset is trajectory data (position, velocity, error)
- Domain mismatch might require different approach
- **Action**: Consider domain adaptation techniques

## Next Steps

1. **Fix LR Display**: ✅ Done - Now shows max LR (head LR)
2. **Add Gradient Debugging**: Check if gradients are flowing
3. **Verify Data**: Check normalized data statistics and distribution
4. **Consider Architecture Changes**:
   - Increase channel projection output (e.g., 9 -> 4 or 9 -> 8)
   - Use separate projection for different feature groups
   - Consider fine-tuning the pretrained model's signal tokenizer

5. **Alternative Approaches**:
   - Train from scratch on robot dataset
   - Use a different pretrained model
   - Apply domain adaptation techniques

## Immediate Actions

1. ✅ **Modified Channel Projection**: Changed from simple Linear(9, 2) to deeper MLP:
   - Architecture: 9 -> 18 -> 9 -> 2
   - Added LayerNorm and GELU activations
   - Should preserve more information during projection
2. Re-run training with improved channel projection
3. Add gradient monitoring to verify learning
4. Check if channel projection weights are updating
5. Evaluate on validation set to see if issue is training-specific

## Channel Projection Improvement

**Before**: Simple linear projection `nn.Linear(9, 2)`
- Direct mapping, might lose information

**After**: Multi-layer projection with normalization
```python
nn.Sequential(
    nn.Linear(9, 18),      # Expand
    nn.LayerNorm(18),
    nn.GELU(),
    nn.Linear(18, 9),      # Intermediate
    nn.LayerNorm(9),
    nn.GELU(),
    nn.Linear(9, 2)        # Compress to target
)
```
- Learns richer representations
- LayerNorm helps with training stability
- GELU provides non-linearity for better feature learning
