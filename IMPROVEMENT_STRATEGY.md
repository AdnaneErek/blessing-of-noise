# Improvement Strategy: From 73% to 85-95%

## Current Status
- **Accuracy**: 73.26% (Epoch 100)
- **Loss**: 0.6321
- **Target**: 85-95%
- **Gap**: ~12-22%

## Strategies to Improve

### Option 1: More Epochs (Easiest)
**Current**: 100 epochs
**Try**: 150-200 epochs

The model is still learning (loss decreasing), so more epochs should help.

```bash
# Use finetune_advanced.yaml with 150 epochs
python train_rmgpt.py \
    --config configs/finetune_advanced.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt
```

### Option 2: Better Diagnosis Head (Recommended)
**Current head**: Simple 2-layer (embed_dim → embed_dim//2 → num_classes)
**Improvement**: Deeper/wider head with more capacity

Current architecture:
```python
embed_dim (512) → embed_dim//2 (256) → num_classes (4)
```

Improved architecture:
```python
embed_dim (512) → embed_dim (512) → embed_dim//2 (256) → num_classes (4)
```

This gives the head more capacity to learn complex decision boundaries.

### Option 3: Different Learning Rates
- **Backbone (pretrained)**: `1.0e-5` (very low, preserve features)
- **Head (new)**: `1.0e-4` (higher, learn faster)

This allows the head to learn quickly while keeping backbone stable.

### Option 4: Learning Rate Restart
1. Load checkpoint from epoch 80-100
2. Restart training with LR = `2.0e-5` (slightly higher)
3. Train for 20-30 more epochs

This can help escape local minima.

### Option 5: Ensemble Multiple Tokens
Currently using only TF token. Could also use:
- Prompt tokens (average)
- Signal tokens (pooled)
- Combined features

## Recommended Approach

**Step 1**: Try more epochs first (easiest)
```bash
python train_rmgpt.py \
    --config configs/finetune_advanced.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt
```

**Step 2**: If still not enough, improve diagnosis head architecture

**Step 3**: If still not enough, try different LRs for backbone vs head

## Expected Results

- **150 epochs**: Should reach 75-80%
- **200 epochs**: Should reach 80-85%
- **With better head**: Should reach 85-90%
- **With different LRs**: Should reach 90-95%
