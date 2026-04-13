# Fine-Tuning Strategy: Full Supervised Fine-Tuning (Not K-Shot)

## Summary

**We did NOT use k-shot training.** Instead, we used **full supervised fine-tuning** with all available training data from the dataset split.

## Fine-Tuning Strategy

### Approach: Full Supervised Fine-Tuning

1. **Data Usage**: Uses **90% of the train80 split** (which equals **72% of the total dataset**)
2. **All Samples**: All samples in the training split are used (no subset limitation)
3. **Epochs**: 3 epochs of fine-tuning (as per paper)
4. **Additional**: 5 epochs of prompt learning (task-specific adaptation)

### Data Split Breakdown

```
Total Dataset (100%)
├── Train80 (80%) ← Used for pretraining + finetuning
│   ├── Pretrain: ALL 80% (unlabeled, for pretraining)
│   ├── Finetune Train: 90% of train80 = 72% of total ✅ USED FOR FINE-TUNING
│   └── Finetune Val: 10% of train80 = 8% of total (validation)
└── Test20 (20%) ← UNTOUCHED until final evaluation
```

### Implementation Details

**Code Location**: `train_rmgpt.py` lines 120-127

```python
# Finetuning: use finetune_train and finetune_val
train_signals, train_labels, train_rul = splits['finetune_train']
val_signals, val_labels, val_rul = splits['finetune_val']

print(f"Finetune train samples: {len(train_signals)} (90% of train80)")
print(f"Finetune val samples: {len(val_signals)} (10% of train80)")
```

**Training Configuration**:
- **Epochs**: 3 (from `config['training']['finetune_epochs']`)
- **Batch Size**: 256
- **Learning Rate**: 3.00 × 10⁻⁷
- **All samples in train split are used** (no k-shot limitation)

### Why Not K-Shot?

K-shot learning (few-shot learning) would involve:
- Using only **k samples per class** (e.g., 1-shot, 5-shot, 10-shot)
- Training with a very limited subset of data
- Typically used to test model's ability to learn from minimal data

**Our approach**:
- Uses **all available training data** (72% of total dataset)
- Standard supervised learning with full dataset
- More similar to "full fine-tuning" or "standard fine-tuning"

### Comparison: K-Shot vs. Our Strategy

| Aspect | K-Shot Training | Our Strategy |
|--------|----------------|--------------|
| **Data Usage** | Limited (k samples per class) | Full training split (72% of total) |
| **Purpose** | Test few-shot capability | Standard supervised fine-tuning |
| **Typical k values** | 1, 5, 10, 20 samples per class | All samples in split |
| **Our approach** | ❌ Not used | ✅ Full fine-tuning |

### Example: CWRU Dataset

For CWRU diagnosis (4 classes):
- **K-shot (if used)**: Might use 5 samples per class = 20 total samples
- **Our approach**: Uses **all samples in the 72% training split** (could be thousands of samples)

### Training Process

1. **Load pretrained model** from checkpoint
2. **Load fine-tuning data**: 90% of train80 (72% of total)
3. **Create task head**: Diagnosis head (classification) or Prognosis head (regression)
4. **Fine-tune for 3 epochs**: 
   - Update both model and task head
   - Use all training samples in each epoch
5. **Prompt learning (optional)**: Additional 5 epochs for prompt token adaptation

### Code Evidence

**No k-shot limitation in code**:
- No sampling or subset selection
- All samples in `train_signals` are used
- DataLoader processes all batches without filtering

```python
train_loader = DataLoader(
    train_dataset,  # Contains ALL samples in finetune_train split
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=config['hardware']['num_workers'],
    pin_memory=config['hardware']['pin_memory']
)
```

### Paper Reference

The paper (arXiv:2409.17604v2) uses:
- **3 epochs** for fine-tuning (matches our implementation)
- **Full dataset** for fine-tuning (not k-shot)
- Standard supervised learning approach

## Conclusion

**Fine-tuning strategy**: **Full supervised fine-tuning** (not k-shot)
- Uses all available training data (72% of total dataset)
- 3 epochs of fine-tuning
- 5 epochs of prompt learning
- Standard supervised learning approach

This is the standard approach for fine-tuning foundation models, where the pretrained model is adapted to a specific task using all available labeled data.
