# Data Split Strategy Update - Paper Compliant

## ✅ Implementation Complete

The data splitting strategy has been **updated to match the paper exactly** and **prevent data leakage**.

## Paper's Methodology

### Split Strategy (80/20)
1. **First Split**: 80% train, 20% test
   - Test set (20%) is **NEVER** used in pretraining or finetuning
   - Only used for final evaluation

2. **Within 80% Train**:
   - **Pretraining**: Uses **ALL 80%** train data (unlabeled, labels ignored)
   - **Finetuning**: Splits the 80% train again:
     - 90% of train80 → Finetuning train
     - 10% of train80 → Finetuning validation

## Effective Data Distribution

```
Total Dataset (100 samples)
├── Train80 (80 samples) ← Used for pretraining + finetuning
│   ├── Pretrain: ALL 80 samples (unlabeled)
│   ├── Finetune Train: 72 samples (90% of train80)
│   └── Finetune Val: 8 samples (10% of train80)
└── Test20 (20 samples) ← UNTOUCHED until final evaluation
```

## Implementation Details

### New Files
1. **`data/split_strategy.py`**:
   - `paper_split_strategy()`: Core splitting function
   - `paper_split_from_phmd()`: Loads PHMD datasets and applies paper split

### Updated Files
1. **`train_rmgpt.py`**:
   - Now uses `paper_split_from_phmd()` instead of PHMD's default splits
   - Automatically applies paper-compliant splits based on task type

2. **`configs/default_config.yaml`**:
   - Added `test_size: 0.2` (20% test)
   - Added `finetune_val_size: 0.1` (10% of train80)
   - Added `random_state: 42` for reproducibility
   - Removed old `fold`, `train_split`, `val_split`, `test_split` parameters

3. **`configs/paper_exact_config.yaml`**:
   - Same updates as default config

## Usage

The split strategy is **automatically applied** - no changes needed to your training commands:

```bash
# Pretraining (uses all 80% train, unlabeled)
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain

# Finetuning (uses 90% of train80 for train, 10% for val)
python train_rmgpt.py --config configs/paper_exact_config.yaml --task diagnosis
```

## Key Features

✅ **No Data Leakage**: Test set completely separate  
✅ **Paper Compliant**: Matches exact methodology  
✅ **Stratified Splits**: Maintains class distribution (for classification)  
✅ **Reproducible**: Fixed random seed  
✅ **Automatic**: No manual split management needed

## Verification

Tested split strategy with 100 samples:
- Pretrain: 80 samples (80%) ✅
- Finetune Train: 72 samples (90% of 80) ✅
- Finetune Val: 8 samples (10% of 80) ✅
- Test: 20 samples (20%) ✅

## Important Notes

- **Test set is preserved**: Available in splits but never used during training
- **Pretraining is unlabeled**: Uses all 80% but ignores labels
- **Finetuning uses labels**: Uses 90/10 split of train80 with labels
- **Final evaluation**: Test set should only be evaluated at the very end

This ensures your results are **comparable to the paper** and **free from data leakage**!
