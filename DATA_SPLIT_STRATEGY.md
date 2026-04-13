# Paper-Compliant Data Split Strategy

## Overview

The RmGPT paper uses a **strict data split strategy** to prevent data leakage. This implementation now follows the **exact methodology** from the paper.

## Paper's Split Strategy

### Step 1: Initial Split (80/20)
- **80% Train**: Used for pretraining and finetuning
- **20% Test**: **NEVER** used in pretraining or finetuning (only for final evaluation)

### Step 2: Within 80% Train
- **Pretraining**: Uses **ALL 80%** train data (unlabeled, labels ignored)
- **Finetuning**: Splits the 80% train again:
  - **90% of train80** → Finetuning train
  - **10% of train80** → Finetuning validation

## Effective Splits

```
Total Dataset (100%)
├── Train80 (80%) ← Used for pretraining + finetuning
│   ├── Pretrain: ALL 80% (unlabeled)
│   ├── Finetune Train: 90% of train80 (72% of total)
│   └── Finetune Val: 10% of train80 (8% of total)
└── Test20 (20%) ← UNTOUCHED until final evaluation
```

## Implementation

### New Function: `paper_split_strategy()`
Located in `data/split_strategy.py`, this function:
1. Performs 80/20 split (stratified for classification)
2. Splits train80 into finetune train/val (90/10)
3. Returns all splits in a dictionary

### Updated Training Script
`train_rmgpt.py` now:
- Uses `paper_split_from_phmd()` to load and split data
- For **pretraining**: uses all 80% train (unlabeled)
- For **finetuning**: uses finetune_train and finetune_val
- **Test set is preserved** for final evaluation only

## Configuration

In `configs/default_config.yaml` or `configs/paper_exact_config.yaml`:

```yaml
data:
  test_size: 0.2  # 20% test (paper requirement)
  finetune_val_size: 0.1  # 10% of train80 for validation
  random_state: 42  # For reproducibility
```

## Usage

The split strategy is **automatically applied** when you run:

```bash
# Pretraining (uses all 80% train, unlabeled)
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain

# Finetuning (uses 90% of train80 for train, 10% for val)
python train_rmgpt.py --config configs/paper_exact_config.yaml --task diagnosis
```

## Key Benefits

1. ✅ **No Data Leakage**: Test set never seen during training
2. ✅ **Paper Compliant**: Matches exact methodology from paper
3. ✅ **Reproducible**: Fixed random seed ensures consistent splits
4. ✅ **Stratified**: Maintains class distribution in splits (for classification)

## Important Notes

- **Test set is completely separate**: Never loaded during pretraining or finetuning
- **Pretraining is unlabeled**: Uses all 80% train but ignores labels
- **Finetuning uses labels**: Uses 90/10 split of train80 with labels
- **Final evaluation**: Test set should only be used at the very end

This ensures your results are comparable to the paper and prevents any data leakage issues!
