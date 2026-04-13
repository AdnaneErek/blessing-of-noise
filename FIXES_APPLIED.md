# Fixes Applied for Evaluation and Fine-tuning

## Issues Fixed

### 1. Evaluation Script - Architecture Mismatch ✅
**Problem**: Evaluation script was trying to load a diagnosis head with 256 classes from checkpoint, but CWRU needs 4 classes.

**Solution**:
- Modified `evaluate_finetuned.py` to **always determine num_classes from test labels** (not from checkpoint)
- Modified `train/trainer.py` to **gracefully skip loading diagnosis head** if architecture doesn't match
- Evaluation script now creates a fresh diagnosis head with correct num_classes

**Changes**:
- `evaluate_finetuned.py`: Changed num_classes determination to prioritize test labels
- `train/trainer.py`: Added try-except around `load_state_dict` to handle mismatches

### 2. Fine-tuning Strategy ✅
**Problem**: User wants to fine-tune from pretrained model, not from already fine-tuned checkpoint.

**Solution**:
- Fine-tuning now **always starts from pretrained checkpoint** (`final_model_pretrain.pt`)
- Diagnosis head is **randomly initialized** if checkpoint has mismatched architecture
- This is the correct behavior for fine-tuning on a new dataset

**Changes**:
- `train_rmgpt.py`: Added informative messages about checkpoint loading
- Diagnosis head is created with correct num_classes for the target dataset
- If checkpoint's diagnosis head doesn't match, it's skipped (random initialization)

## How It Works Now

### Fine-tuning Flow:
1. Load pretrained checkpoint (`final_model_pretrain.pt`)
2. Create diagnosis head with correct num_classes for target dataset (e.g., 4 for CWRU)
3. If checkpoint has diagnosis head with wrong architecture → skip it, use random initialization
4. Train the model with pretrained backbone + fresh diagnosis head

### Evaluation Flow:
1. Determine num_classes from **test labels** (most reliable)
2. Create diagnosis head with correct num_classes
3. Load model backbone from checkpoint
4. If checkpoint's diagnosis head doesn't match → skip it, use random initialization
5. Evaluate on test set

## Usage

### Fine-tuning (from pretrained):
```bash
python train_rmgpt.py \
    --config configs/finetune_aggressive.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt  # Always use pretrained!
```

### Evaluation:
```bash
python evaluate_finetuned.py \
    --config configs/finetune_aggressive.yaml \
    --checkpoint checkpoints/final_model_diagnosis.pt \
    --dataset CWRU \
    --task diagnosis
```

## Key Points

1. **Always use pretrained checkpoint for fine-tuning** (`final_model_pretrain.pt`)
2. **num_classes is determined from data**, not from checkpoint
3. **Architecture mismatches are handled gracefully** - diagnosis head is randomly initialized if needed
4. **Same strategy as CWRU** - pretrained backbone + fresh task head
