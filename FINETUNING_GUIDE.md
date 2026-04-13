# Fine-Tuning Guide

Fine-tune the pretrained model on each dataset and evaluate the results.

## Quick Start

### Fine-Tune on a Single Dataset

```bash
# Fine-tune on CWRU (Diagnosis)
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task diagnosis \
    --dataset CWRU \
    --resume checkpoints/final_model_pretrain.pt

# Fine-tune on XJTU-SY (Prognosis)
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task prognosis \
    --dataset XJTU-SY \
    --resume checkpoints/final_model_pretrain.pt
```

### Fine-Tune All Datasets

```bash
./run_finetune_all.sh
```

This will fine-tune on all datasets sequentially and save checkpoints to `checkpoints/finetuned/`.

## What Happens During Fine-Tuning

1. **Loads pretrained model** from `checkpoints/final_model_pretrain.pt`
2. **Uses paper-compliant split**:
   - 80% train → split 90/10 for fine-tuning train/val
   - 20% test → untouched (for final evaluation)
3. **Fine-tunes for 3 epochs** (paper hyperparameters)
4. **Saves checkpoint** to `checkpoints/final_model_{task}.pt`

## Fine-Tuned Model Checkpoints

After fine-tuning, checkpoints are saved:
- **Diagnosis**: `checkpoints/final_model_diagnosis.pt`
- **Prognosis**: `checkpoints/final_model_prognosis.pt`

Or use the batch script which saves to `checkpoints/finetuned/{DATASET}_final_model_{TASK}.pt`.

## Evaluation After Fine-Tuning

After fine-tuning, you can evaluate on the test set:

```bash
# Evaluate fine-tuned model (will be added in next step)
# For now, check training logs for validation metrics during fine-tuning
```

## Datasets to Fine-Tune

### Diagnosis Datasets (Fault Classification):
- **CWRU** - Bearing fault classification
- **JNUB** - Bearing fault classification  
- **KAUG17** - Gear fault classification
- **HSG18** - Gear fault classification

### Prognosis Datasets (RUL Prediction):
- **XJTU-SY** - Bearing remaining useful life prediction

## Next Steps

After fine-tuning:
1. Evaluate each fine-tuned model on its test set
2. Compare results across datasets
3. Collect metrics (accuracy for diagnosis, MAE/MSE for prognosis)
