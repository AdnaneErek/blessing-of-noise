# Evaluating Pretrained RmGPT Model

This guide explains how to evaluate your pretrained model before fine-tuning.

## What Gets Evaluated

The evaluation script measures:

1. **Next-Token Prediction Loss**: The core pretraining objective - how well the model predicts the next token in a sequence
2. **Feature Quality (Optional)**: Linear probing - trains a simple classifier on frozen pretrained features to assess representation quality

## Quick Start

### Basic Evaluation (Pretraining Loss Only)

```bash
python evaluate_pretrained.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt
```

Or use the quick script:
```bash
./run_evaluation.sh
```

### Full Evaluation (Including Linear Probing)

```bash
python evaluate_pretrained.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt \
    --linear-probe
```

## Command Options

- `--config`: Path to configuration file (e.g., `configs/paper_exact_config.yaml`)
- `--checkpoint`: Path to pretrained model checkpoint (e.g., `checkpoints/final_model_pretrain.pt`)
- `--linear-probe`: Include linear probing evaluation (requires labels, only works for diagnosis datasets)
- `--dataset`: Optional dataset name (defaults to config file value)

## What Happens

1. **Loads Test Data**: Uses the 20% test set that was **never used** during pretraining
2. **Evaluates Pretraining Loss**: Computes next-token prediction MSE loss on test set
3. **Optional Linear Probing**: If `--linear-probe` is used:
   - Freezes pretrained model
   - Trains a simple linear classifier on top
   - Evaluates on validation and test sets
   - This measures how well the pretrained features can be used for classification

## Output

Results are saved to `results/eval_pretrained_<DATASET>.json`:

```json
{
  "checkpoint": "checkpoints/final_model_pretrain.pt",
  "dataset": "CWRU",
  "test_samples": 1234,
  "pretrain_metrics": {
    "pretrain_loss": 0.1253,
    "pretrain_loss_std": 0.0012
  },
  "linear_probe_metrics": {
    "val_accuracy": 0.85,
    "test_accuracy": 0.82,
    "test_f1": 0.81
  }
}
```

## Understanding the Results

### Pretraining Loss

- **Lower is better** - indicates better next-token prediction
- Compares to validation loss from training logs
- If test loss is much higher than training loss → possible overfitting
- If test loss is similar → good generalization

### Linear Probe Accuracy

- **Higher is better** - indicates better learned features
- Measures how well pretrained features can be used for classification
- Good pretrained features → high linear probe accuracy with minimal training
- If linear probe accuracy is high → model learned useful representations

## Examples

### Evaluate on CWRU Dataset

```bash
python evaluate_pretrained.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt \
    --dataset CWRU \
    --linear-probe
```

### Evaluate on Different Dataset

```bash
python evaluate_pretrained.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt \
    --dataset JNUB \
    --linear-probe
```

### Quick Loss Check Only

```bash
python evaluate_pretrained.py \
    --config configs/paper_exact_config.yaml \
    --checkpoint checkpoints/final_model_pretrain.pt
```

## Next Steps After Evaluation

Once you've evaluated the pretrained model:

1. **Check Pretraining Loss**: Should be similar to training loss
2. **Check Linear Probe**: Good features → high accuracy
3. **If results look good** → Proceed to fine-tuning
4. **If results are poor** → May need more pretraining or hyperparameter tuning

## Notes

- **Test Set**: The 20% test set was **never used** during pretraining, so this is a fair evaluation
- **Linear Probing**: Only works for diagnosis datasets (requires labels)
- **Memory**: Evaluation is lighter than training, but still uses GPU if available
- **Time**: Pretraining loss evaluation is fast (~minutes). Linear probing adds ~5-10 minutes.
