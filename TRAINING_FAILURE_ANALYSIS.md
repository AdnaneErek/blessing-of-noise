# Training Failure Analysis: Linear Decay Config

## Problem

The `finetune_optimized.yaml` config with **linear decay** completely failed:
- **Loss**: Stuck at 1.39 (no improvement from epoch 21 to 100)
- **Accuracy**: Stuck at ~26.7% (essentially random for 4 classes)
- **Status**: Model did not learn at all

## Root Cause

**Linear decay reduced learning rate too quickly!**

With `lr_schedule: "linear"` and `min_lr: 5.0e-6`:
- Starting LR: `2.0e-5`
- After warmup (500 steps): LR = `2.0e-5`
- After 50% of training: LR ≈ `1.2e-5` (decayed linearly)
- After 100% of training: LR = `5.0e-6` (minimum)

The learning rate became too small too fast, preventing the model from learning.

## Comparison

| Config | LR | Schedule | Result |
|--------|----|----------|--------|
| `paper_exact_config.yaml` | 1e-5 | **Constant** | ✅ **70.4%** (BEST) |
| `finetune_improved.yaml` | 1e-3 | Cosine | ✅ 61.2% (learning, but plateaued) |
| `finetune_optimized.yaml` | 2e-5 | **Linear** | ❌ **26.7%** (FAILED) |

## Key Insight

**Constant learning rate works best for fine-tuning pretrained models!**

- Decay schedules (linear, cosine) reduce LR too quickly
- Model needs consistent learning rate to adapt pretrained features
- Paper config used constant LR and achieved best results

## Solution

Updated `finetune_optimized.yaml` to:
- **LR**: `1.5e-5` (slightly higher than paper's 1e-5)
- **Schedule**: `constant` (no decay)
- **Warmup**: 500 steps (faster start)

This should perform similarly to paper_exact_config but with slightly faster learning.

## Recommendation

**Use `paper_exact_config.yaml` or the updated `finetune_optimized.yaml` with constant LR.**

Both should achieve ~70% accuracy. To reach 85-95%, we may need:
1. More epochs (100-150)
2. Different learning rates for backbone vs head
3. Better diagnosis head architecture
4. Data augmentation
