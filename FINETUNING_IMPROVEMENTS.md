# Fine-Tuning Improvements

## Problem Analysis

The initial fine-tuning attempt resulted in **worse performance**:
- **Before fine-tuning**: 98.61% sim, 61.11% real
- **After fine-tuning**: 76.67% sim, 53.33% real ❌

### Root Causes:
1. **Catastrophic Forgetting**: Model forgot simulation patterns
2. **Overfitting**: 190 real samples too small for reliable fine-tuning
3. **Learning Rates Too High**: Rapid adaptation caused forgetting
4. **No Regularization**: Model memorized small real dataset

## Improvements Applied

### 1. **Much Lower Learning Rates** (10-100x reduction)
```yaml
training:
  lr: 1.0e-6          # Backbone: 10x lower (was 1.0e-5)
  head_lr: 1.0e-5    # Head: 50x lower (was 5.0e-4)
```

**Why**: Prevents rapid forgetting of simulation features while allowing gradual adaptation.

### 2. **Cosine Learning Rate Schedule**
```yaml
training:
  lr_schedule: "cosine"  # Gradual decay instead of constant
  min_lr: 1.0e-7
```

**Why**: Gradual adaptation prevents sudden changes that cause forgetting.

### 3. **More Epochs with Decay**
```yaml
training:
  finetune_epochs: 30  # More epochs (was 20) but with cosine decay
```

**Why**: More time for gradual adaptation without overfitting.

### 4. **Freeze Backbone Option**
```yaml
training:
  freeze_backbone: false  # Set to true to only fine-tune head
```

**Why**: Option to preserve all simulation features by only adapting the classification head.

## Usage

### Option 1: Fine-tune with Lower LRs (Recommended First Try)
```bash
bash finetune_robot_real.sh checkpoints/final_model_diagnosis.pt
```

Uses the improved config (`configs/finetune_robot_real.yaml`) with:
- Lower learning rates
- Cosine decay
- 30 epochs

### Option 2: Freeze Backbone (Only Fine-tune Head)
Edit `configs/finetune_robot_real.yaml`:
```yaml
training:
  freeze_backbone: true  # Change to true
```

Then run:
```bash
bash finetune_robot_real.sh checkpoints/final_model_diagnosis.pt
```

This will:
- Keep all simulation features intact
- Only adapt the classification head
- Prevent catastrophic forgetting

### Option 3: Use Alternative Config
```bash
python finetune_robot_real.py \
    --config configs/finetune_robot_real_v2.yaml \
    --checkpoint checkpoints/final_model_diagnosis.pt
```

## Expected Results

### With Lower LRs (Option 1):
- **Simulation test**: Should maintain ~90-95% (small drop acceptable)
- **Real test**: Should improve to ~65-70% (better than 61.11%)
- **Domain gap**: Should decrease

### With Frozen Backbone (Option 2):
- **Simulation test**: Should maintain ~98% (no forgetting)
- **Real test**: May improve slightly (~62-65%) or stay similar
- **Trade-off**: Less adaptation but no forgetting

## Monitoring

Watch for:
1. **Validation accuracy**: Should increase gradually
2. **Training accuracy**: Should not reach 100% too quickly (overfitting)
3. **Simulation test**: Should not drop below 90%

## Next Steps if Still Poor

1. **Collect more real data**: 190 samples is very small
2. **Mix simulation + real data**: Train on both simultaneously
3. **Domain adaptation techniques**: DANN, adversarial training
4. **Feature analysis**: Understand what differs between sim and real

## Files Modified

- `configs/finetune_robot_real.yaml`: Updated with lower LRs and cosine decay
- `configs/finetune_robot_real_v2.yaml`: Alternative config (same improvements)
- `finetune_robot_real.py`: Added freeze_backbone support
