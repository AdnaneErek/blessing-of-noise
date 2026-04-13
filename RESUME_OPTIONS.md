# Resume Training Options

## Situation
- Training was killed at epoch 19/20 (95% complete)
- Last saved checkpoint: `checkpoint_epoch_10.pt` (saves every 10 epochs)
- No checkpoint for epoch 18 or 19 exists

## Options

### Option 1: Continue from epoch 10 (RECOMMENDED)
```bash
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt
```
- **What happens**: Trains epochs 10-20 again (repeating 10-19)
- **Pros**: Consistent training, all weights/optimizer/scheduler in sync
- **Cons**: Duplicates training of epochs 10-19 (takes more time)
- **Result**: Complete pretraining with consistent weights

### Option 2: Jump to epoch 19 from epoch 10 weights
```bash
python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt \
    --start-epoch 19
```
- **What happens**: Uses epoch 10 weights for epochs 19-20
- **Pros**: Completes training quickly (only 2 epochs)
- **Cons**: Loses progress from epochs 11-18, optimizer state may be inconsistent
- **Result**: Training completes but weights are from epoch 10, not epoch 18

### Option 3: Accept 19 epochs and move on
Since you completed 95% of epoch 19, you could consider pretraining essentially done (19.95/20 epochs). The final weights aren't saved, but epochs 19-20 represent minimal training.

## Recommendation
**Use Option 1** - It's more consistent and only takes 10 more epochs (you already did 18). The training is stable so repeating epochs 10-19 shouldn't hurt.
