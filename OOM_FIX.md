# Fixing Out-of-Memory (OOM) Error

## Problem

Your job was killed with:
```
oom-kill event(s) in step 11244770.batch cgroup
```

This happened because loading multiple large datasets (CWRU, JNUB, KAUG17, HSG18, XJTU-SY) simultaneously exceeded the 16GB memory limit.

## Solution

I've increased the memory allocation in `run_pretrain_sbatch.sh`:
- **Before**: `--mem=16G`
- **After**: `--mem=128G`

This should be enough for loading all 5 datasets during pretraining.

## Resubmit the Job

```bash
sbatch run_pretrain_sbatch.sh
```

The new job will have 128GB of RAM, which should be sufficient for:
- Loading all 5 datasets simultaneously
- Batch processing with batch_size=256
- DataLoader with num_workers=4

## Why 128GB?

- **5 datasets** being loaded simultaneously
- **Large time series** (especially XJTU-SY with very long sequences)
- **Batch processing** with batch_size=256
- **Multiple DataLoader workers** (num_workers=4)

128GB should provide enough headroom. If it still fails, we can:
1. Reduce `num_workers` in config (less parallel data loading)
2. Further increase memory allocation
3. Load datasets sequentially instead of simultaneously

## Check After Resubmitting

```bash
# Monitor the new job
squeue -u $USER

# Watch output
tail -f logs/pretrain_*.out
```

If you still get OOM errors, we may need to optimize the data loading process.
