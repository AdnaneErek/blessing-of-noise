# Running Training in Background

Since you're connected to a remote host, your training will stop if you disconnect unless you run it in the background. Here are the options:

## Option 1: Using `nohup` (Simplest)

**Run:**
```bash
./run_pretrain_nohup.sh
```

**Or manually:**
```bash
nohup python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt \
    > pretrain_output.log 2>&1 &
```

**Monitor:**
```bash
tail -f pretrain_output.log        # Watch output in real-time
ps aux | grep train_rmgpt         # Check if process is running
```

**Stop:**
```bash
kill <PID>  # Use PID from ps aux | grep train_rmgpt
```

✅ **Pros**: Simple, output goes to log file  
❌ **Cons**: Can't interact with the process

---

## Option 2: Using `screen` (Can Reattach)

**Run:**
```bash
./run_pretrain_screen.sh
```

**Or manually:**
```bash
screen -S pretrain
# Inside screen, run:
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain --resume checkpoints/checkpoint_epoch_10.pt
# Press Ctrl+A, then D to detach
```

**Reattach later:**
```bash
screen -r pretrain
```

**List sessions:**
```bash
screen -ls
```

✅ **Pros**: Can reattach and see live output  
❌ **Cons**: Need screen installed

---

## Option 3: Using `tmux` (Modern Alternative)

**Run:**
```bash
./run_pretrain_tmux.sh
```

**Or manually:**
```bash
tmux new -s pretrain
# Inside tmux, run:
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain --resume checkpoints/checkpoint_epoch_10.pt
# Press Ctrl+B, then D to detach
```

**Reattach:**
```bash
tmux attach -t pretrain
```

✅ **Pros**: Modern, powerful, can reattach  
❌ **Cons**: Need tmux installed

---

## Option 4: Using SLURM `sbatch` (Recommended for HPC)

Since you're on a cluster (ruche-gpu11), use `sbatch`:

**Run:**
```bash
sbatch run_pretrain_sbatch.sh
```

**Monitor:**
```bash
squeue -u $USER          # Check job status
tail -f logs/pretrain_*.out  # Watch output
```

**Cancel:**
```bash
scancel <JOB_ID>
```

✅ **Pros**: Proper job scheduling, automatic resource management  
❌ **Cons**: Need SLURM access

---

## Recommendation

For your setup, I recommend **Option 4 (sbatch)** since you're on a cluster. But if you want something simpler, use **Option 1 (nohup)**.

## Quick Start

**Using nohup (simplest):**
```bash
./run_pretrain_nohup.sh
```

**Using sbatch (best for clusters):**
```bash
sbatch run_pretrain_sbatch.sh
```

Both will continue running even after you disconnect!
