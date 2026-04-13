# Running Training in Background on Ruche Cluster

Based on the [Ruche training guide](mesocentre_training.pdf), here's how to run your training jobs so they continue even after you disconnect.

## ✅ Recommended: Using SLURM `sbatch` (Best for Ruche)

**This is the proper way to run jobs on HPC clusters like Ruche.**

### Quick Start

```bash
# Submit the job
sbatch run_pretrain_sbatch.sh

# Monitor your job
squeue -u $USER

# Check output (wait for job to start)
tail -f logs/pretrain_*.out

# Cancel if needed
scancel <JOB_ID>
```

### How It Works

When you submit a job with `sbatch`:
- ✅ Job runs independently of your SSH connection
- ✅ Continues even if you close your laptop
- ✅ Properly scheduled by SLURM (respects cluster resources)
- ✅ Output saved to log files automatically
- ✅ Can monitor progress anytime by reconnecting

### Checking Job Status

```bash
# List your jobs
squeue -u $USER

# View detailed job info
scontrol show job <JOB_ID>

# Check job accounting (after completion)
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### Viewing Output

The job outputs are automatically saved to:
- **Standard output**: `logs/pretrain_<JOB_ID>.out`
- **Error output**: `logs/pretrain_<JOB_ID>.err`

```bash
# Watch output in real-time
tail -f logs/pretrain_*.out

# View last 50 lines
tail -n 50 logs/pretrain_*.out
```

---

## Alternative: Using `nohup` (Simple, but Less Ideal)

If you want a simpler solution without SLURM:

```bash
./run_pretrain_nohup.sh
```

Or manually:
```bash
nohup python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt \
    > pretrain_output.log 2>&1 &
```

**Note:** `nohup` runs on the front-end node, which may have resource limits. `sbatch` is preferred for GPU jobs.

---

## Alternative: Using `screen` or `tmux`

These allow you to reattach and see live output:

### Screen
```bash
screen -S pretrain
# Inside screen, run your command
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain --resume checkpoints/checkpoint_epoch_10.pt
# Press Ctrl+A, then D to detach

# Later, reattach:
screen -r pretrain
```

### Tmux
```bash
tmux new -s pretrain
# Inside tmux, run your command
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain --resume checkpoints/checkpoint_epoch_10.pt
# Press Ctrl+B, then D to detach

# Later, reattach:
tmux attach -t pretrain
```

---

## Recommendation for Ruche

**Use `sbatch`** - it's the proper way for HPC clusters:

1. ✅ Runs on compute nodes (not front-end)
2. ✅ Proper resource management
3. ✅ Continues after disconnect
4. ✅ Automatic logging
5. ✅ Respects cluster policies

### Steps:

1. **Edit the script** (if needed):
   ```bash
   # Check available partitions
   sinfo
   
   # Update partition in run_pretrain_sbatch.sh if needed
   ```

2. **Submit the job**:
   ```bash
   sbatch run_pretrain_sbatch.sh
   ```

3. **Monitor**:
   ```bash
   squeue -u $USER
   tail -f logs/pretrain_*.out
   ```

4. **You can now disconnect** - the job will continue!

---

## Additional Resources

- **Ruche Documentation**: https://mesocentre.pages.centralesupelec.fr/user_doc/
- **Support Email**: ruche.support@universite-paris-saclay.fr
- **Check partitions**: `sinfo`
- **Check job limits**: `sacctmgr show assoc user=$USER`
