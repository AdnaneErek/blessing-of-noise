# Monitoring Your Training Job on Ruche

Your job has been submitted with ID: **11244770**

## Quick Monitoring Commands

### Check Job Status
```bash
# Check if your job is running
squeue -u $USER

# Check specific job
squeue -j 11244770

# Get detailed job information
scontrol show job 11244770
```

### Watch Output in Real-Time
```bash
# Watch output as it's written (best for monitoring progress)
tail -f logs/pretrain_11244770.out

# Press Ctrl+C to stop watching (job continues running)
```

### View Output Files
```bash
# View last 50 lines of output
tail -n 50 logs/pretrain_11244770.out

# View all output
cat logs/pretrain_11244770.out

# View errors
cat logs/pretrain_11244770.err
```

### Check Job Progress (After Completion)
```bash
# Get job accounting info (CPU, memory, time used)
sacct -j 11244770 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxRSSNode,TotalCPU

# Get detailed accounting
sacct -j 11244770 -l
```

## Using the Monitoring Script

I've created a quick monitoring script for you:

```bash
# Monitor current job (11244770)
./monitor_job.sh

# Or monitor a different job
./monitor_job.sh <JOB_ID>
```

## Common Commands Summary

```bash
# 1. Check if job is running
squeue -u $USER

# 2. Watch live output (recommended)
tail -f logs/pretrain_11244770.out

# 3. Check job details
scontrol show job 11244770

# 4. Cancel job if needed
scancel 11244770

# 5. Check all your jobs
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

## Understanding Job States

- **PENDING (PD)**: Job is waiting for resources
- **RUNNING (R)**: Job is currently running
- **COMPLETED (CD)**: Job finished successfully
- **FAILED (F)**: Job failed
- **CANCELLED (CA)**: Job was cancelled

## Monitoring Tips

1. **Real-time monitoring**: Use `tail -f logs/pretrain_11244770.out` to see progress as it happens
2. **Check periodically**: Use `squeue -u $USER` to see if job is still running
3. **Review errors**: Check `logs/pretrain_11244770.err` if something goes wrong
4. **Job will continue**: Even if you disconnect, the job keeps running

## Next Steps

After training completes (20 epochs from checkpoint):
- Checkpoints will be saved in `checkpoints/`
- Final model will be at `checkpoints/checkpoint_epoch_20.pt` (or similar)
- You can resume from any checkpoint using `--resume`
