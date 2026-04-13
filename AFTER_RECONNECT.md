# Checking Your Job After Reconnecting

When you disconnect and reconnect later, here's how to check on your training job.

## Your Current Job ID: **11244770**

**Note it down** or just use the commands below to find it automatically!

---

## Quick Check (After Reconnecting)

### Option 1: Check All Your Jobs
```bash
# See all your running/pending jobs
squeue -u $USER
```

This will show:
- Job ID
- Job Name (e.g., `rmgpt_pretrain`)
- Status (RUNNING, PENDING, etc.)
- Time running
- Node assigned

### Option 2: Find Your Job ID Automatically
```bash
# List recent output files to find job IDs
ls -lt logs/pretrain_*.out | head -5
```

This shows recent training jobs. The file name contains the job ID (e.g., `pretrain_11244770.out`).

### Option 3: Use the Check Script
```bash
./check_job.sh
```

This script automatically shows your running jobs and recent job IDs.

---

## Check Specific Job Status

Once you have the job ID (e.g., `11244770`):

```bash
# Check if job is running
squeue -j 11244770

# Get detailed information
scontrol show job 11244770
```

**Job States:**
- **RUNNING (R)**: Still training ✅
- **PENDING (PD)**: Waiting in queue (will start soon)
- **COMPLETED (CD)**: Finished successfully ✅
- **FAILED (F)**: Something went wrong ❌
- **CANCELLED (CA)**: Was cancelled ❌

---

## View Output After Reconnecting

### See Recent Output (Last 50 Lines)
```bash
# Replace 11244770 with your job ID
tail -n 50 logs/pretrain_11244770.out

# Or see the latest log file
tail -n 50 logs/pretrain_*.out | head -50
```

### Watch Output Live (If Still Running)
```bash
tail -f logs/pretrain_11244770.out
```

### View All Output
```bash
# Full output
cat logs/pretrain_11244770.out

# Or using less for scrolling
less logs/pretrain_11244770.out
# (Press 'q' to exit)
```

### Check for Errors
```bash
cat logs/pretrain_11244770.err
```

---

## Check Completed Jobs

If job finished (not showing in `squeue`):

```bash
# List all your jobs (including completed)
sacct -u $USER

# Get detailed accounting for specific job
sacct -j 11244770 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxRSSNode

# Get full accounting
sacct -j 11244770 -l
```

This shows:
- Exit code (0 = success, non-zero = error)
- Time used (Elapsed)
- Memory used (MaxRSS)
- CPU efficiency

---

## Step-by-Step: After Reconnecting

1. **Connect to Ruche:**
   ```bash
   ssh your_username@ruche.mesocentre.universite-paris-saclay.fr
   ```

2. **Navigate to project:**
   ```bash
   cd /gpfs/workdir/erekrakead/RmGPT
   ```

3. **Check running jobs:**
   ```bash
   squeue -u $USER
   ```
   If you see your job, it's still running! ✅

4. **If you forgot the job ID:**
   ```bash
   # Find it automatically
   ls -lt logs/pretrain_*.out | head -1
   # Or use the script
   ./check_job.sh
   ```

5. **View recent output:**
   ```bash
   # Replace with your job ID
   tail -n 50 logs/pretrain_11244770.out
   ```

6. **Watch live (if still running):**
   ```bash
   tail -f logs/pretrain_11244770.out
   ```

---

## Example Workflow

```bash
# After reconnecting...
cd /gpfs/workdir/erekrakead/RmGPT

# Check running jobs
squeue -u $USER
# Output:
# JOBID      PARTITION NAME              USER      ST  TIME  NODES NODELIST(REASON)
# 11244770   gpua100   rmgpt_pretrain   erekrakead R   2:34  1     ruche-gpu42

# Job is running! Check progress
tail -n 50 logs/pretrain_11244770.out

# Or watch live
tail -f logs/pretrain_11244770.out
```

---

## Pro Tips

1. **Save your job ID** somewhere (or just remember the date - you can find it)
2. **Check `squeue` first** - quickest way to see if job is running
3. **Output files are persistent** - they're saved on GPFS, so you can check anytime
4. **Jobs run on compute nodes** - not affected by disconnecting from front-end node

---

## If Job Completed

Check the output:
```bash
# View final output
tail -n 100 logs/pretrain_11244770.out

# Check checkpoints created
ls -lt checkpoints/

# Check job accounting
sacct -j 11244770
```

---

## Summary Commands

```bash
# Quick check (all running jobs)
squeue -u $USER

# Find job ID
ls -lt logs/pretrain_*.out | head -1

# Check specific job
squeue -j 11244770

# View output
tail -n 50 logs/pretrain_11244770.out

# Watch live
tail -f logs/pretrain_11244770.out

# Check completed jobs
sacct -u $USER
```

**The job runs independently** - you can disconnect and reconnect anytime to check on it! 🚀
