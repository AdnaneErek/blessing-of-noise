#!/bin/bash
# Quick monitoring script for SLURM job
# Usage: ./monitor_job.sh [JOB_ID]
# If JOB_ID not provided, will show all your jobs

JOB_ID=${1:-"11244770"}  # Default to current job if not specified

echo "=== Job Status ==="
squeue -u $USER -j $JOB_ID

echo ""
echo "=== Job Details ==="
scontrol show job $JOB_ID 2>/dev/null || echo "Job not found or completed"

echo ""
echo "=== Recent Output (last 20 lines) ==="
if [ -f "logs/pretrain_${JOB_ID}.out" ]; then
    tail -n 20 "logs/pretrain_${JOB_ID}.out"
else
    echo "Output file not yet created (job may not have started)"
fi

echo ""
echo "=== Recent Errors (last 10 lines) ==="
if [ -f "logs/pretrain_${JOB_ID}.err" ]; then
    tail -n 10 "logs/pretrain_${JOB_ID}.err"
else
    echo "Error file not yet created"
fi
