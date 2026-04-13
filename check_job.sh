#!/bin/bash
# Script to check your training job after reconnecting
# Usage: ./check_job.sh

echo "=== Checking Your Training Jobs ==="
echo ""

# Check all your running jobs
echo "1. Running/Pending Jobs:"
squeue -u $USER

echo ""
echo "=== Recent Job IDs (if you need to check specific job) ==="
# List recent output files to find job IDs
if [ -d "logs" ]; then
    echo "Recent pretrain jobs:"
    ls -lt logs/pretrain_*.out 2>/dev/null | head -5 | awk '{print "  Job ID:", $NF}' | sed 's|logs/pretrain_||; s|\.out||'
else
    echo "No logs directory found"
fi

echo ""
echo "=== Quick Check Commands ==="
echo "To check a specific job (replace 11244770 with your job ID):"
echo "  squeue -j 11244770              # Check if running"
echo "  tail -n 50 logs/pretrain_*.out  # View recent output"
echo "  sacct -u \$USER                  # View all jobs (including completed)"
echo ""
echo "To watch output live:"
echo "  tail -f logs/pretrain_<JOB_ID>.out"
