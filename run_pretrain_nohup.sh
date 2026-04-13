#!/bin/bash
# Run pretraining in background (continues even if disconnected)

# Option 1: Resume from checkpoint
nohup python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt \
    > pretrain_output.log 2>&1 &

echo "Training started in background!"
echo "Process ID: $!"
echo "Output log: pretrain_output.log"
echo ""
echo "To monitor progress:"
echo "  tail -f pretrain_output.log"
echo ""
echo "To check if process is running:"
echo "  ps aux | grep train_rmgpt"
echo ""
echo "To stop training:"
echo "  kill $!"
