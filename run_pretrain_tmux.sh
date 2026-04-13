#!/bin/bash
# Run pretraining in a tmux session (can reattach later)

# Start tmux session named "pretrain"
tmux new-session -d -s pretrain "python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt; exec bash"

echo "Training started in tmux session 'pretrain'"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t pretrain"
echo ""
echo "To detach (leave it running):"
echo "  Press Ctrl+B, then D"
echo ""
echo "To list all tmux sessions:"
echo "  tmux ls"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t pretrain"
