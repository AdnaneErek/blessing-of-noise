#!/bin/bash
# Run pretraining in a screen session (can reattach later)

# Start screen session named "pretrain"
screen -S pretrain -dm bash -c "python train_rmgpt.py \
    --config configs/paper_exact_config.yaml \
    --task pretrain \
    --resume checkpoints/checkpoint_epoch_10.pt; exec bash"

echo "Training started in screen session 'pretrain'"
echo ""
echo "To attach to the session:"
echo "  screen -r pretrain"
echo ""
echo "To detach (leave it running):"
echo "  Press Ctrl+A, then D"
echo ""
echo "To list all screen sessions:"
echo "  screen -ls"
echo ""
echo "To kill the session:"
echo "  screen -S pretrain -X quit"
