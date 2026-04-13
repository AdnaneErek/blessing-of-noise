#!/bin/bash
# Fine-tune RmGPT on Robot Fault Diagnosis Dataset
# Uses pretrained model and aggressive optimization strategy

PRETRAINED_CKPT="checkpoints/final_model_pretrain.pt"
CONFIG="configs/finetune_robot.yaml"
DATASET="ROBOT"
TASK="diagnosis"

echo "=========================================="
echo "Fine-tuning RmGPT on Robot Dataset"
echo "=========================================="
echo "Config: $CONFIG"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "Dataset: $DATASET"
echo "Task: $TASK"
echo ""
echo "AGGRESSIVE SETTINGS:"
echo "  - Backbone LR: 1.5e-5 (preserve pretrained features)"
echo "  - Head LR: 5.0e-3 (aggressive learning for new head)"
echo "  - Label Smoothing: 0.1"
echo "  - Focal Loss: Enabled (alpha=0.25, gamma=2.0)"
echo "  - Epochs: 200"
echo "  - Improved Diagnosis Head: Enabled"
echo ""
echo "FEATURES:"
echo "  - DesiredTrajectory xyz (3)"
echo "  - RealizedTrajectory xyz (3)"
echo "  - Error xyz (e_x, e_y, e_z) (3)"
echo "  - Total: 9 features"
echo ""

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    echo "Please run pretraining first or specify a different checkpoint."
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Run fine-tuning
echo "Starting fine-tuning..."
python train_rmgpt.py \
    --config "$CONFIG" \
    --task "$TASK" \
    --dataset "$DATASET" \
    --resume "$PRETRAINED_CKPT" \
    2>&1 | tee logs/finetune_robot.log

echo ""
echo "Fine-tuning complete!"
echo "Check logs/finetune_robot.log for details"
echo "Checkpoint saved to: checkpoints/final_model_diagnosis.pt"
