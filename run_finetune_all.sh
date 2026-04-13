#!/bin/bash
# Fine-tune pretrained model on all datasets sequentially

PRETRAINED_CKPT="checkpoints/final_model_pretrain.pt"
CONFIG="configs/paper_exact_config.yaml"

# Diagnosis datasets (classification)
DIAGNOSIS_DATASETS=("CWRU" "JNUB" "KAUG17" "HSG18")

# Prognosis datasets (regression)
PROGNOSIS_DATASETS=("XJTU-SY")

echo "Fine-tuning pretrained model on all datasets..."
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo ""

# Fine-tune on diagnosis datasets
for dataset in "${DIAGNOSIS_DATASETS[@]}"; do
    echo "========================================="
    echo "Fine-tuning on: $dataset (Diagnosis)"
    echo "========================================="
    
    python train_rmgpt.py \
        --config $CONFIG \
        --task diagnosis \
        --resume $PRETRAINED_CKPT \
        2>&1 | tee logs/finetune_${dataset}_diagnosis.log
    
    # Move final checkpoint to dataset-specific location
    if [ -f "checkpoints/final_model_diagnosis.pt" ]; then
        mkdir -p checkpoints/finetuned
        mv checkpoints/final_model_diagnosis.pt \
           checkpoints/finetuned/${dataset}_final_model_diagnosis.pt
        echo "Saved fine-tuned model: checkpoints/finetuned/${dataset}_final_model_diagnosis.pt"
    fi
    
    echo ""
done

# Fine-tune on prognosis datasets
for dataset in "${PROGNOSIS_DATASETS[@]}"; do
    echo "========================================="
    echo "Fine-tuning on: $dataset (Prognosis)"
    echo "========================================="
    
    python train_rmgpt.py \
        --config $CONFIG \
        --task prognosis \
        --resume $PRETRAINED_CKPT \
        2>&1 | tee logs/finetune_${dataset}_prognosis.log
    
    # Move final checkpoint to dataset-specific location
    if [ -f "checkpoints/final_model_prognosis.pt" ]; then
        mkdir -p checkpoints/finetuned
        mv checkpoints/final_model_prognosis.pt \
           checkpoints/finetuned/${dataset}_final_model_prognosis.pt
        echo "Saved fine-tuned model: checkpoints/finetuned/${dataset}_final_model_prognosis.pt"
    fi
    
    echo ""
done

echo "========================================="
echo "Fine-tuning complete!"
echo "========================================="
echo "Fine-tuned models saved in: checkpoints/finetuned/"
echo "Training logs saved in: logs/finetune_*.log"
