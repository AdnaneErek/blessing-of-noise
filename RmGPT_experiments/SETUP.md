# Setup & Run Guide

## 1. Environment Setup

```bash
# Create and activate a virtual environment (Python 3.9+)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the bundled phmd library (dataset loaders)
cd lib && pip install -e . && cd ..
```

## 2. Dataset Preparation

Place the robot dataset under `data/raw/dataset/` with the following structure:

```
data/raw/dataset/
├── trainingDatasets/       # Simulation data (used for pretraining)
│   ├── 20241016/
│   └── 20241017/
├── finetuningDatasets/     # Real robot data (used for fine-tuning)
│   ├── folder1/
│   ├── folder2/
│   └── folder3/
└── testDatasets/           # Real robot test data (held-out evaluation)
    └── 20241016/
```

The simulation training data can be downloaded from:
[https://nextcloud.centralesupelec.fr/s/7AR6aamBZNXcRM8/download](https://nextcloud.centralesupelec.fr/s/7AR6aamBZNXcRM8/download)

## 3. Training Pipeline

The project follows a two-stage approach: **pretrain on simulation**, then **fine-tune on real data**.

### Stage 1 — Pretrain on Simulation Data

Choose one of the two pretraining strategies (both use V2 configs):

**Option A: Multitask pretraining** (masked token prediction + classification)

```bash
python pretrain_robot_sim_multitask.py \
    --config configs/pretrain_robot_sim_multitask_v2.yaml
```

**Option B: Supervised pretraining** (classification only)

```bash
bash pretrain_robot_sim_supervised.sh
# or directly:
python pretrain_robot_sim_supervised.py \
    --config configs/pretrain_robot_sim_supervised_v2.yaml
```

Checkpoints are saved to `checkpoints/`.

### Stage 2 — Fine-tune on Real Robot Data

Fine-tune the pretrained model on real robot data:

```bash
# Using V3 config (recommended — progressive unfreezing, lower LRs)
bash finetune_robot_real_v3.sh checkpoints/pretrain_robot_sim_supervised_final.pt

# Or the standard version
bash finetune_robot_real.sh checkpoints/pretrain_robot_sim_supervised_final.pt
```

### Stage 2 (alternative) — 3-Fold Cross-Experiment Validation

Run all 3 folds:

```bash
bash finetune_3fold_cross.sh checkpoints/pretrain_robot_sim_supervised_best.pt
```

Run a single fold (e.g., fold 2):

```bash
bash finetune_3fold_cross.sh checkpoints/pretrain_robot_sim_supervised_best.pt 2
```

### Robot-B Inverse Experiment

Train on Robot-B (test set), evaluate on Robot-A (fine-tuning set):

```bash
bash finetune_robot_b_inverse.sh checkpoints/pretrain_robot_sim_supervised_epoch_30.pt
```

## 4. Evaluation

### Evaluate on robot data (simulation + real test)

```bash
bash evaluate_robot.sh checkpoints/final_model_finetune_real.pt configs/finetune_robot_real_v3.yaml
```

### Evaluate Robot-B inverse model

```bash
bash evaluate_robot_b_inverse.sh checkpoints/robot_b_inverse/final_model_robot_b_inverse.pt
```

## 5. Paper Baseline (Bearing Datasets)

To reproduce the original RmGPT paper pipeline on bearing datasets (CWRU, JNUB, etc.):

```bash
# Pretrain on all 5 datasets
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain

# Fine-tune on a single dataset (e.g., diagnosis)
python train_rmgpt.py --config configs/paper_exact_config.yaml --task diagnosis
```

## Project Structure

```
├── model/                          # RmGPT model architecture
│   ├── rmgpt.py                    #   Main model, DiagnosisHead, PrognosisHead
│   ├── tokens.py                   #   Token embedding layers
│   └── transformer.py              #   Transformer encoder
├── train/
│   └── trainer.py                  # Training loop
├── data/
│   ├── dataset.py                  # PHM dataset class (bearing benchmarks)
│   ├── robot_dataset_loader.py     # Robot dataset loading utilities
│   └── split_strategy.py           # Paper-compliant data splits
├── configs/                        # YAML configurations
│   ├── pretrain_robot_sim_multitask_v2.yaml
│   ├── pretrain_robot_sim_supervised_v2.yaml
│   ├── finetune_3fold_cross.yaml
│   ├── finetune_robot_real_v3.yaml
│   ├── finetune_robot_b_inverse.yaml
│   └── ...
├── pretrain_robot_sim_multitask.py # Multitask pretraining script
├── pretrain_robot_sim_supervised.py# Supervised pretraining script
├── finetune_3fold_cross.py         # 3-fold cross-experiment fine-tuning
├── finetune_robot_real.py          # Real robot fine-tuning
├── finetune_robot_b_inverse.py     # Robot-B inverse experiment
├── evaluate_robot.py               # Robot evaluation
├── evaluate_robot_b_inverse.py     # Robot-B inverse evaluation
├── train_rmgpt.py                  # Paper baseline training script
├── lib/                            # Bundled phmd library
├── phmd/                           # PHM dataset loaders
└── requirements.txt
```

## Hardware Notes

- Training was performed on an HPC cluster with NVIDIA GPUs.
- Pretraining typically takes 1-3 hours depending on epochs and GPU.
- Fine-tuning on real data is fast (~5-10 min) due to the small dataset size (~190 samples).
- All scripts auto-detect CUDA; they fall back to CPU if unavailable.
