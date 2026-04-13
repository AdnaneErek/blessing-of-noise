# The Blessing of Noise
### Bridging the Sim-to-Real Gap in Robot Fault Diagnosis with Foundation Models and Structured Domain Randomization

**Adnane Erekraken · Daniel Fernandez De La Mela**  
CentraleSupélec, Université Paris-Saclay — MSc in Artificial Intelligence  
Supervisors: Prof. Zhiguo Zeng · April 2026

[![Paper](https://img.shields.io/badge/Paper-April%202026-blue)](./paper.pdf)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

---

## Overview

Industrial robot fault diagnosis faces a fundamental bottleneck: real fault data is scarce and expensive to collect, yet models trained purely on simulation fail badly on real hardware — the **sim-to-real gap**.

This project investigates a multi-stage framework to close that gap:

1. **Structured noise injection** in the Simulink digital twin to produce a physically grounded, diverse training distribution
2. **Supervised pretraining** of [RmGPT](https://arxiv.org/abs/2409.17604) on enriched simulation data, followed by **progressive partial unfreezing** during real-data fine-tuning
3. **Cross-robot generalization** evaluation across two physically distinct robots

| Metric | Value |
|---|---|
| LSTM baseline (real accuracy) | 53.33% |
| **Best real accuracy (3-fold Exp3)** | **75.56%** |
| Simulation accuracy | 87.23% |
| Domain gap reduced | 36 pp → **11.67 pp** |
| Cross-robot transfer (Robot B) | 72.22% |

> **Key finding:** The single most impactful intervention was not a larger model or smarter architecture — it was simply making the simulator noisier.

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/AdnaneErek/blessing-of-noise.git
cd blessing-of-noise

# Create and activate a virtual environment (Python 3.9+)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the bundled phmd library (dataset loaders)
cd lib && pip install -e . && cd ..
```

### Dataset

Place the robot dataset under `data/raw/dataset/`:

```
data/raw/dataset/
├── trainingDatasets/       # Simulation data (pretraining)
│   ├── 20241016/
│   └── 20241017/
├── finetuningDatasets/     # Real robot data (fine-tuning)
│   ├── folder1/
│   ├── folder2/
│   └── folder3/
└── testDatasets/           # Held-out real evaluation
    └── 20241016/
```

Simulation training data download:
```
https://nextcloud.centralesupelec.fr/s/7AR6aamBZNXcRM8/download
```

---

## Training Pipeline

The project follows a two-stage approach: **pretrain on simulation → fine-tune on real data**.

### Stage 1 — Pretrain on Simulation

**Option A — Multitask pretraining** (masked token prediction + classification):

```bash
python pretrain_robot_sim_multitask.py \
    --config configs/pretrain_robot_sim_multitask_v2.yaml
```

**Option B — Supervised pretraining** (classification only, recommended):

```bash
bash pretrain_robot_sim_supervised.sh
# or directly:
python pretrain_robot_sim_supervised.py \
    --config configs/pretrain_robot_sim_supervised_v2.yaml
```

Checkpoints are saved to `checkpoints/`.

### Stage 2 — Fine-tune on Real Robot Data

```bash
# V3 config — progressive unfreezing + decoupled LRs (recommended)
bash finetune_robot_real_v3.sh checkpoints/pretrain_robot_sim_supervised_final.pt

# Standard version
bash finetune_robot_real.sh checkpoints/pretrain_robot_sim_supervised_final.pt
```

### Stage 2 (alternative) — 3-Fold Cross-Robot Validation

```bash
# All 3 folds
bash finetune_3fold_cross.sh checkpoints/pretrain_robot_sim_supervised_best.pt

# Single fold (e.g. fold 2)
bash finetune_3fold_cross.sh checkpoints/pretrain_robot_sim_supervised_best.pt 2
```

### Robot-B Inverse Experiment

Train on Robot B, evaluate on Robot A:

```bash
bash finetune_robot_b_inverse.sh checkpoints/pretrain_robot_sim_supervised_epoch_30.pt
```

---

## Evaluation

```bash
# Evaluate on simulation + real test sets
bash evaluate_robot.sh \
    checkpoints/final_model_finetune_real.pt \
    configs/finetune_robot_real_v3.yaml

# Evaluate Robot-B inverse model
bash evaluate_robot_b_inverse.sh \
    checkpoints/robot_b_inverse/final_model_robot_b_inverse.pt
```

---

## Paper Baseline (Bearing Datasets)

To reproduce the original RmGPT paper pipeline on bearing benchmarks (CWRU, JNUB, etc.):

```bash
# Pretrain on all 5 datasets
python train_rmgpt.py --config configs/paper_exact_config.yaml --task pretrain

# Fine-tune for fault diagnosis
python train_rmgpt.py --config configs/paper_exact_config.yaml --task diagnosis
```

---

## Project Structure

```
├── RmGPT_experiments/                 # All RmGPT-based experiments
│   ├── model/
│   │   ├── rmgpt.py                   # Main model, DiagnosisHead, PrognosisHead
│   │   ├── tokens.py                  # Token embedding layers
│   │   └── transformer.py             # Transformer encoder
│   ├── train/
│   │   └── trainer.py                 # Training loop
│   ├── data/
│   │   ├── dataset.py                 # PHM dataset class (bearing benchmarks)
│   │   ├── robot_dataset_loader.py    # Robot dataset loading utilities
│   │   └── split_strategy.py         # Paper-compliant data splits
│   ├── configs/
│   │   ├── pretrain_robot_sim_multitask_v2.yaml
│   │   ├── pretrain_robot_sim_supervised_v2.yaml
│   │   ├── finetune_robot_real_v3.yaml
│   │   ├── finetune_3fold_cross.yaml
│   │   ├── finetune_robot_b_inverse.yaml
│   │   └── paper_exact_config.yaml
│   ├── pretrain_robot_sim_multitask.py
│   ├── pretrain_robot_sim_supervised.py
│   ├── finetune_robot_real.py
│   ├── finetune_3fold_cross.py
│   ├── finetune_robot_b_inverse.py
│   ├── evaluate_robot.py
│   ├── evaluate_robot_b_inverse.py
│   ├── train_rmgpt.py                 # Paper baseline (bearing datasets)
│   ├── lib/                           # Bundled phmd library
│   └── requirements.txt
└── Moment_experiments/                # All MOMENT-based experiments
    ├── momentfm/                      # MOMENT model library
    ├── scripts/
    │   ├── pretrain_moment_supervised.py
    │   ├── pretrain_robot_moment.py
    │   ├── finetune_robot_moment.py
    │   └── evaluate_robot_moment.py
    └── pretrain_moment_sim.py
```

---

## Method

### Task

9-class robot fault classification from multivariate trajectory signals **X** ∈ ℝ^(T×C), T = 1000 timesteps, C = 9 channels:

```
X = [ x_des (Desired),  x_real (Realized),  e = x_real − x_des (Error) ]
```

Classes: `{Healthy, M1–M4 Stuck, M1–M4 SSE}`

### Step 1 — Foundation Model Selection

RmGPT's **variable-channel inductive bias** (each sensor channel processed as a separate sequence) is critical for distinguishing motor-specific fault signatures. Moment (FLAN-T5 based) flattens all channels, destroying this structure.

| Model | Sim Acc. | Real Acc. | Gap |
|---|---|---|---|
| Moment | 87.69% | 28.11% | 59.6 pp |
| **RmGPT** | **98.15%** | **47.78%** | **50.4 pp** |

### Step 2 — Structured Noise Injection

Replace the deterministic Simulink simulator with a stochastic one, expanding the training distribution toward real-world variability while preserving physical structure.

**Motor steady-state errors** — Gaussian-shaped probabilities (σ = 1.2):
```
p(e_k) = (1/Z) · exp(−e_k² / 2σ²),   e_k ∈ {−3, …, 3}
```

**Random actuation delay:**
```
holdingTimeIdx ~ U(2, 30)
```

| Stage | Sim Acc. | Real Acc. |
|---|---|---|
| No noise | 80.14% | 48.89% |
| Partial noise (3/9 classes) | 96.67% | 53.33% |
| **Full noise (9/9 classes)** | **98.61%** | **61.11%** |

### Step 3 — Pretraining Objective

Two-phase simulation pretraining:
1. **MTP warm-up** — masked token prediction (15% masking, BERT-style)
2. **Supervised fine-tuning on simulation** — combined loss:

```
ℒ = (1 − λ)ℒ_CE + λℒ_focal,   λ = 0.5
```

with label smoothing ε = 0.1, focal loss α = 0.25, γ = 2.0.

### Step 4 — Progressive Fine-Tuning (A1–A4)

| Strategy | Sim | Real | Gap | F1 |
|---|---|---|---|---|
| A1: Scratch + Full FT | 78.15% | 57.78% | 20.4 pp | 0.580 |
| A2: Scratch + Head Only | 90.19% | 61.11% | 29.1 pp | 0.587 |
| A3: Pretrain + 25% Unfreeze | 85.60% | 68.89% | 16.7 pp | 0.684 |
| **A4: Pretrain + 50% Unfreeze** | **86.67%** | **70.00%** | **16.7 pp** | **0.697** |

Best strategy (A4) unfreezes the last 2 of 4 transformer layers with decoupled learning rates:
```
η_backbone = 5×10⁻⁷,   η_head = 5×10⁻⁶   (cosine decay, 20 epochs)
```

---

## Results

### Performance Progression

| Stage | Real Acc. |
|---|---|
| LSTM + error features (baseline) | 53.33% |
| RmGPT scratch | 47.78% |
| + Full noisy simulation | 61.11% |
| + Supervised pretrain + 50% unfreeze | 70.00% |
| Cross-robot (Robot B standard) | 72.22% |
| **3-fold Exp3 (best)** | **75.56%** |

### Cross-Robot Transfer

| Direction | Condition | Accuracy |
|---|---|---|
| Robot A → Robot B | Standard | 70.00% |
| Robot B → Robot A | Standard | 72.22% |
| Robot B → Robot A | Under load | 53.33% |

### 3-Fold Cross-Experiment

| Test Fold | Test Acc. | Best Val Acc. |
|---|---|---|
| Robot A (standard) | 71.11% | 70.59% |
| Robot A (under load) | 44.44% | 67.65% |
| **Robot B** | **75.56%** | **67.65%** |

---

## Hardware

Training ran on **La Ruche Mesocenter** (Université Paris-Saclay HPC), 4× NVIDIA A100 80 GB. All scripts auto-detect CUDA and fall back to CPU.

- Pretraining: ~1–3 h/run (100 epochs, batch 256)
- Fine-tuning: ~5–10 min/run (20 epochs, batch 32, ~190 real samples)
- Total project GPU time: ~261 h (A100)

---

## Citation

```bibtex
@article{erekraken2026blessing,
  title     = {The Blessing of Noise: Bridging the Sim-to-Real Gap in Robot Fault 
               Diagnosis with Foundation Models and Structured Domain Randomization},
  author    = {Erekraken, Adnane and Fernandez De La Mela, Daniel},
  year      = {2026},
  month     = {April},
  institution = {CentraleSupélec, Université Paris-Saclay},
  url       = {https://github.com/AdnaneErek/blessing-of-noise}
}
```

---

## References

- [RmGPT](https://arxiv.org/abs/2409.17604) — Foundation model for PHM with variable-channel transformer
- [MOMENT](https://arxiv.org/abs/2402.03885) — Open time-series foundation models, ICML 2024
- [GPT4TS](https://arxiv.org/abs/2302.11939) — One Fits All, NeurIPS 2023
- [Domain Randomization](https://arxiv.org/abs/1703.06907) — Tobin et al., IROS 2017
- [ULMFiT](https://arxiv.org/abs/1801.06146) — Progressive unfreezing, Howard & Ruder, ACL 2018
- [Digital Twin Simulink](https://github.com/sonic160/dtr_digital_model_simulink)

---

## License

MIT License — see [`LICENSE`](./LICENSE) for details.
