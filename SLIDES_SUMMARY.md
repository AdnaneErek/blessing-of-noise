# RmGPT Project: Slide-Ready Summary

## 🎯 Project Title
**RmGPT: Foundation Model for Rotating Machinery PHM**
*Paper-Compliant Implementation & Evaluation*

---

## 📊 Slide 1: Project Overview

### Objective
- Implement RmGPT (arXiv:2409.17604v2) with exact paper methodology
- Unified foundation model for fault diagnosis and prognosis
- Self-supervised pretraining + supervised fine-tuning

### Key Achievements
✅ Paper-compliant architecture (68.5M parameters)  
✅ Multi-dataset pretraining (5 datasets)  
✅ Fine-tuning for diagnosis and prognosis  
✅ Strict 80/20 data split (no leakage)  
✅ SLURM cluster integration  

---

## 🏗️ Slide 2: Model Architecture

### Core Components
1. **4 Token Types**
   - Signal Tokens (patch-based, P=256)
   - Prompt Tokens (task-adaptive, lp=10)
   - Time-Freq Tokens (FFT + Wavelet)
   - Fault Tokens (learnable prototypes, lt=1)

2. **Transformer Encoder**
   - 4 layers, 512 hidden dim, 8 attention heads
   - Feed-forward: 2048 dim

3. **Task Heads**
   - Diagnosis: Multi-class classification
   - Prognosis: RUL regression

### Model Size
- **Total Parameters**: 68.50M

---

## 📈 Slide 3: Datasets & Preprocessing

### Pretraining Datasets (5)
1. **CWRU** - Bearing diagnosis (12 kHz)
2. **JNUB** - Bearing diagnosis (50 kHz)
3. **KAUG17** - Gear diagnosis (10 kHz)
4. **HSG18** - Gear diagnosis (97.6 kHz)
5. **XJTU-SY** - Bearing RUL (25.6 kHz)

### Paper-Compliant Preprocessing
1. **Downsample** → ~5kHz
2. **Window** → 2048 timesteps (sliding, no overlap)
3. **Channel Standardize** → Pad to max channels
4. **Normalize** → Zero mean, unit variance

---

## 🔄 Slide 4: Data Split Strategy

### Paper-Compliant Split (80/20)

```
Total Dataset (100%)
├── Train80 (80%)
│   ├── Pretrain: ALL 80% (unlabeled)
│   ├── Finetune Train: 72% of total
│   └── Finetune Val: 8% of total
└── Test20 (20%) ← UNTOUCHED until evaluation
```

### Key Features
- ✅ Stratified splitting (maintains class distribution)
- ✅ Fixed random seed (42) for reproducibility
- ✅ **Zero data leakage** (test set isolated)

---

## 🎓 Slide 5: Training Methodology

### 1. Pretraining (Self-Supervised)
- **Objective**: Next-token prediction
- **Epochs**: 20
- **Batch Size**: 256
- **Learning Rate**: 3.00 × 10⁻⁷
- **Data**: Aggregated 80% train from all 5 datasets (unlabeled)

### 2. Fine-tuning (Supervised)
- **Epochs**: 3
- **Tasks**: Diagnosis (classification) / Prognosis (regression)
- **Data**: Individual datasets (72% train, 8% val)
- **Prompt Learning**: +5 epochs for task adaptation

---

## 💻 Slide 6: Implementation Highlights

### Project Structure
- **Model**: Token embeddings, Transformer, Task heads
- **Data**: Preprocessing, splitting, windowing, multi-dataset
- **Training**: Trainer, checkpointing, resumption
- **Evaluation**: Pretrained & fine-tuned model evaluation

### Key Scripts
- `train_rmgpt.py` - Main training (pretrain/finetune)
- `evaluate_pretrained.py` - Pretrained model evaluation
- `evaluate_finetuned.py` - Fine-tuned model evaluation
- `run_pretrain_sbatch.sh` - SLURM job submission

---

## 🔧 Slide 7: Technical Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Dimension Mismatches** | Channel standardization + dynamic inference |
| **Data Leakage** | Strict 80/20 split with isolated test set |
| **Long Time Series** | Sliding windows (2048 timesteps, no overlap) |
| **Checkpoint Resumption** | Infer dimensions from checkpoint state dict |
| **Cluster Training** | SLURM integration with background jobs |
| **OOM Errors** | 128GB memory allocation + optimized batching |
| **Label Encoding** | Fixed windowed label association |

---

## 📊 Slide 8: Results & Status

### Completed ✅
- ✅ Model architecture implementation
- ✅ Paper-compliant preprocessing pipeline
- ✅ Data split strategy (80/20)
- ✅ **Pretraining: 20 epochs COMPLETED**
- ✅ **Fine-tuning: CWRU diagnosis COMPLETED**
- ✅ Evaluation scripts implemented
- ✅ SLURM cluster integration

### Checkpoints
- `checkpoint_epoch_20.pt` - Final pretrained model
- `final_model_diagnosis.pt` - Fine-tuned CWRU diagnosis

### Current Status
- 🔄 **Evaluation in progress** (accuracy metrics)

---

## 📐 Slide 9: Paper-Exact Hyperparameters

### Model Configuration
- Patch length (P): **256**
- Stride (S): **256**
- Embed dim (d): **512**
- Prompt tokens (lp): **10**
- Transformer layers: **4**
- Attention heads: **8**

### Training Configuration
- Batch size: **256**
- Learning rate: **3.00 × 10⁻⁷**
- Pretrain epochs: **20**
- Finetune epochs: **3**
- Prompt learning epochs: **5**

### Data Configuration
- Window length: **2048 timesteps**
- Downsample to: **~5kHz**
- Test split: **20%**
- Finetune val: **10% of train80**

---

## 🎯 Slide 10: Key Contributions

### Methodology
1. **Paper-Compliant Implementation**
   - Exact hyperparameters from paper
   - Reproducible preprocessing pipeline
   - Strict data split strategy

2. **Robust Data Handling**
   - Multi-dataset aggregation
   - Channel standardization
   - Sliding window generation

3. **Production-Ready Training**
   - Checkpoint management
   - SLURM cluster integration
   - Background job execution

4. **Comprehensive Evaluation**
   - Pretrained model evaluation
   - Fine-tuned model evaluation
   - Multiple metrics (accuracy, MSE, etc.)

---

## 📚 Slide 11: References & Resources

### Paper
- **Title**: RmGPT: A Foundation Model with Generative Pre-trained Transformer for Fault Diagnosis and Prognosis in Rotating Machinery
- **arXiv**: 2409.17604v2
- **Year**: 2024

### Implementation
- **Repository**: `/gpfs/workdir/erekrakead/RmGPT`
- **Config**: `configs/paper_exact_config.yaml`
- **Documentation**: Multiple `.md` files

### Datasets
- PHMD library integration
- 5 datasets: CWRU, JNUB, KAUG17, HSG18, XJTU-SY

---

## 🚀 Slide 12: Future Work

### Immediate Next Steps
1. Complete accuracy evaluation for fine-tuned models
2. Investigate label mapping issues (10 classes vs 4 classes)
3. Evaluate on all fine-tuned models (all datasets)

### Potential Extensions
1. Hyperparameter tuning
2. Additional datasets
3. Ablation studies
4. Comparison with baseline methods

---

## 📝 Quick Reference: Key Numbers

- **Model Parameters**: 68.50M
- **Pretraining Epochs**: 20
- **Fine-tuning Epochs**: 3
- **Batch Size**: 256
- **Learning Rate**: 3.00 × 10⁻⁷
- **Window Length**: 2048 timesteps
- **Patch Length**: 256
- **Embed Dim**: 512
- **Transformer Layers**: 4
- **Datasets**: 5 (pretraining), 4 (diagnosis), 1 (prognosis)
- **Train/Test Split**: 80/20
- **Finetune Train/Val**: 90/10 (of train80)

---

**Ready for Presentation** ✨
