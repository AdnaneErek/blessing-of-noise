# PHMD Alternative Datasets for RmGPT

## Overview

Since the original paper datasets **SLIET**, **QPZZ-II**, and **SMU** are not available in PHMD, we've selected **3 alternative datasets** from PHMD that are:
- Similar in application (bearing/gear fault diagnosis)
- Have sampling frequency **>5kHz** (as requested)
- Available directly in PHMD (no custom loaders needed)

## Selected Alternative Datasets

### 1. JNUB (Jiangnan University Bearing) - Replaces SLIET
- **Application**: Bearing fault diagnosis
- **Sampling Frequency**: **50 kHz** ✅ (>5kHz)
- **Fault Types**: 4 classes
  - Health state
  - Inner ring fault
  - Outer ring fault
  - Rolling element fault
- **Number of Units**: 12
- **Similarity to SLIET**: Both are bearing fault diagnosis datasets
- **Status**: ✅ Available in PHMD

### 2. KAUG17 (Korea Aerospace University Gear) - Replaces QPZZ-II
- **Application**: Gear fault diagnosis
- **Sampling Frequency**: **10 kHz** ✅ (>5kHz)
- **Fault Types**: 3 classes
  - Normal
  - Spall
  - Crack
- **Number of Units**: 31
- **Similarity to QPZZ-II**: Both are gear fault diagnosis datasets
- **Status**: ✅ Available in PHMD

### 3. HSG18 (SDOL Wind Generator Gear) - Replaces SMU
- **Application**: Gear fault detection
- **Sampling Frequency**: **97,656 Hz** ✅ (>5kHz)
- **Fault Types**: 2 classes (binary)
  - Normal
  - Fault
- **Number of Units**: 17
- **Note**: This is a detection task (binary), but can be used for diagnosis
- **Similarity to SMU**: Both are gear fault datasets
- **Status**: ✅ Available in PHMD

## Complete Dataset List

### Diagnosis (Fault Classification)
1. **CWRU** - 12 kHz, Bearing (original paper dataset) ✅
2. **JNUB** - 50 kHz, Bearing (replaces SLIET) ✅
3. **KAUG17** - 10 kHz, Gear (replaces QPZZ-II) ✅
4. **HSG18** - 97,656 Hz, Gear (replaces SMU) ✅

### Prognosis (RUL Prediction)
1. **XJTU-SY** - Bearing (original paper dataset) ✅

## Usage

### Using Alternative Datasets

Simply change the `dataset_name` in `configs/default_config.yaml`:

```yaml
data:
  dataset_name: "JNUB"  # or "KAUG17", "HSG18"
  task_name: "Diagnosis"
```

Then run:
```bash
python train_rmgpt.py --config configs/default_config.yaml --task diagnosis
```

### Dataset Characteristics Summary

| Dataset | Type | Freq | Classes | Units | Status |
|---------|------|------|---------|-------|--------|
| CWRU | Bearing | 12 kHz | 4 | 161 | ✅ PHMD |
| JNUB | Bearing | 50 kHz | 4 | 12 | ✅ PHMD |
| KAUG17 | Gear | 10 kHz | 3 | 31 | ✅ PHMD |
| HSG18 | Gear | 97.7 kHz | 2 | 17 | ✅ PHMD |
| XJTU-SY | Bearing | - | RUL | - | ✅ PHMD |

All selected datasets have sampling frequencies **>5kHz** as requested!

## Configuration File

See `configs/rmgpt_phmd_datasets.yaml` for detailed configuration of these alternative datasets.
