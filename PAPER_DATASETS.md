# RmGPT Paper Datasets

## Exact Datasets Used in the Paper (arXiv:2409.17604v2)

### Diagnosis Datasets (Fault Classification)
1. **CWRU** - Case Western Reserve University Bearing Dataset
   - ✅ Available in PHMD library
   - Task: Fault diagnosis (4 classes: Normal, Inner Race, Outer Race, Ball)
   - Sampling frequency: 12 kHz

2. **SLIET** - SLIET Bearing Fault Dataset
   - ❌ NOT in PHMD - Requires custom loader
   - Custom loader implemented in `data/custom_datasets.py`
   - **Alternative**: **JNUB** (50 kHz, bearing) - Available in PHMD ✅

3. **QPZZ-II** - QPZZ-II Gear Fault Dataset  
   - ❌ NOT in PHMD - Requires custom loader
   - Custom loader implemented in `data/custom_datasets.py`
   - **Alternative**: **KAUG17** (10 kHz, gear) - Available in PHMD ✅

4. **SMU** - SMU Gear Fault Dataset
   - ❌ NOT in PHMD - Requires custom loader
   - Custom loader implemented in `data/custom_datasets.py`
   - **Alternative**: **HSG18** (97,656 Hz, gear) - Available in PHMD ✅

### Prognosis Datasets (RUL Prediction)
1. **XJTU-SY** - Xi'an Jiaotong University Bearing RUL Dataset
   - ✅ Available in PHMD library
   - Task: Remaining Useful Life (RUL) prediction

## Alternative PHMD Datasets (All >5kHz)

Since SLIET, QPZZ-II, and SMU are not available, we use these PHMD alternatives:

1. **JNUB** (Jiangnan University Bearing) - 50 kHz
   - Similar to SLIET (bearing fault diagnosis)
   - 4 fault types: health state, inner ring, outer ring, rolling element
   
2. **KAUG17** (Korea Aerospace University Gear) - 10 kHz
   - Similar to QPZZ-II (gear fault diagnosis)
   - 3 fault types: normal, spall, crack
   
3. **HSG18** (SDOL Wind Generator Gear) - 97,656 Hz
   - Similar to SMU (gear fault diagnosis)
   - Binary classification: normal, fault

## Dataset Loading

### PHMD Datasets (CWRU, XJTU-SY)
These can be loaded directly using the PHMD integration:
```python
python train_rmgpt.py --config configs/default_config.yaml --task diagnosis --dataset CWRU
python train_rmgpt.py --config configs/default_config.yaml --task prognosis --dataset XJTU-SY
```

### Custom Datasets (SLIET, QPZZ-II, SMU)
These require the datasets to be placed in `./data/raw/` with the following structure:

```
data/raw/
├── sliet/
│   ├── normal/
│   ├── inner_race/
│   ├── outer_race/
│   └── ball/
├── qpzz-ii/qpzz-ii/
│   ├── normal/
│   ├── tooth_wear/
│   ├── tooth_breakage/
│   └── pitting/
└── smu/
    ├── normal/
    ├── tooth_wear/
    ├── tooth_breakage/
    └── crack/
```

The custom loaders in `data/custom_datasets.py` will automatically detect and load signals from these directories.

**Note**: The actual file structure may vary. You may need to adjust the loaders in `data/custom_datasets.py` based on the actual dataset organization.

## Download Scripts

Download scripts are available in `data/`:
- `data/download_cwru.py` - CWRU dataset
- `data/download_sliet.py` - SLIET dataset
- `data/download_qpzz2.py` - QPZZ-II dataset
- `data/download_smu.py` - SMU dataset
- `data/download_xjtu.py` - XJTU-SY dataset

Run `python data/download_all.py` to download all datasets.

## Configuration

Use `configs/rmgpt_paper_datasets.yaml` for paper-specific configuration, or modify `configs/default_config.yaml` to specify the dataset:

```yaml
data:
  dataset_name: "SLIET"  # or QPZZ-II, SMU
  task_name: "Diagnosis"
  data_dir: "./data/raw"
```

## Implementation Status

✅ **CWRU** - Fully supported via PHMD  
✅ **XJTU-SY** - Fully supported via PHMD  
⚠️ **SLIET** - Custom loader implemented, requires dataset files  
⚠️ **QPZZ-II** - Custom loader implemented, requires dataset files  
⚠️ **SMU** - Custom loader implemented, requires dataset files

The custom loaders are template implementations that should work with common dataset structures, but may need adjustment based on the actual file format and organization of these datasets.

## Paper Hyperparameters

**EXACT hyperparameters from the paper are now configured** in `configs/paper_exact_config.yaml`:

- **Batch Size**: 256
- **Learning Rate**: 3.00 × 10^-7
- **Pretraining Epochs**: 20
- **Finetuning Epochs**: 3
- **Prompt Learning Epochs**: 5
- **Tokenizer Patch Length (P)**: 256
- **Tokenizer Stride (S)**: 256
- **Transformer Layers**: 4
- **Hidden Size (d)**: 512
- **Prompt Token Length (lp)**: 10
- **Fault Token Length (lt)**: 1
- **Total Model Parameters**: 68.50M

Use `configs/paper_exact_config.yaml` to reproduce the paper exactly!
