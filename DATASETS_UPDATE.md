# Dataset Implementation Update

## Summary

The implementation has been updated to support the **exact datasets used in the RmGPT paper** (arXiv:2409.17604v2).

## Paper Datasets

### Diagnosis (Fault Classification):
1. ✅ **CWRU** - Available in PHMD (fully supported)
2. ⚠️ **SLIET** - Custom loader implemented (requires dataset files)
3. ⚠️ **QPZZ-II** - Custom loader implemented (requires dataset files)
4. ⚠️ **SMU** - Custom loader implemented (requires dataset files)

### Prognosis (RUL Prediction):
1. ✅ **XJTU-SY** - Available in PHMD (fully supported)

## Implementation Changes

### 1. Custom Dataset Loaders (`data/custom_datasets.py`)
- Created loaders for SLIET, QPZZ-II, and SMU
- These datasets are NOT in PHMD, so custom loaders are required
- Loaders support common file formats (.csv, .txt, .mat)

### 2. Updated Dataset Loading (`data/dataset.py`)
- Enhanced `load_phmd_dataset()` to handle both PHMD and custom datasets
- Automatically routes to appropriate loader based on dataset name
- Supports custom data directories via `data_dir` parameter

### 3. Configuration Files
- Created `configs/rmgpt_paper_datasets.yaml` with paper datasets
- Updated `configs/default_config.yaml` to support custom datasets

### 4. Documentation
- Created `PAPER_DATASETS.md` with detailed dataset information
- Includes usage instructions and data structure requirements

## Usage

### For PHMD Datasets (CWRU, XJTU-SY):
```bash
# Diagnosis with CWRU
python train_rmgpt.py --config configs/default_config.yaml --task diagnosis

# Prognosis with XJTU-SY  
python train_rmgpt.py --config configs/default_config.yaml --task prognosis
```

Update `configs/default_config.yaml`:
```yaml
data:
  dataset_name: "CWRU"  # or "XJTU-SY"
  task_name: "Diagnosis"  # or "Prognosis"
```

### For Custom Datasets (SLIET, QPZZ-II, SMU):
1. Download datasets to `./data/raw/` directory
2. Organize according to expected structure (see `PAPER_DATASETS.md`)
3. Run training:

```bash
python train_rmgpt.py --config configs/default_config.yaml --task diagnosis
```

Update config:
```yaml
data:
  dataset_name: "SLIET"  # or "QPZZ-II", "SMU"
  task_name: "Diagnosis"
  data_dir: "./data/raw"
```

## Custom Loader Notes

The custom loaders are **template implementations** that:
- Load signals from fault-type folders
- Support common file formats
- Pad signals to same length
- Return labels/RUL values

**Important**: You may need to adjust the loaders in `data/custom_datasets.py` based on the actual file structure and format of your datasets.

## Status

- ✅ CWRU - Fully working via PHMD
- ✅ XJTU-SY - Fully working via PHMD  
- ⚠️ SLIET - Loader ready, needs dataset files
- ⚠️ QPZZ-II - Loader ready, needs dataset files
- ⚠️ SMU - Loader ready, needs dataset files

Once you have the dataset files and organize them correctly, all datasets should work!
