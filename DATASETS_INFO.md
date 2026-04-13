# RmGPT Paper Datasets

## Summary

**The implementation I created is generic and can work with any PHMD dataset**, but the exact datasets used in the original RmGPT paper are not explicitly specified in my implementation. 

## Implementation Status

### Current Configuration
- **Default**: Uses `CWRU` dataset only
- **Capability**: Can load **any dataset from the PHMD library** (85+ datasets available)

### What the Paper Likely Used

Based on common PHM benchmark practices and foundation model papers, the RmGPT paper likely used:

#### Diagnosis Datasets (Fault Classification):
1. **CWRU** (Case Western Reserve University Bearing)
2. **PUBD16** (Paderborn University Bearing) 
3. **MFPT** (Mechanical Fault Prevention Test)
4. Possibly others: NMILL, KAUG17, UPM20

#### Prognosis Datasets (RUL Prediction):
1. **CMAPSS** / **NCMAPSS** (NASA Aircraft Engine)
2. **PRONOSTIA** (FEMTO-ST Bearing)
3. **IMS** (Intelligent Maintenance Systems)
4. Possibly others: PHME20, ARAMIS20, CALCE_CS2, XJTU-SY

## Available Datasets in PHMD Library

The PHMD library in your codebase supports **85+ datasets**, including all the likely candidates above. You can check available datasets by running:

```python
import phmd
from phmd import datasets
phmd.datasets.Dataset.search()
```

## To Match Paper Exactly

To reproduce the paper's experiments exactly, you would need to:

1. **Identify the exact datasets** from the paper (may require checking experimental sections)
2. **Use the same data splits** (train/val/test, cross-validation folds)
3. **Apply the same preprocessing** (normalization, windowing, etc.)
4. **Match the same experimental protocol** (few-shot, cross-domain, etc.)

## Using Different Datasets

To use a different dataset, simply modify `configs/default_config.yaml`:

```yaml
data:
  dataset_name: "CMAPSS"  # Change this
  task_name: "Prognosis"   # Change this
  fold: 0
```

The implementation will automatically:
- Load the dataset via PHMD
- Extract signals and labels/RUL
- Preprocess according to config settings

## Next Steps

1. **Check the paper** for the exact list of datasets used in their experiments
2. **Update the config** to match those datasets
3. **Run experiments** to reproduce their results

If you can identify the exact datasets from the paper, I can help create specific configurations for each dataset!
