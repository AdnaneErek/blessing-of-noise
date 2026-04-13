# Robot Fault Diagnosis Dataset Setup

## Overview

Fine-tune pretrained RmGPT on robot fault diagnosis dataset, following the same approach as the LSTM-based method in `trainOnSimulationData.md` and `trainSimTestReal.md`, but using RmGPT instead of LSTM.

## Dataset Structure

- **Training Data**: Simulation data from digital twin (`data/raw/dataset/trainingDatasets/20241017/`)
- **Test Data**: Real robot data (`data/raw/dataset/testDatasets/20241016/`)
- **Classes**: 9 fault types
  - Healthy
  - Motor_1_Stuck, Motor_2_Stuck, Motor_3_Stuck, Motor_4_Stuck
  - Motor_1_Steady_state_error, Motor_2_Steady_state_error, Motor_3_Steady_state_error, Motor_4_Steady_state_error

## Features Extracted

From each sample's `.mat` file:
- **DesiredTrajectory** (x, y, z): 3 features - from `trajCmds` (digital twin simulation)
- **RealizedTrajectory** (x, y, z): 3 features - from `trajResps` (actual measurements)
- **Error Features** (e_x, e_y, e_z): 3 features - calculated as `Realized - Desired`

**Total: 9 features per time step**

**Note**: Motor commands (Motor1Cmd-5Cmd) are stored as MATLAB timeseries objects which are complex to extract. They can be added later if needed. The trajectory and error features are the key discriminative features according to the original paper.

## Data Loading

The dataset is loaded from individual `.mat` files in fault type folders:
- `data/raw/dataset/trainingDatasets/20241017/{FaultType}/dataset_*.mat`

Each `.mat` file contains:
- `trajCmds`: [1000, 3] - Desired trajectory
- `trajResps`: [1000, 3] - Realized trajectory
- `motorCmdsRadius`: MATLAB timeseries objects (5 motors)

## Fine-tuning Configuration

Uses the same aggressive strategy as CWRU:
- **Backbone LR**: 1.5e-5 (preserve pretrained features)
- **Head LR**: 5.0e-3 (aggressive learning for new head)
- **Label Smoothing**: 0.1
- **Focal Loss**: Enabled (alpha=0.25, gamma=2.0)
- **Epochs**: 200
- **Improved Diagnosis Head**: Enabled (deeper architecture)

## Usage

### 1. Load and Test Data

```python
from data.robot_dataset_loader import load_robot_training_data
signals, labels = load_robot_training_data('data/raw/dataset', '20241017')
print(f"Signals shape: {signals.shape}")  # [3600, 1000, 9]
print(f"Labels shape: {labels.shape}")    # [3600]
```

### 2. Run Fine-tuning

```bash
bash finetune_robot.sh
```

Or manually:
```bash
python train_rmgpt.py \
    --config configs/finetune_robot.yaml \
    --task diagnosis \
    --dataset ROBOT \
    --resume checkpoints/final_model_pretrain.pt
```

## Expected Results

Based on the original LSTM results:
- **Simulation test**: ~89% accuracy (with digital twin features)
- **Real robot test**: ~63% accuracy (domain shift)

With RmGPT's pretrained features and aggressive fine-tuning, we expect:
- Better generalization from simulation to real data
- Potentially higher accuracy on both simulation and real data

## Files Created

1. **`data/robot_dataset_loader.py`**: Data loader for robot dataset
2. **`configs/finetune_robot.yaml`**: Fine-tuning configuration
3. **`finetune_robot.sh`**: Training script
4. **`ROBOT_DATASET_SETUP.md`**: This documentation

## Next Steps

1. Run fine-tuning and evaluate on test set
2. Compare with LSTM baseline results
3. If needed, add motor command features (requires MATLAB timeseries extraction)
4. Experiment with different feature combinations
