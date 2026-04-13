"""
Custom dataset loaders for RmGPT paper datasets

Datasets not available in PHMD:
- SLIET (bearing diagnosis)
- QPZZ-II (gear diagnosis) 
- SMU (gear diagnosis)

These datasets need custom loaders.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import os
import glob


def load_sliet_dataset(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SLIET bearing fault diagnosis dataset
    
    Args:
        data_dir: Directory containing SLIET dataset
        split: 'train', 'val', or 'test'
        
    Returns:
        signals: [num_samples, seq_len, num_channels]
        labels: [num_samples] - fault labels
    """
    data_path = Path(data_dir) / "sliet"
    
    # SLIET dataset structure (adjust based on actual structure)
    # Typically organized by fault type folders
    signals_list = []
    labels_list = []
    
    fault_types = ['normal', 'inner_race', 'outer_race', 'ball']  # Typical bearing faults
    label_map = {fault: idx for idx, fault in enumerate(fault_types)}
    
    # Load signals from each fault type folder
    for fault_type, label in label_map.items():
        fault_dir = data_path / fault_type
        if not fault_dir.exists():
            continue
            
        # Load signal files (adjust pattern based on actual file format)
        signal_files = glob.glob(str(fault_dir / "*.csv")) + glob.glob(str(fault_dir / "*.txt"))
        
        for file_path in signal_files:
            try:
                # Load signal data (adjust based on format)
                if file_path.endswith('.csv'):
                    signal = pd.read_csv(file_path).values
                else:
                    signal = np.loadtxt(file_path)
                
                # Ensure 2D: [seq_len, channels]
                if signal.ndim == 1:
                    signal = signal[:, np.newaxis]
                
                signals_list.append(signal)
                labels_list.append(label)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
    
    if len(signals_list) == 0:
        raise ValueError(f"No signals found in SLIET dataset at {data_path}")
    
    # Pad signals to same length or use variable length
    max_len = max(s.shape[0] for s in signals_list)
    num_channels = signals_list[0].shape[1]
    
    signals_padded = np.zeros((len(signals_list), max_len, num_channels))
    for i, signal in enumerate(signals_list):
        signals_padded[i, :signal.shape[0], :] = signal
    
    labels = np.array(labels_list)
    
    return signals_padded, labels


def load_qpzz2_dataset(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load QPZZ-II gear fault diagnosis dataset
    
    Args:
        data_dir: Directory containing QPZZ-II dataset
        split: 'train', 'val', or 'test'
        
    Returns:
        signals: [num_samples, seq_len, num_channels]
        labels: [num_samples] - fault labels
    """
    data_path = Path(data_dir) / "qpzz-ii" / "qpzz-ii"
    
    signals_list = []
    labels_list = []
    
    # QPZZ-II gear fault types (adjust based on actual structure)
    fault_types = ['normal', 'tooth_wear', 'tooth_breakage', 'pitting']
    label_map = {fault: idx for idx, fault in enumerate(fault_types)}
    
    # Load signals from each fault type
    for fault_type, label in label_map.items():
        fault_dir = data_path / fault_type
        if not fault_dir.exists():
            continue
        
        signal_files = glob.glob(str(fault_dir / "*.csv")) + glob.glob(str(fault_dir / "*.txt")) + \
                       glob.glob(str(fault_dir / "*.mat"))
        
        for file_path in signal_files:
            try:
                if file_path.endswith('.csv'):
                    signal = pd.read_csv(file_path).values
                elif file_path.endswith('.mat'):
                    # Use scipy.io.loadmat if needed
                    from scipy.io import loadmat
                    mat_data = loadmat(file_path)
                    # Extract signal (adjust key based on actual structure)
                    signal_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
                    signal = mat_data[signal_key]
                else:
                    signal = np.loadtxt(file_path)
                
                if signal.ndim == 1:
                    signal = signal[:, np.newaxis]
                
                signals_list.append(signal)
                labels_list.append(label)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
    
    if len(signals_list) == 0:
        raise ValueError(f"No signals found in QPZZ-II dataset at {data_path}")
    
    # Pad to same length
    max_len = max(s.shape[0] for s in signals_list)
    num_channels = signals_list[0].shape[1]
    
    signals_padded = np.zeros((len(signals_list), max_len, num_channels))
    for i, signal in enumerate(signals_list):
        signals_padded[i, :signal.shape[0], :] = signal
    
    labels = np.array(labels_list)
    
    return signals_padded, labels


def load_smu_dataset(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SMU gear fault diagnosis dataset
    
    Args:
        data_dir: Directory containing SMU dataset
        split: 'train', 'val', or 'test'
        
    Returns:
        signals: [num_samples, seq_len, num_channels]
        labels: [num_samples] - fault labels
    """
    data_path = Path(data_dir) / "smu"
    
    signals_list = []
    labels_list = []
    
    # SMU gear fault types (adjust based on actual structure)
    fault_types = ['normal', 'tooth_wear', 'tooth_breakage', 'crack']
    label_map = {fault: idx for idx, fault in enumerate(fault_types)}
    
    # Load signals from each fault type
    for fault_type, label in label_map.items():
        fault_dir = data_path / fault_type
        if not fault_dir.exists():
            continue
        
        signal_files = glob.glob(str(fault_dir / "*.csv")) + glob.glob(str(fault_dir / "*.txt")) + \
                       glob.glob(str(fault_dir / "*.mat"))
        
        for file_path in signal_files:
            try:
                if file_path.endswith('.csv'):
                    signal = pd.read_csv(file_path).values
                elif file_path.endswith('.mat'):
                    from scipy.io import loadmat
                    mat_data = loadmat(file_path)
                    signal_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
                    signal = mat_data[signal_key]
                else:
                    signal = np.loadtxt(file_path)
                
                if signal.ndim == 1:
                    signal = signal[:, np.newaxis]
                
                signals_list.append(signal)
                labels_list.append(label)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
    
    if len(signals_list) == 0:
        raise ValueError(f"No signals found in SMU dataset at {data_path}")
    
    # Pad to same length
    max_len = max(s.shape[0] for s in signals_list)
    num_channels = signals_list[0].shape[1]
    
    signals_padded = np.zeros((len(signals_list), max_len, num_channels))
    for i, signal in enumerate(signals_list):
        signals_padded[i, :signal.shape[0], :] = signal
    
    labels = np.array(labels_list)
    
    return signals_padded, labels


def load_xjtu_dataset(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load XJTU bearing RUL dataset for prognosis
    
    Note: XJTU-SY is available in PHMD, but this is a fallback custom loader
    
    Args:
        data_dir: Directory containing XJTU dataset
        split: 'train', 'val', or 'test'
        
    Returns:
        signals: [num_samples, seq_len, num_channels]
        rul: [num_samples] - remaining useful life
    """
    # Prefer PHMD if available
    try:
        import phmd
        from phmd import datasets
        dataset = datasets.Dataset("XJTU-SY")
        task = dataset["Prognosis"]
        sets = task.load_fold(0)
        data = sets[split]
        
        # Extract signals and RUL
        feature_cols = task.features
        signals = data[feature_cols].values
        
        # Group by unit
        identifier_cols = task.meta.get('identifier', [])
        if identifier_cols:
            # Reshape for unit-based sequences
            signals_list = []
            rul_list = []
            for unit_id, group in data.groupby(identifier_cols):
                unit_signals = group[feature_cols].values
                if len(unit_signals) > 0:
                    signals_list.append(unit_signals)
                    rul_list.append(group['rul'].values[-1] if 'rul' in group.columns else 0)
            
            if len(signals_list) > 0:
                max_len = max(s.shape[0] for s in signals_list)
                num_channels = signals_list[0].shape[1]
                signals_padded = np.zeros((len(signals_list), max_len, num_channels))
                for i, signal in enumerate(signals_list):
                    signals_padded[i, :signal.shape[0], :] = signal
                return signals_padded, np.array(rul_list)
    except Exception as e:
        print(f"PHMD loading failed, trying custom loader: {e}")
    
    # Custom loader fallback
    data_path = Path(data_dir) / "xjtu-sy" / "xjtu-sy"
    
    # Load from raw data files
    # This is a placeholder - adjust based on actual XJTU data structure
    signals_list = []
    rul_list = []
    
    # Implementation depends on actual file structure
    # This is a template that should be adjusted
    
    if len(signals_list) == 0:
        raise ValueError(f"No signals found in XJTU dataset at {data_path}")
    
    max_len = max(s.shape[0] for s in signals_list)
    num_channels = signals_list[0].shape[1]
    signals_padded = np.zeros((len(signals_list), max_len, num_channels))
    for i, signal in enumerate(signals_list):
        signals_padded[i, :signal.shape[0], :] = signal
    
    return signals_padded, np.array(rul_list)
