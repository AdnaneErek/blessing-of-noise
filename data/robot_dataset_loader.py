"""
Robot Fault Diagnosis Dataset Loader

Loads robot fault diagnosis data from .mat files and extracts features including:
- Motor commands (Motor1Cmd - Motor5Cmd) - from timeseries objects
- Desired trajectory (x, y, z) - from trajCmds
- Realized trajectory (x, y, z) - from trajResps
- Error features (e_x, e_y, e_z) = Realized - Desired
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import scipy.io
from sklearn.model_selection import train_test_split
import glob
import os


def extract_motor_commands_from_timeseries(motor_cmds_ts):
    """
    Extract motor command values from MATLAB timeseries object
    
    Args:
        motor_cmds_ts: Array of MATLAB timeseries objects (5 motors)
        
    Returns:
        motor_commands: [seq_len, 5] numpy array
    """
    # motor_cmds_ts is an array of 5 timeseries objects
    # Each timeseries has data and time
    motor_data = []
    
    for i, ts_obj in enumerate(motor_cmds_ts):
        # Try to extract data from timeseries
        # MATLAB timeseries structure: has 'Data' and 'Time' fields
        try:
            # Access the timeseries data
            arr = np.array(ts_obj)
            # The timeseries data might be in the structure
            # For now, we'll need to handle this differently
            # Let's try accessing via the array structure
            if arr.dtype.names:
                # Structured array - might have 'Data' field
                if 'Data' in arr.dtype.names:
                    data = arr['Data']
                else:
                    # Try to get data from the structure
                    data = arr[0] if len(arr) > 0 else None
            else:
                data = arr
            motor_data.append(data)
        except Exception as e:
            print(f"Warning: Could not extract motor {i+1} command: {e}")
            # Use zeros as fallback
            motor_data.append(np.zeros(1000))  # Default length
    
    # Stack into [5, seq_len] then transpose to [seq_len, 5]
    if len(motor_data) > 0:
        # Find max length
        max_len = max(len(d) if hasattr(d, '__len__') else 1000 for d in motor_data)
        motor_array = np.zeros((max_len, 5))
        
        for i, data in enumerate(motor_data):
            if hasattr(data, '__len__'):
                data_len = len(data)
                motor_array[:data_len, i] = data[:data_len] if data_len <= max_len else data[:max_len]
        
        return motor_array
    else:
        # Return zeros if extraction failed
        return np.zeros((1000, 5))


def load_robot_dataset_from_individual_files(data_dir: str, folder_name: str = '20241017', test_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load robot dataset from individual .mat files in fault type folders
    
    This is more reliable than trying to parse the consolidated training_dataset.mat
    which contains complex MATLAB table structures.
    
    Args:
        data_dir: Base data directory
        folder_name: Dataset folder name
        test_data: If True, load from testDatasets instead of trainingDatasets
        
    Returns:
        signals: [num_samples, seq_len, num_features] numpy array
        labels: [num_samples] numpy array of integer labels
    """
    if test_data:
        base_path = Path(data_dir) / 'testDatasets' / folder_name
    else:
        base_path = Path(data_dir) / 'trainingDatasets' / folder_name
    
    # Define fault types (folders)
    fault_types = [
        'Healthy',
        'Motor_1_Stuck',
        'Motor_2_Stuck',
        'Motor_3_Stuck',
        'Motor_4_Stuck',
        'Motor_1_Steady_state_error',
        'Motor_2_Steady_state_error',
        'Motor_3_Steady_state_error',
        'Motor_4_Steady_state_error'
    ]
    
    signals_list = []
    labels_list = []
    
    for fault_idx, fault_type in enumerate(fault_types):
        fault_dir = base_path / fault_type
        if not fault_dir.exists():
            print(f"Warning: Fault type folder not found: {fault_dir}")
            continue
        
        # Load all dataset_*.mat files (skip hidden_dataset_*.mat)
        mat_files = sorted([f for f in os.listdir(fault_dir) 
                           if f.startswith('dataset_') and f.endswith('.mat')])
        
        print(f"Loading {len(mat_files)} files from {fault_type}...")
        
        for mat_file in mat_files:
            mat_path = fault_dir / mat_file
            try:
                mat = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
                
                # Extract trajectory data
                traj_cmds = mat.get('trajCmds', None)  # Desired trajectory [seq_len, 3]
                traj_resps = mat.get('trajResps', None)  # Realized trajectory [seq_len, 3]
                
                if traj_cmds is None or traj_resps is None:
                    print(f"Warning: Missing trajectory data in {mat_file}")
                    continue
                
                # Ensure they're numpy arrays
                traj_cmds = np.array(traj_cmds)
                traj_resps = np.array(traj_resps)
                
                # Calculate error features: e = Realized - Desired
                error_traj = traj_resps - traj_cmds
                
                # For now, use only trajectory features (9 features total)
                # Motor commands extraction is complex (MATLAB timeseries), can add later
                # Features: Desired xyz (3), Realized xyz (3), Error xyz (3) = 9 features
                features = np.concatenate([
                    traj_cmds,      # Desired trajectory [seq_len, 3]
                    traj_resps,     # Realized trajectory [seq_len, 3]
                    error_traj      # Error trajectory [seq_len, 3]
                ], axis=1)  # [seq_len, 9]
                
                signals_list.append(features)
                labels_list.append(fault_idx)
                
            except Exception as e:
                print(f"Warning: Could not load {mat_file}: {e}")
                continue
    
    if len(signals_list) == 0:
        raise ValueError(f"No valid signals loaded from {base_path}")
    
    # Pad/truncate to same length
    seq_lengths = [s.shape[0] for s in signals_list]
    target_length = max(seq_lengths)  # Use max length
    num_features = signals_list[0].shape[1]
    
    signals_array = np.zeros((len(signals_list), target_length, num_features))
    for i, signal in enumerate(signals_list):
        seq_len = signal.shape[0]
        if seq_len <= target_length:
            signals_array[i, :seq_len, :] = signal
        else:
            signals_array[i, :, :] = signal[:target_length, :]
    
    labels = np.array(labels_list)
    
    print(f"\nLoaded {len(signals_array)} samples")
    print(f"Signal shape: {signals_array.shape}")
    print(f"Number of classes: {len(fault_types)}")
    print(f"Features: DesiredTraj xyz (3), RealizedTraj xyz (3), Error xyz (3) = 9 total")
    print(f"Note: Motor commands not included (MATLAB timeseries extraction complex)")
    
    return signals_array, labels


def extract_features_with_errors(data_tables: List[pd.DataFrame]) -> np.ndarray:
    """
    Extract features from data tables including error features
    
    Features extracted:
    - Motor1Cmd, Motor2Cmd, Motor3Cmd, Motor4Cmd, Motor5Cmd (5 features)
    - DesiredTrajectory-x, DesiredTrajectory-y, DesiredTrajectory-z (3 features)
    - RealizedTrajectory-x, RealizedTrajectory-y, RealizedTrajectory-z (3 features)
    - e_x, e_y, e_z = Realized - Desired (3 error features)
    
    Total: 14 features
    
    Args:
        data_tables: List of DataFrames
        
    Returns:
        signals: [num_samples, seq_len, 14] numpy array
    """
    signals_list = []
    
    for df in data_tables:
        # Extract motor commands
        motor_cols = [f'Motor{i}Cmd' for i in range(1, 6)]
        motor_data = df[motor_cols].values if all(col in df.columns for col in motor_cols) else None
        
        # Extract desired trajectory
        desired_cols = ['DesiredTrajectory-x', 'DesiredTrajectory-y', 'DesiredTrajectory-z']
        desired_data = df[desired_cols].values if all(col in df.columns for col in desired_cols) else None
        
        # Extract realized trajectory
        realized_cols = ['RealizedTrajectory-x', 'RealizedTrajectory-y', 'RealizedTrajectory-z']
        realized_data = df[realized_cols].values if all(col in df.columns for col in realized_cols) else None
        
        if motor_data is None or desired_data is None or realized_data is None:
            print(f"Warning: Missing columns in DataFrame. Available: {df.columns.tolist()}")
            continue
        
        # Calculate error features: e = Realized - Desired
        error_data = realized_data - desired_data
        
        # Concatenate all features: [seq_len, 14]
        # Order: Motor1-5, Desired xyz, Realized xyz, Error xyz
        features = np.concatenate([
            motor_data,           # 5 features
            desired_data,         # 3 features
            realized_data,        # 3 features
            error_data            # 3 features (e_x, e_y, e_z)
        ], axis=1)
        
        signals_list.append(features)
    
    # Convert to numpy array: [num_samples, seq_len, num_features]
    # Pad or truncate to same length
    if len(signals_list) == 0:
        raise ValueError("No valid signals extracted from data tables")
    
    seq_lengths = [s.shape[0] for s in signals_list]
    target_length = max(seq_lengths)  # Use max length, or could use median/mean
    
    # Pad shorter sequences
    num_features = signals_list[0].shape[1]
    signals_array = np.zeros((len(signals_list), target_length, num_features))
    
    for i, signal in enumerate(signals_list):
        seq_len = signal.shape[0]
        if seq_len <= target_length:
            signals_array[i, :seq_len, :] = signal
        else:
            # Truncate if longer
            signals_array[i, :, :] = signal[:target_length, :]
    
    return signals_array


def load_robot_training_data(data_dir: str, folder_name: str = None, use_individual_files: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load robot training dataset from one or multiple folders
    
    Args:
        data_dir: Base data directory (e.g., 'data/raw/dataset')
        folder_name: Training dataset folder name (e.g., '20241017') or list of folder names.
                    If None, automatically discovers all folders in trainingDatasets
        use_individual_files: If True, load from individual .mat files (more reliable)
                            If False, try to load from consolidated training_dataset.mat
        
    Returns:
        signals: [num_samples, seq_len, num_features] numpy array
        labels: [num_samples] numpy array of integer labels
    """
    training_datasets_path = Path(data_dir) / 'trainingDatasets'
    
    # Determine which folders to load from
    if folder_name is None:
        # Auto-discover all folders in trainingDatasets
        folder_names = [d.name for d in training_datasets_path.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        print(f"Auto-discovered {len(folder_names)} training folders: {folder_names}")
    elif isinstance(folder_name, str):
        folder_names = [folder_name]
    elif isinstance(folder_name, list):
        folder_names = folder_name
    else:
        raise ValueError(f"folder_name must be str, list, or None, got {type(folder_name)}")
    
    if use_individual_files:
        # Load from individual .mat files (more reliable)
        all_signals_list = []
        all_labels_list = []
        
        for folder in folder_names:
            print(f"\nLoading data from folder: {folder}")
            try:
                signals, labels = load_robot_dataset_from_individual_files(data_dir, folder)
                all_signals_list.append(signals)
                all_labels_list.append(labels)
                print(f"  Loaded {len(signals)} samples from {folder}")
            except Exception as e:
                print(f"  Warning: Failed to load from {folder}: {e}")
                print(f"  Skipping folder {folder}")
                continue
        
        if len(all_signals_list) == 0:
            raise ValueError(f"No valid data loaded from any folder: {folder_names}")
        
        # Concatenate all signals and labels
        all_signals = np.concatenate(all_signals_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        
        print(f"\nTotal loaded: {len(all_signals)} samples from {len(folder_names)} folder(s)")
        print(f"Signal shape: {all_signals.shape}")
        print(f"Labels shape: {all_labels.shape}")
        
        return all_signals, all_labels
    else:
        # Try to load from consolidated .mat files (one per folder)
        all_signals_list = []
        all_labels_list = []
        
        for folder in folder_names:
            mat_path = training_datasets_path / folder / 'training_dataset.mat'
            
            if not mat_path.exists():
                print(f"Warning: Consolidated file not found: {mat_path}")
                continue
            
            try:
                # Load data
                data_tables, labels_str = load_robot_dataset_mat(str(mat_path))
                
                # Extract features
                signals = extract_features_with_errors(data_tables)
                
                # Map string labels to integers
                unique_labels = sorted(set(labels_str))
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                labels = np.array([label_map[label] for label in labels_str])
                
                all_signals_list.append(signals)
                all_labels_list.append(labels)
                
                print(f"Loaded {len(signals)} samples from {folder}")
            except Exception as e:
                print(f"Warning: Failed to load from {folder}: {e}")
                continue
        
        if len(all_signals_list) == 0:
            raise ValueError(f"No valid data loaded from any folder: {folder_names}")
        
        # Concatenate all signals and labels
        all_signals = np.concatenate(all_signals_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        
        print(f"\nTotal loaded: {len(all_signals)} samples from {len(folder_names)} folder(s)")
        print(f"Signal shape: {all_signals.shape}")
        
        return all_signals, all_labels


def load_robot_test_data(data_dir: str, folder_name: str = '20241016', use_individual_files: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load robot test dataset (real robot data)
    
    Args:
        data_dir: Base data directory
        folder_name: Test dataset folder name
        use_individual_files: If True, load from individual files (preferred, more reliable)
        
    Returns:
        signals: [num_samples, seq_len, num_features] numpy array
        labels: [num_samples] numpy array of integer labels
    """
    base_path = Path(data_dir) / 'testDatasets' / folder_name
    
    # Try individual files first (more reliable, same structure as training)
    if use_individual_files:
        # Check if individual folders exist (same structure as training)
        fault_types = [
            'Healthy',
            'Motor_1_Stuck',
            'Motor_2_Stuck',
            'Motor_3_Stuck',
            'Motor_4_Stuck',
            'Motor_1_Steady_state_error',
            'Motor_2_Steady_state_error',
            'Motor_3_Steady_state_error',
            'Motor_4_Steady_state_error'
        ]
        
        # Check if at least one fault folder exists
        if any((base_path / fault_type).exists() for fault_type in fault_types):
            print(f"Loading test data from individual files in {base_path}...")
            return load_robot_dataset_from_individual_files(data_dir, folder_name=folder_name, test_data=True)
        else:
            print(f"Warning: Individual fault folders not found in {base_path}")
    
    # Fallback: try consolidated file
    mat_path = base_path / 'real_testing_dataset.mat'
    if mat_path.exists():
        print(f"Warning: Consolidated file loading not fully implemented.")
        print(f"Please ensure individual files are available in {base_path}")
        raise NotImplementedError("Consolidated file loading requires additional implementation. Please use individual files.")
    
    raise FileNotFoundError(f"Test dataset not found: {base_path}. Expected individual folders or {mat_path}")


def load_robot_finetuning_data(data_dir: str, folder_name: str = None, use_individual_files: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load robot fine-tuning dataset from finetuningDatasets folder (real robot data)
    
    This function loads real robot data for fine-tuning a model trained on simulation data.
    It handles folders that may have different numbers of fault types (e.g., some folders
    may only have Healthy class).
    
    Args:
        data_dir: Base data directory (e.g., 'data/raw/dataset')
        folder_name: Fine-tuning dataset folder name (e.g., 'robot_a_20251127') or list of folder names.
                    If None, automatically discovers all folders in finetuningDatasets
        use_individual_files: If True, load from individual .mat files (more reliable)
                            If False, try to load from consolidated files
        
    Returns:
        signals: [num_samples, seq_len, num_features] numpy array
        labels: [num_samples] numpy array of integer labels
    """
    finetuning_datasets_path = Path(data_dir) / 'finetuningDatasets'
    
    # Determine which folders to load from
    if folder_name is None:
        # Auto-discover all folders in finetuningDatasets
        folder_names = [d.name for d in finetuning_datasets_path.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        print(f"Auto-discovered {len(folder_names)} fine-tuning folders: {folder_names}")
    elif isinstance(folder_name, str):
        folder_names = [folder_name]
    elif isinstance(folder_name, list):
        folder_names = folder_name
    else:
        raise ValueError(f"folder_name must be str, list, or None, got {type(folder_name)}")
    
    if use_individual_files:
        # Load from individual .mat files (more reliable)
        all_signals_list = []
        all_labels_list = []
        
        # Define all possible fault types (for consistent label mapping)
        all_fault_types = [
            'Healthy',
            'Motor_1_Stuck',
            'Motor_2_Stuck',
            'Motor_3_Stuck',
            'Motor_4_Stuck',
            'Motor_1_Steady_state_error',
            'Motor_2_Steady_state_error',
            'Motor_3_Steady_state_error',
            'Motor_4_Steady_state_error'
        ]
        fault_type_to_idx = {fault_type: idx for idx, fault_type in enumerate(all_fault_types)}
        
        for folder in folder_names:
            print(f"\nLoading fine-tuning data from folder: {folder}")
            base_path = finetuning_datasets_path / folder
            
            # Check which fault types exist in this folder
            available_fault_types = [ft for ft in all_fault_types 
                                   if (base_path / ft).exists() and (base_path / ft).is_dir()]
            
            if len(available_fault_types) == 0:
                print(f"  Warning: No fault type folders found in {folder}")
                continue
            
            print(f"  Found {len(available_fault_types)} fault types: {available_fault_types}")
            
            for fault_type in available_fault_types:
                fault_dir = base_path / fault_type
                fault_idx = fault_type_to_idx[fault_type]
                
                # Load all dataset_*.mat files (skip hidden_dataset_*.mat)
                mat_files = sorted([f for f in os.listdir(fault_dir) 
                                   if f.startswith('dataset_') and f.endswith('.mat')])
                
                print(f"  Loading {len(mat_files)} files from {fault_type}...")
                
                for mat_file in mat_files:
                    mat_path = fault_dir / mat_file
                    try:
                        mat = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
                        
                        # Extract trajectory data
                        traj_cmds = mat.get('trajCmds', None)  # Desired trajectory [seq_len, 3]
                        traj_resps = mat.get('trajResps', None)  # Realized trajectory [seq_len, 3]
                        
                        if traj_cmds is None or traj_resps is None:
                            print(f"    Warning: Missing trajectory data in {mat_file}")
                            continue
                        
                        # Ensure they're numpy arrays
                        traj_cmds = np.array(traj_cmds)
                        traj_resps = np.array(traj_resps)
                        
                        # Calculate error features: e = Realized - Desired
                        error_traj = traj_resps - traj_cmds
                        
                        # Features: Desired xyz (3), Realized xyz (3), Error xyz (3) = 9 features
                        features = np.concatenate([
                            traj_cmds,      # Desired trajectory [seq_len, 3]
                            traj_resps,     # Realized trajectory [seq_len, 3]
                            error_traj      # Error trajectory [seq_len, 3]
                        ], axis=1)  # [seq_len, 9]
                        
                        all_signals_list.append(features)
                        all_labels_list.append(fault_idx)
                        
                    except Exception as e:
                        print(f"    Warning: Could not load {mat_file}: {e}")
                        continue
            
            print(f"  Loaded data from {folder}")
        
        if len(all_signals_list) == 0:
            raise ValueError(f"No valid signals loaded from any folder: {folder_names}")
        
        # Pad/truncate to same length
        seq_lengths = [s.shape[0] for s in all_signals_list]
        target_length = max(seq_lengths) if seq_lengths else 1000
        num_features = all_signals_list[0].shape[1] if all_signals_list else 9
        
        signals_array = np.zeros((len(all_signals_list), target_length, num_features))
        for i, signal in enumerate(all_signals_list):
            seq_len = signal.shape[0]
            if seq_len <= target_length:
                signals_array[i, :seq_len, :] = signal
            else:
                signals_array[i, :, :] = signal[:target_length, :]
        
        labels = np.array(all_labels_list)
        
        print(f"\nTotal loaded: {len(signals_array)} samples from {len(folder_names)} folder(s)")
        print(f"Signal shape: {signals_array.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        print(f"Class distribution:")
        for fault_idx, fault_type in enumerate(all_fault_types):
            count = np.sum(labels == fault_idx)
            if count > 0:
                print(f"  {fault_type}: {count} samples")
        
        return signals_array, labels
    else:
        raise NotImplementedError("Consolidated file loading for finetuning data not implemented. Please use individual files.")


def split_robot_dataset(signals: np.ndarray, 
                       labels: np.ndarray,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split robot dataset into train/val/test sets
    
    Args:
        signals: [num_samples, seq_len, num_features]
        labels: [num_samples]
        test_size: Fraction for test set
        val_size: Fraction for validation (from remaining after test split)
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (signals, labels)
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        signals, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, random_state=random_state, stratify=y_trainval
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
