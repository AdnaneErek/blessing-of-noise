"""
Paper-compliant data splitting strategy for RmGPT

Implements the exact split methodology from the paper:
1. First split: 80% train, 20% test (test NEVER used in pretraining/finetuning)
2. Within 80% train:
   - Pretraining: uses all 80% (unlabeled, labels ignored)
   - Finetuning: split 80% again (e.g., 90/10) for train/val

This prevents data leakage and matches the paper's methodology.
"""
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
import pandas as pd


def paper_split_strategy(signals: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        rul: Optional[np.ndarray] = None,
                        test_size: float = 0.2,
                        finetune_val_size: float = 0.1,
                        random_state: int = 42,
                        stratify: bool = True) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Paper-compliant data splitting strategy
    
    Args:
        signals: All signals [num_samples, seq_len, num_channels]
        labels: Labels for diagnosis [num_samples] (optional)
        rul: RUL values for prognosis [num_samples] (optional)
        test_size: Test set size (default 0.2 = 20%)
        finetune_val_size: Validation size within train80 (default 0.1 = 10% of train80)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by labels (for classification)
        
    Returns:
        Dictionary with keys:
        - 'pretrain': (signals, None, None) - All 80% train data (unlabeled)
        - 'finetune_train': (signals, labels/rul, None) - 90% of train80
        - 'finetune_val': (signals, labels/rul, None) - 10% of train80
        - 'test': (signals, labels/rul, None) - 20% test (untouched)
    """
    num_samples = len(signals)
    
    # Step 1: Split into 80% train, 20% test (paper requirement)
    if stratify and labels is not None:
        # Stratified split for classification
        train80_idx, test20_idx = train_test_split(
            np.arange(num_samples),
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
    else:
        # Random split (for regression or when labels not available)
        train80_idx, test20_idx = train_test_split(
            np.arange(num_samples),
            test_size=test_size,
            random_state=random_state
        )
    
    # Extract test set (20%) - NEVER used in pretraining/finetuning
    test_signals = signals[test20_idx]
    test_labels = labels[test20_idx] if labels is not None else None
    test_rul = rul[test20_idx] if rul is not None else None
    
    # Extract train80 set (80%)
    train80_signals = signals[train80_idx]
    train80_labels = labels[train80_idx] if labels is not None else None
    train80_rul = rul[train80_idx] if rul is not None else None
    
    # Step 2: Split train80 for finetuning (90/10 split within train80)
    if stratify and train80_labels is not None:
        finetune_train_idx, finetune_val_idx = train_test_split(
            np.arange(len(train80_signals)),
            test_size=finetune_val_size,
            stratify=train80_labels,
            random_state=random_state
        )
    else:
        finetune_train_idx, finetune_val_idx = train_test_split(
            np.arange(len(train80_signals)),
            test_size=finetune_val_size,
            random_state=random_state
        )
    
    # Finetuning train (90% of train80)
    finetune_train_signals = train80_signals[finetune_train_idx]
    finetune_train_labels = train80_labels[finetune_train_idx] if train80_labels is not None else None
    finetune_train_rul = train80_rul[finetune_train_idx] if train80_rul is not None else None
    
    # Finetuning val (10% of train80)
    finetune_val_signals = train80_signals[finetune_val_idx]
    finetune_val_labels = train80_labels[finetune_val_idx] if train80_labels is not None else None
    finetune_val_rul = train80_rul[finetune_val_idx] if train80_rul is not None else None
    
    # Pretraining uses ALL of train80 (unlabeled)
    pretrain_signals = train80_signals  # All 80%
    # Labels are ignored for pretraining (self-supervised)
    
    return {
        'pretrain': (pretrain_signals, None, None),
        'finetune_train': (finetune_train_signals, finetune_train_labels, finetune_train_rul),
        'finetune_val': (finetune_val_signals, finetune_val_labels, finetune_val_rul),
        'test': (test_signals, test_labels, test_rul)
    }


def paper_split_from_phmd(dataset_name: str,
                          task_name: str = None,
                          test_size: float = 0.2,
                          finetune_val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Load dataset from PHMD and apply paper-compliant split
    
    Args:
        dataset_name: Dataset name
        task_name: Task name (e.g., 'Diagnosis', 'Prognosis', 'fault', 'rul') 
                   If None, uses first available task
        test_size: Test set size (default 0.2)
        finetune_val_size: Validation size within train80 (default 0.1)
        random_state: Random seed
        
    Returns:
        Dictionary with paper-compliant splits
    """
    from .dataset import load_phmd_dataset
    
    # Load ALL data first (no split)
    # We'll do our own split following paper methodology
    try:
        import phmd
        from phmd import datasets
        
        dataset = datasets.Dataset(dataset_name)
        
        # Map common task names to PHMD task names
        available_task_names = [t.name for t in dataset.tasks]
        task_name_map = {
            'Diagnosis': 'fault',  # CWRU uses 'fault' not 'Diagnosis'
            'diagnosis': 'fault',
            'Prognosis': 'rul',
            'prognosis': 'rul'
        }
        
        # If task_name is a mapped name, use the actual PHMD task name
        actual_task_name = task_name_map.get(task_name, task_name) if task_name else None
        
        # Try to find the task
        if actual_task_name and actual_task_name in available_task_names:
            task = dataset[actual_task_name]
        elif task_name and task_name in available_task_names:
            task = dataset[task_name]
        else:
            # Use first available task as fallback
            task = dataset.tasks[0]
            print(f"Warning: Task '{task_name}' not found. Using first available task: '{task.name}'")
        
        # Load full dataset without splitting
        load_result = task.load()  # Load all data - returns tuple, list, or DataFrame
        
        # Handle different return types
        # task.load() can return: tuple, list, or DataFrame
        # Tuple might contain: (DataFrame,) or (list,) where list contains DataFrames
        # List might contain: [DataFrame, ...] or other structures
        
        full_data = None
        
        if isinstance(load_result, tuple):
            if len(load_result) > 0:
                first_item = load_result[0]
                # If first item is DataFrame, use it
                if isinstance(first_item, pd.DataFrame):
                    full_data = first_item
                # If first item is a list, check what's inside
                elif isinstance(first_item, list):
                    if len(first_item) > 0 and isinstance(first_item[0], pd.DataFrame):
                        # List of DataFrames - concatenate
                        full_data = pd.concat(first_item, ignore_index=True)
                    else:
                        # List of other things - try to use first element of tuple as DataFrame
                        full_data = pd.DataFrame(first_item) if len(first_item) > 0 else None
                else:
                    # Try to convert first item to DataFrame
                    try:
                        full_data = pd.DataFrame(first_item)
                    except:
                        pass
            if full_data is None:
                raise ValueError(f"task.load() returned tuple but couldn't extract DataFrame for {dataset_name}")
        elif isinstance(load_result, list):
            if len(load_result) > 0:
                # Check if list contains DataFrames
                if all(isinstance(item, pd.DataFrame) for item in load_result):
                    # List of DataFrames - concatenate
                    full_data = pd.concat(load_result, ignore_index=True)
                elif isinstance(load_result[0], pd.DataFrame):
                    # At least first is DataFrame
                    full_data = load_result[0]
                else:
                    # List doesn't contain DataFrames - try to convert
                    try:
                        full_data = pd.DataFrame(load_result)
                    except:
                        # Try first element
                        if len(load_result) > 0:
                            try:
                                full_data = pd.DataFrame(load_result[0])
                            except:
                                pass
            if full_data is None:
                raise ValueError(f"task.load() returned list but couldn't extract DataFrame for {dataset_name}")
        elif isinstance(load_result, pd.DataFrame):
            full_data = load_result
        else:
            # Try to convert to DataFrame
            try:
                full_data = pd.DataFrame(load_result)
            except Exception as e:
                raise TypeError(f"Cannot convert load_result to DataFrame. Type: {type(load_result)} for {dataset_name}. Error: {e}")
        
        # Final check
        if not isinstance(full_data, pd.DataFrame):
            raise TypeError(f"Failed to get DataFrame. Got {type(full_data)} for {dataset_name}")
        
        # Extract features and target
        feature_cols = task.features
        signal_cols = [col for col in full_data.columns if col in feature_cols]
        
        # Get target column - task.target returns the key name ('target' or 'target_id'), 
        # not the actual column name. The actual column name is in task.meta['target'] or task.meta['target_id']
        # Use task.meta to get the actual column name
        if 'target_id' in task.meta:
            target_col = task.meta['target_id']
        elif 'target' in task.meta:
            # target might be a list or string
            target_val = task.meta['target']
            if isinstance(target_val, list):
                target_col = target_val[0]  # Use first target if list
            else:
                target_col = target_val
        else:
            # Fallback to task.target (which might be 'target' or 'target_id')
            target_col = task.target
        
        # Debug: Check task metadata
        task_type_meta = task.meta.get('type', '')
        print(f"DEBUG: task.meta['type'] = {task_type_meta}")
        print(f"DEBUG: task.target = {task.target}")
        print(f"DEBUG: task.meta.get('target') = {task.meta.get('target')}")
        print(f"DEBUG: Using target_col = {target_col}")
        print(f"DEBUG: Available columns: {list(full_data.columns)[:10]}...")  # Show first 10 columns
        targets = full_data[target_col].values if target_col in full_data.columns else None
        print(f"DEBUG: targets is None? {targets is None}")
        print(f"DEBUG: target_col in full_data.columns? {target_col in full_data.columns if hasattr(full_data, 'columns') else 'N/A'}")
        
        # Group by unit identifier
        identifier_cols = task.meta.get('identifier', [])
        if identifier_cols:
            signals_list = []
            is_classification = task.meta.get('type', '').startswith('classification')
            is_regression = task.meta.get('type') == 'regression'
            labels_list = [] if is_classification else None
            rul_list = [] if is_regression else None
            print(f"DEBUG: is_classification = {is_classification}, is_regression = {is_regression}")
            print(f"DEBUG: labels_list initialized as: {type(labels_list)}")
            
            for unit_id, group in full_data.groupby(identifier_cols):
                unit_signals = group[signal_cols].values
                if len(unit_signals) > 0:
                    signals_list.append(unit_signals)
                    
                    if labels_list is not None and targets is not None:
                        unit_labels = group[target_col].values if target_col in group.columns else None
                        if unit_labels is not None and len(unit_labels) > 0:
                            labels_list.append(unit_labels[0])  # Use first label for unit
                    
                    if rul_list is not None and targets is not None:
                        unit_rul = group[target_col].values if target_col in group.columns else None
                        if unit_rul is not None and len(unit_rul) > 0:
                            rul_list.append(unit_rul[-1])  # Use last RUL for unit
            
            if len(signals_list) > 0:
                # Apply paper preprocessing: create sliding windows of 2048 timesteps
                from .windowing import create_sliding_windows
                
                window_length = 2048  # Paper standard: 2048 time steps per window
                
                # Create sliding windows from each unit's long signal
                # signals_list: list of [timesteps, channels] arrays (can be very long, e.g., 204M points)
                windowed_signals_list = []
                windowed_labels_list = []
                windowed_rul_list = []
                
                for i, signal in enumerate(signals_list):
                    # signal: [timesteps, channels] - can be very long (e.g., 204M points for XJTU-SY)
                    # Create multiple 2048-length windows from this long sequence
                    windows = create_sliding_windows(signal, window_length=window_length, 
                                                   stride=window_length, overlap=0.0)
                    # windows: [num_windows_from_this_unit, 2048, channels]
                    
                    windowed_signals_list.append(windows)
                    
                    # Assign same label/RUL to all windows from this unit
                    if labels_list is not None and i < len(labels_list):
                        windowed_labels_list.extend([labels_list[i]] * len(windows))
                    
                    if rul_list is not None and i < len(rul_list):
                        windowed_rul_list.extend([rul_list[i]] * len(windows))
                
                # Concatenate all windows from all units
                if len(windowed_signals_list) > 0:
                    signals_padded = np.concatenate(windowed_signals_list, axis=0)
                else:
                    signals_padded = np.array(windowed_signals_list)
                
                # Update labels/rul to match windowed signals
                print(f"DEBUG: len(windowed_labels_list) = {len(windowed_labels_list) if windowed_labels_list else 0}")
                print(f"DEBUG: len(labels_list) = {len(labels_list) if labels_list else 0}")
                labels = np.array(windowed_labels_list) if windowed_labels_list else None
                rul = np.array(windowed_rul_list) if windowed_rul_list else None
                print(f"DEBUG: Final labels shape: {labels.shape if labels is not None else None}")
                
                # Apply paper split strategy
                return paper_split_strategy(
                    signals_padded,
                    labels=labels,
                    rul=rul,
                    test_size=test_size,
                    finetune_val_size=finetune_val_size,
                    random_state=random_state,
                    stratify=(labels is not None)
                )
        else:
            # No grouping - treat each row as sample
            signals = full_data[signal_cols].values
            if signals.ndim == 2:
                signals = signals[:, np.newaxis, :]  # [num_samples, 1, num_channels]
            
            labels = targets if task.meta.get('type', '').startswith('classification') else None
            rul = targets if task.meta.get('type') == 'regression' else None
            
            return paper_split_strategy(
                signals,
                labels=labels,
                rul=rul,
                test_size=test_size,
                finetune_val_size=finetune_val_size,
                random_state=random_state,
                stratify=(labels is not None)
            )
    
    except Exception as e:
        print(f"Error loading dataset with paper split: {e}")
        raise
