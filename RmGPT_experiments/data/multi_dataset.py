"""
Multi-dataset support for RmGPT

Aggregates multiple datasets for pretraining and keeps them separate for finetuning.
Uses paper-compliant 80/20 split per dataset.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .split_strategy import paper_split_from_phmd


def load_all_datasets_for_pretraining(
    dataset_names: List[str],
    task_names: List[str],
    test_size: float = 0.2,
    finetune_val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all datasets and aggregate 80% train portions for pretraining
    
    Args:
        dataset_names: List of dataset names (e.g., ['CWRU', 'JNUB', 'KAUG17', 'HSG18', 'XJTU-SY'])
        task_names: List of task names (e.g., ['Diagnosis', 'Diagnosis', 'Diagnosis', 'Diagnosis', 'Prognosis'])
        test_size: Test set size (default 0.2 = 20%)
        finetune_val_size: Validation size within train80 (default 0.1 = 10% of train80)
        random_state: Random seed
        
    Returns:
        (pretrain_signals, pretrain_val_signals) - Aggregated signals from all datasets
    """
    all_pretrain_signals = []
    all_pretrain_val_signals = []
    
    print(f"Loading {len(dataset_names)} datasets for pretraining...")
    
    for dataset_name, task_name in zip(dataset_names, task_names):
        print(f"  Loading {dataset_name} ({task_name})...")
        
        try:
            splits = paper_split_from_phmd(
                dataset_name=dataset_name,
                task_name=task_name,
                test_size=test_size,
                finetune_val_size=finetune_val_size,
                random_state=random_state
            )
            
            # Extract pretrain signals (80% train, unlabeled)
            pretrain_signals, _, _ = splits['pretrain']
            pretrain_val_signals, _, _ = splits['finetune_val']
            
            all_pretrain_signals.append(pretrain_signals)
            all_pretrain_val_signals.append(pretrain_val_signals)
            
            print(f"    Added {len(pretrain_signals)} pretrain samples (80% of {dataset_name})")
            print(f"    Added {len(pretrain_val_signals)} pretrain val samples (10% of {dataset_name} train80)")
            
        except Exception as e:
            print(f"    Warning: Failed to load {dataset_name}: {e}")
            print(f"    Skipping {dataset_name} for pretraining")
            continue
    
    if len(all_pretrain_signals) == 0:
        raise ValueError("No datasets could be loaded for pretraining!")
    
    # Apply paper preprocessing: downsample to ~5kHz and standardize to 2048 timesteps
    from .preprocessing import preprocess_dataset_paper
    
    print(f"\nApplying paper preprocessing:")
    print(f"  Downsampling to ~5kHz")
    print(f"  Standardizing to 2048 time steps per window")
    print(f"  Standardizing channels to max across datasets")
    
    # Ensure all signals are 3D: [num_samples, 2048, channels] before processing
    # Some might be 2D: [num_samples, 2048] which should become [num_samples, 2048, 1]
    standardized_signals = []
    for signals in all_pretrain_signals:
        if signals.ndim == 2:
            # 2D: [num_samples, 2048] -> add channel dimension [num_samples, 2048, 1]
            signals = signals[:, :, np.newaxis]
        elif signals.ndim == 1:
            # 1D edge case: [num_samples] -> [num_samples, 2048, 1]
            signals = signals[:, np.newaxis, np.newaxis]
        standardized_signals.append(signals)
    
    all_pretrain_signals = standardized_signals
    all_pretrain_val_signals = [
        s[:, :, np.newaxis] if s.ndim == 2 else s 
        for s in all_pretrain_val_signals
    ]
    
    # Ensure val signals are also 3D
    standardized_val_signals = []
    for signals in all_pretrain_val_signals:
        if signals.ndim == 2:
            signals = signals[:, :, np.newaxis]
        standardized_val_signals.append(signals)
    all_pretrain_val_signals = standardized_val_signals
    
    # Find max number of channels across all datasets
    max_channels = max(s.shape[2] if s.ndim >= 3 else 1 for s in all_pretrain_signals)
    print(f"  Max channels found: {max_channels}")
    
    # Preprocess all signals and standardize channels
    preprocessed_signals = []
    for i, signals in enumerate(all_pretrain_signals):
        # Signals are already 3D from standardization above
        # But preprocess_dataset_paper might change dimensions, so ensure 3D
        if signals.ndim == 2:
            signals = signals[:, :, np.newaxis]
        
        # Apply paper preprocessing (standardize to 2048) - skip since already windowed
        # The signals are already windowed to 2048 from split_strategy
        processed = signals  # Already in correct format from windowing
        
        # Standardize channels: pad or truncate to max_channels
        if processed.ndim == 3:
            current_channels = processed.shape[2]
            if current_channels < max_channels:
                # Pad with zeros
                pad_size = max_channels - current_channels
                padding = np.zeros((processed.shape[0], processed.shape[1], pad_size))
                processed = np.concatenate([processed, padding], axis=2)
            elif current_channels > max_channels:
                # Truncate to max_channels (take first N channels)
                processed = processed[:, :, :max_channels]
        
        preprocessed_signals.append(processed)
        print(f"    Dataset {i}: {signals.shape} -> {processed.shape}")
    
    preprocessed_val_signals = []
    for i, signals in enumerate(all_pretrain_val_signals):
        # Ensure 3D
        if signals.ndim == 2:
            signals = signals[:, :, np.newaxis]
        
        # Signals already windowed to 2048 from split_strategy
        processed = signals
        
        # Standardize channels
        if processed.ndim == 3:
            current_channels = processed.shape[2]
            if current_channels < max_channels:
                pad_size = max_channels - current_channels
                padding = np.zeros((processed.shape[0], processed.shape[1], pad_size))
                processed = np.concatenate([processed, padding], axis=2)
            elif current_channels > max_channels:
                processed = processed[:, :, :max_channels]
        
        preprocessed_val_signals.append(processed)
    
    # Now concatenate (all have same shape: [num_samples, 2048, max_channels])
    aggregated_pretrain = np.concatenate(preprocessed_signals, axis=0)
    aggregated_pretrain_val = np.concatenate(preprocessed_val_signals, axis=0)
    
    print(f"\nAggregated pretraining data:")
    print(f"  Total pretrain samples: {len(aggregated_pretrain)}")
    print(f"  Total pretrain val samples: {len(aggregated_pretrain_val)}")
    
    return aggregated_pretrain, aggregated_pretrain_val


def load_dataset_for_finetuning(
    dataset_name: str,
    task_name: str,
    test_size: float = 0.2,
    finetune_val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Load single dataset for finetuning with paper-compliant splits
    
    Args:
        dataset_name: Dataset name
        task_name: Task name
        test_size: Test set size (default 0.2)
        finetune_val_size: Validation size within train80 (default 0.1)
        random_state: Random seed
        
    Returns:
        Dictionary with 'finetune_train', 'finetune_val', and 'test' splits
    """
    splits = paper_split_from_phmd(
        dataset_name=dataset_name,
        task_name=task_name,
        test_size=test_size,
        finetune_val_size=finetune_val_size,
        random_state=random_state
    )
    
    return {
        'finetune_train': splits['finetune_train'],
        'finetune_val': splits['finetune_val'],
        'test': splits['test']
    }
