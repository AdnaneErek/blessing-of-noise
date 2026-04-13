"""
Dataset classes for RmGPT training

Provides data loading and preprocessing for PHM datasets.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import os
from .split_strategy import paper_split_strategy, paper_split_from_phmd


class PHMSignalDataset(Dataset):
    """
    Dataset for PHM signals
    
    Handles loading and preprocessing of multivariate time-series signals
    from PHM datasets.
    """
    
    def __init__(self,
                 signals: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 rul: Optional[np.ndarray] = None,
                 normalize: bool = True,
                 signal_window: Optional[int] = None,
                 overlap: int = 0,
                 add_noise: bool = False,
                 noise_std: float = 0.01,
                 noise_type: str = 'gaussian'):
        """
        Args:
            signals: Signal data [num_samples, seq_len, num_channels] or [num_samples, num_channels]
            labels: Classification labels [num_samples] (for diagnosis)
            rul: RUL values [num_samples] (for prognosis)
            normalize: Whether to normalize signals
            signal_window: Window size for sliding window (if None, use full signal)
            overlap: Overlap between windows
            add_noise: Whether to add noise augmentation (typically only for training)
            noise_std: Standard deviation of noise (relative to signal std if normalize=True)
            noise_type: Type of noise ('gaussian', 'uniform', 'laplace')
        """
        self.signals = signals
        self.labels = labels
        self.rul = rul
        self.normalize = normalize
        self.signal_window = signal_window
        self.overlap = overlap
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.noise_type = noise_type
        
        # Normalize signals
        if normalize:
            self.scaler = StandardScaler()
            # Reshape for scaler: [num_samples * seq_len, num_channels]
            original_shape = signals.shape
            signals_flat = signals.reshape(-1, signals.shape[-1])
            self.scaler.fit(signals_flat)
            signals_scaled = self.scaler.transform(signals_flat)
            self.signals = signals_scaled.reshape(original_shape)
        
        # Apply sliding window if specified
        if signal_window is not None:
            self.signals, self.labels, self.rul = self._apply_sliding_window(
                self.signals, labels, rul, signal_window, overlap
            )
        
        # Convert to numpy if needed
        if isinstance(self.signals, pd.DataFrame):
            self.signals = self.signals.values
        
        # Ensure 3D: [num_samples, seq_len, num_channels]
        if self.signals.ndim == 2:
            # Assume single channel with sequence
            self.signals = self.signals[:, :, np.newaxis]
        elif self.signals.ndim == 1:
            # Reshape to [1, seq_len, 1]
            self.signals = self.signals.reshape(1, -1, 1)
        
    def _apply_sliding_window(self,
                             signals: np.ndarray,
                             labels: Optional[np.ndarray],
                             rul: Optional[np.ndarray],
                             window_size: int,
                             overlap: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply sliding window to signals"""
        stride = window_size - overlap
        windowed_signals = []
        windowed_labels = []
        windowed_rul = []
        
        if signals.ndim == 2:
            # [num_samples, features] - assume single time step
            signals = signals[:, np.newaxis, :]  # [num_samples, 1, features]
        
        num_samples = signals.shape[0]
        seq_len = signals.shape[1]
        
        for i in range(num_samples):
            signal = signals[i]  # [seq_len, num_channels]
            
            # Extract windows
            windows = []
            for start in range(0, seq_len - window_size + 1, stride):
                window = signal[start:start + window_size, :]
                windows.append(window)
            
            if len(windows) > 0:
                windowed_signals.extend(windows)
                
                if labels is not None:
                    windowed_labels.extend([labels[i]] * len(windows))
                
                if rul is not None:
                    windowed_rul.extend([rul[i]] * len(windows))
        
        windowed_signals = np.array(windowed_signals)
        windowed_labels = np.array(windowed_labels) if labels is not None else None
        windowed_rul = np.array(windowed_rul) if rul is not None else None
        
        return windowed_signals, windowed_labels, windowed_rul
    
    def __len__(self) -> int:
        return self.signals.shape[0]
    
    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add noise to signal for data augmentation
        
        Args:
            signal: Input signal [seq_len, num_channels]
            
        Returns:
            Noisy signal [seq_len, num_channels]
        """
        # Ensure noise has same dtype as signal (float32)
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_std, signal.shape).astype(signal.dtype)
        elif self.noise_type == 'uniform':
            # Uniform noise: [-noise_std*sqrt(3), +noise_std*sqrt(3)] to match variance
            noise = np.random.uniform(-self.noise_std * np.sqrt(3), 
                                     self.noise_std * np.sqrt(3), 
                                     signal.shape).astype(signal.dtype)
        elif self.noise_type == 'laplace':
            noise = np.random.laplace(0, self.noise_std / np.sqrt(2), signal.shape).astype(signal.dtype)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return signal + noise
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary with:
            - 'signals': [seq_len, num_channels]
            - 'labels': (optional) label value
            - 'rul': (optional) RUL value
        """
        signal = self.signals[idx].astype(np.float32)
        
        # Add noise augmentation if enabled (typically only for training)
        if self.add_noise:
            signal = self._add_noise(signal)
        
        sample = {
            'signals': torch.from_numpy(signal)
        }
        
        if self.labels is not None:
            sample['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.rul is not None:
            sample['rul'] = torch.tensor(self.rul[idx], dtype=torch.float32)
        
        return sample


def load_phmd_dataset(dataset_name: str,
                     task_name: str,
                     fold: int = 0,
                     split: str = 'train',
                     data_dir: str = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load dataset from PHMD library or custom loaders
    
    Args:
        dataset_name: Name of the dataset (CWRU, XJTU-SY, SLIET, QPZZ-II, SMU)
        task_name: Task name ('Diagnosis' or 'Prognosis')
        fold: Cross-validation fold
        split: Data split ('train', 'val', 'test')
        data_dir: Custom data directory for datasets not in PHMD
        
    Returns:
        signals: [num_samples, seq_len, num_channels] or processed format
        labels: [num_samples] (for diagnosis) or None
        rul: [num_samples] (for prognosis) or None
    """
    # Check for custom datasets first
    dataset_name_upper = dataset_name.upper()
    
    if dataset_name_upper == 'SLIET':
        from .custom_datasets import load_sliet_dataset
        if data_dir is None:
            data_dir = os.getenv('RMGPT_DATA_DIR', './data/raw')
        signals, labels = load_sliet_dataset(data_dir, split)
        return signals, labels, None
    
    elif dataset_name_upper in ['QPZZ-II', 'QPZZ2', 'QPZZ_II']:
        from .custom_datasets import load_qpzz2_dataset
        if data_dir is None:
            data_dir = os.getenv('RMGPT_DATA_DIR', './data/raw')
        signals, labels = load_qpzz2_dataset(data_dir, split)
        return signals, labels, None
    
    elif dataset_name_upper == 'SMU':
        from .custom_datasets import load_smu_dataset
        if data_dir is None:
            data_dir = os.getenv('RMGPT_DATA_DIR', './data/raw')
        signals, labels = load_smu_dataset(data_dir, split)
        return signals, labels, None
    
    elif dataset_name_upper in ['XJTU', 'XJTU-SY']:
        # Try PHMD first, fallback to custom
        try:
            import phmd
            from phmd import datasets
            dataset = datasets.Dataset("XJTU-SY")
            task = dataset["Prognosis"] if task_name == "Prognosis" else dataset["Diagnosis"]
            sets = task.load_fold(fold)
            data = sets[split]
            
            # Extract features and target
            feature_cols = task.features
            signal_cols = [col for col in data.columns if col in feature_cols]
            
            target_col = task.target
            if target_col in data.columns:
                targets = data[target_col].values
            else:
                targets = None
            
            # Group by unit identifier
            identifier_cols = task.meta.get('identifier', [])
            if identifier_cols:
                signals_list = []
                target_list = []
                for unit_id, group in data.groupby(identifier_cols):
                    unit_signals = group[signal_cols].values
                    if len(unit_signals) > 0:
                        signals_list.append(unit_signals)
                        if targets is not None:
                            unit_targets = group[target_col].values if target_col in group.columns else None
                            if unit_targets is not None:
                                if task_name == "Prognosis":
                                    target_list.append(unit_targets[-1])  # Last RUL value
                                else:
                                    target_list.append(unit_targets[0])  # First label
                
                if len(signals_list) > 0:
                    max_len = max(s.shape[0] for s in signals_list)
                    num_channels = signals_list[0].shape[1]
                    signals_padded = np.zeros((len(signals_list), max_len, num_channels))
                    for i, signal in enumerate(signals_list):
                        signals_padded[i, :signal.shape[0], :] = signal
                    
                    if task_name == "Prognosis":
                        return signals_padded, None, np.array(target_list)
                    else:
                        return signals_padded, np.array(target_list), None
        except Exception as e:
            print(f"PHMD loading failed for XJTU, trying custom loader: {e}")
            # Fallback to custom loader
            from .custom_datasets import load_xjtu_dataset
            if data_dir is None:
                data_dir = os.getenv('RMGPT_DATA_DIR', './data/raw')
            signals, rul = load_xjtu_dataset(data_dir, split)
            return signals, None, rul
    
    # Default: Try PHMD library
    try:
        import phmd
        from phmd import datasets
        
        # Load dataset
        dataset = datasets.Dataset(dataset_name)
        task = dataset[task_name]
        
        # Load fold
        sets = task.load_fold(fold)
        
        # Get data
        data = sets[split]
        
        # Extract features (signal columns)
        feature_cols = task.features
        signal_cols = [col for col in data.columns if col in feature_cols or col in data.columns[:len(feature_cols)]]
        
        # Extract target
        target_col = task.target
        if target_col in data.columns:
            targets = data[target_col].values
        else:
            targets = None
        
        # Group by unit identifier
        identifier_cols = task.meta.get('identifier', [])
        if identifier_cols:
            signals_list = []
            labels_list = [] if task.meta.get('type', '').startswith('classification') else None
            rul_list = [] if task.meta.get('type') == 'regression' else None
            
            for unit_id, group in data.groupby(identifier_cols):
                # Extract signals for this unit
                unit_signals = group[signal_cols].values  # [seq_len, num_channels]
                
                if unit_signals.shape[0] > 0:
                    signals_list.append(unit_signals)
                    
                    if labels_list is not None and targets is not None:
                        # Use the most common label for this unit
                        unit_labels = group[target_col].values if target_col in group.columns else None
                        if unit_labels is not None and len(unit_labels) > 0:
                            labels_list.append(unit_labels[0])
                    
                    if rul_list is not None and targets is not None:
                        # Use the last RUL value
                        unit_rul = group[target_col].values if target_col in group.columns else None
                        if unit_rul is not None and len(unit_rul) > 0:
                            rul_list.append(unit_rul[-1])
            
            signals = np.array(signals_list, dtype=object) if len(signals_list) > 0 else np.array([])
            
            # Convert to numpy arrays with padding if needed
            if len(signals_list) > 0:
                max_len = max(s.shape[0] for s in signals_list)
                num_channels = signals_list[0].shape[1]
                signals_padded = np.zeros((len(signals_list), max_len, num_channels))
                
                for i, s in enumerate(signals_list):
                    signals_padded[i, :s.shape[0], :] = s
                
                signals = signals_padded
            
            labels = np.array(labels_list) if labels_list else None
            rul = np.array(rul_list) if rul_list else None
            
            return signals, labels, rul
        else:
            # No grouping - treat each row as a sample
            signals = data[signal_cols].values
            
            # Reshape if needed
            if signals.ndim == 2:
                signals = signals[:, np.newaxis, :]  # [num_samples, 1, num_channels]
            
            labels = targets if task.meta.get('type', '').startswith('classification') else None
            rul = targets if task.meta.get('type') == 'regression' else None
            
            return signals, labels, rul
    
    except Exception as e:
        print(f"Error loading PHMD dataset: {e}")
        print("Returning empty dataset. Please check dataset name and task.")
        return np.array([]), None, None
