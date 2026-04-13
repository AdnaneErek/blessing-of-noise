"""
Sliding window utilities for time series data

Creates multiple samples from long time series sequences using sliding windows.
Paper uses 2048 timestep windows.
"""
import numpy as np
from typing import Optional, Tuple


def create_sliding_windows(signal: np.ndarray,
                          window_length: int = 2048,
                          stride: Optional[int] = None,
                          overlap: float = 0.0) -> np.ndarray:
    """
    Create sliding windows from a long time series signal
    
    Args:
        signal: Input signal [timesteps, channels] or [timesteps]
        window_length: Length of each window (default 2048, per paper)
        stride: Stride between windows. If None, stride = window_length * (1 - overlap)
        overlap: Overlap ratio between windows (0.0 = no overlap, 0.5 = 50% overlap)
                 Used only if stride is None
        
    Returns:
        windows: [num_windows, window_length, channels] or [num_windows, window_length]
    """
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]  # [timesteps, 1]
    
    timesteps, channels = signal.shape
    
    # Calculate stride
    if stride is None:
        stride = int(window_length * (1 - overlap))
    
    # Create windows
    windows = []
    start_idx = 0
    
    while start_idx + window_length <= timesteps:
        window = signal[start_idx:start_idx + window_length, :]
        windows.append(window)
        start_idx += stride
    
    if len(windows) == 0:
        # Signal shorter than window_length - pad or use whole signal
        if timesteps < window_length:
            # Pad to window_length
            pad_size = window_length - timesteps
            padded = np.pad(signal, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            windows = [padded]
        else:
            # Use full signal (shouldn't happen but safety)
            windows = [signal[:window_length, :]]
    
    windows_array = np.array(windows)
    
    # Keep as 3D: [num_windows, window_length, channels]
    # Don't squeeze - always maintain 3D format for consistency
    return windows_array


def window_time_series_signals(signals: np.ndarray,
                               labels: Optional[np.ndarray] = None,
                               rul: Optional[np.ndarray] = None,
                               window_length: int = 2048,
                               stride: Optional[int] = None,
                               overlap: float = 0.0) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create sliding windows for multiple time series signals
    
    Args:
        signals: Input signals [num_units, timesteps, channels] where timesteps can be very long
        labels: Labels per unit [num_units] (optional)
        rul: RUL values per unit [num_units] (optional)
        window_length: Length of each window (default 2048)
        stride: Stride between windows
        overlap: Overlap ratio (used if stride is None)
        
    Returns:
        windowed_signals: [total_windows, window_length, channels]
        windowed_labels: [total_windows] (if labels provided)
        windowed_rul: [total_windows] (if rul provided)
    """
    windowed_signals_list = []
    windowed_labels_list = []
    windowed_rul_list = []
    
    num_units = len(signals)
    
    for i in range(num_units):
        signal = signals[i]  # [timesteps, channels] - can be very long
        
        # Create windows for this unit's signal
        windows = create_sliding_windows(signal, window_length=window_length, 
                                        stride=stride, overlap=overlap)
        # windows: [num_windows_from_this_unit, window_length, channels]
        
        windowed_signals_list.append(windows)
        
        # Assign same label/RUL to all windows from this unit
        if labels is not None:
            windowed_labels_list.extend([labels[i]] * len(windows))
        
        if rul is not None:
            windowed_rul_list.extend([rul[i]] * len(windows))
    
    # Concatenate all windows
    windowed_signals = np.concatenate(windowed_signals_list, axis=0)
    
    windowed_labels = np.array(windowed_labels_list) if labels is not None else None
    windowed_rul = np.array(windowed_rul_list) if rul is not None else None
    
    return windowed_signals, windowed_labels, windowed_rul
