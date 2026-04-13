"""
Signal preprocessing utilities for RmGPT

Implements paper preprocessing:
1. Downsample all signals to ~5kHz
2. Standardize input windows to 2048 time steps
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Optional, Tuple


def downsample_to_5khz(signal: np.ndarray, 
                       original_fs: float,
                       target_fs: float = 5000.0) -> np.ndarray:
    """
    Downsample signal to approximately 5kHz
    
    Args:
        signal: Input signal [timesteps, channels] or [timesteps]
        original_fs: Original sampling frequency (Hz)
        target_fs: Target sampling frequency (Hz), default 5000.0 (5kHz)
        
    Returns:
        Downsampled signal
    """
    if original_fs <= target_fs:
        # Already at or below target frequency, return as-is
        return signal
    
    # Calculate downsampling factor
    downsample_factor = original_fs / target_fs
    
    # Apply anti-aliasing filter before downsampling
    # Butterworth filter with cutoff at target_fs/2
    nyquist = original_fs / 2
    cutoff = target_fs / 2
    
    if signal.ndim == 1:
        # 1D signal
        b, a = scipy_signal.butter(4, cutoff / nyquist, btype='low')
        filtered = scipy_signal.filtfilt(b, a, signal)
        downsampled = scipy_signal.decimate(filtered, int(downsample_factor), ftype='iir')
        return downsampled
    else:
        # Multi-channel signal [timesteps, channels]
        downsampled_channels = []
        for ch in range(signal.shape[1]):
            channel_signal = signal[:, ch]
            b, a = scipy_signal.butter(4, cutoff / nyquist, btype='low')
            filtered = scipy_signal.filtfilt(b, a, channel_signal)
            downsampled = scipy_signal.decimate(filtered, int(downsample_factor), ftype='iir')
            downsampled_channels.append(downsampled)
        return np.column_stack(downsampled_channels)


def standardize_window_length(signal: np.ndarray, 
                               target_length: int = 2048,
                               mode: str = 'truncate') -> np.ndarray:
    """
    Standardize signal window to target length (2048 time steps as per paper)
    
    Args:
        signal: Input signal [timesteps, channels] or [timesteps]
        target_length: Target window length (default 2048)
        mode: 'truncate' (take first N) or 'pad' (zero-pad to N) or 'crop_center'
        
    Returns:
        Standardized signal with target_length timesteps
    """
    current_length = signal.shape[0]
    
    if current_length == target_length:
        return signal
    elif current_length > target_length:
        # Truncate or crop center
        if mode == 'truncate':
            return signal[:target_length, ...]
        elif mode == 'crop_center':
            start = (current_length - target_length) // 2
            return signal[start:start + target_length, ...]
        else:
            return signal[:target_length, ...]  # Default to truncate
    else:
        # Pad with zeros
        pad_length = target_length - current_length
        if signal.ndim == 1:
            return np.pad(signal, (0, pad_length), mode='constant', constant_values=0)
        else:
            # Multi-channel: pad along first axis
            return np.pad(signal, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)


def preprocess_signal_paper(signal: np.ndarray,
                            original_fs: Optional[float] = None,
                            target_fs: float = 5000.0,
                            window_length: int = 2048,
                            downsample: bool = True) -> np.ndarray:
    """
    Apply paper preprocessing pipeline:
    1. Downsample to ~5kHz (if original_fs provided and > target_fs)
    2. Standardize to 2048 time steps
    
    Args:
        signal: Input signal [timesteps, channels] or [timesteps]
        original_fs: Original sampling frequency (Hz). If None, skip downsampling.
        target_fs: Target sampling frequency (Hz), default 5000.0
        window_length: Target window length, default 2048
        downsample: Whether to downsample (if False, only standardize length)
        
    Returns:
        Preprocessed signal [window_length, channels]
    """
    processed = signal.copy()
    
    # Step 1: Downsample to ~5kHz (if needed and original_fs provided)
    if downsample and original_fs is not None and original_fs > target_fs:
        processed = downsample_to_5khz(processed, original_fs, target_fs)
    
    # Step 2: Standardize to 2048 time steps
    processed = standardize_window_length(processed, target_length=window_length, mode='truncate')
    
    return processed


def preprocess_dataset_paper(signals: np.ndarray,
                             original_fs: Optional[float] = None,
                             target_fs: float = 5000.0,
                             window_length: int = 2048,
                             downsample: bool = True) -> np.ndarray:
    """
    Apply paper preprocessing to a batch of signals
    
    Args:
        signals: Input signals [num_samples, timesteps, channels] or [num_samples, timesteps]
        original_fs: Original sampling frequency (Hz)
        target_fs: Target sampling frequency (Hz), default 5000.0
        window_length: Target window length, default 2048
        downsample: Whether to downsample
        
    Returns:
        Preprocessed signals [num_samples, window_length, channels]
    """
    preprocessed = []
    
    for i in range(len(signals)):
        signal = signals[i]
        processed = preprocess_signal_paper(
            signal, 
            original_fs=original_fs,
            target_fs=target_fs,
            window_length=window_length,
            downsample=downsample
        )
        preprocessed.append(processed)
    
    return np.array(preprocessed)
