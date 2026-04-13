import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy import interpolate

def make_dir_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def sliding_windows(signal: np.ndarray, window: int, stride: int):
    n = len(signal)
    for start in range(0, n - window + 1, stride):
        yield signal[start:start + window, :]

class LBNL_FCU_Dataset(Dataset):
    def __init__(self, config, phase: str = "eval"):
        self.base_path = config.base_path
        self.cache_dir = config.cache_dir
        self.seq_len = config.seq_len
        self.window = getattr(config, "window", 1024)
        self.stride = getattr(config, "stride", 1024)
        
        # We always need FCU_SPD to filter, plus the user's selected columns
        raw_cols = getattr(config, "sensor_column", "RM_TEMP")
        if isinstance(raw_cols, str):
            self.input_columns = [raw_cols]
        else:
            self.input_columns = raw_cols
            
        self.binary_labels = getattr(config, "binary_labels", True)
        self.load_cache = getattr(config, "load_cache", True)

        make_dir_if_not_exists(self.cache_dir)
        
        # Unique cache name including 'FILTERED'
        col_str = "_".join(self.input_columns)
        if len(col_str) > 50: col_str = str(hash(col_str))
        cache_name = f"lbnl_fcu_FILTERED_{col_str}_{self.window}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_name)

        if os.path.exists(cache_path) and self.load_cache:
            print(f"[INFO] Loading cached FILTERED dataset from {cache_path}...")
            with open(cache_path, "rb") as f:
                self._timeseries, self._labels = pickle.load(f)
        else:
            print(f"[INFO] Building FILTERED dataset (Fan > 0)...")
            self._timeseries, self._labels = self._build_dataset()
            with open(cache_path, "wb") as f:
                pickle.dump((self._timeseries, self._labels), f)
            print(f"[INFO] Cache saved to {cache_path}")

        # Interpolate to seq_len
        n_samples, n_channels, timesteps = self._timeseries.shape
        if timesteps != self.seq_len:
            x_old = np.linspace(0, 1, timesteps)
            x_new = np.linspace(0, 1, self.seq_len)
            f = interpolate.interp1d(x_old, self._timeseries, axis=2)
            self._timeseries = f(x_new).astype(np.float32)

        # Instance Normalization
        mean = np.mean(self._timeseries, axis=2, keepdims=True)
        std = np.std(self._timeseries, axis=2, keepdims=True)
        self._timeseries = (self._timeseries - mean) / (std + 1e-8)

        # Label Mapping
        unique_labels = np.unique(self._labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        self._labels = np.array([label_mapping[l] for l in self._labels], dtype=np.int64)
        self.num_classes = len(unique_labels)
        self._length = len(self._timeseries)

    def _build_dataset(self):
        X, y = [], []
        all_files = sorted([f for f in os.listdir(self.base_path) if f.endswith(".csv")])
        
        # Columns to load: User request + FCU_SPD for filtering
        load_cols = list(set(self.input_columns + ['FCU_SPD']))

        for file_idx, fname in enumerate(all_files):
            if self.binary_labels:
                normal_kws = ["fault_free", "faultfree", "normal", "healthy"]
                label = 0 if any(k in fname.lower() for k in normal_kws) else 1
            else:
                label = file_idx

            df = pd.read_csv(os.path.join(self.base_path, fname))
            
            # --- CRITICAL FILTERING STEP ---
            # 1. Filter rows where Fan is ON (Speed > 0.1)
            if 'FCU_SPD' in df.columns:
                initial_len = len(df)
                # Keep only active times
                df = df[df['FCU_SPD'] > 0.1].copy()
                
                # If file becomes empty (e.g. always off), skip it
                if len(df) < self.window:
                    continue
            else:
                print(f"[WARN] FCU_SPD not found in {fname}, skipping filter.")

            # 2. Extract only the input signal columns
            try:
                signal = df[self.input_columns].astype(np.float32).values
            except KeyError:
                continue

            # Sliding window
            for w in sliding_windows(signal, self.window, self.stride):
                X.append(w.T) 
                y.append(label)

        if len(X) == 0:
            raise ValueError("No data left after filtering! Check Fan Speed threshold.")

        return np.stack(X), np.array(y, dtype=np.int64)

    def __len__(self): return self._length
    def __getitem__(self, idx): return self._timeseries[idx], self._labels[idx]