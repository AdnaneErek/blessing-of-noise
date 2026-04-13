import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset

class DigitalTwinDataset(Dataset):
    def __init__(self, file_path, seq_len=512):
        self.seq_len = seq_len
        
        # Load .mat file
        try:
            mat = scipy.io.loadmat(file_path)
        except ImportError:
            raise ImportError("Please install scipy: pip install scipy")
            
        # Extract object arrays
        X_obj = mat['X_test_array']  # Shape (1, 90)
        y_obj = mat['y_test_array']  # Shape (1, 90)
        
        # Process X: Convert object array of (1000, 6) -> Tensor (90, 6, 1000)
        X_list = []
        for i in range(X_obj.shape[1]):
            # Get the sample (1000, 6)
            sample = X_obj[0, i]
            
            # Transpose to (6, 1000) for PyTorch (Channels, Time)
            sample = sample.T 
            
            # Resize/Pad to seq_len
            if sample.shape[1] > seq_len:
                sample = sample[:, :seq_len]
            elif sample.shape[1] < seq_len:
                pad = np.zeros((sample.shape[0], seq_len - sample.shape[1]))
                sample = np.concatenate([sample, pad], axis=1)
                
            X_list.append(sample)
            
        self.X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        
        # Instance Normalization (Crucial for Transfer Learning)
        # We normalize each robot axis independently to mean=0, std=1
        mean = self.X.mean(dim=2, keepdim=True)
        std = self.X.std(dim=2, keepdim=True)
        self.X = (self.X - mean) / (std + 1e-8)

        # Process y: String -> Integer
        y_raw = []
        for i in range(y_obj.shape[1]):
            val = y_obj[0, i]
            # Handle numpy scalar string extraction
            if isinstance(val, np.ndarray):
                val = val.item()
            y_raw.append(str(val))
            
        self.classes = sorted(list(set(y_raw)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[l] for l in y_raw]
        self.y = torch.tensor(self.labels, dtype=torch.long)
        
        print(f"[INFO] Loaded DigitalTwin: {self.X.shape} samples. Classes: {len(self.classes)}")
        print(f"[INFO] Class Map: {self.class_to_idx}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]