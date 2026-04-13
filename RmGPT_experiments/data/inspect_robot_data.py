"""Inspect robot dataset structure"""
import scipy.io
import numpy as np
import os
from pathlib import Path

# Check individual dataset files
data_dir = Path('data/raw/dataset/trainingDatasets/20241017/Healthy')
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')][:3]

print("Sample .mat files:", mat_files)
print()

if mat_files:
    mat_path = data_dir / mat_files[0]
    print(f"Loading: {mat_path}")
    mat = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    print("Keys:", [k for k in mat.keys() if not k.startswith('__')])
    
    for key in mat.keys():
        if not key.startswith('__'):
            data = mat[key]
            print(f"\n{key}:")
            print(f"  Type: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"  Shape: {data.shape}")
            if isinstance(data, np.ndarray) and data.size < 20:
                print(f"  Data: {data}")
