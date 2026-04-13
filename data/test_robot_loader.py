"""Test script to understand robot dataset structure"""
import scipy.io
import numpy as np
import pandas as pd

mat_path = 'data/raw/dataset/trainingDatasets/20241017/training_dataset.mat'
mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

print("Keys in mat file:", list(mat.keys()))
print("\nDataTables type:", type(mat['dataTables']))
print("DataTables shape:", mat['dataTables'].shape if hasattr(mat['dataTables'], 'shape') else 'N/A')
print("Labels type:", type(mat['y']))
print("Labels shape:", mat['y'].shape if hasattr(mat['y'], 'shape') else 'N/A')
print("First 5 labels:", mat['y'][:5] if len(mat['y']) > 0 else 'N/A')

# Try to access first dataTable
dt = mat['dataTables'][0]
print("\nFirst dataTable type:", type(dt))
print("First dataTable:", dt)

# Try different conversion methods
try:
    arr = np.array(dt)
    print("\nAs numpy array:")
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    if arr.dtype.names:
        print("Field names:", arr.dtype.names)
        for name in arr.dtype.names:
            print(f"  {name}: {arr[name]}")
except Exception as e:
    print(f"\nError converting to array: {e}")

# Try to convert to DataFrame
try:
    if hasattr(dt, '__array__'):
        arr = np.array(dt)
        if arr.dtype.names:
            # Structured array - convert to dict then DataFrame
            data_dict = {name: arr[name] for name in arr.dtype.names}
            df = pd.DataFrame(data_dict)
            print("\nAs DataFrame:")
            print("Shape:", df.shape)
            print("Columns:", df.columns.tolist())
            print("\nFirst few rows:")
            print(df.head())
except Exception as e:
    print(f"\nError converting to DataFrame: {e}")
