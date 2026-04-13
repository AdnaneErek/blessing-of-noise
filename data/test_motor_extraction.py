"""Test motor command extraction"""
import scipy.io
import numpy as np

mat_path = 'data/raw/dataset/trainingDatasets/20241017/Healthy/dataset_140.mat'
mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

motor_cmds = mat['motorCmdsRadius']
print("Motor commands type:", type(motor_cmds))
print("Motor commands shape:", motor_cmds.shape)
print("Motor commands:", motor_cmds)

# Try to access timeseries data
for i, ts in enumerate(motor_cmds):
    print(f"\nMotor {i+1}:")
    print(f"  Type: {type(ts)}")
    arr = np.array(ts)
    print(f"  Array shape: {arr.shape}")
    print(f"  Array dtype: {arr.dtype}")
    if arr.dtype.names:
        print(f"  Field names: {arr.dtype.names}")
        for name in arr.dtype.names:
            field_data = arr[name]
            print(f"    {name}: {type(field_data)}, shape: {field_data.shape if hasattr(field_data, 'shape') else 'N/A'}")
