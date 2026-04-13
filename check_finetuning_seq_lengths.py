import scipy.io
import numpy as np
from pathlib import Path
import os

data_dir = '/gpfs/workdir/erekrakead/RmGPT/data/raw/dataset'
finetuning_path = Path(data_dir) / 'finetuningDatasets'

# Get all finetuning folders
folders = [d.name for d in finetuning_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

print("=" * 80)
print("Sequence Lengths in Finetuning Dataset .mat Files")
print("=" * 80)

for folder in sorted(folders):
    print(f"\n📁 Folder: {folder}")
    print("-" * 80)
    
    folder_path = finetuning_path / folder
    
    # Get all fault type folders
    fault_folders = [d for d in folder_path.iterdir() if d.is_dir()]
    
    all_lengths = []
    
    for fault_folder in sorted(fault_folders):
        fault_name = fault_folder.name
        
        # Get all dataset_*.mat files
        mat_files = sorted([f for f in os.listdir(fault_folder) 
                           if f.startswith('dataset_') and f.endswith('.mat')])
        
        if len(mat_files) == 0:
            continue
        
        lengths_for_fault = []
        
        for mat_file in mat_files:
            mat_path = fault_folder / mat_file
            try:
                mat = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
                
                traj_cmds = mat.get('trajCmds', None)
                traj_resps = mat.get('trajResps', None)
                
                if traj_cmds is not None and traj_resps is not None:
                    traj_cmds = np.array(traj_cmds)
                    traj_resps = np.array(traj_resps)
                    
                    # Get sequence length (should be same for both)
                    seq_len = traj_cmds.shape[0]
                    lengths_for_fault.append(seq_len)
                    all_lengths.append(seq_len)
            except Exception as e:
                print(f"  ⚠️  Error loading {mat_file}: {e}")
                continue
        
        if lengths_for_fault:
            min_len = min(lengths_for_fault)
            max_len = max(lengths_for_fault)
            mean_len = np.mean(lengths_for_fault)
            print(f"  {fault_name:30s}: {len(lengths_for_fault):2d} files, "
                  f"lengths: min={min_len:5d}, max={max_len:5d}, mean={mean_len:7.1f}")
    
    if all_lengths:
        print(f"\n  📊 Overall for {folder}:")
        print(f"     Total files: {len(all_lengths)}")
        print(f"     Sequence lengths: min={min(all_lengths):5d}, max={max(all_lengths):5d}, "
              f"mean={np.mean(all_lengths):7.1f}, median={np.median(all_lengths):7.1f}")

print("\n" + "=" * 80)