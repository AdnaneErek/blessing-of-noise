#!/usr/bin/env python3
"""
Main training script for RmGPT

Usage:
    python train_rmgpt.py --config configs/default_config.yaml --task diagnosis
    python train_rmgpt.py --config configs/default_config.yaml --task prognosis
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
import numpy as np

from model.rmgpt import RmGPT, DiagnosisHead, PrognosisHead
from train.trainer import RmGPTTrainer, train_epoch
from data.dataset import PHMSignalDataset, load_phmd_dataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train RmGPT model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--task', type=str, choices=['diagnosis', 'prognosis', 'pretrain'],
                       default='diagnosis', help='Task type')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=None,
                       help='Start epoch number (if resuming and checkpoint has no epoch info)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config file value)')
    parser.add_argument('--task-name', type=str, default=None,
                       help='Task name: Diagnosis or Prognosis (overrides config, auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override dataset name if provided
    if args.dataset:
        config['data']['dataset_name'] = args.dataset
        print(f"Using dataset from command line: {args.dataset}")
    
    # Override task name if provided, or auto-detect for fine-tuning
    if args.task_name:
        config['data']['task_name'] = args.task_name
        print(f"Using task name from command line: {args.task_name}")
    elif args.dataset and args.task != 'pretrain':
        # Auto-detect task name from pretrain_datasets list
        pretrain_datasets = config['data'].get('pretrain_datasets', [])
        pretrain_task_names = config['data'].get('pretrain_task_names', [])
        if args.dataset in pretrain_datasets:
            idx = pretrain_datasets.index(args.dataset)
            if idx < len(pretrain_task_names):
                config['data']['task_name'] = pretrain_task_names[idx]
                print(f"Auto-detected task name: {config['data']['task_name']}")
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['evaluation']['save_dir'], exist_ok=True)
    
    # Initialize checkpoint_signal_dim (will be set if resuming from checkpoint)
    checkpoint_signal_dim = None
    
    # Load dataset(s) with paper-compliant split strategy
    if args.task == 'pretrain':
        # Pretraining: Aggregate ALL datasets (CWRU, XJTU-SY, JNUB, KAUG17, HSG18)
        from data.multi_dataset import load_all_datasets_for_pretraining
        
        # All datasets for pretraining
        dataset_names = config['data'].get('pretrain_datasets', [
            'CWRU', 'JNUB', 'KAUG17', 'HSG18', 'XJTU-SY'
        ])
        task_names = config['data'].get('pretrain_task_names', [
            'Diagnosis', 'Diagnosis', 'Diagnosis', 'Diagnosis', 'Prognosis'
        ])
        
        print(f"Pretraining on {len(dataset_names)} datasets: {', '.join(dataset_names)}")
        print("Applying paper-compliant split: 80% train per dataset (aggregated, unlabeled)")
        
        train_signals, val_signals = load_all_datasets_for_pretraining(
            dataset_names=dataset_names,
            task_names=task_names,
            test_size=config['data'].get('test_size', 0.2),
            finetune_val_size=config['data'].get('finetune_val_size', 0.1),
            random_state=config['data'].get('random_state', 42)
        )
        train_labels = None
        train_rul = None
        val_labels = None
        val_rul = None
    else:
        # Finetuning: Use single dataset with paper-compliant splits
        print(f"Loading dataset: {config['data']['dataset_name']}")
        print("Applying paper-compliant split: 80% train (for finetune), 20% test (untouched)")
        
        # Check if this is the robot dataset
        dataset_name_upper = config['data']['dataset_name'].upper()
        if dataset_name_upper == 'ROBOT' or dataset_name_upper == 'ROBOT_MIXED':
            from data.robot_dataset_loader import load_robot_training_data, load_robot_finetuning_data
            from data.split_strategy import paper_split_strategy
            
            data_dir = config['data'].get('data_dir', './data/raw/dataset')
            use_individual_files = config['data'].get('use_individual_files', True)
            
            # Check if we should mix real data
            use_real_data = config['data'].get('use_real_data', False)
            real_data_ratio = config['data'].get('real_data_ratio', 0.2)  # Default 20% real
            
            if dataset_name_upper == 'ROBOT_MIXED' and use_real_data:
                # MIXED TRAINING: Combine simulation and real data
                print("\n" + "="*60)
                print("MIXED TRAINING: Simulation + Real Data")
                print("="*60)
                
                # Load simulation data
                print("\n1. Loading simulation data (trainingDatasets/)...")
                training_folder = config['data'].get('robot_training_folder', None)
                sim_signals, sim_labels = load_robot_training_data(
                    data_dir=data_dir,
                    folder_name=training_folder,  # None = auto-discover all folders
                    use_individual_files=use_individual_files
                )
                print(f"   Loaded {len(sim_signals)} simulation samples")
                
                # Load real data (from finetuningDatasets - NOT testDatasets)
                print("\n2. Loading real data (finetuningDatasets/)...")
                finetuning_folder = config['data'].get('finetuning_folder', None)
                real_signals, real_labels = load_robot_finetuning_data(
                    data_dir=data_dir,
                    folder_name=finetuning_folder,  # None = auto-discover all folders
                    use_individual_files=True
                )
                print(f"   Loaded {len(real_signals)} real samples")
                
                # Use ALL available data - don't force a specific ratio
                # If we have limited real data, use all of it and all sim data
                available_real = len(real_signals)
                available_sim = len(sim_signals)
                
                # Calculate what ratio we'd get if we used all data
                total_if_all = available_sim + available_real
                actual_real_ratio_if_all = available_real / total_if_all if total_if_all > 0 else 0
                
                # Check if we have enough real data to meet the target ratio
                min_real_needed = int(available_sim * real_data_ratio / (1 - real_data_ratio))
                
                if available_real < min_real_needed:
                    # We don't have enough real data to meet the target ratio
                    # Use ALL available data instead of sampling down
                    print(f"\n3. Using ALL available data (not enough real data for {real_data_ratio*100:.1f}% ratio):")
                    print(f"   Target ratio: {real_data_ratio*100:.1f}% real would need {min_real_needed} real samples")
                    print(f"   Available: {available_real} real samples")
                    print(f"   Using all {available_sim} simulation samples")
                    print(f"   Using all {available_real} real samples")
                    print(f"   Actual ratio: {actual_real_ratio_if_all*100:.2f}% real, {100-actual_real_ratio_if_all*100:.2f}% simulation")
                    # Don't sample - use all data
                    actual_real_ratio = actual_real_ratio_if_all
                else:
                    # We have enough real data, can sample to match target ratio
                    print(f"\n3. Sampling to match target ratio ({real_data_ratio*100:.1f}% real):")
                    target_sim_samples = int(available_real * (1.0 / real_data_ratio - 1.0))
                    
                    if available_sim > target_sim_samples:
                        print(f"   Sampling simulation: {available_sim} -> {target_sim_samples} samples")
                        indices = np.random.RandomState(config['data'].get('random_state', 42)).choice(
                            available_sim, size=target_sim_samples, replace=False
                        )
                        sim_signals = sim_signals[indices]
                        sim_labels = sim_labels[indices]
                    else:
                        print(f"   Using all simulation data: {available_sim} samples")
                    
                    # Use all available real data
                    print(f"   Using all real data: {available_real} samples")
                    actual_real_ratio = real_data_ratio
                
                # Combine simulation and real data
                total_mixed = len(sim_signals) + len(real_signals)
                print(f"\n4. Combining data:")
                print(f"   Simulation: {len(sim_signals)} samples ({len(sim_signals)/total_mixed*100:.1f}%)")
                print(f"   Real: {len(real_signals)} samples ({len(real_signals)/total_mixed*100:.1f}%)")
                print(f"   Actual real data ratio: {actual_real_ratio*100:.2f}% (target: {real_data_ratio*100:.1f}%)")
                
                all_signals = np.concatenate([sim_signals, real_signals], axis=0)
                all_labels = np.concatenate([sim_labels, real_labels], axis=0)
                
                # Shuffle the combined data
                print(f"   Shuffling combined data...")
                shuffle_idx = np.random.RandomState(config['data'].get('random_state', 42)).permutation(len(all_signals))
                all_signals = all_signals[shuffle_idx]
                all_labels = all_labels[shuffle_idx]
                
                print(f"   Total: {len(all_signals)} samples")
                
            else:
                # STANDARD: Only simulation data
                training_folder = config['data'].get('robot_training_folder', None)
                all_signals, all_labels = load_robot_training_data(
                    data_dir=data_dir,
                    folder_name=training_folder,  # None = auto-discover all folders
                    use_individual_files=use_individual_files
                )
            
            # Apply paper-compliant split (only on simulation data for test set)
            # For mixed training, we still want test set from simulation only
            if dataset_name_upper == 'ROBOT_MIXED' and use_real_data:
                # Split simulation data first to get test set
                print("\n5. Splitting simulation data for test set...")
                # Use sklearn directly to split sim data into test and train (no val split needed)
                from sklearn.model_selection import train_test_split as sk_train_test_split
                test_size = config['data'].get('test_size', 0.2)
                sim_train_idx, sim_test_idx = sk_train_test_split(
                    np.arange(len(sim_signals)),
                    test_size=test_size,
                    random_state=config['data'].get('random_state', 42),
                    stratify=sim_labels
                )
                test_signals = sim_signals[sim_test_idx]
                test_labels = sim_labels[sim_test_idx]
                sim_train_signals = sim_signals[sim_train_idx]
                sim_train_labels = sim_labels[sim_train_idx]
                
                # Combine sim train with real data for training
                print(f"   Test set (simulation only): {len(test_signals)} samples")
                print(f"   Training data (sim + real): {len(sim_train_signals)} sim + {len(real_signals)} real = {len(sim_train_signals) + len(real_signals)} total")
                
                # Combine sim train and real for mixed training
                mixed_train_signals = np.concatenate([sim_train_signals, real_signals], axis=0)
                mixed_train_labels = np.concatenate([sim_train_labels, real_labels], axis=0)
                
                # Shuffle mixed training data
                shuffle_idx = np.random.RandomState(config['data'].get('random_state', 42) + 1).permutation(len(mixed_train_signals))
                mixed_train_signals = mixed_train_signals[shuffle_idx]
                mixed_train_labels = mixed_train_labels[shuffle_idx]
                
                # Split mixed training data into train/val
                from sklearn.model_selection import train_test_split
                train_signals, val_signals, train_labels, val_labels = train_test_split(
                    mixed_train_signals, mixed_train_labels,
                    test_size=config['data'].get('finetune_val_size', 0.1),
                    random_state=config['data'].get('random_state', 42),
                    stratify=mixed_train_labels
                )
                train_rul = None
                val_rul = None
                test_rul = None
                
                print(f"\nFinal splits:")
                print(f"  Train (mixed): {len(train_signals)} samples")
                print(f"  Val (mixed): {len(val_signals)} samples")
                print(f"  Test (simulation only): {len(test_signals)} samples")
            else:
                # Standard split for simulation-only training
                splits = paper_split_strategy(
                    signals=all_signals,
                    labels=all_labels,
                    rul=None,
                    test_size=config['data'].get('test_size', 0.2),
                    finetune_val_size=config['data'].get('finetune_val_size', 0.1),
                    random_state=config['data'].get('random_state', 42),
                    stratify=True
                )
                
                # Finetuning: use finetune_train and finetune_val
                train_signals, train_labels, train_rul = splits['finetune_train']
                val_signals, val_labels, val_rul = splits['finetune_val']
                # Test set is available but not used during training
                test_signals, test_labels, test_rul = splits['test']
                print(f"Finetune train samples: {len(train_signals)} (90% of train80)")
                print(f"Finetune val samples: {len(val_signals)} (10% of train80)")
                print(f"Test samples: {len(test_signals)} (20% - untouched until evaluation)")
        else:
            # Use PHMD dataset loader
            from data.split_strategy import paper_split_from_phmd
            
            splits = paper_split_from_phmd(
                dataset_name=config['data']['dataset_name'],
                task_name=config['data']['task_name'],
                test_size=config['data'].get('test_size', 0.2),
                finetune_val_size=config['data'].get('finetune_val_size', 0.1),
                random_state=config['data'].get('random_state', 42)
            )
            
            # Finetuning: use finetune_train and finetune_val
            train_signals, train_labels, train_rul = splits['finetune_train']
            val_signals, val_labels, val_rul = splits['finetune_val']
            # Test set is available but not used during training
            test_signals, test_labels, test_rul = splits['test']
            print(f"Finetune train samples: {len(train_signals)} (90% of train80)")
            print(f"Finetune val samples: {len(val_signals)} (10% of train80)")
            print(f"Test samples: {len(test_signals)} (20% - untouched until evaluation)")
    
    # If resuming from pretrained checkpoint, pad signals BEFORE creating datasets
    # This ensures datasets use the padded signals with correct signal_dim
    if args.resume and os.path.exists(args.resume):
        print(f"Inferring model dimensions from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Infer signal_dim from checkpoint weights
        patch_embed_shape = state_dict['signal_tokenizer.patch_embed.weight'].shape
        patch_embed_input_dim = patch_embed_shape[1]  # Input dimension to patch_embed
        patch_length = config['model']['patch_length']  # Should be 256
        
        # Calculate signal_dim: patch_embed_input_dim = patch_length * signal_dim
        signal_dim_from_ckpt = patch_embed_input_dim // patch_length
        
        # Verify the division is exact
        if patch_embed_input_dim % patch_length != 0:
            # Try to infer patch_length from checkpoint instead
            # Common values: 256, 512
            possible_patch_lengths = [256, 512, 128]
            inferred_patch_length = None
            for pl in possible_patch_lengths:
                if patch_embed_input_dim % pl == 0:
                    inferred_signal_dim = patch_embed_input_dim // pl
                    print(f"  Note: patch_embed input dim ({patch_embed_input_dim}) is divisible by {pl}")
                    print(f"        This would give signal_dim = {inferred_signal_dim}")
                    if inferred_patch_length is None:
                        inferred_patch_length = pl
            
            raise ValueError(
                f"Cannot infer signal_dim: patch_embed input dim ({patch_embed_input_dim}) "
                f"is not divisible by patch_length ({patch_length}). "
                f"Checkpoint may have been trained with different patch_length.\n"
                f"  Checkpoint patch_embed.weight shape: {patch_embed_shape}\n"
                f"  Config patch_length: {patch_length}\n"
                f"  If checkpoint used patch_length={inferred_patch_length}, signal_dim would be {patch_embed_input_dim // inferred_patch_length if inferred_patch_length else 'unknown'}"
            )
        
        print(f"Inferred signal_dim={signal_dim_from_ckpt} from checkpoint (to match pretrained model)")
        print(f"  patch_embed.weight shape: {patch_embed_shape}")
        print(f"  patch_embed input dim: {patch_embed_input_dim}")
        print(f"  patch_length (from config): {patch_length}")
        print(f"  Calculated signal_dim: {signal_dim_from_ckpt} = {patch_embed_input_dim} / {patch_length}")
        print(f"  Expected patch features per patch: {patch_length} * {signal_dim_from_ckpt} = {patch_length * signal_dim_from_ckpt}")
        
        # Pad signals to match checkpoint dimensions if needed (BEFORE creating datasets)
        if len(train_signals) > 0:
            sample_signal = train_signals[0]
            if sample_signal.ndim >= 2:
                current_signal_dim = sample_signal.shape[-1]
                if current_signal_dim < signal_dim_from_ckpt:
                    print(f"Padding signals from {current_signal_dim} to {signal_dim_from_ckpt} channels to match pretrained model...")
                    # Convert to numpy array if needed
                    if not isinstance(train_signals, np.ndarray):
                        train_signals = np.array(train_signals)
                    if not isinstance(val_signals, np.ndarray):
                        val_signals = np.array(val_signals)
                    
                    # Ensure 3D shape
                    if train_signals.ndim == 2:
                        train_signals = train_signals[:, :, np.newaxis]
                    if val_signals.ndim == 2:
                        val_signals = val_signals[:, :, np.newaxis]
                    
                    # Pad along last dimension
                    if train_signals.shape[-1] < signal_dim_from_ckpt:
                        pad_size = signal_dim_from_ckpt - train_signals.shape[-1]
                        padding = np.zeros((train_signals.shape[0], train_signals.shape[1], pad_size))
                        train_signals = np.concatenate([train_signals, padding], axis=2)
                    
                    if val_signals.shape[-1] < signal_dim_from_ckpt:
                        pad_size = signal_dim_from_ckpt - val_signals.shape[-1]
                        padding = np.zeros((val_signals.shape[0], val_signals.shape[1], pad_size))
                        val_signals = np.concatenate([val_signals, padding], axis=2)
                    
                    print(f"  Padded train signals: {train_signals.shape}")
                    print(f"  Padded val signals: {val_signals.shape}")
                    
                    # Verify the actual signal_dim after padding matches what we set
                    actual_signal_dim_after_padding = train_signals.shape[-1]
                    if actual_signal_dim_after_padding != signal_dim_from_ckpt:
                        raise ValueError(
                            f"Signal dimension mismatch after padding: "
                            f"expected {signal_dim_from_ckpt} (from checkpoint), "
                            f"but got {actual_signal_dim_after_padding} after padding"
                        )
                    print(f"  Verified: Actual signal_dim after padding = {actual_signal_dim_after_padding}")
        
        # Store signal_dim_from_ckpt for later use when creating the model
        # We'll set model_config['signal_dim'] later, but we need to remember this value
        checkpoint_signal_dim = signal_dim_from_ckpt
    else:
        checkpoint_signal_dim = None
    
    # Get noise augmentation settings (if any)
    noise_config = config['data'].get('noise_augmentation', {})
    noise_enabled = noise_config.get('enabled', False)
    noise_std = noise_config.get('std', 0.01)
    noise_type = noise_config.get('type', 'gaussian')
    
    if noise_enabled:
        print(f"Noise augmentation enabled for training data:")
        print(f"  Type: {noise_type}")
        print(f"  Std: {noise_std}")
    
    # Create datasets (using padded signals if checkpoint was loaded)
    if args.task == 'diagnosis':
        train_dataset = PHMSignalDataset(
            train_signals, labels=train_labels,
            normalize=config['data']['normalize'],
            signal_window=config['data']['signal_window'],
            overlap=config['data']['overlap'],
            add_noise=noise_enabled,  # Only add noise to training data
            noise_std=noise_std,
            noise_type=noise_type
        )
        val_dataset = PHMSignalDataset(
            val_signals, labels=val_labels,
            normalize=config['data']['normalize'],
            signal_window=config['data']['signal_window'],
            overlap=config['data']['overlap'],
            add_noise=False  # No noise for validation
        )
    elif args.task == 'prognosis':
        train_dataset = PHMSignalDataset(
            train_signals, rul=train_rul,
            normalize=config['data']['normalize'],
            signal_window=config['data']['signal_window'],
            overlap=config['data']['overlap'],
            add_noise=noise_enabled,  # Only add noise to training data
            noise_std=noise_std,
            noise_type=noise_type
        )
        val_dataset = PHMSignalDataset(
            val_signals, rul=val_rul,
            normalize=config['data']['normalize'],
            signal_window=config['data']['signal_window'],
            overlap=config['data']['overlap'],
            add_noise=False  # No noise for validation
        )
    else:  # pretrain
        train_dataset = PHMSignalDataset(
            train_signals, normalize=config['data']['normalize'],
            signal_window=config['data']['signal_window'],
            overlap=config['data']['overlap'],
            add_noise=noise_enabled,  # Only add noise to training data
            noise_std=noise_std,
            noise_type=noise_type
        )
        val_dataset = PHMSignalDataset(
            val_signals, normalize=config['data']['normalize'],
            signal_window=config['data']['signal_window'],
            overlap=config['data']['overlap'],
            add_noise=False  # No noise for validation
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Determine model parameters from dataset
    model_config = config['model'].copy()
    
    # Extract tokenizer_stride if provided
    tokenizer_stride = model_config.pop('tokenizer_stride', model_config.get('patch_length', 256))
    
    # Detect actual input channels from data
    actual_input_channels = None
    if len(train_signals) > 0:
        sample_signal = train_signals[0]
        if sample_signal.ndim >= 2:
            actual_input_channels = sample_signal.shape[-1]
            print(f"Detected actual input channels: {actual_input_channels}")
    
    # If resuming from pretrained checkpoint, use the signal_dim we inferred earlier
    # (Signals were already padded above if needed)
    if checkpoint_signal_dim is not None:
        model_config['signal_dim'] = checkpoint_signal_dim
        print(f"Setting model signal_dim={checkpoint_signal_dim} to match checkpoint")
        
        # If input channels differ from signal_dim, add projection
        if actual_input_channels is not None and actual_input_channels != checkpoint_signal_dim:
            model_config['input_channels'] = actual_input_channels
            print(f"  Input channels ({actual_input_channels}) != signal_dim ({checkpoint_signal_dim})")
            print(f"  Will use channel projection: {actual_input_channels} -> {checkpoint_signal_dim}")
    else:
        # Determine signal_dim from actual data (for training from scratch)
        if actual_input_channels is not None:
            model_config['signal_dim'] = actual_input_channels
            print(f"Setting model signal_dim={actual_input_channels} from data")
        else:
            actual_input_channels = model_config.get('signal_dim', 24)
    
    # Add stride back
    model_config['tokenizer_stride'] = tokenizer_stride
    
    # For diagnosis, num_faults should match number of classes
    # Paper uses lt=1 but this may be per-fault-type, so we adapt per task
    num_classes = None  # Initialize
    if args.task == 'diagnosis' and train_labels is not None:
        num_classes = len(set(train_labels)) if train_labels is not None else 10
        # Paper uses lt=1, but for multi-class we need one token per class
        # Keep base value from config (1 per paper) but this gets adapted per task
        model_config['num_faults'] = num_classes
    else:
        model_config['num_faults'] = config['model'].get('num_faults', 1)
    
    # Create model
    print("Creating RmGPT model...")
    print(f"  Model config signal_dim: {model_config['signal_dim']}")
    print(f"  Model config patch_length: {model_config['patch_length']}")
    print(f"  Expected patch_embed input dim: {model_config['signal_dim'] * model_config['patch_length']}")
    
    # Filter out non-RmGPT parameters (like improved_diagnosis_head)
    rmgpt_params = {
        'signal_dim', 'patch_length', 'tokenizer_stride',
        'embed_dim', 'num_prompts', 'num_faults',
        'num_layers', 'num_heads', 'ff_dim', 'dropout',
        'n_fft', 'wavelet', 'wavelet_levels', 'input_channels'
    }
    filtered_model_config = {k: v for k, v in model_config.items() if k in rmgpt_params}
    model = RmGPT(**filtered_model_config)
    
    # Verify model was created with correct dimensions
    actual_patch_embed_input = model.signal_tokenizer.patch_embed.in_features
    expected_patch_embed_input = model_config['signal_dim'] * model_config['patch_length']
    if actual_patch_embed_input != expected_patch_embed_input:
        raise ValueError(
            f"Model patch_embed input dimension mismatch: "
            f"expected {expected_patch_embed_input} (signal_dim={model_config['signal_dim']} * patch_length={model_config['patch_length']}), "
            f"but got {actual_patch_embed_input}"
        )
    print(f"  Verified: Model patch_embed input dim = {actual_patch_embed_input}")
    
    # Create task heads
    diagnosis_head = None
    prognosis_head = None
    
    if args.task == 'diagnosis':
        # Ensure num_classes is defined
        if num_classes is None and train_labels is not None:
            num_classes = len(set(train_labels))
        elif num_classes is None:
            num_classes = 10  # Default fallback
        # Use improved head if specified in config
        use_improved_head = config.get('model', {}).get('improved_diagnosis_head', False)
        diagnosis_head = DiagnosisHead(
            embed_dim=model_config['embed_dim'],
            num_classes=num_classes,
            improved=use_improved_head
        ).to(device)
    elif args.task == 'prognosis':
        prognosis_head = PrognosisHead(
            embed_dim=model_config['embed_dim']
        ).to(device)
    
    # Calculate total training steps for scheduler
    total_steps = None
    if args.task != 'pretrain':
        # For fine-tuning, calculate total steps
        steps_per_epoch = len(train_loader)
        num_epochs = config['training']['finetune_epochs']
        total_steps = steps_per_epoch * num_epochs
    
    # Get head learning rate (if specified, otherwise use same as backbone)
    head_lr = config['training'].get('head_lr', None)
    head_params = None
    if args.task == 'diagnosis' and diagnosis_head is not None:
        head_params = list(diagnosis_head.parameters())
    elif args.task == 'prognosis' and prognosis_head is not None:
        head_params = list(prognosis_head.parameters())
    
    # Create trainer
    trainer = RmGPTTrainer(
        model=model,
        device=device,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        total_steps=total_steps,
        lr_schedule=config['training'].get('lr_schedule', 'constant'),
        min_lr=config['training'].get('min_lr', 1e-6),
        head_lr=head_lr,
        head_params=head_params,
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        use_focal_loss=config['training'].get('use_focal_loss', False),
        focal_alpha=config['training'].get('focal_alpha', 0.25),
        focal_gamma=config['training'].get('focal_gamma', 2.0)
    )
    
    # Training loop
    num_epochs = config['training']['pretrain_epochs'] if args.task == 'pretrain' else config['training']['finetune_epochs']
    
    # Resume from checkpoint if provided
    # IMPORTANT: For fine-tuning, always start from PRETRAINED checkpoint, not fine-tuned checkpoint
    # The diagnosis head will be randomly initialized if checkpoint has mismatched architecture
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        print("Note: If checkpoint has mismatched diagnosis head architecture, it will be randomly initialized")
        print("      This is expected when fine-tuning from pretrained model on a new dataset")
        saved_epoch = trainer.load_checkpoint(args.resume, diagnosis_head, prognosis_head)
        
        # Determine if this is a fine-tuning checkpoint or pretraining checkpoint
        # If saved_epoch is high (e.g., > 10), it's likely from a previous fine-tuning run
        # If saved_epoch is low (e.g., < 20) and we're fine-tuning, it might be from pretraining
        # For now, we'll check: if resuming and saved_epoch exists, resume from that epoch
        if args.task in ['diagnosis', 'prognosis']:
            if saved_epoch is not None:
                # Check if this looks like a fine-tuning checkpoint (epoch >= 10) or pretraining (epoch < 20)
                if saved_epoch >= 10:
                    # Likely a fine-tuning checkpoint - resume from saved epoch
                    start_epoch = saved_epoch + 1  # Resume from next epoch
                    if start_epoch >= num_epochs:
                        print(f"Warning: Checkpoint was at epoch {saved_epoch}, but config only has {num_epochs} epochs.")
                        print(f"Resetting to epoch 0.")
                        start_epoch = 0
                    else:
                        print(f"Resuming fine-tuning from epoch {start_epoch}/{num_epochs} (checkpoint was at epoch {saved_epoch})")
                else:
                    # Likely a pretraining checkpoint - start fresh fine-tuning
                    print(f"Note: Checkpoint was at epoch {saved_epoch} (likely from pretraining)")
                    print(f"Starting fine-tuning from epoch 0/{num_epochs} (fresh fine-tuning, pretrained weights loaded)")
                    start_epoch = 0
            else:
                # No epoch info in checkpoint
                print(f"Warning: No epoch info in checkpoint. Starting from epoch 0.")
                start_epoch = 0
        elif args.task == 'pretrain':
            # For pretraining, resume from saved epoch
            if saved_epoch is not None:
                start_epoch = saved_epoch + 1  # Resume from next epoch
                if start_epoch >= num_epochs:
                    print(f"Warning: Checkpoint was at epoch {saved_epoch}, but config only has {num_epochs} epochs.")
                    print(f"Resetting to epoch 0.")
                    start_epoch = 0
                else:
                    print(f"Resuming pretraining from epoch {start_epoch}/{num_epochs}")
        elif args.start_epoch is not None:
            # Manual start epoch specified
            start_epoch = args.start_epoch
            if start_epoch >= num_epochs:
                print(f"Warning: --start-epoch {start_epoch} >= num_epochs {num_epochs}. Resetting to 0.")
                start_epoch = 0
            print(f"Checkpoint loaded. Starting from epoch {start_epoch}/{num_epochs} (manually specified)")
        else:
            print(f"Warning: Checkpoint loaded but no epoch info found.")
            print(f"Starting from epoch 0 (weights loaded, but epochs will repeat).")
            print(f"To start from specific epoch, use --start-epoch N")
    
    print(f"Starting {args.task} training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            trainer, train_loader, args.task,
            diagnosis_head=diagnosis_head,
            prognosis_head=prognosis_head,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        )
        
        # Validate (simplified - just for monitoring)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                signals = batch['signals'].to(device)
                # Simple validation - compute loss without backward pass
                # (This is simplified; full validation would use proper evaluation)
                pass
        
        print(f"Epoch {epoch+1}: Train Loss = {train_metrics.get('loss', 0):.4f}")
        if 'accuracy' in train_metrics:
            print(f"  Train Accuracy = {train_metrics['accuracy']:.4f}")
        if 'mae' in train_metrics:
            print(f"  Train MAE = {train_metrics['mae']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, diagnosis_head, prognosis_head, epoch=epoch)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(
        config['logging']['checkpoint_dir'],
        f"final_model_{args.task}.pt"
    )
    trainer.save_checkpoint(final_checkpoint_path, diagnosis_head, prognosis_head, epoch=num_epochs-1)
    print(f"Training complete! Final model saved to {final_checkpoint_path}")


if __name__ == '__main__':
    main()
