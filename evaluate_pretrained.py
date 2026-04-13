#!/usr/bin/env python3
"""
Evaluation script for pretrained RmGPT model

Evaluates the pretrained model on:
1. Next-token prediction loss (pretraining objective) on test set
2. Feature quality via linear probing (optional)

Usage:
    python evaluate_pretrained.py --config configs/paper_exact_config.yaml --checkpoint checkpoints/final_model_pretrain.pt
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error

from model.rmgpt import RmGPT, DiagnosisHead, PrognosisHead
from train.trainer import RmGPTTrainer
from data.dataset import PHMSignalDataset
from data.split_strategy import paper_split_from_phmd


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_pretrain_loss(model: RmGPT, dataloader: DataLoader, device: torch.device):
    """
    Evaluate next-token prediction loss (pretraining objective)
    
    Args:
        model: Pretrained RmGPT model
        dataloader: Test data loader
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating pretrain loss"):
            signals = batch['signals'].to(device)
            batch_size, seq_len, signal_dim = signals.shape
            
            # Forward pass
            output = model(signals, task_type='pretrain', return_tokens=False)
            features = output['features']  # [batch, total_seq_len, embed_dim]
            
            # Next-token prediction loss
            input_features = features[:, :-1, :]
            target_features = features[:, 1:, :]
            
            # Predict next token
            pred_next = model.transformer(input_features)
            
            # MSE loss
            loss = F.mse_loss(pred_next, target_features, reduction='mean')
            losses.append(loss.item())
    
    return {
        'pretrain_loss': np.mean(losses),
        'pretrain_loss_std': np.std(losses)
    }


def evaluate_linear_probe(model: RmGPT, 
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          test_loader: DataLoader,
                          num_classes: int,
                          device: torch.device,
                          epochs: int = 10,
                          lr: float = 0.001):
    """
    Evaluate learned features via linear probing
    
    Adds a simple classifier on top of frozen pretrained features
    and trains only the classifier to assess feature quality.
    
    Args:
        model: Pretrained RmGPT model (frozen)
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device to run on
        epochs: Number of epochs for linear probing
        lr: Learning rate for linear probe
        
    Returns:
        Dictionary of metrics
    """
    model.eval()  # Freeze pretrained model
    
    # Create linear probe classifier
    embed_dim = model.embed_dim
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train linear probe
    print(f"Training linear probe for {epochs} epochs...")
    for epoch in range(epochs):
        classifier.train()
        for batch in tqdm(train_loader, desc=f"Linear probe epoch {epoch+1}/{epochs}", leave=False):
            signals = batch['signals'].to(device)
            labels = batch['labels'].to(device).long()
            
            # Extract features from pretrained model
            with torch.no_grad():
                output = model(signals, task_type='diagnosis', return_tokens=False)
                features = output['features']
                # Use time-freq token features
                tf_token_idx = model.num_prompts
                tf_features = features[:, tf_token_idx, :]  # [batch, embed_dim]
            
            # Classify
            logits = classifier(tf_features)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation and test sets
    results = {}
    
    for split_name, loader in [('val', val_loader), ('test', test_loader)]:
        classifier.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating on {split_name}"):
                signals = batch['signals'].to(device)
                labels = batch['labels'].to(device).long()
                
                # Extract features
                output = model(signals, task_type='diagnosis', return_tokens=False)
                features = output['features']
                tf_token_idx = model.num_prompts
                tf_features = features[:, tf_token_idx, :]
                
                # Classify
                logits = classifier(tf_features)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        results[f'{split_name}_accuracy'] = accuracy
        results[f'{split_name}_precision'] = precision
        results[f'{split_name}_recall'] = recall
        results[f'{split_name}_f1'] = f1
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained RmGPT model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--linear-probe', action='store_true',
                       help='Also evaluate via linear probing (requires labels)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name for evaluation (default: from config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['evaluation']['save_dir'], exist_ok=True)
    
    # Determine dataset for evaluation
    dataset_name = args.dataset or config['data']['dataset_name']
    
    # Determine task_name for this specific dataset
    # Check if dataset is in pretrain datasets list
    pretrain_datasets = config['data'].get('pretrain_datasets', [])
    pretrain_task_names = config['data'].get('pretrain_task_names', [])
    
    if dataset_name in pretrain_datasets:
        # Get corresponding task name
        idx = pretrain_datasets.index(dataset_name)
        task_name = pretrain_task_names[idx] if idx < len(pretrain_task_names) else config['data']['task_name']
    else:
        # Use default task name from config
        task_name = config['data']['task_name']
    
    print(f"Evaluating pretrained model on dataset: {dataset_name}")
    print(f"Task type: {task_name}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load test data with paper-compliant split
    from data.split_strategy import paper_split_from_phmd
    
    splits = paper_split_from_phmd(
        dataset_name=dataset_name,
        task_name=task_name,
        test_size=config['data'].get('test_size', 0.2),
        finetune_val_size=config['data'].get('finetune_val_size', 0.1),
        random_state=config['data'].get('random_state', 42)
    )
    
    # Use test set for evaluation (the 20% that was untouched)
    test_signals, test_labels, test_rul = splits['test']
    
    print(f"Test samples: {len(test_signals)}")
    print(f"Test set was NOT used during pretraining (20% untouched)")
    
    # Load checkpoint FIRST to infer model dimensions BEFORE creating datasets
    print("Loading checkpoint to infer model dimensions...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Infer signal_dim from checkpoint weights
    # patch_embed weight shape is [embed_dim, patch_length * signal_dim]
    patch_embed_shape = state_dict['signal_tokenizer.patch_embed.weight'].shape
    embed_dim_from_ckpt = patch_embed_shape[0]
    patch_length_x_signal_dim = patch_embed_shape[1]
    
    # Infer signal_dim from checkpoint
    patch_length = config['model']['patch_length']  # Should be 256
    signal_dim_from_ckpt = patch_length_x_signal_dim // patch_length
    
    # Infer tf_proj dimensions to validate
    tf_proj_shape = state_dict['tf_tokenizer.tf_proj.weight'].shape
    tf_input_dim = tf_proj_shape[1]
    
    print(f"Inferred from checkpoint:")
    print(f"  embed_dim: {embed_dim_from_ckpt}")
    print(f"  patch_length * signal_dim: {patch_length_x_signal_dim}")
    print(f"  signal_dim: {signal_dim_from_ckpt}")
    print(f"  tf_input_dim: {tf_input_dim}")
    
    # Pad test signals to match model's expected signal_dim (if needed)
    # During pretraining, signals were padded to max_channels for consistency
    # First, ensure signals are in numpy array format
    if not isinstance(test_signals, np.ndarray):
        test_signals = np.array(test_signals)
    
    if len(test_signals) > 0:
        # Handle different input shapes
        if test_signals.ndim == 2:
            # [num_samples, seq_len] -> [num_samples, seq_len, 1]
            test_signals = test_signals[:, :, np.newaxis]
        elif test_signals.ndim == 1:
            # [seq_len] -> [1, seq_len, 1]
            test_signals = test_signals.reshape(1, -1, 1)
        # If already 3D, keep as is
        
        # Now test_signals should be [num_samples, seq_len, num_channels]
        current_signal_dim = test_signals.shape[-1]
        
        if current_signal_dim < signal_dim_from_ckpt:
            print(f"Padding test signals from {current_signal_dim} to {signal_dim_from_ckpt} channels...")
            # Pad with zeros along the last dimension
            pad_size = signal_dim_from_ckpt - current_signal_dim
            padding = np.zeros((test_signals.shape[0], test_signals.shape[1], pad_size))
            test_signals = np.concatenate([test_signals, padding], axis=2)
            print(f"  Padded test signals shape: {test_signals.shape}")
        elif current_signal_dim > signal_dim_from_ckpt:
            print(f"Truncating test signals from {current_signal_dim} to {signal_dim_from_ckpt} channels...")
            # Truncate to match model's expected signal_dim
            test_signals = test_signals[..., :signal_dim_from_ckpt]
            print(f"  Truncated test signals shape: {test_signals.shape}")
    
    # NOW create test dataset and loader (after padding)
    test_dataset = PHMSignalDataset(
        test_signals, 
        labels=test_labels if args.linear_probe else None,
        normalize=config['data']['normalize'],
        signal_window=config['data']['signal_window'],
        overlap=config['data']['overlap']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Determine model parameters - use checkpoint dimensions, not data dimensions
    model_config = config['model'].copy()
    model_config['signal_dim'] = signal_dim_from_ckpt  # Use checkpoint dimension
    
    # Extract tokenizer_stride
    tokenizer_stride = model_config.pop('tokenizer_stride', model_config.get('patch_length', 256))
    model_config['tokenizer_stride'] = tokenizer_stride
    
    # Create model with checkpoint dimensions
    print("Creating model with checkpoint dimensions...")
    model = RmGPT(**model_config).to(device)
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluate pretraining loss on test set
    print("\n=== Evaluating Next-Token Prediction Loss (Pretraining Objective) ===")
    pretrain_metrics = evaluate_pretrain_loss(model, test_loader, device)
    
    print(f"\nPretrain Loss on Test Set: {pretrain_metrics['pretrain_loss']:.6f} ± {pretrain_metrics['pretrain_loss_std']:.6f}")
    
    results = {
        'checkpoint': args.checkpoint,
        'dataset': dataset_name,
        'test_samples': len(test_signals),
        'pretrain_metrics': pretrain_metrics
    }
    
    # Optional: Linear probing evaluation
    # Only run linear probing for diagnosis datasets (which have labels)
    # Prognosis datasets have RUL, not classification labels
    if args.linear_probe:
        if test_labels is None:
            print(f"\nWarning: test_labels is None for {dataset_name}. Cannot run linear probing.")
            print(f"  Task type: {task_name}")
            if task_name.startswith('Prognosis'):
                print(f"  This is a prognosis dataset - linear probing (classification) is not applicable.")
        elif not task_name.startswith('Diagnosis'):
            print(f"\nWarning: {dataset_name} has task type '{task_name}', not 'Diagnosis'. Skipping linear probing.")
        else:  # test_labels is not None and task_name.startswith('Diagnosis')
            print("\n=== Evaluating Feature Quality via Linear Probing ===")
            
            # Load train and val splits for linear probe training
            train_signals, train_labels, _ = splits['finetune_train']
            val_signals, val_labels, _ = splits['finetune_val']
            
            # Pad train and val signals to match model's expected signal_dim (if needed)
            for signal_list, split_name in [(train_signals, 'train'), (val_signals, 'val')]:
                if len(signal_list) > 0:
                    sample_signal = signal_list[0]
                    if sample_signal.ndim >= 2:
                        current_signal_dim = sample_signal.shape[-1]
                        if current_signal_dim < signal_dim_from_ckpt:
                            print(f"Padding {split_name} signals from {current_signal_dim} to {signal_dim_from_ckpt} channels...")
                            padded_signals = []
                            for signal in signal_list:
                                if signal.ndim == 2:
                                    pad_size = signal_dim_from_ckpt - signal.shape[-1]
                                    padding = np.zeros((signal.shape[0], pad_size))
                                    padded = np.concatenate([signal, padding], axis=1)
                                else:
                                    padded = np.zeros((signal.shape[0], signal_dim_from_ckpt))
                                    padded[:, 0] = signal if signal.ndim == 1 else signal.squeeze()
                                padded_signals.append(padded)
                            if split_name == 'train':
                                train_signals = np.array(padded_signals)
                            else:
                                val_signals = np.array(padded_signals)
                        elif current_signal_dim > signal_dim_from_ckpt:
                            if split_name == 'train':
                                train_signals = train_signals[..., :signal_dim_from_ckpt]
                            else:
                                val_signals = val_signals[..., :signal_dim_from_ckpt]
            
            train_dataset = PHMSignalDataset(
                train_signals, labels=train_labels,
                normalize=config['data']['normalize'],
                signal_window=config['data']['signal_window'],
                overlap=config['data']['overlap']
            )
            val_dataset = PHMSignalDataset(
                val_signals, labels=val_labels,
                normalize=config['data']['normalize'],
                signal_window=config['data']['signal_window'],
                overlap=config['data']['overlap']
            )
            
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
            
            num_classes = len(set(test_labels))
            linear_probe_metrics = evaluate_linear_probe(
                model, train_loader, val_loader, test_loader,
                num_classes, device, epochs=10
            )
            
            print("\nLinear Probe Results:")
            print(f"  Validation Accuracy: {linear_probe_metrics['val_accuracy']:.4f}")
            print(f"  Test Accuracy: {linear_probe_metrics['test_accuracy']:.4f}")
            print(f"  Test F1-Score: {linear_probe_metrics['test_f1']:.4f}")
            
            results['linear_probe_metrics'] = linear_probe_metrics
    
    # Save results
    results_path = os.path.join(
        config['evaluation']['save_dir'],
        f'eval_pretrained_{dataset_name}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
