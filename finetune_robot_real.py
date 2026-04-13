#!/usr/bin/env python3
"""
Fine-tune RmGPT model on real robot data

This script:
1. Loads a checkpoint from simulation training
2. Fine-tunes on real robot data from finetuningDatasets
3. Uses different learning rates for backbone (low) and head (high)
4. Saves the fine-tuned model

Usage:
    python finetune_robot_real.py --config configs/finetune_robot_real.yaml --checkpoint checkpoints/final_model_diagnosis.pt
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split

from model.rmgpt import RmGPT, DiagnosisHead
from train.trainer import RmGPTTrainer, train_epoch
from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import load_robot_finetuning_data


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Fine-tune RmGPT model on real robot data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to simulation-trained checkpoint to fine-tune from')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume fine-tuning from (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['evaluation']['save_dir'], exist_ok=True)
    
    # Load checkpoint to infer model dimensions
    print(f"\nLoading simulation checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer signal_dim from checkpoint weights
    patch_embed_shape = state_dict['signal_tokenizer.patch_embed.weight'].shape
    patch_embed_input_dim = patch_embed_shape[1]
    patch_length = config['model']['patch_length']
    signal_dim_from_ckpt = patch_embed_input_dim // patch_length
    
    # Infer num_classes from diagnosis_head
    num_classes_from_ckpt = None
    if 'diagnosis_head_state_dict' in checkpoint:
        diag_head_state = checkpoint['diagnosis_head_state_dict']
        for key in sorted(diag_head_state.keys(), reverse=True):
            if key.endswith('.weight') and len(diag_head_state[key].shape) == 2:
                num_classes_from_ckpt = diag_head_state[key].shape[0]
                break
    
    print(f"Inferred from checkpoint:")
    print(f"  signal_dim: {signal_dim_from_ckpt}")
    print(f"  num_classes: {num_classes_from_ckpt}")
    
    # Load real robot fine-tuning data
    print("\n" + "="*60)
    print("Loading real robot fine-tuning data")
    print("="*60)
    
    data_dir = config['data'].get('data_dir', './data/raw/dataset')
    finetuning_folder = config['data'].get('finetuning_folder', None)
    
    all_signals, all_labels = load_robot_finetuning_data(
        data_dir=data_dir,
        folder_name=finetuning_folder,  # None = auto-discover all folders
        use_individual_files=True
    )
    
    print(f"\nTotal real data loaded: {len(all_signals)} samples")
    print(f"Signal shape: {all_signals.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Unique labels: {np.unique(all_labels)}")
    
    # Split into train/val (simple split, not paper-compliant since this is fine-tuning data)
    val_size = config['data'].get('finetune_val_size', 0.1)
    random_state = config['data'].get('random_state', 42)
    
    train_signals, val_signals, train_labels, val_labels = train_test_split(
        all_signals, all_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=all_labels  # Stratified split to maintain class distribution
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_signals)} samples")
    print(f"  Val: {len(val_signals)} samples")
    
    # Create datasets (no noise augmentation for real data)
    train_dataset = PHMSignalDataset(
        train_signals, labels=train_labels,
        normalize=config['data']['normalize'],
        signal_window=config['data']['signal_window'],
        overlap=config['data']['overlap'],
        add_noise=False  # No noise for real data
    )
    val_dataset = PHMSignalDataset(
        val_signals, labels=val_labels,
        normalize=config['data']['normalize'],
        signal_window=config['data']['signal_window'],
        overlap=config['data']['overlap'],
        add_noise=False
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
    
    # Determine model parameters
    model_config = config['model'].copy()
    model_config['signal_dim'] = signal_dim_from_ckpt
    model_config['input_channels'] = signal_dim_from_ckpt
    
    # Extract tokenizer_stride
    tokenizer_stride = model_config.pop('tokenizer_stride', model_config.get('patch_length', 256))
    model_config['tokenizer_stride'] = tokenizer_stride
    
    # Filter out non-RmGPT parameters
    rmgpt_params = {
        'signal_dim', 'patch_length', 'tokenizer_stride',
        'embed_dim', 'num_prompts', 'num_faults',
        'num_layers', 'num_heads', 'ff_dim', 'dropout',
        'n_fft', 'wavelet', 'wavelet_levels', 'input_channels'
    }
    filtered_model_config = {k: v for k, v in model_config.items() if k in rmgpt_params}
    
    # Create model
    print("\nCreating RmGPT model...")
    model = RmGPT(**filtered_model_config)
    
    # Create diagnosis head
    num_classes = num_classes_from_ckpt or len(np.unique(all_labels))
    use_improved_head = config.get('model', {}).get('improved_diagnosis_head', False)
    diagnosis_head = DiagnosisHead(
        embed_dim=model_config['embed_dim'],
        num_classes=num_classes,
        improved=use_improved_head
    ).to(device)
    
    print(f"Created diagnosis head: num_classes={num_classes}, improved={use_improved_head}")
    
    # Calculate total training steps for scheduler
    steps_per_epoch = len(train_loader)
    num_epochs = config['training']['finetune_epochs']
    total_steps = steps_per_epoch * num_epochs
    
    # Get head learning rate
    head_lr = config['training'].get('head_lr', None)
    head_params = list(diagnosis_head.parameters())
    
    # Check if we should freeze backbone (before creating trainer)
    freeze_backbone = config['training'].get('freeze_backbone', False)
    
    # Create trainer with different LRs for backbone and head
    # If backbone is frozen, we only need to optimize the head
    if freeze_backbone:
        # When backbone is mostly frozen, we still need diagnosis_head in optimizer.
        # We pass head_params so the head is explicitly trainable.
        head_only_lr = head_lr if head_lr is not None else config['training']['lr']
        print(f"\nCreating trainer (backbone frozen):")
        print(f"  Head LR: {head_only_lr}")
        trainer = RmGPTTrainer(
            model=model,
            device=device,
            lr=head_only_lr,  # Keeps unfreezed backbone layers at conservative LR
            weight_decay=config['training']['weight_decay'],
            warmup_steps=config['training']['warmup_steps'],
            max_grad_norm=config['training']['max_grad_norm'],
            total_steps=total_steps,
            lr_schedule=config['training'].get('lr_schedule', 'constant'),
            min_lr=config['training'].get('min_lr', 1e-6),
            head_lr=head_only_lr,
            head_params=head_params,
            label_smoothing=config['training'].get('label_smoothing', 0.0),
            use_focal_loss=config['training'].get('use_focal_loss', False),
            focal_alpha=config['training'].get('focal_alpha', 0.25),
            focal_gamma=config['training'].get('focal_gamma', 2.0)
        )
    else:
        print(f"\nCreating trainer with:")
        print(f"  Backbone LR: {config['training']['lr']}")
        print(f"  Head LR: {head_lr}")
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
    
    # Load checkpoint (simulation-trained model)
    print(f"\nLoading simulation checkpoint: {args.checkpoint}")
    saved_epoch = trainer.load_checkpoint(args.checkpoint, diagnosis_head, None)
    print(f"Loaded checkpoint (was at epoch {saved_epoch})")
    
    # Optionally freeze backbone (only fine-tune head) - AFTER loading checkpoint
    if freeze_backbone:
        # Get number of layers to unfreeze
        num_layers = config['model']['num_layers']  # Should be 4
        unfreeze_percentage = config['training'].get('unfreeze_percentage', 0.25)  # Default: last 25%
        num_layers_to_unfreeze = max(1, int(num_layers * unfreeze_percentage))  # At least 1 layer
        
        print("\n" + "="*60)
        print(f"PROGRESSIVE UNFREEZING - Unfreezing last {num_layers_to_unfreeze}/{num_layers} transformer layers")
        print("="*60)
        
        # Freeze everything first
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer layers
        total_layers = len(model.transformer.layers)
        layers_to_unfreeze = total_layers - num_layers_to_unfreeze
        
        for i in range(layers_to_unfreeze, total_layers):
            for param in model.transformer.layers[i].parameters():
                param.requires_grad = True
            print(f"  Unfrozen transformer layer {i+1}/{total_layers}")
        
        # Unfreeze diagnosis head
        for param in diagnosis_head.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params += sum(p.numel() for p in diagnosis_head.parameters() if p.requires_grad)
        print(f"\nTrainable parameters: {trainable_params:,}")
        print(f"  - Last {num_layers_to_unfreeze} transformer layers")
        print(f"  - Diagnosis head")
    else:
        print("\n" + "="*60)
        print("FINE-TUNING ENTIRE MODEL (backbone + head)")
        print("="*60)
        print("Using different learning rates:")
        print(f"  Backbone LR: {config['training']['lr']}")
        print(f"  Head LR: {head_lr}")
    
    # Determine start epoch
    start_epoch = 0
    if args.resume:
        print(f"Resuming from fine-tuning checkpoint: {args.resume}")
        saved_epoch = trainer.load_checkpoint(args.resume, diagnosis_head, None)
        if saved_epoch is not None:
            start_epoch = saved_epoch + 1
            if start_epoch >= num_epochs:
                print(f"Warning: Checkpoint was at epoch {saved_epoch}, but config only has {num_epochs} epochs.")
                print(f"Resetting to epoch 0.")
                start_epoch = 0
            else:
                print(f"Resuming fine-tuning from epoch {start_epoch}/{num_epochs}")
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    print(f"Training on {len(train_signals)} real samples")
    print(f"Validating on {len(val_signals)} real samples")
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            trainer, train_loader, 'diagnosis',
            diagnosis_head=diagnosis_head,
            prognosis_head=None,
            desc=f"Epoch {epoch+1}/{num_epochs} [Fine-tune Train]"
        )
        
        # Validate
        model.eval()
        diagnosis_head.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                signals = batch['signals'].to(device)
                labels = batch['labels'].to(device).long()
                
                # Forward pass
                output = model(signals, task_type='diagnosis', return_tokens=False)
                features = output['features']
                
                # Use time-freq token features for classification
                tf_token_idx = model.num_prompts
                tf_features = features[:, tf_token_idx, :]
                
                # Classification
                logits = diagnosis_head(tf_features.unsqueeze(1))
                logits = logits.squeeze(1)
                
                # Loss
                if trainer.use_focal_loss:
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits, labels, reduction='none',
                        label_smoothing=trainer.label_smoothing
                    )
                    pt = torch.exp(-ce_loss)
                    loss = trainer.focal_alpha * (1 - pt) ** trainer.focal_gamma * ce_loss
                    loss = loss.mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits, labels, label_smoothing=trainer.label_smoothing
                    )
                
                val_losses.append(loss.item())
                
                # Accuracy
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        print(f"Epoch {epoch+1}: Train Loss = {train_metrics.get('loss', 0):.4f}, "
              f"Train Acc = {train_metrics.get('accuracy', 0):.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f"best_finetune_real_epoch_{epoch+1}.pt"
            )
            trainer.save_checkpoint(best_checkpoint_path, diagnosis_head, None, epoch=epoch)
            print(f"  → Saved best model (val_acc={val_acc:.4f}) to {best_checkpoint_path}")
        
        # Save checkpoint periodically
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f"finetune_real_epoch_{epoch+1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, diagnosis_head, None, epoch=epoch)
            print(f"  → Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(
        config['logging']['checkpoint_dir'],
        f"final_model_finetune_real.pt"
    )
    trainer.save_checkpoint(final_checkpoint_path, diagnosis_head, None, epoch=num_epochs-1)
    print(f"\nFine-tuning complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final model saved to: {final_checkpoint_path}")


if __name__ == '__main__':
    main()
