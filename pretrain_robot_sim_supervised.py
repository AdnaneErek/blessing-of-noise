"""
Supervised pretraining: Train RmGPT (backbone + head) on robot simulation data with classification labels.
This learns class-discriminative features from simulation data, then fine-tune on real data.
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path

from model.rmgpt import RmGPT, DiagnosisHead
from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import load_robot_training_data
from data.split_strategy import paper_split_strategy
from train.trainer import RmGPTTrainer, train_epoch


def main():
    parser = argparse.ArgumentParser(description='Supervised pretraining: Train RmGPT on robot simulation data')
    parser.add_argument('--config', type=str, default='configs/pretrain_robot_sim_supervised.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load robot simulation data
    print("\n" + "="*60)
    print("Loading Robot Simulation Data for Supervised Pretraining")
    print("="*60)
    
    data_dir = config['data'].get('data_dir', './data/raw/dataset')
    training_folder = config['data'].get('robot_training_folder', None)
    use_individual_files = config['data'].get('use_individual_files', True)
    
    all_signals, all_labels = load_robot_training_data(
        data_dir=data_dir,
        folder_name=training_folder,  # None = auto-discover all folders
        use_individual_files=use_individual_files
    )
    
    print(f"\nTotal loaded: {len(all_signals)} samples")
    print(f"Signal shape: {all_signals.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Number of classes: {len(set(all_labels))}")
    
    # Split data: 80% train, 20% test (paper-compliant)
    # For supervised pretraining, we use 'finetune_train' and 'finetune_val' splits
    splits = paper_split_strategy(
        signals=all_signals,
        labels=all_labels,
        rul=None,
        test_size=config['data']['test_size'],
        finetune_val_size=config['data']['finetune_val_size'],
        random_state=config['data']['random_state'],
        stratify=True
    )
    
    train_signals, train_labels, _ = splits['finetune_train']
    val_signals, val_labels, _ = splits['finetune_val']
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_signals)} samples")
    print(f"  Val: {len(val_signals)} samples")
    
    # Get noise augmentation settings (if any)
    noise_config = config['data'].get('noise_augmentation', {})
    noise_enabled = noise_config.get('enabled', False)
    noise_std = noise_config.get('std', 0.01)
    noise_type = noise_config.get('type', 'gaussian')
    
    if noise_enabled:
        print(f"\nNoise augmentation enabled for training data:")
        print(f"  Type: {noise_type}")
        print(f"  Std: {noise_std}")
    
    # Create datasets
    train_dataset = PHMSignalDataset(
        train_signals,
        labels=train_labels,
        normalize=config['data']['normalize'],
        signal_window=config['data']['signal_window'],
        overlap=config['data']['overlap'],
        add_noise=noise_enabled,  # Only add noise to training data
        noise_std=noise_std,
        noise_type=noise_type
    )
    val_dataset = PHMSignalDataset(
        val_signals,
        labels=val_labels,
        normalize=config['data']['normalize'],
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
    
    # Determine model parameters from data
    model_config = config['model'].copy()
    
    # Detect actual input channels from data
    sample_signal = train_signals[0]
    actual_input_channels = sample_signal.shape[-1] if sample_signal.ndim >= 2 else model_config.get('signal_dim', 9)
    model_config['signal_dim'] = actual_input_channels
    
    # Determine number of classes
    num_classes = len(set(train_labels))
    model_config['num_faults'] = num_classes
    
    print(f"\nModel configuration:")
    print(f"  signal_dim: {model_config['signal_dim']}")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print(f"  num_classes: {num_classes}")
    
    # Filter out non-RmGPT parameters
    rmgpt_params = {
        'signal_dim', 'patch_length', 'tokenizer_stride',
        'embed_dim', 'num_prompts', 'num_faults',
        'num_layers', 'num_heads', 'ff_dim', 'dropout',
        'n_fft', 'wavelet', 'wavelet_levels', 'input_channels'
    }
    filtered_model_config = {k: v for k, v in model_config.items() if k in rmgpt_params}
    
    # Create model
    model = RmGPT(**filtered_model_config)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Create diagnosis head
    use_improved_head = config.get('model', {}).get('improved_diagnosis_head', False)
    diagnosis_head = DiagnosisHead(
        embed_dim=model_config['embed_dim'],
        num_classes=num_classes,
        improved=use_improved_head
    ).to(device)
    print(f"Diagnosis head created: {sum(p.numel() for p in diagnosis_head.parameters())/1e6:.2f}M parameters")
    
    # Create trainer
    num_epochs = config['training']['pretrain_epochs']
    total_steps = len(train_loader) * num_epochs
    
    # Get head learning rate (if specified, otherwise use same as backbone)
    head_lr = config['training'].get('head_lr', None)
    head_params = list(diagnosis_head.parameters())
    
    trainer = RmGPTTrainer(
        model=model,
        device=device,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        total_steps=total_steps,
        lr_schedule=config['training']['lr_schedule'],
        min_lr=config['training']['min_lr'],
        head_lr=head_lr,
        head_params=head_params,
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        use_focal_loss=config['training'].get('use_focal_loss', False),
        focal_alpha=config['training'].get('focal_alpha', 0.25),
        focal_gamma=config['training'].get('focal_gamma', 2.0)
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        saved_epoch = trainer.load_checkpoint(args.resume, diagnosis_head=diagnosis_head)
        if saved_epoch is not None:
            start_epoch = saved_epoch + 1
            if start_epoch >= num_epochs:
                print(f"Warning: Checkpoint was at epoch {saved_epoch}, but config only has {num_epochs} epochs.")
                print(f"Resetting to epoch 0.")
                start_epoch = 0
            else:
                print(f"\nResuming supervised pretraining from epoch {start_epoch}/{num_epochs}")
        else:
            print(f"\nWarning: No epoch info in checkpoint. Starting from epoch 0.")
            start_epoch = 0
    else:
        print(f"\nStarting supervised pretraining from scratch for {num_epochs} epochs")
    
    # Training loop
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    early_stopping_cfg = config.get('training', {}).get('early_stopping', {})
    early_stopping_enabled = early_stopping_cfg.get('enabled', False)
    early_stopping_patience = int(early_stopping_cfg.get('patience', 10))
    early_stopping_min_delta = float(early_stopping_cfg.get('min_delta', 0.0))
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        train_metrics = train_epoch(
            trainer, train_loader, 'diagnosis',
            diagnosis_head=diagnosis_head,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        )
        
        # Validation
        model.eval()
        diagnosis_head.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                signals = batch['signals'].to(device)
                labels = batch['labels'].to(device).long()
                
                # Forward pass
                output = model(signals, task_type='diagnosis', return_tokens=False)
                features = output['features']
                
                # Use time-freq token for classification
                tf_token_idx = model.num_prompts
                tf_features = features[:, tf_token_idx, :]  # [batch, embed_dim]
                
                # Classification
                logits = diagnosis_head(tf_features.unsqueeze(1))  # [batch, num_classes]
                logits = logits.squeeze(1)  # [batch, num_classes]
                
                # Loss
                if config['training'].get('use_focal_loss', False):
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits, labels, reduction='none',
                        label_smoothing=config['training'].get('label_smoothing', 0.0)
                    )
                    pt = torch.exp(-ce_loss)
                    loss = config['training'].get('focal_alpha', 0.25) * (1 - pt) ** config['training'].get('focal_gamma', 2.0) * ce_loss
                    loss = loss.mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits, labels,
                        label_smoothing=config['training'].get('label_smoothing', 0.0)
                    )
                
                # Accuracy
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == labels).float().mean()
                
                val_losses.append(loss.item())
                val_accs.append(accuracy.item())
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy.item():.4f}"
                })
        
        avg_train_loss = train_metrics.get('loss', 0.0)
        avg_train_acc = train_metrics.get('accuracy', 0.0)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_every'] == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = checkpoint_dir / f"pretrain_robot_sim_supervised_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(
                str(checkpoint_path),
                diagnosis_head=diagnosis_head,
                epoch=epoch
            )
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_val_acc > (best_val_acc + early_stopping_min_delta):
            best_val_acc = avg_val_acc
            epochs_without_improvement = 0
            best_checkpoint_path = checkpoint_dir / "pretrain_robot_sim_supervised_best.pt"
            trainer.save_checkpoint(
                str(best_checkpoint_path),
                diagnosis_head=diagnosis_head,
                epoch=epoch
            )
            print(f"  Saved best model (val_acc={avg_val_acc:.4f}): {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1

        # Early stopping when validation accuracy has not improved for N epochs
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            print(
                f"  Early stopping triggered at epoch {epoch+1}: "
                f"no val_acc improvement > {early_stopping_min_delta} for "
                f"{early_stopping_patience} consecutive epochs."
            )
            break
    
    # Save final model
    final_checkpoint_path = checkpoint_dir / "pretrain_robot_sim_supervised_final.pt"
    trainer.save_checkpoint(
        str(final_checkpoint_path),
        diagnosis_head=diagnosis_head,
        epoch=num_epochs - 1
    )
    print(f"\nSupervised pretraining complete! Final model saved: {final_checkpoint_path}")


if __name__ == '__main__':
    main()
