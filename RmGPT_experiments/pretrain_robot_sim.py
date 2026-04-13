"""
Pretrain RmGPT on robot simulation data using next-token prediction (self-supervised learning)
This learns signal patterns from simulation data, then the head will be trained on real data.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path

from model.rmgpt import RmGPT
from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import load_robot_training_data
from data.split_strategy import paper_split_strategy
from train.trainer import RmGPTTrainer


def main():
    parser = argparse.ArgumentParser(description='Pretrain RmGPT on robot simulation data')
    parser.add_argument('--config', type=str, default='configs/pretrain_robot_sim.yaml',
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
    print("Loading Robot Simulation Data for Pretraining")
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
    
    # Split data: 80% train, 20% test (paper-compliant)
    # For pretraining, we use 'pretrain' split (all 80% train data) and 'finetune_val' for validation
    splits = paper_split_strategy(
        all_signals, all_labels,
        test_size=config['data']['test_size'],
        finetune_val_size=config['data']['finetune_val_size'],
        random_state=config['data']['random_state']
    )
    
    train_signals = splits['pretrain'][0]  # All 80% train data (unlabeled for pretraining)
    val_signals = splits['finetune_val'][0]  # Use validation split for pretraining validation
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_signals)} samples")
    print(f"  Val: {len(val_signals)} samples")
    
    # Create datasets (no noise for pretraining - clean signals)
    train_dataset = PHMSignalDataset(
        train_signals,
        normalize=config['data']['normalize'],
        signal_window=config['data']['signal_window'],
        overlap=config['data']['overlap'],
        add_noise=False  # No noise for pretraining
    )
    val_dataset = PHMSignalDataset(
        val_signals,
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
    
    # Determine model parameters from data
    model_config = config['model'].copy()
    
    # Detect actual input channels from data
    sample_signal = train_signals[0]
    actual_input_channels = sample_signal.shape[-1] if sample_signal.ndim >= 2 else model_config.get('signal_dim', 9)
    model_config['signal_dim'] = actual_input_channels
    model_config['num_faults'] = 1  # Not used in pretraining
    
    print(f"\nModel configuration:")
    print(f"  signal_dim: {model_config['signal_dim']}")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  num_layers: {model_config['num_layers']}")
    
    # Create model (no diagnosis head for pretraining)
    model = RmGPT(**model_config)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Create trainer
    num_epochs = config['training']['pretrain_epochs']
    total_steps = len(train_loader) * num_epochs
    
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
        head_lr=None,  # No head in pretraining
        head_params=None
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.load_checkpoint(checkpoint)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"\nResuming pretraining from epoch {start_epoch}/{num_epochs}")
    else:
        print(f"\nStarting pretraining from scratch for {num_epochs} epochs")
    
    # Training loop
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Get mask probability from config
    mask_prob = config['training'].get('mask_prob', 0.15)
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_pbar:
            metrics = trainer.pretrain_step(batch, mask_prob=mask_prob)
            train_losses.append(metrics['loss'])
            train_pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'lr': f"{metrics['lr']:.2e}"})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                signals = batch['signals'].to(device)
                batch_size = signals.shape[0]
                
                # Forward pass to get token embeddings
                output = model(signals, task_type='pretrain', return_tokens=True)
                features = output['features']  # [batch, total_seq_len, embed_dim]
                signal_tokens = output['signal_tokens']  # [batch, num_patches, embed_dim]
                
                # Get token sequence structure: [Prompt] [Time-Freq] [Signal Patches]
                num_prompts = model.num_prompts
                num_patches = signal_tokens.shape[1]
                signal_start_idx = num_prompts + 1  # After prompts and time-freq token
                
                # Extract signal token embeddings from the full sequence
                signal_token_features = features[:, signal_start_idx:signal_start_idx + num_patches, :]  # [batch, num_patches, embed_dim]
                
                # Create mask for signal tokens (mask 15% randomly per sample)
                num_tokens_to_mask = max(1, int(num_patches * mask_prob))
                mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
                for batch_idx in range(batch_size):
                    # Randomly select tokens to mask for this sample
                    mask_indices = torch.randperm(num_patches, device=device)[:num_tokens_to_mask]
                    mask[batch_idx, mask_indices] = True
                
                # Store original embeddings for masked positions
                masked_token_embeddings = signal_token_features[mask]  # [num_masked, embed_dim]
                
                # Replace masked tokens with [MASK] token (zero embedding)
                masked_features = features.clone()
                mask_embedding = torch.zeros_like(signal_token_features[0, 0])  # [embed_dim]
                for batch_idx in range(batch_size):
                    # Get masked token indices for this sample
                    sample_mask_indices = torch.where(mask[batch_idx])[0]
                    for token_idx in sample_mask_indices:
                        masked_features[batch_idx, signal_start_idx + token_idx, :] = mask_embedding
                
                # Pass masked sequence through transformer
                pred_features = model.transformer(masked_features)
                
                # Extract predictions for masked positions
                pred_masked_tokens = pred_features[:, signal_start_idx:signal_start_idx + num_patches, :]  # [batch, num_patches, embed_dim]
                pred_masked = pred_masked_tokens[mask]  # [num_masked, embed_dim]
                
                # MSE loss between predicted and original masked token embeddings
                loss = nn.functional.mse_loss(pred_masked, masked_token_embeddings)
                
                val_losses.append(loss.item())
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_val_loss = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_every'] == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = checkpoint_dir / f"pretrain_robot_sim_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = checkpoint_dir / "pretrain_robot_sim_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, best_checkpoint_path)
            print(f"  Saved best model (val_loss={avg_val_loss:.4f}): {best_checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = checkpoint_dir / "pretrain_robot_sim_final.pt"
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'config': config
    }, final_checkpoint_path)
    print(f"\nPretraining complete! Final model saved: {final_checkpoint_path}")


if __name__ == '__main__':
    main()
