"""
Pretrain MOMENT on robot simulation data using masked patch reconstruction.
Adapted from RmGPT pipeline settings to utilize the same data split and schedule.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
import math

from momentfm.models.moment import MOMENT
from momentfm.common import TASKS
from argparse import Namespace

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from data.robot_dataset_loader import load_robot_training_data
from sklearn.model_selection import train_test_split


class MomentPretrainDataset(Dataset):
    """
    Dataset wrapper for robot data arrays adapted for MOMENT.
    Signals shape comes in as: [num_samples, seq_len, num_features]
    MOMENT expects: [batch, features, seq_len]
    """
    def __init__(self, signals, normalize=True):
        self.signals = torch.from_numpy(signals).float()
        self.num_samples = self.signals.shape[0]
        self.seq_len = self.signals.shape[1]
        self.num_features = self.signals.shape[2]
        
        # We can apply physical normalization here or rely on MOMENT's RevIN 
        # MOMENT has its own internal normalizer which we use during the forward pass.

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Permute [seq_len, features] -> [features, seq_len]
        return self.signals[idx].permute(1, 0)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=1e-2):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description='Pretrain MOMENT on robot simulation data using external pipeline')
    parser.add_argument('--config', type=str, default='pretrain_moment_sim.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # -----------------------------------------------------
    # 1. Load Robot Simulation Data (using the same loader and splits as RmGPT)
    # -----------------------------------------------------
    print("\n" + "="*60)
    print("Loading Robot Simulation Data for Pretraining")
    print("="*60)
    
    data_dir = config['data'].get('data_dir', './data/raw/dataset')
    training_folder = config['data'].get('robot_training_folder', None)
    
    all_signals, all_labels = load_robot_training_data(
        data_dir=data_dir,
        folder_name=training_folder,
        use_individual_files=True
    )
    print(f"\nTotal loaded: {len(all_signals)} samples")
    print(f"Signal shape: {all_signals.shape}")
    
    val_size = config['data'].get('finetune_val_size', 0.1)
    train_signals, val_signals = train_test_split(
        all_signals,
        test_size=val_size,
        random_state=config['data'].get('random_state', 42)
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_signals)} samples")
    print(f"  Val: {len(val_signals)} samples")
    
    train_dataset = MomentPretrainDataset(train_signals)
    val_dataset = MomentPretrainDataset(val_signals)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # -----------------------------------------------------
    # 2. Configure MOMENT Model
    # -----------------------------------------------------
    model_config = Namespace(
        task_name=TASKS.RECONSTRUCTION,
        n_channels=config['model']['n_channels'],
        seq_len=config['model']['seq_len'],
        patch_len=config['model']['patch_len'],
        patch_stride_len=config['model']['patch_stride_len'],
        d_model=config['model']['d_model'],
        transformer_backbone=config['model']['transformer_backbone'],
        transformer_type=config['model']['transformer_type'],
        mask_ratio=config['model']['mask_ratio'],
        t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
        
        # MOMENT expects num_class natively, even for reconstruction
        num_class=config['model']['num_class'],
        
        randomly_initialize_backbone=False,
        freeze_embedder=False,
        freeze_encoder=False,
        freeze_head=False,
        enable_gradient_checkpointing=True,
    )
    
    print("\nModel configuration:")
    print(f"  Architecture: MOMENT ({model_config.transformer_backbone})")
    print(f"  n_channels: {model_config.n_channels}")
    print(f"  mask_ratio: {model_config.mask_ratio}")
    
    model = MOMENT(model_config).to(device)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters")
    
    # -----------------------------------------------------
    # 3. Trainer Setup (optimizer, scheduler matching RmGPT)
    # -----------------------------------------------------
    num_epochs = config['training']['pretrain_epochs']
    total_steps = len(train_loader) * num_epochs
    warmup_steps = config['training']['warmup_steps']
    lr = config['training']['lr']
    min_lr = config['training']['min_lr']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['training']['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps, 
        min_lr_ratio=min_lr/lr
    )
    
    criterion = nn.MSELoss(reduction="none")
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"\nResuming pretraining from epoch {start_epoch}/{num_epochs}")
    else:
        print(f"\nStarting pretraining from scratch for {num_epochs} epochs")
        
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # -----------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for signals in train_pbar:
            signals = signals.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type == 'cuda')):
                input_mask = torch.ones((signals.shape[0], signals.shape[2])).to(device)
                
                # MOMENT self-supervised reconstruction (internally applies dropout/mask)
                outputs = model.reconstruction(x_enc=signals, input_mask=input_mask)
                
                reconstructed = outputs.reconstruction
                mask = outputs.pretrain_mask
                expanded_mask = mask.unsqueeze(1).float()
                
                # Target is normalized physical signal
                target_norm = model.normalizer(x=signals, mask=mask * outputs.input_mask, mode="norm")
                target_norm = torch.nan_to_num(target_norm, nan=0.0)
                
                loss_matrix = criterion(reconstructed, target_norm)
                
                # Calculate average physical MSE loss across masked positions
                loss = (loss_matrix * expanded_mask).sum() / (expanded_mask.sum() * model_config.n_channels + 1e-8)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping (as in RmGPT pipeline)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
            
        avg_train_loss = np.mean(train_losses)
        
        # Validation Pipeline
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for signals in val_pbar:
                signals = signals.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    input_mask = torch.ones((signals.shape[0], signals.shape[2])).to(device)
                    outputs = model.reconstruction(x_enc=signals, input_mask=input_mask)
                    
                    reconstructed = outputs.reconstruction
                    mask = outputs.pretrain_mask
                    expanded_mask = mask.unsqueeze(1).float()
                    
                    target_norm = model.normalizer(x=signals, mask=mask * outputs.input_mask, mode="norm")
                    target_norm = torch.nan_to_num(target_norm, nan=0.0)
                    
                    loss_matrix = criterion(reconstructed, target_norm)
                    loss = (loss_matrix * expanded_mask).sum() / (expanded_mask.sum() * model_config.n_channels + 1e-8)
                    
                val_losses.append(loss.item())
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Physical MSE)")
        print(f"  Val Loss: {avg_val_loss:.4f} (Physical MSE)")
        
        # Save checkpoints
        if (epoch + 1) % config['logging']['save_every'] == 0 or (epoch + 1) == num_epochs:
            ckpt_path = checkpoint_dir / f"pretrain_moment_sim_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, ckpt_path)
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = checkpoint_dir / "pretrain_moment_sim_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, best_ckpt_path)
            print(f"  Saved best model (val_loss={avg_val_loss:.4f}): {best_ckpt_path}")
            
    # Final save
    final_path = checkpoint_dir / "pretrain_moment_sim_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nPretraining complete! Final MOMENT model saved to: {final_path}")


if __name__ == '__main__':
    main()
