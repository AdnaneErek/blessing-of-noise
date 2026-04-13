"""
Training script for RmGPT model

Implements:
1. Self-supervised pretraining with next-token prediction
2. Supervised fine-tuning for diagnosis and prognosis tasks
3. Prompt learning for task adaptation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from typing import Dict, Optional, Callable
import numpy as np

from model.rmgpt import RmGPT, DiagnosisHead, PrognosisHead



class RmGPTTrainer:
    """Trainer for RmGPT model"""
    
    def __init__(self,
                 model: RmGPT,
                 device: torch.device,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_grad_norm: float = 1.0,
                 total_steps: Optional[int] = None,
                 lr_schedule: str = "cosine",
                 min_lr: float = 1e-6,
                 head_lr: Optional[float] = None,
                 head_params: Optional[list] = None,
                 label_smoothing: float = 0.0,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Args:
            model: RmGPT model
            device: Training device
            lr: Learning rate for backbone (pretrained model)
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            total_steps: Total training steps (for cosine annealing)
            lr_schedule: Learning rate schedule ("cosine", "constant", "linear")
            min_lr: Minimum learning rate for cosine decay
            head_lr: Learning rate for task head (if None, uses lr)
            head_params: Parameters of task head (for separate LR)
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            use_focal_loss: Whether to use focal loss instead of cross-entropy
            focal_alpha: Focal loss alpha parameter (class weighting)
            focal_gamma: Focal loss gamma parameter (focusing parameter)
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.head_lr = head_lr if head_lr is not None else lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.lr_schedule = lr_schedule
        self.min_lr = min_lr
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Create parameter groups with different learning rates
        # Separate channel projection (if exists) from backbone - it's NEW and needs higher LR
        channel_proj_params = []
        backbone_params = []
        
        # Get head param IDs for comparison (if provided)
        head_param_ids = set(id(p) for p in head_params) if head_params is not None else set()
        
        for name, param in self.model.named_parameters():
            # Skip if this param is in head_params
            if id(param) in head_param_ids:
                continue
            if 'channel_proj' in name:
                channel_proj_params.append(param)
            else:
                backbone_params.append(param)
        
        # Build parameter groups
        param_groups = []
        
        # Backbone (pretrained) - low LR to preserve features
        if len(backbone_params) > 0:
            param_groups.append({
                'params': backbone_params, 
                'lr': lr, 
                'weight_decay': weight_decay
            })
        
        # Channel projection (new layer) - use higher LR (same as head or intermediate)
        if len(channel_proj_params) > 0:
            # Use head_lr if available, otherwise use 10x backbone LR
            proj_lr = head_lr if head_lr is not None else lr * 10
            param_groups.append({
                'params': channel_proj_params,
                'lr': proj_lr,
                'weight_decay': weight_decay
            })
            print(f"Channel projection layer found: using LR={proj_lr} (higher than backbone LR={lr})")
        
        # Head (new) - high LR
        if head_lr is not None and head_params is not None:
            param_groups.append({
                'params': head_params, 
                'lr': head_lr, 
                'weight_decay': weight_decay
            })
            if len(channel_proj_params) > 0:
                print(f"Using different LRs: backbone={lr}, projection={proj_lr}, head={head_lr}")
            else:
                print(f"Using different LRs: backbone={lr}, head={head_lr}")
        elif head_params is not None:
            # Include head params but with same LR
            param_groups.append({
                'params': head_params,
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        # Fallback: if no groups created, use all params
        if len(param_groups) == 0:
            all_params = list(self.model.parameters())
            if head_params is not None:
                all_params.extend(head_params)
            param_groups = [{'params': all_params, 'lr': lr, 'weight_decay': weight_decay}]
        
        # Optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler with warmup and decay
        if lr_schedule == "cosine" and total_steps is not None:
            # Cosine annealing with warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    # Warmup: linear increase from 0 to 1
                    return step / warmup_steps
                else:
                    # Cosine decay from 1 to min_lr/lr
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                    return cosine_decay * (1 - min_lr / lr) + min_lr / lr
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif lr_schedule == "linear" and total_steps is not None:
            # Linear decay with warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return max(0.0, 1.0 - progress)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            # Constant with warmup (original behavior)
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
            )
        
        self.global_step = 0
        
    def pretrain_step(self, batch: Dict[str, torch.Tensor], mask_prob: float = 0.15) -> Dict[str, float]:
        """
        Self-supervised pretraining with masked token prediction (BERT-style)
        
        Args:
            batch: Dictionary with 'signals' [batch, seq_len, signal_dim]
            mask_prob: Probability of masking each signal token (default 0.15 like BERT)
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        signals = batch['signals'].to(self.device)
        batch_size, seq_len, signal_dim = signals.shape
        
        # Forward pass to get token embeddings
        output = self.model(signals, task_type='pretrain', return_tokens=True)
        features = output['features']  # [batch, total_seq_len, embed_dim]
        signal_tokens = output['signal_tokens']  # [batch, num_patches, embed_dim]
        
        # Get token sequence structure: [Prompt] [Time-Freq] [Signal Patches]
        num_prompts = self.model.num_prompts
        num_patches = signal_tokens.shape[1]
        signal_start_idx = num_prompts + 1  # After prompts and time-freq token
        
        # Extract signal token embeddings from the full sequence
        signal_token_features = features[:, signal_start_idx:signal_start_idx + num_patches, :]  # [batch, num_patches, embed_dim]
        
        # Create mask for signal tokens (mask 15% randomly per sample)
        num_tokens_to_mask = max(1, int(num_patches * mask_prob))
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=self.device)
        for batch_idx in range(batch_size):
            # Randomly select tokens to mask for this sample
            mask_indices = torch.randperm(num_patches, device=self.device)[:num_tokens_to_mask]
            mask[batch_idx, mask_indices] = True
        
        # Store original embeddings for masked positions
        masked_token_embeddings = signal_token_features[mask]  # [num_masked, embed_dim]
        
        # Replace masked tokens with [MASK] token (use learnable mask embedding)
        # For now, use zero embedding as mask (can be made learnable later)
        masked_features = features.clone()
        mask_embedding = torch.zeros_like(signal_token_features[0, 0])  # [embed_dim]
        for batch_idx in range(batch_size):
            # Get masked token indices for this sample
            sample_mask_indices = torch.where(mask[batch_idx])[0]
            for token_idx in sample_mask_indices:
                masked_features[batch_idx, signal_start_idx + token_idx, :] = mask_embedding
        
        # Pass masked sequence through transformer
        pred_features = self.model.transformer(masked_features)
        
        # Extract predictions for masked positions
        pred_masked_tokens = pred_features[:, signal_start_idx:signal_start_idx + num_patches, :]  # [batch, num_patches, embed_dim]
        pred_masked = pred_masked_tokens[mask]  # [num_masked, embed_dim]
        
        # MSE loss between predicted and original masked token embeddings
        loss = nn.functional.mse_loss(pred_masked, masked_token_embeddings)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def diagnosis_step(self, 
                      batch: Dict[str, torch.Tensor],
                      diagnosis_head: DiagnosisHead) -> Dict[str, float]:
        """
        Supervised fine-tuning for fault diagnosis
        
        Args:
            batch: Dictionary with 'signals' and 'labels'
            diagnosis_head: Diagnosis classification head
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        diagnosis_head.train()
        
        signals = batch['signals'].to(self.device)
        labels = batch['labels'].to(self.device).long()
        
        # Forward pass
        output = self.model(signals, task_type='diagnosis', return_tokens=False)
        features = output['features']
        
        # Use time-freq token features (after prompt tokens) for classification
        # Prompt tokens are at the beginning, TF token is after them
        tf_token_idx = self.model.num_prompts  # Index of TF token
        tf_features = features[:, tf_token_idx, :]  # [batch, embed_dim]
        
        # Classification
        logits = diagnosis_head(tf_features.unsqueeze(1))  # [batch, num_classes]
        logits = logits.squeeze(1)  # [batch, num_classes]
        
        # Loss calculation with optional label smoothing and focal loss
        if self.use_focal_loss:
            # Focal loss for handling class imbalance
            ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none', label_smoothing=self.label_smoothing)
            pt = torch.exp(-ce_loss)
            loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
            loss = loss.mean()
        else:
            # Standard cross-entropy with optional label smoothing
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for both model and head
        all_params = list(self.model.parameters()) + list(diagnosis_head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        
        # Get learning rates for all parameter groups
        lrs = self.scheduler.get_last_lr()
        # Display the maximum LR (usually the head LR) for visibility
        displayed_lr = max(lrs) if len(lrs) > 0 else 0.0
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'lr': displayed_lr
        }
    
    def prognosis_step(self,
                      batch: Dict[str, torch.Tensor],
                      prognosis_head: PrognosisHead) -> Dict[str, float]:
        """
        Supervised fine-tuning for RUL prediction
        
        Args:
            batch: Dictionary with 'signals' and 'rul'
            prognosis_head: Prognosis regression head
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        prognosis_head.train()
        
        signals = batch['signals'].to(self.device)
        rul = batch['rul'].to(self.device).float()
        
        # Forward pass
        output = self.model(signals, task_type='prognosis', return_tokens=False)
        features = output['features']
        
        # Use time-freq token for RUL prediction
        tf_token_idx = self.model.num_prompts
        tf_features = features[:, tf_token_idx, :]  # [batch, embed_dim]
        
        # Regression
        rul_pred = prognosis_head(tf_features.unsqueeze(1))  # [batch, 1]
        rul_pred = rul_pred.squeeze(1)  # [batch]
        
        # MSE loss
        loss = nn.functional.mse_loss(rul_pred, rul)
        
        # Compute MAE
        mae = torch.abs(rul_pred - rul).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for both model and head
        all_params = list(self.model.parameters()) + list(prognosis_head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'mae': mae.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def save_checkpoint(self, path: str, diagnosis_head: Optional[nn.Module] = None,
                       prognosis_head: Optional[nn.Module] = None, epoch: Optional[int] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if diagnosis_head is not None:
            checkpoint['diagnosis_head_state_dict'] = diagnosis_head.state_dict()
        
        if prognosis_head is not None:
            checkpoint['prognosis_head_state_dict'] = prognosis_head.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, diagnosis_head: Optional[nn.Module] = None,
                       prognosis_head: Optional[nn.Module] = None):
        """Load model checkpoint
        
        Handles dimension mismatches (e.g., fault_tokenizer when num_faults differs)
        by filtering out incompatible parameters.
        
        Returns:
            epoch: Epoch number from checkpoint, or None if not found
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state dict, handling dimension mismatches
        model_state_dict = checkpoint['model_state_dict']
        model_state = self.model.state_dict()
        
        # Check critical dimensions before loading
        if 'signal_tokenizer.patch_embed.weight' in model_state_dict:
            ckpt_patch_embed = model_state_dict['signal_tokenizer.patch_embed.weight']
            model_patch_embed = model_state['signal_tokenizer.patch_embed.weight']
            if ckpt_patch_embed.shape != model_patch_embed.shape:
                raise RuntimeError(
                    f"Critical dimension mismatch in signal_tokenizer.patch_embed.weight:\n"
                    f"  Checkpoint: {ckpt_patch_embed.shape} (input_dim={ckpt_patch_embed.shape[1]})\n"
                    f"  Model: {model_patch_embed.shape} (input_dim={model_patch_embed.shape[1]})\n"
                    f"  This indicates the model was created with wrong signal_dim or patch_length.\n"
                    f"  Expected input_dim: {ckpt_patch_embed.shape[1]}, got: {model_patch_embed.shape[1]}"
                )
        
        # Filter out parameters with dimension mismatches
        filtered_state_dict = {}
        skipped_params = []
        
        for key, value in model_state_dict.items():
            if key in model_state:
                # Check if shapes match
                if model_state[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    skipped_params.append(f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}")
            else:
                skipped_params.append(f"{key}: not in current model")
        
        if skipped_params:
            print(f"Warning: Skipped {len(skipped_params)} incompatible parameters:")
            for param in skipped_params[:5]:  # Show first 5
                print(f"  - {param}")
            if len(skipped_params) > 5:
                print(f"  ... and {len(skipped_params) - 5} more")
        
        # Load filtered state dict
        self.model.load_state_dict(filtered_state_dict, strict=False)
        
        # Load optimizer and scheduler (may have mismatches too, but usually fine)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Optimizer will be reinitialized.")
        
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
            print("Scheduler will be reinitialized.")
        
        # Reset global_step to 0 when loading pretrained checkpoint for fine-tuning
        # This ensures the learning rate scheduler starts fresh
        # (The checkpoint's global_step might be from a different training run)
        old_global_step = checkpoint.get('global_step', 0)
        self.global_step = 0
        print(f"Reset global_step from {old_global_step} to 0 for fresh training")
        
        # Load diagnosis head if compatible, otherwise skip (will use randomly initialized head)
        if diagnosis_head is not None and 'diagnosis_head_state_dict' in checkpoint:
            try:
                diagnosis_head.load_state_dict(checkpoint['diagnosis_head_state_dict'], strict=False)
                print("Loaded diagnosis head from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load diagnosis head state dict: {e}")
                print("Using randomly initialized diagnosis head (architecture mismatch or different num_classes)")
        
        # Load prognosis head if compatible
        if prognosis_head is not None and 'prognosis_head_state_dict' in checkpoint:
            try:
                prognosis_head.load_state_dict(checkpoint['prognosis_head_state_dict'], strict=False)
                print("Loaded prognosis head from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load prognosis head state dict: {e}")
                print("Using randomly initialized prognosis head")
        
        return checkpoint.get('epoch', None)


def train_epoch(trainer: RmGPTTrainer,
                dataloader: DataLoader,
                task_type: str,
                diagnosis_head: Optional[DiagnosisHead] = None,
                prognosis_head: Optional[PrognosisHead] = None,
                desc: str = "Training"):
    """Train for one epoch"""
    epoch_metrics = {}
    
    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        if task_type == 'pretrain':
            metrics = trainer.pretrain_step(batch)
        elif task_type == 'diagnosis':
            metrics = trainer.diagnosis_step(batch, diagnosis_head)
        elif task_type == 'prognosis':
            metrics = trainer.prognosis_step(batch, prognosis_head)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Accumulate metrics
        for key, value in metrics.items():
            if key not in epoch_metrics:
                epoch_metrics[key] = []
            epoch_metrics[key].append(value)
        
        # Update progress bar
        pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in metrics.items()})
    
    # Average metrics
    return {k: np.mean(v) for k, v in epoch_metrics.items()}
