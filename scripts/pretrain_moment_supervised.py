import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.model_selection import train_test_split
from argparse import Namespace

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from momentfm.models.moment import MOMENT
from momentfm.common import TASKS
from data.robot_dataset_loader import load_robot_training_data

class SupervisedRobotDataset(Dataset):
    def __init__(self, signals, labels, add_noise=False, noise_std=0.01):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        self.num_samples = self.signals.shape[0]
        self.seq_len = self.signals.shape[1]
        self.num_features = self.signals.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        signal = self.signals[idx].clone()
        if self.add_noise:
            # V2 config: Very mild gaussian noise
            noise = torch.randn_like(signal) * self.noise_std
            signal += noise
        
        # [seq_len, num_features] -> [num_features, seq_len] for MOMENT
        return signal.permute(1, 0), self.labels[idx]


def main():
    parser = argparse.ArgumentParser(description="Supervised Pretraining using MOMENT")
    parser.add_argument("--config", type=str, default="pretrain_robot_sim_supervised_v2.yaml", help="Path to YAML config")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # 1. Load Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('hardware', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Set seeds based on config
    seed = config.get('data', {}).get('random_state', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Data Loading
    data_dir = config.get('data', {}).get('data_dir', 'data/raw/dataset')
    training_folder = config.get('data', {}).get('robot_training_folder', None)
    use_individual_files = config.get('data', {}).get('use_individual_files', True)

    print(f"[INFO] Loading robot training simulation data from {data_dir}...")
    signals, labels = load_robot_training_data(
        data_dir=data_dir,
        folder_name=training_folder,
        use_individual_files=use_individual_files
    )
    
    num_classes = len(np.unique(labels))
    print(f"[INFO] Total loaded: {len(signals)} samples. Classes: {num_classes}")

    # 3. Data Splits
    val_size = config.get('data', {}).get('test_size', 0.2)
    
    X_train, X_val, y_train, y_val = train_test_split(
        signals, labels, test_size=val_size, random_state=seed, stratify=labels
    )
    
    print(f"[INFO] Train split: {len(y_train)}, Val split: {len(y_val)}")

    # 4. Datasets & Lodaers
    noise_cfg = config.get('data', {}).get('noise_augmentation', {})
    noise_enabled = noise_cfg.get('enabled', False)
    noise_std = noise_cfg.get('std', 0.01)

    train_ds = SupervisedRobotDataset(X_train, y_train, add_noise=noise_enabled, noise_std=noise_std)
    val_ds = SupervisedRobotDataset(X_val, y_val, add_noise=False)

    if args.debug:
        print("[INFO] DEBUG MODE: Using small subset of data.")
        train_ds = torch.utils.data.Subset(train_ds, range(min(128, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(64, len(val_ds))))
        config['training']['pretrain_epochs'] = 2

    batch_size = config.get('training', {}).get('batch_size', 256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. MOMENT Initialization
    print(f"[INFO] Initializing fully unfrozen MOMENT model for CLASSIFICATION...")
    moment_config = Namespace(
        task_name=TASKS.CLASSIFICATION,
        n_channels=train_ds.dataset.num_features if args.debug else train_ds.num_features,
        num_class=num_classes,
        seq_len=train_ds.dataset.seq_len if args.debug else train_ds.seq_len,
        patch_len=8,
        patch_stride_len=8,
        d_model=256,
        transformer_backbone="google/flan-t5-small",
        transformer_type="encoder_only",
        t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
        randomly_initialize_backbone=False,
        freeze_embedder=False,
        freeze_encoder=False,
        freeze_head=False,
        enable_gradient_checkpointing=True,
    )
    model = MOMENT(moment_config).to(device)

    # We do NOT load any pretrained MOMENT checkpoint, because this IS the pretraining script!
    
    # 6. Training Configuration
    base_lr = float(config.get('training', {}).get('lr', 8e-5))
    head_lr = float(config.get('training', {}).get('head_lr', 2e-4))
    weight_decay = float(config.get('training', {}).get('weight_decay', 0.01))
    epochs = config.get('training', {}).get('pretrain_epochs', 80)
    label_smoothing = float(config.get('training', {}).get('label_smoothing', 0.05))

    # Define param groups so head gets different LR from backbone
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('head.') and p.requires_grad]
    
    param_groups = [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": head_lr}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Cosine scheduling down to min_lr
    min_lr = float(config.get('training', {}).get('min_lr', 1e-6))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # Early stopping config
    es_cfg = config.get('training', {}).get('early_stopping', {})
    es_enabled = es_cfg.get('enabled', True)
    es_patience = es_cfg.get('patience', 12)
    es_min_delta = float(es_cfg.get('min_delta', 0.001))
    
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"[INFO] Base LR: {base_lr}, Head LR: {head_lr}")
    print(f"[INFO] Launching Supervised Pretraining loop for {epochs} epochs...")

    CHECKPOINT_DIR = config.get('logging', {}).get('checkpoint_dir', 'checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "moment_robot_sim_supervised_best.pt")

    best_val_loss = float("inf")
    no_improve = 0

    # 7. Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_correct, train_total = 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                input_mask = torch.ones(X.shape[0], X.shape[2], device=device)
                logits = model.classify(x_enc=X, input_mask=input_mask).logits.squeeze()
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            
            # optional max_grad_norm from config could be applied here
            max_grad_norm = float(config.get('training', {}).get('max_grad_norm', 1.0))
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        curr_lr = scheduler.get_last_lr()[0]

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast(enabled=use_amp):
                    input_mask = torch.ones(X.shape[0], X.shape[2], device=device)
                    logits = model.classify(x_enc=X, input_mask=input_mask).logits.squeeze()
                    loss = criterion(logits, y)
                    
                    total_val_loss += loss.item()
                    val_correct += (logits.argmax(1) == y).sum().item()
                    val_total += y.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        print(f"[Epoch {epoch+1:2d}/{epochs}] "
              f"Train Loss {avg_train_loss:.5f} ({train_acc:.1f}%) | "
              f"Val Loss {avg_val_loss:.5f} ({val_acc:.1f}%) | "
              f"LR {curr_lr:.2e}")

        # Early Stopping & Checkpointing
        if avg_val_loss < (best_val_loss - es_min_delta):
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  ✅ New best val loss: {best_val_loss:.5f} — checkpoint saved.")
        else:
            no_improve += 1
            if es_enabled and no_improve >= es_patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}. Patience ({es_patience}) exceeded without {es_min_delta} improvement.")
                break

    print(f"\n[INFO] Supervised Pretraining Complete. Best Val Loss: {best_val_loss:.5f}")
    print(f"[INFO] Final Model saved to {best_checkpoint_path}")

if __name__ == "__main__":
    main()
