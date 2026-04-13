import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from argparse import Namespace, ArgumentParser
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from momentfm.models.moment import MOMENT
from momentfm.common import TASKS
import sys

# Ensure data package is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.robot_dataset_loader import load_robot_training_data


class RobotPretrainDataset(Dataset):
    """
    Dataset wrapper for robot data arrays.
    Signals shape: [num_samples, seq_len, num_features]
    """
    def __init__(self, signals):
        self.signals = torch.from_numpy(signals).float()
        self.num_samples = self.signals.shape[0]
        self.seq_len = self.signals.shape[1]
        self.num_features = self.signals.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Moment expects [batch, channels, seq_len], so permute [seq_len, features] -> [features, seq_len]
        signal = self.signals[idx].permute(1, 0)
        return signal


def build_parser():
    parser = ArgumentParser(description="Pretrain MOMENT on robot data using masked reconstruction.")
    parser.add_argument("--data_dir", type=str, default="data/raw/dataset", help="Path to raw dataset directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="Fraction of patches to mask.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer steps).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # -----------------------------------------------------
    # 0. Setup
    # -----------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -----------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------
    print(f"[INFO] Loading robot training data from {args.data_dir}...")
    try:
        # Load all folders (None auto-discovers)
        signals, _ = load_robot_training_data(args.data_dir, folder_name=None, use_individual_files=True)
    except Exception as e:
        print(f"[ERROR] Failed to load training data: {e}")
        return

    print(f"[INFO] Loaded signals shape: {signals.shape}")
    dataset = RobotPretrainDataset(signals)
    
    # Validation split (90/10)
    total_len = len(dataset)
    train_size = int(0.9 * total_len)
    val_size = total_len - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    
    print(f"[INFO] Dataset split: {len(train_dataset)} train, {len(val_dataset)} val.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # If debug mode, truncate dataset
    if args.debug:
        print("[INFO] DEBUG MODE: Using small subset of data.")
        args.epochs = 2
        # just modify the length so we don't iterate forever
        train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(args.batch_size * 2)), batch_size=args.batch_size)
        val_loader = DataLoader(torch.utils.data.Subset(val_dataset, range(args.batch_size * 2)), batch_size=args.batch_size)

    # -----------------------------------------------------
    # 2. Configure and Initialize MOMENT Model
    # -----------------------------------------------------
    seq_len = dataset.seq_len
    n_channels = dataset.num_features

    model_config = Namespace(
        task_name=TASKS.RECONSTRUCTION,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_len=8,
        patch_stride_len=8,
        d_model=256,
        transformer_backbone="google/flan-t5-small",
        transformer_type="encoder_only",
        t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
        
        # We start with initialization of parameters
        randomly_initialize_backbone=False,
        mask_ratio=args.mask_ratio,  

        # Allow full network updates for pretraining
        freeze_embedder=False,
        freeze_encoder=False,
        freeze_head=False,
        enable_gradient_checkpointing=True,
    )

    print(f"[INFO] Initializing MOMENT model for reconstruction...")
    model = MOMENT(model_config).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model initialized. Trainable parameters: {n_trainable}")

    # -----------------------------------------------------
    # 3. Training Setup
    # -----------------------------------------------------
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "moment_robot_pretrained.pt")

    # -----------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------
    print("[INFO] Beginning pretraining...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for X in train_loader:
            X = X.to(device)
            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                # Using MOMENT's self-supervised reconstruction method
                # This forwards through encoder-decoder and automatically applies mask
                input_mask = torch.ones((X.shape[0], X.shape[2])).to(device)
                outputs = model.reconstruction(x_enc=X, input_mask=input_mask)
                
                # outputs.reconstruction: [batch, channels, seq_len]
                # outputs.pretrain_mask: [batch, seq_len]
                reconstructed = outputs.reconstruction
                mask = outputs.pretrain_mask
                
                # Compute MSE loss specifically on the masked elements
                # Reshape mask to broadcast across channels -> [batch, 1, seq_len]
                expanded_mask = mask.unsqueeze(1).float()
                
                # We normalize the targets as is done inside model.reconstruction
                # so the loss is computed on the normalized space
                target_norm = model.normalizer(x=X, mask=mask * outputs.input_mask, mode="norm")
                target_norm = torch.nan_to_num(target_norm, nan=0.0)

                loss_matrix = criterion(reconstructed, target_norm)
                
                # Average loss only on masked positions
                # sum of losses on masked positions / total number of masked elements
                masked_loss = (loss_matrix * expanded_mask).sum() / (expanded_mask.sum() * n_channels + 1e-8)

            scaler.scale(masked_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += masked_loss.item()
            num_train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / num_train_batches
        current_lr = scheduler.get_last_lr()[0]

        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for X in val_loader:
                X = X.to(device)
                with autocast(enabled=use_amp):
                    input_mask = torch.ones((X.shape[0], X.shape[2])).to(device)
                    outputs = model.reconstruction(x_enc=X, input_mask=input_mask)
                    
                    reconstructed = outputs.reconstruction
                    mask = outputs.pretrain_mask
                    expanded_mask = mask.unsqueeze(1).float()
                    
                    target_norm = model.normalizer(x=X, mask=mask * outputs.input_mask, mode="norm")
                    target_norm = torch.nan_to_num(target_norm, nan=0.0)
                    
                    loss_matrix = criterion(reconstructed, target_norm)
                    masked_loss = (loss_matrix * expanded_mask).sum() / (expanded_mask.sum() * n_channels + 1e-8)
                    
                    total_val_loss += masked_loss.item()
                    num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0

        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

    # -----------------------------------------------------
    # 5. Save Checkpoint
    # -----------------------------------------------------
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] ✅ Pretrained model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
