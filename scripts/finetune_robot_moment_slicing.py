import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace, ArgumentParser
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

from momentfm.models.moment import MOMENT
from momentfm.common import TASKS
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.robot_dataset_loader import load_robot_finetuning_data


class RobotFinetuneDataset(Dataset):
    def __init__(self, signals, labels, window_size=512, training=True):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()
        self.seq_len = self.signals.shape[1]
        self.num_features = self.signals.shape[2]
        self.window_size = window_size
        self.training = training

        if self.window_size > self.seq_len:
            raise ValueError("window_size cannot be larger than actual seq_len.")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].clone()
        
        # Slicing logic
        if self.training:
            # Massive Data Augmentation: Random offset
            max_start = self.seq_len - self.window_size
            start_idx = np.random.randint(0, max_start + 1)
        else:
            # Deterministic middle slice for validation
            start_idx = (self.seq_len - self.window_size) // 2
            
        sliced_signal = signal[start_idx : start_idx + self.window_size, :]
        return sliced_signal.permute(1, 0), self.labels[idx]


def build_parser():
    parser = ArgumentParser(description="Finetune MOMENT with Random Window Slicing and Progressive Unfreezing.")
    parser.add_argument("--data_dir", type=str, default="data/raw/dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200, help="Max total epochs.")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Epochs to train head only before unfreezing the last T5 block.")
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for the head.")
    parser.add_argument("--encoder_lr_factor", type=float, default=0.1,
                        help="LR multiplier for the unfrozen encoder block relative to --lr.")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (epochs without val_acc improvement).")
    parser.add_argument("--window_size", type=int, default=512,
                        help="Sequence length for data augmentation slicing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if args.debug:
        args.epochs = 10
        args.warmup_epochs = 3
        args.patience = 3

    print(f"[INFO] Loading robot finetuning data from {args.data_dir}...")
    try:
        signals, labels = load_robot_finetuning_data(args.data_dir, folder_name=None, use_individual_files=True)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    num_classes = len(np.unique(labels))
    X_train, X_val, y_train, y_val = train_test_split(
        signals, labels, test_size=0.5, random_state=args.seed, stratify=labels
    )

    train_ds = RobotFinetuneDataset(X_train, y_train, window_size=args.window_size, training=True)
    val_ds   = RobotFinetuneDataset(X_val, y_val, window_size=args.window_size, training=False)
    
    print(f"[INFO] Dataset split: {len(train_ds)} train / {len(val_ds)} val. Sequence length: {args.window_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model_config = Namespace(
        task_name=TASKS.CLASSIFICATION,
        n_channels=train_ds.num_features,
        num_class=num_classes,
        seq_len=args.window_size, # <--- IMPORTANT: Classifier dimension scaling!
        patch_len=8,
        patch_stride_len=8,
        d_model=256,
        transformer_backbone="google/flan-t5-small",
        transformer_type="encoder_only",
        t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
        randomly_initialize_backbone=False,
        freeze_embedder=True,
        freeze_encoder=True,   # everything frozen initially
        freeze_head=False,     # only head trains
        enable_gradient_checkpointing=True,
    )
    model = MOMENT(model_config).to(device)

    pretrained_path = os.path.join("checkpoints", "moment_robot_pretrained.pt")
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        keys_to_drop = [k for k in state_dict if k.startswith("head.")]
        for k in keys_to_drop:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARNING] Pretrained checkpoint not found. Using T5 weights.")

    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "moment_robot_finetuned.pt")

    criterion = nn.CrossEntropyLoss()
    use_amp = (device.type == "cuda")
    scaler   = GradScaler(enabled=use_amp)

    def make_optimizer(model, base_lr, encoder_lr_factor):
        head_params    = list(model.head.parameters())
        param_groups = [{"params": [p for p in head_params if p.requires_grad], "lr": base_lr}]
        encoder_trainable = [p for name, p in model.encoder.named_parameters() if p.requires_grad]
        if encoder_trainable:
            param_groups.append({"params": encoder_trainable, "lr": base_lr * encoder_lr_factor})
        return optim.AdamW(param_groups, weight_decay=1e-4)

    optimizer = make_optimizer(model, args.lr, args.encoder_lr_factor)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve    = 0
    encoder_block_unfrozen = False
    last_block_idx = len(model.encoder.block) - 1

    print(f"\n[INFO] Starting Finetuning: Warmup {args.warmup_epochs} epochs. Total: {args.epochs}")
    for epoch in range(args.epochs):
        
        # ----- PROGRESSIVE UNFREEZING ACTIVATION -----
        if not encoder_block_unfrozen and epoch == args.warmup_epochs:
            print(f"\n[INFO] ── Warmup done. Unfreezing encoder block {last_block_idx}. ──")
            for param in model.encoder.block[last_block_idx].parameters():
                param.requires_grad = True

            # Restart optimizer and scaler to pick up new gradients
            optimizer = make_optimizer(model, args.lr, args.encoder_lr_factor)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            scaler = GradScaler(enabled=use_amp)
            encoder_block_unfrozen = True

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                input_mask = torch.ones(X.shape[0], X.shape[2], device=device)
                logits = model.classify(x_enc=X, input_mask=input_mask).logits
                loss   = criterion(logits, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc  = 100.0 * correct / total
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast(enabled=use_amp):
                    input_mask = torch.ones(X.shape[0], X.shape[2], device=device)
                    logits = model.classify(x_enc=X, input_mask=input_mask).logits
                    val_loss_sum += criterion(logits, y).item()
                    val_correct  += (logits.argmax(1) == y).sum().item()
                    val_total    += y.size(0)
                    
        val_loss = val_loss_sum / len(val_loader)
        val_acc  = 100.0 * val_correct / val_total

        phase_tag = "Phase 2 (head+block)" if encoder_block_unfrozen else "Phase 1 (head only)"
        print(f"[Epoch {epoch+1:>3}/{args.epochs} | {phase_tag}] "
              f"Train {train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val {val_loss:.4f} ({val_acc:.1f}%) | LR {current_lr:.6f}")

        # Early Stopping strictly on val_acc first, then val_loss to resolve ties
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            no_improve    = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✅ New best val acc: {best_val_acc:.2f}% (loss: {val_loss:.4f}) — checkpoint saved.")
        else:
            no_improve += 1
            if encoder_block_unfrozen and no_improve >= args.patience:
                print(f"\n[INFO] Early stopping triggered. ({args.patience} epochs without val_acc improvement).")
                break

    print(f"\n[INFO] Training complete. Best val acc: {best_val_acc:.2f}% (loss: {best_val_loss:.4f})")
    print(f"[INFO] ✅ Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
