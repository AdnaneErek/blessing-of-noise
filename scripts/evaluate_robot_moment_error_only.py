import os
import torch
import torch.nn as nn
from argparse import Namespace, ArgumentParser
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from momentfm.models.moment import MOMENT
from momentfm.common import TASKS
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.robot_dataset_loader import load_robot_test_data

class RobotTestDataset(Dataset):
    def __init__(self, signals, labels, window_size=512):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()
        self.seq_len = self.signals.shape[1]
        self.num_features = self.signals.shape[2]
        self.window_size = window_size
        
        if self.window_size > self.seq_len:
            raise ValueError(f"Window size {self.window_size} is larger than seq_len {self.seq_len}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].clone()
        # Grab the deterministic middle window for optimal feature representation
        start_idx = (self.seq_len - self.window_size) // 2
        sliced_signal = signal[start_idx : start_idx + self.window_size, :]
        
        # Only use Error xyz
        error_only = sliced_signal[:, -3:]
        return error_only.permute(1, 0), self.labels[idx]


def build_parser():
    parser = ArgumentParser(description="Evaluate MOMENT on robot testing data.")
    parser.add_argument("--data_dir", type=str, default="data/raw/dataset/testDatasets/20241016")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=512)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    # 1. Load Data
    print(f"[INFO] Loading robot testing data from {args.data_dir}...")
    try:
        signals, labels = load_robot_test_data(args.data_dir, use_individual_files=True)
    except Exception as e:
        print(f"[ERROR] Failed to load testing data: {e}")
        return

    num_classes = len(np.unique(labels))
    dataset = RobotTestDataset(signals, labels, window_size=args.window_size)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Loaded {len(dataset)} testing samples. Sequence length: {args.window_size}")

    # 2. Configure Model
    model_config = Namespace(
        task_name=TASKS.CLASSIFICATION,
        n_channels=3, # Error X, Y, Z only
        num_class=num_classes,
        seq_len=args.window_size,
        patch_len=8,
        patch_stride_len=8,
        d_model=256,
        transformer_backbone="google/flan-t5-small",
        transformer_type="encoder_only",
        t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
        randomly_initialize_backbone=False,
    )

    model = MOMENT(model_config).to(device)

    # 3. Load Checkpoint
    checkpoint_path = os.path.join("checkpoints", "moment_robot_finetuned_error_only.pt")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Finetuned checkpoint not found at {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded trained weights from {checkpoint_path}")

    # 4. Evaluation Loop
    all_preds = []
    all_labels = []

    print(f"[INFO] Running inference on validation dataset...")
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            # Classification inference
            input_mask = torch.ones(X.shape[0], X.shape[2], device=device)
            logits = model.classify(x_enc=X, input_mask=input_mask).logits
            
            # Record loss for debugging
            loss = criterion(logits, y).item()
            total_loss += loss

            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    print(f"\n[RESULTS] Final Test Loss: {avg_loss:.4f}")
    print(f"[RESULTS] Final Test Accuracy: {accuracy * 100:.2f}%")

    # The mapping from string labels to integers used during loading
    class_names = [
        "Healthy",
        "Motor_1_Stuck", "Motor_2_Stuck", "Motor_3_Stuck", "Motor_4_Stuck",
        "Motor_1_Steady_state_error", "Motor_2_Steady_state_error",
        "Motor_3_Steady_state_error", "Motor_4_Steady_state_error"
    ]

    print("\n[RESULTS] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - MOMENT Model (Window Sliced - Error Only)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_moment_sliced_error_only.png'))
    print(f"\n[INFO] Confusion matrix saved to {RESULTS_DIR}/confusion_matrix_moment_sliced_error_only.png")


if __name__ == "__main__":
    main()
