"""
evaluate_robot_moment_tta.py

Test-Time Augmentation (TTA) evaluation on the existing finetuned checkpoint.
Instead of taking one deterministic center window, we take N random windows,
run each through the model, average the softmax probabilities, and take argmax.
This is free accuracy (no retraining needed).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class RobotTestDatasetTTA(Dataset):
    """Dataset that returns ALL the signal so we can slice in the eval loop."""
    def __init__(self, signals, labels):
        self.signals = torch.from_numpy(signals).float()  # [N, seq_len, feats]
        self.labels  = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Return full signal — slicing happens per-augmentation in eval loop
        return self.signals[idx], self.labels[idx]


def get_window(signal, window_size, random=False):
    """Extract a window from [seq_len, feats] signal."""
    seq_len = signal.shape[0]
    if random:
        max_start = seq_len - window_size
        start_idx = np.random.randint(0, max_start + 1)
    else:
        start_idx = (seq_len - window_size) // 2  # center
    return signal[start_idx : start_idx + window_size, :]


def build_parser():
    parser = ArgumentParser(description="TTA Evaluation of MOMENT on robot test data.")
    parser.add_argument("--data_dir",    type=str,   default="data/raw/dataset",
                        help="Base dataset directory (same as used during finetuning).")
    parser.add_argument("--test_folder", type=str,   default="20241016",
                        help="Name of the test folder inside testDatasets/.")
    parser.add_argument("--checkpoint",  type=str,   default="checkpoints/moment_robot_finetuned.pt",
                        help="Path to finetuned checkpoint.")
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--window_size", type=int,   default=512)
    parser.add_argument("--n_augments",  type=int,   default=20,
                        help="Number of random windows to average per sample (TTA).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")
    print(f"[INFO] TTA: {args.n_augments} random windows per sample")

    # 1. Load Data
    print(f"[INFO] Loading robot testing data from {args.data_dir}/testDatasets/{args.test_folder}...")
    try:
        signals, labels = load_robot_test_data(args.data_dir, folder_name=args.test_folder, use_individual_files=True)
    except Exception as e:
        print(f"[ERROR] Failed to load testing data: {e}")
        return

    num_classes = len(np.unique(labels))
    dataset = RobotTestDatasetTTA(signals, labels)
    # Use batch_size=1 for TTA since we process each sample with N augmentations
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"[INFO] Loaded {len(dataset)} testing samples.")

    # 2. Configure & Load Model
    model_config = Namespace(
        task_name=TASKS.CLASSIFICATION,
        n_channels=dataset.signals.shape[2],
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

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found at {args.checkpoint}")
        return

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded weights from {args.checkpoint}")

    # 3. TTA Evaluation Loop
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for signal_batch, label_batch in test_loader:
            # signal_batch: [1, seq_len, feats]
            signal = signal_batch[0]  # [seq_len, feats]
            label  = label_batch[0].item()

            # Accumulate soft probabilities over N random windows
            probs_accumulator = torch.zeros(num_classes, device=device)

            for aug_i in range(args.n_augments):
                use_random = (aug_i > 0)  # First window is always center, rest are random
                window = get_window(signal.numpy(), args.window_size, random=use_random)
                window_t = torch.from_numpy(window).float().unsqueeze(0).permute(0, 2, 1).to(device)
                # window_t: [1, feats, window_size]

                input_mask = torch.ones(1, window_t.shape[2], device=device)
                logits = model.classify(x_enc=window_t, input_mask=input_mask).logits  # [1, num_class]
                probs  = F.softmax(logits[0], dim=0)  # [num_class]
                probs_accumulator += probs

            avg_probs = probs_accumulator / args.n_augments
            pred = avg_probs.argmax().item()

            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(avg_probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()
    print(f"\n[RESULTS] TTA ({args.n_augments} windows) Test Accuracy: {accuracy * 100:.2f}%")

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - MOMENT (TTA {args.n_augments} windows, {accuracy*100:.1f}%)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'confusion_matrix_tta_{args.n_augments}.png')
    plt.savefig(out_path)
    print(f"\n[INFO] Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
