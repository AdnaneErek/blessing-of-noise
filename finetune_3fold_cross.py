#!/usr/bin/env python3
"""
3-Fold Cross-Experiment Fine-tuning.

Three real-robot folders:
  A1 = robot_a_20251127       (finetuningDatasets)
  A2 = robot_a_20251127_load  (finetuningDatasets)
  B  = 20241016               (testDatasets)

For each experiment i ∈ {1,2,3}:
  - The i-th folder is split 50/50 (stratified)
  - Train on: the other 2 full folders + first half of folder i
  - Test  on: second half of folder i
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import load_robot_finetuning_data, load_robot_test_data
from model.rmgpt import DiagnosisHead, RmGPT
from train.trainer import RmGPTTrainer, train_epoch


FAULT_NAMES = [
    "Healthy",
    "Motor_1_Stuck",
    "Motor_2_Stuck",
    "Motor_3_Stuck",
    "Motor_4_Stuck",
    "Motor_1_SSE",
    "Motor_2_SSE",
    "Motor_3_SSE",
    "Motor_4_SSE",
]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_folder(data_dir: str, parent: str, name: str):
    """Load signals & labels from a folder under finetuningDatasets or testDatasets."""
    if parent == "testDatasets":
        return load_robot_test_data(data_dir, folder_name=name, use_individual_files=True)
    else:
        return load_robot_finetuning_data(data_dir, folder_name=name, use_individual_files=True)


def build_model_from_checkpoint(config, checkpoint_path, device):
    """Build model + head from config and load checkpoint weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    patch_length = config["model"]["patch_length"]
    signal_dim = state_dict["signal_tokenizer.patch_embed.weight"].shape[1] // patch_length

    num_classes = None
    if "diagnosis_head_state_dict" in checkpoint:
        for key in sorted(checkpoint["diagnosis_head_state_dict"].keys(), reverse=True):
            if key.endswith(".weight") and len(checkpoint["diagnosis_head_state_dict"][key].shape) == 2:
                num_classes = checkpoint["diagnosis_head_state_dict"][key].shape[0]
                break
    num_classes = num_classes or 9

    model_cfg = config["model"].copy()
    model_cfg["signal_dim"] = signal_dim
    model_cfg["input_channels"] = signal_dim
    model_cfg["tokenizer_stride"] = model_cfg.pop("tokenizer_stride", model_cfg.get("patch_length", 256))

    allowed = {
        "signal_dim", "patch_length", "tokenizer_stride", "embed_dim",
        "num_prompts", "num_faults", "num_layers", "num_heads",
        "ff_dim", "dropout", "n_fft", "wavelet", "wavelet_levels", "input_channels",
    }
    model = RmGPT(**{k: v for k, v in model_cfg.items() if k in allowed}).to(device)

    diagnosis_head = DiagnosisHead(
        embed_dim=model_cfg["embed_dim"],
        num_classes=num_classes,
        improved=config.get("model", {}).get("improved_diagnosis_head", False),
    ).to(device)

    return model, diagnosis_head, checkpoint


def run_experiment(exp_idx, folders, config, checkpoint_path, device, base_ckpt_dir, base_log_dir, base_save_dir):
    """Run a single fold experiment.

    exp_idx: 0, 1, or 2 — which folder is held out for testing.
    """
    test_folder = folders[exp_idx]
    train_folders = [f for i, f in enumerate(folders) if i != exp_idx]

    print("\n" + "=" * 70)
    print(f"EXPERIMENT {exp_idx + 1}/3")
    print(f"  Test folder (50% held out): {test_folder['name']} ({test_folder['parent']})")
    print(f"  Train folders (full):        {[f['name'] for f in train_folders]}")
    print("=" * 70)

    data_dir = config["data"]["data_dir"]
    random_state = config["data"].get("random_state", 42)
    test_split = config["data"].get("test_split", 0.5)

    # ---- Load held-out folder and split 50/50 ----
    held_signals, held_labels = load_folder(data_dir, test_folder["parent"], test_folder["name"])
    print(f"Held-out folder '{test_folder['name']}': {len(held_signals)} samples")

    held_train_sig, held_test_sig, held_train_lab, held_test_lab = train_test_split(
        held_signals, held_labels,
        test_size=test_split,
        random_state=random_state,
        stratify=held_labels,
    )
    print(f"  -> train-half: {len(held_train_sig)}, test-half: {len(held_test_sig)}")

    # ---- Load the other 2 full folders ----
    full_signals_list, full_labels_list = [], []
    for f in train_folders:
        sig, lab = load_folder(data_dir, f["parent"], f["name"])
        print(f"Full train folder '{f['name']}': {len(sig)} samples")
        full_signals_list.append(sig)
        full_labels_list.append(lab)

    # ---- Combine training data: 2 full + train-half ----
    all_train_sig = np.concatenate(full_signals_list + [held_train_sig], axis=0)
    all_train_lab = np.concatenate(full_labels_list + [held_train_lab], axis=0)

    # Shuffle
    perm = np.random.RandomState(random_state).permutation(len(all_train_sig))
    all_train_sig = all_train_sig[perm]
    all_train_lab = all_train_lab[perm]

    # Split a small validation set from training data
    val_ratio = 0.15
    train_sig, val_sig, train_lab, val_lab = train_test_split(
        all_train_sig, all_train_lab,
        test_size=val_ratio,
        random_state=random_state,
        stratify=all_train_lab,
    )
    print(f"Final split: train={len(train_sig)}, val={len(val_sig)}, test={len(held_test_sig)}")

    # ---- Datasets & loaders ----
    train_dataset = PHMSignalDataset(
        train_sig, labels=train_lab,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
        add_noise=False,
    )
    val_dataset = PHMSignalDataset(
        val_sig, labels=val_lab,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
        add_noise=False,
    )
    test_dataset = PHMSignalDataset(
        held_test_sig, labels=held_test_lab,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
        add_noise=False,
    )

    bs = config["training"]["batch_size"]
    nw = config["hardware"]["num_workers"]
    pm = config["hardware"]["pin_memory"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    # ---- Model (fresh from pretrained checkpoint each time) ----
    model, diagnosis_head, ckpt_data = build_model_from_checkpoint(config, checkpoint_path, device)

    num_epochs = config["training"]["finetune_epochs"]
    total_steps = len(train_loader) * num_epochs
    head_lr = config["training"].get("head_lr", None)
    head_params = list(diagnosis_head.parameters())
    freeze_backbone = config["training"].get("freeze_backbone", False)

    if freeze_backbone:
        head_only_lr = head_lr if head_lr is not None else config["training"]["lr"]
        trainer = RmGPTTrainer(
            model=model, device=device, lr=head_only_lr,
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_grad_norm=config["training"]["max_grad_norm"],
            total_steps=total_steps,
            lr_schedule=config["training"].get("lr_schedule", "constant"),
            min_lr=config["training"].get("min_lr", 1e-6),
            head_lr=head_only_lr, head_params=head_params,
            label_smoothing=config["training"].get("label_smoothing", 0.0),
            use_focal_loss=config["training"].get("use_focal_loss", False),
            focal_alpha=config["training"].get("focal_alpha", 0.25),
            focal_gamma=config["training"].get("focal_gamma", 2.0),
        )
    else:
        trainer = RmGPTTrainer(
            model=model, device=device, lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_grad_norm=config["training"]["max_grad_norm"],
            total_steps=total_steps,
            lr_schedule=config["training"].get("lr_schedule", "constant"),
            min_lr=config["training"].get("min_lr", 1e-6),
            head_lr=head_lr, head_params=head_params,
            label_smoothing=config["training"].get("label_smoothing", 0.0),
            use_focal_loss=config["training"].get("use_focal_loss", False),
            focal_alpha=config["training"].get("focal_alpha", 0.25),
            focal_gamma=config["training"].get("focal_gamma", 2.0),
        )

    # Load pretrained weights
    trainer.load_checkpoint(checkpoint_path, diagnosis_head, None)
    print("Loaded pretrained checkpoint.")

    # Progressive unfreezing
    if freeze_backbone:
        num_layers = config["model"]["num_layers"]
        unfreeze_pct = config["training"].get("unfreeze_percentage", 0.25)
        num_unfreeze = max(1, int(num_layers * unfreeze_pct))
        for _, param in model.named_parameters():
            param.requires_grad = False
        total_layers = len(model.transformer.layers)
        for i in range(total_layers - num_unfreeze, total_layers):
            for param in model.transformer.layers[i].parameters():
                param.requires_grad = True
        for param in diagnosis_head.parameters():
            param.requires_grad = True
        print(f"Progressive unfreezing: last {num_unfreeze}/{total_layers} layers + head")

    # ---- Training loop ----
    exp_ckpt_dir = Path(base_ckpt_dir) / f"exp{exp_idx + 1}"
    exp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_ckpt_path = None

    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            trainer, train_loader, "diagnosis",
            diagnosis_head=diagnosis_head, prognosis_head=None,
            desc=f"Exp{exp_idx+1} Epoch {epoch+1}/{num_epochs} [Train]",
        )

        # Validation
        model.eval()
        diagnosis_head.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for batch in val_loader:
                signals = batch["signals"].to(device)
                labels = batch["labels"].to(device).long()
                output = model(signals, task_type="diagnosis", return_tokens=False)
                tf_feats = output["features"][:, model.num_prompts, :]
                logits = diagnosis_head(tf_feats.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=trainer.label_smoothing)
                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"  Exp{exp_idx+1} Epoch {epoch+1}: Train Loss={train_metrics.get('loss',0):.4f} "
              f"Acc={train_metrics.get('accuracy',0):.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = str(exp_ckpt_dir / f"best_exp{exp_idx+1}.pt")
            trainer.save_checkpoint(best_ckpt_path, diagnosis_head, None, epoch=epoch)

        if (epoch + 1) % config["logging"]["save_every"] == 0:
            ckpt_p = str(exp_ckpt_dir / f"exp{exp_idx+1}_epoch_{epoch+1}.pt")
            trainer.save_checkpoint(ckpt_p, diagnosis_head, None, epoch=epoch)

    # Save final
    final_path = str(exp_ckpt_dir / f"final_exp{exp_idx+1}.pt")
    trainer.save_checkpoint(final_path, diagnosis_head, None, epoch=num_epochs - 1)

    # ---- Evaluate on test half ----
    # Reload best checkpoint
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        print(f"\nLoading best checkpoint for evaluation: {best_ckpt_path}")
        trainer.load_checkpoint(best_ckpt_path, diagnosis_head, None)

    model.eval()
    diagnosis_head.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            signals = batch["signals"].to(device)
            labels = batch["labels"].to(device).long()
            output = model(signals, task_type="diagnosis", return_tokens=False)
            tf_feats = output["features"][:, model.num_prompts, :]
            logits = diagnosis_head(tf_feats.unsqueeze(1)).squeeze(1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_labels_list)
    test_acc = (all_preds == all_true).mean()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_idx+1} RESULTS — Test on '{test_folder['name']}' (50% held out)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f} ({(all_preds == all_true).sum()}/{len(all_true)})")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")

    # Classification report
    present = sorted(set(all_true) | set(all_preds))
    target_names = [FAULT_NAMES[i] if i < len(FAULT_NAMES) else f"Class_{i}" for i in present]
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, labels=present, target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds, labels=present)
    print("Confusion Matrix:")
    header = "".join(f"{FAULT_NAMES[i][:8]:>9s}" if i < len(FAULT_NAMES) else f"{'C'+str(i):>9s}" for i in present)
    print(f"{'Pred->':>18s}{header}")
    for row_idx, row_label in enumerate(present):
        row_name = FAULT_NAMES[row_label][:16] if row_label < len(FAULT_NAMES) else f"Class_{row_label}"
        row_vals = "".join(f"{cm[row_idx, c]:>9d}" for c in range(len(present)))
        print(f"{row_name:>18s}{row_vals}")

    # Save results
    exp_save_dir = Path(base_save_dir) / f"exp{exp_idx + 1}"
    exp_save_dir.mkdir(parents=True, exist_ok=True)
    import json
    results = {
        "experiment": exp_idx + 1,
        "test_folder": test_folder["name"],
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "num_test_samples": int(len(all_true)),
        "confusion_matrix": cm.tolist(),
        "labels": [FAULT_NAMES[i] for i in present],
    }
    with open(exp_save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {exp_save_dir / 'results.json'}")

    return test_acc, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="3-Fold Cross-Experiment Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Pretrained checkpoint path")
    parser.add_argument("--exp", type=int, default=None, help="Run only experiment N (1, 2, or 3). Default: run all.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    folders = config["data"]["folders"]
    base_ckpt_dir = config["logging"]["checkpoint_dir"]
    base_log_dir = config["logging"]["log_dir"]
    base_save_dir = config["evaluation"]["save_dir"]

    os.makedirs(base_ckpt_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)
    os.makedirs(base_save_dir, exist_ok=True)

    if args.exp is not None:
        # Run single experiment
        exp_idx = args.exp - 1
        assert 0 <= exp_idx < len(folders), f"--exp must be 1..{len(folders)}"
        run_experiment(exp_idx, folders, config, args.checkpoint, device, base_ckpt_dir, base_log_dir, base_save_dir)
    else:
        # Run all 3
        results = []
        for exp_idx in range(len(folders)):
            acc, val_acc = run_experiment(exp_idx, folders, config, args.checkpoint, device, base_ckpt_dir, base_log_dir, base_save_dir)
            results.append((exp_idx + 1, folders[exp_idx]["name"], acc, val_acc))

        print("\n" + "=" * 70)
        print("SUMMARY — 3-Fold Cross-Experiment Results")
        print("=" * 70)
        for exp_num, folder_name, acc, val_acc in results:
            print(f"  Exp {exp_num} (test={folder_name:30s}): Test Acc = {acc:.4f}  |  Best Val = {val_acc:.4f}")
        avg_acc = np.mean([r[2] for r in results])
        print(f"\n  Average Test Accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
