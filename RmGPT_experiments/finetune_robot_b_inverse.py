#!/usr/bin/env python3
"""
Fine-tune RmGPT in inverse setting:
- Train on Robot-B data from testDatasets/<folder> (few-shot)
- Save separate Robot-B inverse checkpoints
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import load_robot_test_data
from model.rmgpt import DiagnosisHead, RmGPT
from train.trainer import RmGPTTrainer, train_epoch


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RmGPT on Robot-B (inverse setup)")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretraining checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["evaluation"]["save_dir"], exist_ok=True)

    # Infer dimensions from checkpoint
    print(f"\nLoading base checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    patch_embed_shape = state_dict["signal_tokenizer.patch_embed.weight"].shape
    patch_length = config["model"]["patch_length"]
    signal_dim_from_ckpt = patch_embed_shape[1] // patch_length

    num_classes_from_ckpt = None
    if "diagnosis_head_state_dict" in checkpoint:
        diag_head_state = checkpoint["diagnosis_head_state_dict"]
        for key in sorted(diag_head_state.keys(), reverse=True):
            if key.endswith(".weight") and len(diag_head_state[key].shape) == 2:
                num_classes_from_ckpt = diag_head_state[key].shape[0]
                break

    print(f"Inferred from checkpoint: signal_dim={signal_dim_from_ckpt}, num_classes={num_classes_from_ckpt}")

    # Load Robot-B train source from testDatasets/<folder>
    data_dir = config["data"].get("data_dir", "./data/raw/dataset")
    train_folder = config["data"].get("robot_b_train_folder", "20241016")
    print("\n" + "=" * 60)
    print(f"Loading Robot-B few-shot train data from testDatasets/{train_folder}")
    print("=" * 60)
    all_signals, all_labels = load_robot_test_data(data_dir=data_dir, folder_name=train_folder, use_individual_files=True)
    print(f"Loaded Robot-B train source: {len(all_signals)} samples, labels={np.unique(all_labels)}")

    val_size = config["data"].get("finetune_val_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    train_signals, val_signals, train_labels, val_labels = train_test_split(
        all_signals,
        all_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=all_labels,
    )
    print(f"Split: train={len(train_signals)}, val={len(val_signals)}")

    train_dataset = PHMSignalDataset(
        train_signals,
        labels=train_labels,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
        add_noise=False,
    )
    val_dataset = PHMSignalDataset(
        val_signals,
        labels=val_labels,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
        add_noise=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    model_config = config["model"].copy()
    model_config["signal_dim"] = signal_dim_from_ckpt
    model_config["input_channels"] = signal_dim_from_ckpt
    tokenizer_stride = model_config.pop("tokenizer_stride", model_config.get("patch_length", 256))
    model_config["tokenizer_stride"] = tokenizer_stride

    allowed = {
        "signal_dim",
        "patch_length",
        "tokenizer_stride",
        "embed_dim",
        "num_prompts",
        "num_faults",
        "num_layers",
        "num_heads",
        "ff_dim",
        "dropout",
        "n_fft",
        "wavelet",
        "wavelet_levels",
        "input_channels",
    }
    model = RmGPT(**{k: v for k, v in model_config.items() if k in allowed})
    model = model.to(device)

    num_classes = num_classes_from_ckpt or len(np.unique(all_labels))
    diagnosis_head = DiagnosisHead(
        embed_dim=model_config["embed_dim"],
        num_classes=num_classes,
        improved=config.get("model", {}).get("improved_diagnosis_head", False),
    ).to(device)
    print(f"Created diagnosis head: num_classes={num_classes}")

    steps_per_epoch = len(train_loader)
    num_epochs = config["training"]["finetune_epochs"]
    total_steps = steps_per_epoch * num_epochs
    head_lr = config["training"].get("head_lr", None)
    head_params = list(diagnosis_head.parameters())
    freeze_backbone = config["training"].get("freeze_backbone", False)

    if freeze_backbone:
        head_only_lr = head_lr if head_lr is not None else config["training"]["lr"]
        trainer = RmGPTTrainer(
            model=model,
            device=device,
            lr=head_only_lr,
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_grad_norm=config["training"]["max_grad_norm"],
            total_steps=total_steps,
            lr_schedule=config["training"].get("lr_schedule", "constant"),
            min_lr=config["training"].get("min_lr", 1e-6),
            head_lr=head_only_lr,
            head_params=head_params,
            label_smoothing=config["training"].get("label_smoothing", 0.0),
            use_focal_loss=config["training"].get("use_focal_loss", False),
            focal_alpha=config["training"].get("focal_alpha", 0.25),
            focal_gamma=config["training"].get("focal_gamma", 2.0),
        )
    else:
        trainer = RmGPTTrainer(
            model=model,
            device=device,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_grad_norm=config["training"]["max_grad_norm"],
            total_steps=total_steps,
            lr_schedule=config["training"].get("lr_schedule", "constant"),
            min_lr=config["training"].get("min_lr", 1e-6),
            head_lr=head_lr,
            head_params=head_params,
            label_smoothing=config["training"].get("label_smoothing", 0.0),
            use_focal_loss=config["training"].get("use_focal_loss", False),
            focal_alpha=config["training"].get("focal_alpha", 0.25),
            focal_gamma=config["training"].get("focal_gamma", 2.0),
        )

    saved_epoch = trainer.load_checkpoint(args.checkpoint, diagnosis_head, None)
    print(f"Loaded base checkpoint epoch={saved_epoch}")

    if freeze_backbone:
        num_layers = config["model"]["num_layers"]
        unfreeze_percentage = config["training"].get("unfreeze_percentage", 0.25)
        num_layers_to_unfreeze = max(1, int(num_layers * unfreeze_percentage))
        for _, param in model.named_parameters():
            param.requires_grad = False
        total_layers = len(model.transformer.layers)
        first_unfreeze = total_layers - num_layers_to_unfreeze
        for i in range(first_unfreeze, total_layers):
            for param in model.transformer.layers[i].parameters():
                param.requires_grad = True
        for param in diagnosis_head.parameters():
            param.requires_grad = True
        print(f"Progressive unfreezing: last {num_layers_to_unfreeze}/{total_layers} layers + head")

    start_epoch = 0
    if args.resume:
        resumed_epoch = trainer.load_checkpoint(args.resume, diagnosis_head, None)
        if resumed_epoch is not None:
            start_epoch = min(resumed_epoch + 1, num_epochs - 1)

    best_val_acc = 0.0
    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_epoch(
            trainer,
            train_loader,
            "diagnosis",
            diagnosis_head=diagnosis_head,
            prognosis_head=None,
            desc=f"Epoch {epoch+1}/{num_epochs} [Robot-B Inverse Train]",
        )

        model.eval()
        diagnosis_head.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                signals = batch["signals"].to(device)
                labels = batch["labels"].to(device).long()
                output = model(signals, task_type="diagnosis", return_tokens=False)
                features = output["features"]
                tf_features = features[:, model.num_prompts, :]
                logits = diagnosis_head(tf_features.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=trainer.label_smoothing)
                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(
            f"Epoch {epoch+1}: Train Loss={train_metrics.get('loss', 0):.4f}, "
            f"Train Acc={train_metrics.get('accuracy', 0):.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(config["logging"]["checkpoint_dir"], f"best_robot_b_inverse_epoch_{epoch+1}.pt")
            trainer.save_checkpoint(best_path, diagnosis_head, None, epoch=epoch)
            print(f"  -> Saved best checkpoint: {best_path}")

        if (epoch + 1) % config["logging"]["save_every"] == 0:
            ckpt_path = os.path.join(config["logging"]["checkpoint_dir"], f"robot_b_inverse_epoch_{epoch+1}.pt")
            trainer.save_checkpoint(ckpt_path, diagnosis_head, None, epoch=epoch)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(config["logging"]["checkpoint_dir"], "final_model_robot_b_inverse.pt")
    trainer.save_checkpoint(final_path, diagnosis_head, None, epoch=num_epochs - 1)
    print(f"\nRobot-B inverse fine-tuning complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
