#!/usr/bin/env python3
"""
Multitask pretraining on robot simulation data:
 - Supervised classification (diagnosis) using labels
 - Self-supervised masked token prediction

Implementation:
Both losses are combined with lambda weights in a single backward pass:
  total_loss = lambda_cls * cls_loss + lambda_mask * mask_loss
This ensures the config lambdas actually control the gradient contributions.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import load_robot_training_data
from data.split_strategy import paper_split_strategy
from model.rmgpt import DiagnosisHead, RmGPT
from train.trainer import RmGPTTrainer, train_epoch


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Multitask pretraining (CLS + Masked) on robot simulation data")
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data load
    data_dir = config["data"].get("data_dir", "./data/raw/dataset")
    training_folder = config["data"].get("robot_training_folder", None)
    use_individual_files = config["data"].get("use_individual_files", True)

    print("\n" + "=" * 60)
    print("Loading Robot Simulation Data for Multitask Pretraining")
    print("=" * 60)

    all_signals, all_labels = load_robot_training_data(
        data_dir=data_dir,
        folder_name=training_folder,
        use_individual_files=use_individual_files,
    )
    print(f"Total loaded: {len(all_signals)} samples | signals: {all_signals.shape} | labels: {all_labels.shape}")

    splits = paper_split_strategy(
        signals=all_signals,
        labels=all_labels,
        rul=None,
        test_size=config["data"]["test_size"],
        finetune_val_size=config["data"]["finetune_val_size"],
        random_state=config["data"]["random_state"],
        stratify=True,
    )
    train_signals, train_labels, _ = splits["finetune_train"]
    val_signals, val_labels, _ = splits["finetune_val"]
    print(f"Splits -> train: {len(train_signals)}, val: {len(val_signals)}")

    # Augmentation/noise for training
    noise_cfg = config["data"].get("noise_augmentation", {})
    add_noise = bool(noise_cfg.get("enabled", False))
    noise_std = float(noise_cfg.get("std", 0.01))
    noise_type = str(noise_cfg.get("type", "gaussian"))
    if add_noise:
        print(f"Noise augmentation: enabled (type={noise_type}, std={noise_std})")

    train_dataset = PHMSignalDataset(
        train_signals,
        labels=train_labels,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
        add_noise=add_noise,
        noise_std=noise_std,
        noise_type=noise_type,
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

    # Model
    model_cfg = config["model"].copy()
    # Confirm channels via a sample
    sample = train_signals[0]
    inferred_channels = sample.shape[-1] if sample.ndim >= 2 else model_cfg.get("signal_dim", 9)
    model_cfg["signal_dim"] = inferred_channels
    model_cfg["input_channels"] = inferred_channels
    model_cfg["tokenizer_stride"] = model_cfg.pop("tokenizer_stride", model_cfg.get("patch_length", 256))

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
    model = RmGPT(**{k: v for k, v in model_cfg.items() if k in allowed})
    model = model.to(device)
    num_classes = len(set(train_labels))
    diagnosis_head = DiagnosisHead(
        embed_dim=model_cfg["embed_dim"],
        num_classes=num_classes,
        improved=config.get("model", {}).get("improved_diagnosis_head", False),
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"Head params: {sum(p.numel() for p in diagnosis_head.parameters())/1e6:.2f}M")

    # Trainer
    num_epochs = int(config["training"]["pretrain_epochs"])
    total_steps = len(train_loader) * num_epochs
    head_lr = config["training"].get("head_lr", None)

    trainer = RmGPTTrainer(
        model=model,
        device=device,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        total_steps=total_steps,
        lr_schedule=config["training"]["lr_schedule"],
        min_lr=config["training"]["min_lr"],
        head_lr=head_lr,
        head_params=list(diagnosis_head.parameters()),
        label_smoothing=config["training"].get("label_smoothing", 0.0),
        use_focal_loss=config["training"].get("use_focal_loss", False),
        focal_alpha=config["training"].get("focal_alpha", 0.25),
        focal_gamma=config["training"].get("focal_gamma", 2.0),
    )

    # Resume
    start_epoch = 0
    if args.resume:
        saved_epoch = trainer.load_checkpoint(args.resume, diagnosis_head=diagnosis_head)
        if saved_epoch is not None:
            start_epoch = min(saved_epoch + 1, num_epochs - 1)
            print(f"Resuming from epoch {start_epoch}")

    # Multitask settings
    mt_cfg = config["training"].get("multitask", {})
    mt_enabled = bool(mt_cfg.get("enabled", True))
    mask_every_k = int(mt_cfg.get("mask_every_k_steps", 1))
    mask_prob = float(mt_cfg.get("mask_prob", 0.2))
    lambda_cls = float(mt_cfg.get("lambda_cls", 1.0))
    lambda_mask = float(mt_cfg.get("lambda_mask", 0.5))
    span_cfg = mt_cfg.get("span_mask", {})
    span_enabled = bool(span_cfg.get("enabled", False))
    avg_span = int(span_cfg.get("avg_span", 24))
    ch_drop_prob = float(mt_cfg.get("channel_drop", {}).get("prob", 0.0))
    print(f"\nMultitask: enabled={mt_enabled} mask_every_k_steps={mask_every_k} "
          f"mask_prob={mask_prob} lambdas(cls={lambda_cls}, mask={lambda_mask}) "
          f"span_mask={span_enabled} avg_span={avg_span} channel_drop_prob={ch_drop_prob}")

    # Output dirs
    log_dir = Path(config["logging"]["log_dir"])
    ckpt_dir = Path(config["logging"]["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Read label smoothing for manual cls loss
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))

    best_val_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        diagnosis_head.train()
        epoch_cls_losses = []
        epoch_cls_accs = []
        epoch_mask_losses = []
        epoch_total_losses = []

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Multitask Train]")
        for step_idx, batch in pbar:
            trainer.optimizer.zero_grad()

            # ---- 1) Classification forward ----
            signals_cls = batch["signals"].to(device)
            labels = batch["labels"].to(device).long()
            out_cls = model(signals_cls, task_type="diagnosis", return_tokens=False)
            tf_features = out_cls["features"][:, model.num_prompts, :]  # [B, D]
            logits = diagnosis_head(tf_features.unsqueeze(1)).squeeze(1)
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean().item()
            epoch_cls_losses.append(cls_loss.item())
            epoch_cls_accs.append(accuracy)

            # ---- 2) Masked-token forward (every k steps) ----
            do_mask = mt_enabled and ((step_idx + 1) % mask_every_k == 0)
            mask_loss = torch.tensor(0.0, device=device)

            if do_mask:
                signals_m = batch["signals"].clone()
                # Optional channel-drop (on CPU tensor, before .to(device))
                if ch_drop_prob > 0.0:
                    bsz_m, seq_len_m, ch_m = signals_m.shape
                    drop_mask = (torch.rand(bsz_m, ch_m) < ch_drop_prob)
                    for i in range(bsz_m):
                        if drop_mask[i].any():
                            signals_m[i, :, drop_mask[i]] = 0.0
                # Optional span masking on input
                if span_enabled and avg_span > 0:
                    bsz_m, seq_len_m, ch_m = signals_m.shape
                    num_spans = max(1, int(mask_prob * seq_len_m / max(1, avg_span)))
                    for i in range(bsz_m):
                        for _ in range(num_spans):
                            span_len = max(1, int(round(
                                np.random.normal(loc=float(avg_span),
                                                 scale=float(max(1.0, avg_span * 0.2))))))
                            start = torch.randint(0, max(1, seq_len_m - span_len + 1), (1,)).item()
                            signals_m[i, start:start+span_len, :] = 0.0

                signals_m = signals_m.to(device)
                bsz_m, seq_len_m, _ = signals_m.shape

                # Forward through model
                out_m = model(signals_m, task_type="pretrain", return_tokens=True)
                features_m = out_m["features"]
                signal_tokens_m = out_m["signal_tokens"]
                num_patches = signal_tokens_m.shape[1]
                sig_start = model.num_prompts + 1

                sig_feats = features_m[:, sig_start:sig_start + num_patches, :]

                # Random token mask
                n_mask = max(1, int(num_patches * mask_prob))
                tok_mask = torch.zeros(bsz_m, num_patches, dtype=torch.bool, device=device)
                for bi in range(bsz_m):
                    idx = torch.randperm(num_patches, device=device)[:n_mask]
                    tok_mask[bi, idx] = True

                orig_masked = sig_feats[tok_mask]  # [N_masked, D]

                # Zero-out masked positions and re-run transformer
                masked_feats = features_m.clone()
                zero_emb = torch.zeros(features_m.shape[-1], device=device)
                for bi in range(bsz_m):
                    for ti in torch.where(tok_mask[bi])[0]:
                        masked_feats[bi, sig_start + ti, :] = zero_emb

                pred_feats = model.transformer(masked_feats)
                pred_sig = pred_feats[:, sig_start:sig_start + num_patches, :]
                pred_masked = pred_sig[tok_mask]

                mask_loss = F.mse_loss(pred_masked, orig_masked)
                epoch_mask_losses.append(mask_loss.item())

            # ---- 3) Combined loss with lambda weighting ----
            total_loss = lambda_cls * cls_loss + lambda_mask * mask_loss
            total_loss.backward()

            all_params = list(model.parameters()) + list(diagnosis_head.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, trainer.max_grad_norm)
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.global_step += 1
            epoch_total_losses.append(total_loss.item())

            # Update progress
            disp = {
                "total": f"{np.mean(epoch_total_losses):.4f}",
                "cls": f"{np.mean(epoch_cls_losses):.4f}",
                "acc": f"{np.mean(epoch_cls_accs):.4f}",
            }
            if epoch_mask_losses:
                disp["mask"] = f"{np.mean(epoch_mask_losses):.4f}"
            pbar.set_postfix(disp)

        # Validation (classification only)
        model.eval()
        diagnosis_head.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for vbatch in val_loader:
                signals = vbatch["signals"].to(device)
                labels = vbatch["labels"].to(device).long()
                out = model(signals, task_type="diagnosis", return_tokens=False)
                feats = out["features"][:, model.num_prompts, :]
                logits = diagnosis_head(feats.unsqueeze(1)).squeeze(1)
                vloss = torch.nn.functional.cross_entropy(
                    logits, labels, label_smoothing=config["training"].get("label_smoothing", 0.0)
                )
                preds = torch.argmax(logits, dim=1)
                vacc = (preds == labels).float().mean().item()
                val_losses.append(vloss.item())
                val_accs.append(vacc)

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        avg_val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        print(f"\nEpoch {epoch+1}/{num_epochs} summary:")
        print(f"  Train total loss: {np.mean(epoch_total_losses):.4f}")
        print(f"  Train CLS loss: {np.mean(epoch_cls_losses):.4f}  (lambda={lambda_cls})")
        print(f"  Train CLS acc:  {np.mean(epoch_cls_accs):.4f}")
        if epoch_mask_losses:
            print(f"  Train MASK loss: {np.mean(epoch_mask_losses):.4f}  (lambda={lambda_mask})")
        print(f"  Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.4f}")

        # Save periodic
        if ((epoch + 1) % config["logging"]["save_every"] == 0) or ((epoch + 1) == num_epochs):
            ckpt_path = ckpt_dir / f"pretrain_robot_sim_multitask_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(str(ckpt_path), diagnosis_head=diagnosis_head, epoch=epoch)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_path = ckpt_dir / "pretrain_robot_sim_multitask_best.pt"
            trainer.save_checkpoint(str(best_path), diagnosis_head=diagnosis_head, epoch=epoch)
            print(f"  Saved best (val_acc={avg_val_acc:.4f}): {best_path}")

    # Save final
    final_path = ckpt_dir / "pretrain_robot_sim_multitask_final.pt"
    trainer.save_checkpoint(str(final_path), diagnosis_head=diagnosis_head, epoch=num_epochs - 1)
    print(f"\nMultitask pretraining complete! Final: {final_path}")


if __name__ == "__main__":
    main()

