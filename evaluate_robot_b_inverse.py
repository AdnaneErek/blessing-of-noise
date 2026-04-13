#!/usr/bin/env python3
"""
Evaluate Robot-B inverse fine-tuned model:
1) Robot-B held-out simulation-like split from testDatasets/<robot_b_train_folder>
2) Robot-A target domain from finetuningDatasets (all folders or configured subset)
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import load_robot_finetuning_data, load_robot_test_data
from model.rmgpt import DiagnosisHead, RmGPT
from train.trainer import RmGPTTrainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_diagnosis(model: RmGPT, diagnosis_head: DiagnosisHead, dataloader: DataLoader, device: torch.device, name: str):
    model.eval()
    diagnosis_head.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            signals = batch["signals"].to(device)
            labels = batch["labels"].to(device).long()
            out = model(signals, task_type="diagnosis", return_tokens=False)
            features = out["features"]
            tf_features = features[:, model.num_prompts, :]
            logits = diagnosis_head(tf_features.unsqueeze(1)).squeeze(1)
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="weighted", zero_division=0)
    cm = confusion_matrix(labels_all, preds_all)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Robot-B inverse model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(config["evaluation"]["save_dir"], exist_ok=True)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    patch_embed_shape = state_dict["signal_tokenizer.patch_embed.weight"].shape
    patch_length = config["model"]["patch_length"]
    signal_dim = patch_embed_shape[1] // patch_length

    num_classes = 9
    if "diagnosis_head_state_dict" in checkpoint:
        diag_head_state = checkpoint["diagnosis_head_state_dict"]
        for key in sorted(diag_head_state.keys(), reverse=True):
            if key.endswith(".weight") and len(diag_head_state[key].shape) == 2:
                num_classes = diag_head_state[key].shape[0]
                break

    model_cfg = config["model"].copy()
    model_cfg["signal_dim"] = signal_dim
    model_cfg["input_channels"] = signal_dim
    model_cfg["tokenizer_stride"] = model_cfg.pop("tokenizer_stride", model_cfg.get("patch_length", 256))
    improved_head = model_cfg.pop("improved_diagnosis_head", False)

    model = RmGPT(**model_cfg).to(device)
    diagnosis_head = DiagnosisHead(embed_dim=model_cfg["embed_dim"], num_classes=num_classes, improved=improved_head).to(device)
    trainer = RmGPTTrainer(model=model, device=device, lr=0.0)
    trainer.load_checkpoint(args.checkpoint, diagnosis_head, None)

    # Eval domain 1: Robot-B folder used for fine-tuning source (full set, reported for reference)
    data_dir = config["data"].get("data_dir", "./data/raw/dataset")
    robot_b_folder = config["data"].get("robot_b_train_folder", "20241016")
    b_signals, b_labels = load_robot_test_data(data_dir=data_dir, folder_name=robot_b_folder, use_individual_files=True)
    b_dataset = PHMSignalDataset(
        b_signals,
        labels=b_labels,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
    )
    b_loader = DataLoader(
        b_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )
    metrics_robot_b = evaluate_diagnosis(model, diagnosis_head, b_loader, device, "Robot-B Source")

    # Eval domain 2: Robot-A from finetuningDatasets
    eval_folders = config["data"].get("robot_a_eval_folders", None)
    a_signals, a_labels = load_robot_finetuning_data(data_dir=data_dir, folder_name=eval_folders, use_individual_files=True)
    a_dataset = PHMSignalDataset(
        a_signals,
        labels=a_labels,
        normalize=config["data"]["normalize"],
        signal_window=config["data"]["signal_window"],
        overlap=config["data"]["overlap"],
    )
    a_loader = DataLoader(
        a_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )
    metrics_robot_a = evaluate_diagnosis(model, diagnosis_head, a_loader, device, "Robot-A Target")

    # Per-folder Robot-A metrics + confusion matrices
    if eval_folders is None:
        finetune_root = os.path.join(data_dir, "finetuningDatasets")
        per_folder_names = sorted(
            [d for d in os.listdir(finetune_root) if os.path.isdir(os.path.join(finetune_root, d)) and not d.startswith(".")]
        )
    elif isinstance(eval_folders, list):
        per_folder_names = eval_folders
    else:
        per_folder_names = [eval_folders]

    per_folder_results = {}
    for folder_name in per_folder_names:
        folder_signals, folder_labels = load_robot_finetuning_data(
            data_dir=data_dir, folder_name=folder_name, use_individual_files=True
        )
        folder_dataset = PHMSignalDataset(
            folder_signals,
            labels=folder_labels,
            normalize=config["data"]["normalize"],
            signal_window=config["data"]["signal_window"],
            overlap=config["data"]["overlap"],
        )
        folder_loader = DataLoader(
            folder_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["hardware"]["num_workers"],
            pin_memory=config["hardware"]["pin_memory"],
        )
        per_folder_results[folder_name] = {
            "samples": int(len(folder_signals)),
            "metrics": evaluate_diagnosis(model, diagnosis_head, folder_loader, device, f"Robot-A Target ({folder_name})"),
        }

    print("\nRobot-B Source Metrics:")
    print(f"  Accuracy: {metrics_robot_b['accuracy']:.4f}")
    print(f"  Precision: {metrics_robot_b['precision']:.4f}")
    print(f"  Recall: {metrics_robot_b['recall']:.4f}")
    print(f"  F1: {metrics_robot_b['f1']:.4f}")
    print("  Confusion Matrix:")
    print(np.array(metrics_robot_b["confusion_matrix"]))

    print("\nRobot-A Target Metrics:")
    print(f"  Accuracy: {metrics_robot_a['accuracy']:.4f}")
    print(f"  Precision: {metrics_robot_a['precision']:.4f}")
    print(f"  Recall: {metrics_robot_a['recall']:.4f}")
    print(f"  F1: {metrics_robot_a['f1']:.4f}")
    print("  Confusion Matrix (TOTAL across Robot-A folders):")
    print(np.array(metrics_robot_a["confusion_matrix"]))

    print("\nRobot-A Target Per-Folder Confusion Matrices:")
    for folder_name, folder_result in per_folder_results.items():
        print(f"\n  Folder: {folder_name}")
        print(f"    Samples: {folder_result['samples']}")
        fm = folder_result["metrics"]
        print(f"    Accuracy: {fm['accuracy']:.4f}")
        print("    Confusion Matrix:")
        print(np.array(fm["confusion_matrix"]))

    out = {
        "checkpoint": args.checkpoint,
        "setting": "robot_b_inverse",
        "robot_b_source": {"folder": robot_b_folder, "samples": int(len(b_signals)), "metrics": metrics_robot_b},
        "robot_a_target": {
            "folders": eval_folders if eval_folders is not None else "all_finetuningDatasets",
            "samples": int(len(a_signals)),
            "metrics": metrics_robot_a,
        },
        "robot_a_target_per_folder": per_folder_results,
    }
    out_path = os.path.join(
        config["evaluation"]["save_dir"],
        f"eval_robot_b_inverse_{os.path.basename(args.checkpoint).replace('.pt', '')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
