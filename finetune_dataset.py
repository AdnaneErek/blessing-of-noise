#!/usr/bin/env python3
"""
Fine-tune pretrained model on a specific dataset and evaluate

Usage:
    python finetune_dataset.py --dataset CWRU --checkpoint checkpoints/final_model_pretrain.pt
"""
import argparse
import yaml
import json
import os
import subprocess
import sys


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Fine-tune pretrained model on a dataset and evaluate')
    parser.add_argument('--config', type=str, default='configs/paper_exact_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model_pretrain.pt',
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., CWRU, JNUB, KAUG17, HSG18, XJTU-SY)')
    parser.add_argument('--task', type=str, default=None,
                       help='Task type: diagnosis or prognosis (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine task type
    if args.task is None:
        # Auto-detect from config
        pretrain_datasets = config['data'].get('pretrain_datasets', [])
        pretrain_task_names = config['data'].get('pretrain_task_names', [])
        
        if args.dataset in pretrain_datasets:
            idx = pretrain_datasets.index(args.dataset)
            task_type = pretrain_task_names[idx].lower() if idx < len(pretrain_task_names) else 'diagnosis'
        else:
            task_type = 'diagnosis'  # default
    else:
        task_type = args.task.lower()
    
    # Map task name to training task
    if 'diagnosis' in task_type:
        train_task = 'diagnosis'
    elif 'prognosis' in task_type:
        train_task = 'prognosis'
    else:
        train_task = 'diagnosis'  # default
    
    print(f"Fine-tuning on dataset: {args.dataset}")
    print(f"Task type: {train_task}")
    print(f"Pretrained checkpoint: {args.checkpoint}")
    
    # Create output directories
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['evaluation']['save_dir'], exist_ok=True)
    
    # Update config with dataset name (temporarily)
    # Save original config
    original_dataset = config['data']['dataset_name']
    config['data']['dataset_name'] = args.dataset
    
    # Fine-tune
    print(f"\n=== Fine-tuning on {args.dataset} ===")
    finetune_cmd = [
        sys.executable,
        "train_rmgpt.py",
        "--config", args.config,
        "--task", train_task,
        "--resume", args.checkpoint
    ]
    
    try:
        result = subprocess.run(finetune_cmd, check=True, text=True)
        print("Fine-tuning completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during fine-tuning: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Find the fine-tuned model checkpoint
    checkpoint_dir = config['logging']['checkpoint_dir']
    finetuned_checkpoint = os.path.join(checkpoint_dir, f"final_model_{train_task}.pt")
    
    if not os.path.exists(finetuned_checkpoint):
        print(f"Warning: Fine-tuned checkpoint not found at {finetuned_checkpoint}")
        print("Checking for other checkpoints...")
        # Look for most recent checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and args.dataset.lower() in f.lower()]
        if checkpoints:
            finetuned_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            print(f"Using: {finetuned_checkpoint}")
        else:
            print("No fine-tuned checkpoint found. Fine-tuning may have failed.")
            sys.exit(1)
    
    print(f"\nFine-tuned model saved to: {finetuned_checkpoint}")
    print(f"\nTo evaluate this model, run:")
    print(f"  python train_rmgpt.py --config {args.config} --task {train_task} --resume {finetuned_checkpoint}")


if __name__ == '__main__':
    main()
