#!/usr/bin/env python3
"""
Evaluate fine-tuned RmGPT model on test set

Usage:
    python evaluate_finetuned.py --checkpoint checkpoints/final_model_diagnosis.pt --dataset CWRU --task diagnosis
"""
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error

from model.rmgpt import RmGPT, DiagnosisHead, PrognosisHead
from train.trainer import RmGPTTrainer
from data.dataset import PHMSignalDataset
from data.split_strategy import paper_split_from_phmd


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_diagnosis(model: RmGPT, diagnosis_head: DiagnosisHead, 
                      dataloader: DataLoader, device: torch.device):
    """Evaluate diagnosis model (classification)"""
    model.eval()
    diagnosis_head.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            signals = batch['signals'].to(device)
            
            # Get labels if available
            if 'labels' in batch:
                labels = batch['labels'].to(device).long()
            else:
                raise ValueError("Labels not found in batch. Check that test_labels are provided to PHMSignalDataset.")
            
            # Forward pass
            output = model(signals, task_type='diagnosis', return_tokens=False)
            features = output['features']
            
            # Use time-freq token features for classification
            tf_token_idx = model.num_prompts
            tf_features = features[:, tf_token_idx, :]  # [batch, embed_dim]
            
            # Classification
            logits = diagnosis_head(tf_features.unsqueeze(1))
            logits = logits.squeeze(1)  # [batch, num_classes]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def evaluate_prognosis(model: RmGPT, prognosis_head: PrognosisHead,
                      dataloader: DataLoader, device: torch.device):
    """Evaluate prognosis model (regression)"""
    model.eval()
    prognosis_head.eval()
    
    all_preds = []
    all_rul = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            signals = batch['signals'].to(device)
            rul = batch['rul'].to(device).float()
            
            # Forward pass
            output = model(signals, task_type='prognosis', return_tokens=False)
            features = output['features']
            
            # Use time-freq token for RUL prediction
            tf_token_idx = model.num_prompts
            tf_features = features[:, tf_token_idx, :]  # [batch, embed_dim]
            
            # Regression
            rul_pred = prognosis_head(tf_features.unsqueeze(1))
            rul_pred = rul_pred.squeeze(1)  # [batch]
            
            all_preds.extend(rul_pred.cpu().numpy())
            all_rul.extend(rul.cpu().numpy())
    
    # Compute metrics
    mse = mean_squared_error(all_rul, all_preds)
    mae = mean_absolute_error(all_rul, all_preds)
    rmse = np.sqrt(mse)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned RmGPT model')
    parser.add_argument('--config', type=str, default='configs/paper_exact_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (default: from config)')
    parser.add_argument('--task', type=str, choices=['diagnosis', 'prognosis'], default=None,
                       help='Task type: diagnosis or prognosis (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['evaluation']['save_dir'], exist_ok=True)
    
    # Determine dataset and task
    dataset_name = args.dataset or config['data']['dataset_name']
    
    # Determine task type
    if args.task:
        task_type = args.task
    else:
        # Auto-detect from config
        pretrain_datasets = config['data'].get('pretrain_datasets', [])
        pretrain_task_names = config['data'].get('pretrain_task_names', [])
        
        if dataset_name in pretrain_datasets:
            idx = pretrain_datasets.index(dataset_name)
            task_name = pretrain_task_names[idx] if idx < len(pretrain_task_names) else 'Diagnosis'
            task_type = 'diagnosis' if 'Diagnosis' in task_name else 'prognosis'
        else:
            task_type = config['data']['task_name'].lower()
    
    # Get corresponding task_name for data loading
    if 'diagnosis' in task_type:
        task_name = 'Diagnosis'
    else:
        task_name = 'Prognosis'
    
    print(f"Evaluating fine-tuned model on dataset: {dataset_name}")
    print(f"Task type: {task_type}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load test data with paper-compliant split
    splits = paper_split_from_phmd(
        dataset_name=dataset_name,
        task_name=task_name,
        test_size=config['data'].get('test_size', 0.2),
        finetune_val_size=config['data'].get('finetune_val_size', 0.1),
        random_state=config['data'].get('random_state', 42)
    )
    
    # Use test set for evaluation (the 20% that was untouched)
    test_signals, test_labels, test_rul = splits['test']
    
    print(f"Test samples: {len(test_signals)}")
    print(f"Test labels: {type(test_labels)}, None? {test_labels is None}, shape: {test_labels.shape if test_labels is not None else None}")
    print(f"Test set was NOT used during fine-tuning (20% untouched)")
    
    # Load checkpoint to infer model dimensions
    print("Loading checkpoint to infer model dimensions...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer signal_dim from checkpoint weights
    patch_embed_shape = state_dict['signal_tokenizer.patch_embed.weight'].shape
    patch_length_x_signal_dim = patch_embed_shape[1]
    patch_length = config['model']['patch_length']  # Should be 256
    signal_dim_from_ckpt = patch_length_x_signal_dim // patch_length
    
    # Infer num_classes from diagnosis_head (if fine-tuned model)
    num_classes_from_ckpt = None
    if 'diagnosis_head_state_dict' in checkpoint:
        diag_head_state = checkpoint['diagnosis_head_state_dict']
        # DiagnosisHead uses Sequential with head.4 as final linear layer
        # head.4.weight shape: [num_classes, embed_dim // 2]
        if 'head.4.weight' in diag_head_state:
            num_classes_from_ckpt = diag_head_state['head.4.weight'].shape[0]
            print(f"Found num_classes={num_classes_from_ckpt} from diagnosis_head (head.4.weight)")
        else:
            # Fallback: find last linear layer (smallest output dimension)
            for key in sorted(diag_head_state.keys()):
                if key.endswith('.weight') and len(diag_head_state[key].shape) == 2:
                    # Use the last one found (should be the final layer)
                    num_classes_from_ckpt = diag_head_state[key].shape[0]
            if num_classes_from_ckpt:
                print(f"Found num_classes={num_classes_from_ckpt} from diagnosis_head (inferred)")
    
    # Always infer num_faults from model (for model config)
    # Note: Fine-tuned models may have num_faults=1 (pretrained base) but num_classes=10 (diagnosis head)
    num_faults_from_ckpt = None
    fault_tokens_key = 'fault_tokenizer.fault_tokens'
    if fault_tokens_key in state_dict:
        fault_tokens_shape = state_dict[fault_tokens_key].shape
        num_faults_from_ckpt = fault_tokens_shape[1]  # [1, num_faults, embed_dim]
    else:
        print(f"Warning: {fault_tokens_key} not found in checkpoint")
    
    print(f"Inferred from checkpoint:")
    print(f"  signal_dim: {signal_dim_from_ckpt}")
    if num_classes_from_ckpt is not None:
        print(f"  num_classes (from head): {num_classes_from_ckpt}")
    elif num_faults_from_ckpt is not None:
        print(f"  num_faults: {num_faults_from_ckpt}")
    
    # Pad test signals to match model's expected signal_dim (if needed)
    if not isinstance(test_signals, np.ndarray):
        test_signals = np.array(test_signals)
    
    if len(test_signals) > 0:
        # Ensure 3D shape
        if test_signals.ndim == 2:
            test_signals = test_signals[:, :, np.newaxis]
        elif test_signals.ndim == 1:
            test_signals = test_signals.reshape(1, -1, 1)
        
        current_signal_dim = test_signals.shape[-1]
        if current_signal_dim < signal_dim_from_ckpt:
            print(f"Padding test signals from {current_signal_dim} to {signal_dim_from_ckpt} channels...")
            pad_size = signal_dim_from_ckpt - current_signal_dim
            padding = np.zeros((test_signals.shape[0], test_signals.shape[1], pad_size))
            test_signals = np.concatenate([test_signals, padding], axis=2)
            print(f"  Padded test signals shape: {test_signals.shape}")
        elif current_signal_dim > signal_dim_from_ckpt:
            print(f"Truncating test signals from {current_signal_dim} to {signal_dim_from_ckpt} channels...")
            test_signals = test_signals[..., :signal_dim_from_ckpt]
    
    # Create test dataset and loader
    # For diagnosis, ensure labels are provided
    dataset_labels = test_labels if (task_type == 'diagnosis' and test_labels is not None) else None
    dataset_rul = test_rul if (task_type == 'prognosis' and test_rul is not None) else None
    
    if task_type == 'diagnosis' and dataset_labels is None:
        raise ValueError(f"test_labels is None for diagnosis task. Cannot evaluate without labels.")
    
    test_dataset = PHMSignalDataset(
        test_signals,
        labels=dataset_labels,
        rul=dataset_rul,
        normalize=config['data']['normalize'],
        signal_window=config['data']['signal_window'],
        overlap=config['data']['overlap']
    )
    
    # Verify dataset has labels (for diagnosis)
    if task_type == 'diagnosis' and test_dataset.labels is None:
        raise ValueError(f"PHMSignalDataset created without labels for diagnosis task.")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Determine model parameters from checkpoint
    model_config = config['model'].copy()
    model_config['signal_dim'] = signal_dim_from_ckpt
    
    # Extract tokenizer_stride
    tokenizer_stride = model_config.pop('tokenizer_stride', model_config.get('patch_length', 256))
    model_config['tokenizer_stride'] = tokenizer_stride
    
    # Determine num_classes for diagnosis
    # ALWAYS use the actual number of classes from the test labels (dataset-specific)
    # Don't trust checkpoint's num_classes as it may be from a different dataset
    num_classes = None
    if task_type == 'diagnosis':
        if test_labels is not None:
            # Use actual number of classes from test labels (most reliable)
            num_classes = len(set(test_labels))
            print(f"Using num_classes={num_classes} from test labels (dataset-specific)")
        elif num_classes_from_ckpt is not None:
            # Fallback: use checkpoint's num_classes if test labels not available
            num_classes = num_classes_from_ckpt
            print(f"Warning: Using num_classes={num_classes} from checkpoint (test labels not available)")
        else:
            num_classes = 4  # Default for CWRU
            print(f"Warning: Using default num_classes={num_classes} (CWRU default)")
        
        # Model config uses num_faults (for pretrained base, this is 1)
        model_config['num_faults'] = num_faults_from_ckpt if num_faults_from_ckpt is not None else 1
        print(f"Model num_faults={model_config['num_faults']} (from checkpoint, for pretrained base)")
    
    # Create model
    print("Loading fine-tuned model...")
    model = RmGPT(**model_config).to(device)
    
    # Create task heads
    diagnosis_head = None
    prognosis_head = None
    
    if task_type == 'diagnosis':
        # Check if config specifies improved head
        improved_head = config.get('model', {}).get('improved_diagnosis_head', False)
        diagnosis_head = DiagnosisHead(
            embed_dim=model_config['embed_dim'],
            num_classes=num_classes,
            improved=improved_head
        ).to(device)
        print(f"Created diagnosis head: num_classes={num_classes}, improved={improved_head}")
    elif task_type == 'prognosis':
        prognosis_head = PrognosisHead(
            embed_dim=model_config['embed_dim']
        ).to(device)
    
    # Load checkpoint
    trainer = RmGPTTrainer(model=model, device=device, lr=0.0)  # Dummy trainer for loading
    trainer.load_checkpoint(args.checkpoint, diagnosis_head, prognosis_head)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluate on test set
    print(f"\n=== Evaluating on Test Set ===")
    
    if task_type == 'diagnosis':
        metrics = evaluate_diagnosis(model, diagnosis_head, test_loader, device)
        print(f"\nTest Set Results (Diagnosis):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
    else:
        metrics = evaluate_prognosis(model, prognosis_head, test_loader, device)
        print(f"\nTest Set Results (Prognosis):")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'dataset': dataset_name,
        'task': task_type,
        'test_samples': len(test_signals),
        'metrics': metrics
    }
    
    results_path = os.path.join(
        config['evaluation']['save_dir'],
        f'eval_finetuned_{dataset_name}_{task_type}.json'
    )
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
