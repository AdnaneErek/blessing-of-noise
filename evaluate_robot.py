#!/usr/bin/env python3
"""
Evaluate fine-tuned RmGPT model on robot dataset
Evaluates on both:
1. Simulation test data (from training data split)
2. Real robot test data

Usage:
    python evaluate_robot.py --checkpoint checkpoints/final_model_diagnosis.pt --config configs/finetune_robot_from_scratch.yaml
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from model.rmgpt import RmGPT, DiagnosisHead
from train.trainer import RmGPTTrainer
from data.dataset import PHMSignalDataset
from data.robot_dataset_loader import (
    load_robot_training_data, 
    load_robot_test_data
)
from data.split_strategy import paper_split_strategy


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_diagnosis(model: RmGPT, diagnosis_head: DiagnosisHead, 
                      dataloader: DataLoader, device: torch.device,
                      dataset_name: str = ""):
    """Evaluate diagnosis model (classification)"""
    model.eval()
    diagnosis_head.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            signals = batch['signals'].to(device)
            
            # Get labels if available
            if 'labels' in batch:
                labels = batch['labels'].to(device).long()
            else:
                raise ValueError("Labels not found in batch.")
            
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
    
    # Per-class metrics
    cm = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned RmGPT model on robot dataset')
    parser.add_argument('--config', type=str, default='configs/finetune_robot_from_scratch.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test_folder', type=str, default='20241016',
                       help='Test dataset folder name (for real robot data)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['evaluation']['save_dir'], exist_ok=True)
    
    print(f"Evaluating fine-tuned model on robot dataset")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load checkpoint to infer model dimensions
    print("\nLoading checkpoint to infer model dimensions...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer signal_dim from checkpoint weights
    patch_embed_shape = state_dict['signal_tokenizer.patch_embed.weight'].shape
    patch_length_x_signal_dim = patch_embed_shape[1]
    patch_length = config['model']['patch_length']  # Should be 256
    signal_dim_from_ckpt = patch_length_x_signal_dim // patch_length
    
    # Infer num_classes from diagnosis_head
    num_classes_from_ckpt = None
    if 'diagnosis_head_state_dict' in checkpoint:
        diag_head_state = checkpoint['diagnosis_head_state_dict']
        # Find last linear layer
        for key in sorted(diag_head_state.keys(), reverse=True):
            if key.endswith('.weight') and len(diag_head_state[key].shape) == 2:
                num_classes_from_ckpt = diag_head_state[key].shape[0]
                print(f"Found num_classes={num_classes_from_ckpt} from diagnosis_head ({key})")
                break
    
    print(f"Inferred from checkpoint:")
    print(f"  signal_dim: {signal_dim_from_ckpt}")
    print(f"  num_classes: {num_classes_from_ckpt}")
    
    # Determine model parameters
    model_config = config['model'].copy()
    model_config['signal_dim'] = signal_dim_from_ckpt
    model_config['input_channels'] = signal_dim_from_ckpt  # Robot dataset uses 9 channels
    
    # Extract tokenizer_stride
    tokenizer_stride = model_config.pop('tokenizer_stride', model_config.get('patch_length', 256))
    model_config['tokenizer_stride'] = tokenizer_stride
    
    # Remove improved_diagnosis_head from model_config (it's for DiagnosisHead, not RmGPT)
    improved_head = model_config.pop('improved_diagnosis_head', False)
    
    # Create model
    print("\nLoading fine-tuned model...")
    model = RmGPT(**model_config).to(device)
    
    # Create diagnosis head (improved_head was already extracted above)
    diagnosis_head = DiagnosisHead(
        embed_dim=model_config['embed_dim'],
        num_classes=num_classes_from_ckpt or 9,  # Robot dataset has 9 classes
        improved=improved_head
    ).to(device)
    print(f"Created diagnosis head: num_classes={num_classes_from_ckpt or 9}, improved={improved_head}")
    
    # Load checkpoint
    trainer = RmGPTTrainer(model=model, device=device, lr=0.0)  # Dummy trainer for loading
    trainer.load_checkpoint(args.checkpoint, diagnosis_head, None)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # ========== EVALUATION 1: Simulation Test Data ==========
    print("\n" + "="*60)
    print("EVALUATION 1: Simulation Test Data (from training split)")
    print("="*60)
    
    # Load training data and split
    data_dir = config['data'].get('data_dir', './data/raw/dataset')
    training_folder = config['data'].get('robot_training_folder', None)
    
    # Use load_robot_training_data to auto-discover all folders if folder_name is None
    from data.robot_dataset_loader import load_robot_training_data
    
    print(f"\nLoading training data...")
    if training_folder is None:
        print("Auto-discovering all training folders...")
    else:
        print(f"Loading from folder: {training_folder}")
    
    all_signals, all_labels = load_robot_training_data(
        data_dir=data_dir,
        folder_name=training_folder,  # None = auto-discover all folders
        use_individual_files=True
    )
    
    # Apply paper-compliant split to get test set
    splits = paper_split_strategy(
        signals=all_signals,
        labels=all_labels,
        rul=None,
        test_size=config['data'].get('test_size', 0.2),
        finetune_val_size=config['data'].get('finetune_val_size', 0.1),
        random_state=config['data'].get('random_state', 42),
        stratify=True
    )
    
    # splits['test'] returns (signals, labels, rul) tuple
    test_signals_sim, test_labels_sim, _ = splits['test']
    
    print(f"Simulation test samples: {len(test_signals_sim)}")
    print(f"Simulation test labels shape: {test_labels_sim.shape}")
    print(f"Unique labels: {np.unique(test_labels_sim)}")
    
    # Create test dataset and loader for simulation
    test_dataset_sim = PHMSignalDataset(
        test_signals_sim,
        labels=test_labels_sim,
        rul=None,
        normalize=config['data']['normalize'],
        signal_window=config['data']['signal_window'],
        overlap=config['data']['overlap']
    )
    
    test_loader_sim = DataLoader(
        test_dataset_sim,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Evaluate on simulation test data
    metrics_sim = evaluate_diagnosis(model, diagnosis_head, test_loader_sim, device, "Simulation")
    
    print(f"\nSimulation Test Set Results:")
    print(f"  Accuracy: {metrics_sim['accuracy']:.4f} ({metrics_sim['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics_sim['precision']:.4f}")
    print(f"  Recall: {metrics_sim['recall']:.4f}")
    print(f"  F1-Score: {metrics_sim['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(metrics_sim['confusion_matrix']))
    
    # ========== EVALUATION 2: Real Robot Test Data ==========
    print("\n" + "="*60)
    print("EVALUATION 2: Real Robot Test Data")
    print("="*60)
    
    try:
        # Try to load real test data
        print(f"\nLoading real test data from {args.test_folder}...")
        test_signals_real, test_labels_real = load_robot_test_data(
            data_dir, folder_name=args.test_folder, use_individual_files=True
        )
        
        print(f"Real test samples: {len(test_signals_real)}")
        print(f"Real test labels shape: {test_labels_real.shape}")
        print(f"Unique labels: {np.unique(test_labels_real)}")
        
        # Create test dataset and loader for real data
        test_dataset_real = PHMSignalDataset(
            test_signals_real,
            labels=test_labels_real,
            rul=None,
            normalize=config['data']['normalize'],
            signal_window=config['data']['signal_window'],
            overlap=config['data']['overlap']
        )
        
        test_loader_real = DataLoader(
            test_dataset_real,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )
        
        # Evaluate on real test data
        metrics_real = evaluate_diagnosis(model, diagnosis_head, test_loader_real, device, "Real")
        
        print(f"\nReal Test Set Results:")
        print(f"  Accuracy: {metrics_real['accuracy']:.4f} ({metrics_real['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics_real['precision']:.4f}")
        print(f"  Recall: {metrics_real['recall']:.4f}")
        print(f"  F1-Score: {metrics_real['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(np.array(metrics_real['confusion_matrix']))
        
        # Save results with both evaluations
        results = {
            'checkpoint': args.checkpoint,
            'dataset': 'ROBOT',
            'task': 'diagnosis',
            'simulation_test': {
                'samples': len(test_signals_sim),
                'metrics': {k: v for k, v in metrics_sim.items() if k != 'confusion_matrix' and k != 'classification_report'}
            },
            'real_test': {
                'samples': len(test_signals_real),
                'metrics': {k: v for k, v in metrics_real.items() if k != 'confusion_matrix' and k != 'classification_report'}
            }
        }
        
    except FileNotFoundError as e:
        print(f"\nWarning: Real test data not found: {e}")
        print("Skipping real test data evaluation.")
        
        # Save results with only simulation evaluation
        results = {
            'checkpoint': args.checkpoint,
            'dataset': 'ROBOT',
            'task': 'diagnosis',
            'simulation_test': {
                'samples': len(test_signals_sim),
                'metrics': {k: v for k, v in metrics_sim.items() if k != 'confusion_matrix' and k != 'classification_report'}
            },
            'real_test': {
                'samples': 0,
                'metrics': None,
                'error': str(e)
            }
        }
    
    # Save results
    results_path = os.path.join(
        config['evaluation']['save_dir'],
        f'eval_robot_{os.path.basename(args.checkpoint).replace(".pt", "")}.json'
    )
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
