#!/usr/bin/env python3
"""
Evaluate pretrained model on all datasets and collect accuracy metrics

Creates a summary file with accuracy for each dataset's test set.
"""
import argparse
import yaml
import json
import os
from pathlib import Path
import subprocess
import sys


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_dataset(dataset_name: str, config_path: str, checkpoint: str, results_dir: str) -> dict:
    """
    Evaluate a single dataset and return results
    
    Returns:
        Dictionary with dataset name and accuracy metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Run evaluation with linear probe
    cmd = [
        sys.executable,
        "evaluate_pretrained.py",
        "--config", config_path,
        "--checkpoint", checkpoint,
        "--dataset", dataset_name,
        "--linear-probe"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {dataset_name}:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return {
            "dataset": dataset_name,
            "status": "error",
            "error": str(e)
        }
    
    # Load results from JSON file
    results_file = os.path.join(results_dir, f"eval_pretrained_{dataset_name}.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract accuracy metrics
        summary = {
            "dataset": dataset_name,
            "status": "success",
            "test_samples": results.get("test_samples", 0)
        }
        
        # Add pretrain metrics
        if "pretrain_metrics" in results:
            summary["pretrain_loss"] = results["pretrain_metrics"].get("pretrain_loss")
        
        # Add linear probe metrics (accuracy)
        if "linear_probe_metrics" in results:
            lp_metrics = results["linear_probe_metrics"]
            summary["val_accuracy"] = lp_metrics.get("val_accuracy")
            summary["test_accuracy"] = lp_metrics.get("test_accuracy")
            summary["test_precision"] = lp_metrics.get("test_precision")
            summary["test_recall"] = lp_metrics.get("test_recall")
            summary["test_f1"] = lp_metrics.get("test_f1")
        else:
            summary["status"] = "warning"
            summary["warning"] = "No linear_probe_metrics found (did --linear-probe run?)"
        
        return summary
    else:
        return {
            "dataset": dataset_name,
            "status": "error",
            "error": f"Results file not found: {results_file}"
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained model on all datasets')
    parser.add_argument('--config', type=str, default='configs/paper_exact_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model_pretrain.pt',
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--output', type=str, default='results/pretrained_accuracy_summary.json',
                       help='Output summary file path')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='List of datasets to evaluate (default: all pretrain datasets from config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine datasets to evaluate
    if args.datasets:
        dataset_names = args.datasets
    else:
        # Use pretrain datasets from config
        dataset_names = config['data'].get('pretrain_datasets', [
            'CWRU', 'JNUB', 'KAUG17', 'HSG18', 'XJTU-SY'
        ])
    
    print(f"Evaluating pretrained model on {len(dataset_names)} datasets:")
    for ds in dataset_names:
        print(f"  - {ds}")
    
    # Get results directory from config
    results_dir = config['evaluation'].get('save_dir', 'results/')
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate each dataset
    all_results = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "datasets": []
    }
    
    for dataset_name in dataset_names:
        result = evaluate_dataset(dataset_name, args.config, args.checkpoint, results_dir)
        all_results["datasets"].append(result)
    
    # Save summary
    summary_path = args.output
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"\nResults saved to: {summary_path}")
    print(f"\nTest Set Accuracy by Dataset:")
    print(f"{'Dataset':<15} {'Test Accuracy':<15} {'Test F1':<15} {'Status'}")
    print("-" * 60)
    
    for result in all_results["datasets"]:
        dataset = result["dataset"]
        status = result["status"]
        if status == "success" and "test_accuracy" in result:
            acc = result.get("test_accuracy", 0.0)
            f1 = result.get("test_f1", 0.0)
            print(f"{dataset:<15} {acc:<15.4f} {f1:<15.4f} {status}")
        else:
            error_msg = result.get("error", "unknown error")
            print(f"{dataset:<15} {'N/A':<15} {'N/A':<15} {status} ({error_msg})")
    
    # Also create a simple CSV-like summary
    csv_path = summary_path.replace('.json', '.txt')
    with open(csv_path, 'w') as f:
        f.write("Dataset,Test_Accuracy,Test_F1,Test_Precision,Test_Recall,Val_Accuracy,Pretrain_Loss,Test_Samples\n")
        for result in all_results["datasets"]:
            if result["status"] == "success":
                dataset = result["dataset"]
                test_acc = result.get("test_accuracy", "N/A")
                test_f1 = result.get("test_f1", "N/A")
                test_prec = result.get("test_precision", "N/A")
                test_rec = result.get("test_recall", "N/A")
                val_acc = result.get("val_accuracy", "N/A")
                pretrain_loss = result.get("pretrain_loss", "N/A")
                test_samples = result.get("test_samples", "N/A")
                
                f.write(f"{dataset},{test_acc},{test_f1},{test_prec},{test_rec},{val_acc},{pretrain_loss},{test_samples}\n")
    
    print(f"\nCSV summary saved to: {csv_path}")
    print(f"\nComplete! Evaluated {len([r for r in all_results['datasets'] if r['status'] == 'success'])}/{len(dataset_names)} datasets successfully.")


if __name__ == '__main__':
    main()
