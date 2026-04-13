# RmGPT: A Foundation Model for Rotating Machinery PHM

This is a PyTorch implementation of **RmGPT: A Foundation Model with Generative Pre-trained Transformer for Fault Diagnosis and Prognosis in Rotating Machinery** (arXiv:2409.17604v2).

## Overview

RmGPT is a unified foundation model for both fault diagnosis and prognosis tasks in rotating machinery. It uses a generative token-based framework with four types of tokens:

1. **Signal Tokens**: Encode raw sensor signals into patch-based embeddings
2. **Prompt Tokens**: Task-specific learnable tokens for adaptive task adaptation
3. **Time-Frequency Task Tokens**: Extract health status semantics from time-frequency domain (FFT + Wavelet)
4. **Fault Tokens**: Learnable fault prototypes for comparison-based diagnosis

## Architecture

The model consists of:
- **Token Embedding Layers**: Four tokenizers for different token types
- **Transformer Encoder**: Multi-layer transformer with self-attention
- **Task Heads**: 
  - Diagnosis Head (classification)
  - Prognosis Head (regression for RUL prediction)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the `phmd` library is accessible (already in the repository).

## Project Structure

```
RmGPT/
├── model/              # Model architecture
│   ├── tokens.py      # Token embedding layers
│   ├── transformer.py # Transformer encoder
│   └── rmgpt.py       # Main RmGPT model
├── train/              # Training utilities
│   └── trainer.py     # Training loop and trainer class
├── data/               # Data preprocessing
│   └── dataset.py     # Dataset classes and PHMD integration
├── configs/            # Configuration files
│   └── default_config.yaml
├── train_rmgpt.py      # Main training script
└── requirements.txt    # Dependencies
```

## Usage

### Training

#### Pretraining (Self-supervised)
```bash
python train_rmgpt.py \
    --config configs/default_config.yaml \
    --task pretrain
```

#### Fine-tuning for Diagnosis
```bash
python train_rmgpt.py \
    --config configs/default_config.yaml \
    --task diagnosis
```

#### Fine-tuning for Prognosis
```bash
python train_rmgpt.py \
    --config configs/default_config.yaml \
    --task prognosis
```

### Configuration

Edit `configs/default_config.yaml` to customize:
- Model architecture (embedding dimension, number of layers, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Dataset settings (dataset name, task, fold, etc.)

### Using PHMD Datasets

The implementation integrates with the PHMD library for dataset loading. Supported datasets include:
- CWRU (Bearing fault diagnosis)
- CMAPSS (Aircraft engine RUL prediction)
- NCMAPSS
- And many more...

Example dataset configuration:
```yaml
data:
  dataset_name: "CWRU"  # PHM dataset name
  task_name: "Diagnosis"  # Task name
  fold: 0  # Cross-validation fold
```

## Model Details

### Token Types

1. **Signal Tokens**: 
   - Patch-based embedding of raw signals
   - Configurable patch length and overlap
   - Learnable positional embeddings

2. **Prompt Tokens**:
   - Task-adaptive learnable tokens
   - Enable efficient task-specific adaptation

3. **Time-Frequency Task Tokens**:
   - FFT for frequency domain analysis
   - Wavelet transform for time-frequency analysis
   - Encodes health status semantics

4. **Fault Tokens**:
   - Learnable fault prototypes
   - Used for comparison-based diagnosis

### Training Strategy

1. **Self-supervised Pretraining**:
   - Next-token prediction objective
   - Learns generalizable signal representations

2. **Supervised Fine-tuning**:
   - Task-specific heads for diagnosis/prognosis
   - Prompt learning for task adaptation

## Implementation Notes

This implementation follows the paper's architecture as closely as possible. Key features:

- Token-based unified framework
- Time-frequency domain processing (FFT + Wavelet)
- Self-supervised pretraining with next-token prediction
- Prompt learning for task adaptation
- Support for both diagnosis and prognosis tasks

## Citation

If you use this code, please cite the original paper:

```
@article{wang2024rmgpt,
  title={RmGPT: A Foundation Model with Generative Pre-trained Transformer for Fault Diagnosis and Prognosis in Rotating Machinery},
  author={Wang, Yilin and Yu, Yifei and Sun, Kong and Lei, Peixuan and Zhang, Yuxuan and Zio, Enrico and Xia, Aiguo and Li, Yuanxiang},
  journal={arXiv preprint arXiv:2409.17604},
  year={2024}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper and repository for licensing details.

## Contact

For questions or issues, please refer to the original repository: https://github.com/Pandalin98/RmGPT
