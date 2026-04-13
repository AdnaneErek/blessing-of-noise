"""
RmGPT: Main Model Architecture

Unified foundation model for fault diagnosis and prognosis in rotating machinery.
Combines Signal Tokens, Prompt Tokens, Time-Frequency Task Tokens, and Fault Tokens
within a transformer architecture.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .tokens import SignalTokenizer, PromptTokenizer, TimeFreqTaskTokenizer, FaultTokenizer
from .transformer import TransformerEncoder


class RmGPT(nn.Module):
    """
    RmGPT: Foundation Model for Rotating Machinery PHM
    
    Architecture:
    1. Signal Tokens: Raw signal patches
    2. Prompt Tokens: Task-specific adaptation
    3. Time-Frequency Task Tokens: Health status representation
    4. Transformer Encoder: Unified feature extraction
    5. Task Heads: Diagnosis (classification) and Prognosis (regression)
    """
    
    def __init__(self,
                 # Signal configuration
                 signal_dim: int = 24,  # Number of sensor channels
                 patch_length: int = 256,  # Temporal patch size (P) - PAPER: 256
                 tokenizer_stride: int = 256,  # Tokenizer stride (S) - PAPER: 256
                 
                 # Token configuration
                 embed_dim: int = 512,  # Hidden Size (d) - PAPER: 512
                 num_prompts: int = 10,  # Prompt Token Length (lp) - PAPER: 10
                 num_faults: int = 1,  # Fault Token Length (lt) - PAPER: 1 (will be expanded per task)
                 
                 # Transformer configuration
                 num_layers: int = 4,  # Transformer Layers - PAPER: 4
                 num_heads: int = 8,  # Number of attention heads
                 ff_dim: int = 2048,  # Feed-forward dimension
                 dropout: float = 0.1,
                 
                 # Time-frequency configuration
                 n_fft: int = 256,
                 wavelet: str = 'db4',
                 wavelet_levels: int = 4,
                 
                 # Channel projection (for datasets with different channel counts)
                 input_channels: int = None):
        
        super().__init__()
        
        self.signal_dim = signal_dim
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        self.num_faults = num_faults
        self.input_channels = input_channels if input_channels is not None else signal_dim
        
        # Channel projection layer (if input_channels != signal_dim)
        # Use a deeper projection to learn better mapping: 9 -> 18 -> 9 -> 2
        # This preserves more information than direct 9 -> 2 projection
        if self.input_channels != signal_dim:
            # Multi-layer projection for better feature learning
            if self.input_channels > signal_dim:
                # Expanding then compressing helps learn better representations
                # Architecture: input -> 2*input -> input -> output
                expand_dim = self.input_channels * 2  # Expand first
                self.channel_proj = nn.Sequential(
                    nn.Linear(self.input_channels, expand_dim),
                    nn.LayerNorm(expand_dim),
                    nn.GELU(),
                    nn.Linear(expand_dim, self.input_channels),  # Back to input size
                    nn.LayerNorm(self.input_channels),
                    nn.GELU(),
                    nn.Linear(self.input_channels, signal_dim)  # Final compression
                )
            else:
                # If input < output, just use simple projection
                self.channel_proj = nn.Linear(self.input_channels, signal_dim)
        else:
            self.channel_proj = None
        
        # Tokenizers
        self.signal_tokenizer = SignalTokenizer(
            signal_dim=signal_dim,
            patch_length=patch_length,
            embed_dim=embed_dim,
            stride=tokenizer_stride,
            input_channels=None  # Will receive already-projected signals
        )
        
        self.prompt_tokenizer = PromptTokenizer(
            num_prompts=num_prompts,
            embed_dim=embed_dim
        )
        
        self.tf_tokenizer = TimeFreqTaskTokenizer(
            signal_dim=signal_dim,
            embed_dim=embed_dim,
            n_fft=n_fft,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels
        )
        
        self.fault_tokenizer = FaultTokenizer(
            num_faults=num_faults,
            embed_dim=embed_dim
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                signals: torch.Tensor,
                task_type: str = 'diagnosis',  # 'diagnosis' or 'prognosis'
                return_tokens: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            signals: Input signals [batch_size, seq_len, signal_dim]
            task_type: Task type ('diagnosis' or 'prognosis')
            return_tokens: Whether to return intermediate token representations
            
        Returns:
            Dictionary containing:
            - 'features': Combined token features [batch_size, total_seq_len, embed_dim]
            - 'signal_tokens': Signal token embeddings
            - 'tf_tokens': Time-frequency token embeddings
            - 'prompt_tokens': Prompt token embeddings
            - 'fault_tokens': Fault token embeddings (if task_type == 'diagnosis')
        """
        batch_size = signals.shape[0]
        
        # Project channels if needed (before passing to tokenizers)
        if self.channel_proj is not None:
            signals = self.channel_proj(signals)  # [batch, seq_len, input_channels] -> [batch, seq_len, signal_dim]
        
        # 1. Create Signal Tokens
        signal_tokens, num_patches = self.signal_tokenizer(signals)
        # signal_tokens: [batch, num_patches, embed_dim]
        
        # 2. Create Time-Frequency Task Tokens
        tf_tokens = self.tf_tokenizer(signals)
        # tf_tokens: [batch, 1, embed_dim]
        
        # 3. Create Prompt Tokens
        prompt_tokens = self.prompt_tokenizer(batch_size)
        # prompt_tokens: [batch, num_prompts, embed_dim]
        
        # 4. Concatenate tokens: [Prompt] [Time-Freq] [Signal Patches]
        token_sequence = torch.cat([
            prompt_tokens,  # [batch, num_prompts, embed_dim]
            tf_tokens,      # [batch, 1, embed_dim]
            signal_tokens   # [batch, num_patches, embed_dim]
        ], dim=1)  # [batch, num_prompts + 1 + num_patches, embed_dim]
        
        token_sequence = self.dropout(token_sequence)
        
        # 5. Apply Transformer Encoder
        encoded_features = self.transformer(token_sequence)
        # encoded_features: [batch, total_seq_len, embed_dim]
        
        # Prepare output
        output = {
            'features': encoded_features,
            'num_patches': num_patches
        }
        
        if return_tokens:
            output['signal_tokens'] = signal_tokens
            output['tf_tokens'] = tf_tokens
            output['prompt_tokens'] = prompt_tokens
        
        # 6. Add Fault Tokens for diagnosis task (for comparison-based diagnosis)
        if task_type == 'diagnosis':
            fault_tokens = self.fault_tokenizer(batch_size)
            # Use fault tokens for comparison
            # Extract prompt/tf token features for comparison
            prompt_tf_features = encoded_features[:, :self.num_prompts + 1, :]
            output['fault_tokens'] = fault_tokens
            output['prompt_tf_features'] = prompt_tf_features
        
        return output
    
    def get_sequence_length(self, signal_seq_len: int) -> int:
        """Calculate total sequence length after tokenization"""
        num_patches = (signal_seq_len - self.signal_tokenizer.patch_length) // self.signal_tokenizer.stride + 1
        total_length = self.num_prompts + 1 + num_patches  # prompts + tf_token + signal_patches
        return total_length


class DiagnosisHead(nn.Module):
    """Classification head for fault diagnosis"""
    
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1, improved: bool = False):
        super().__init__()
        if improved:
            # Improved head: deeper and wider for better capacity
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),  # Wider first layer
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim // 2),  # Intermediate layer
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes)
            )
        else:
            # Original head: simple 2-layer
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes)
            )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Token features [batch, seq_len, embed_dim]
            
        Returns:
            logits: [batch, num_classes]
        """
        # Use the Time-Frequency task token (after prompt tokens) for prediction
        # This represents the health status semantics
        tf_token_feat = features[:, 0, :]  # First token is time-freq token after prompts
        
        logits = self.head(tf_token_feat)
        return logits


class PrognosisHead(nn.Module):
    """Regression head for RUL prediction"""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Token features [batch, seq_len, embed_dim]
            
        Returns:
            rul_pred: [batch, 1]
        """
        # Use the Time-Frequency task token for RUL prediction
        tf_token_feat = features[:, 0, :]
        
        rul_pred = self.head(tf_token_feat)
        return rul_pred
