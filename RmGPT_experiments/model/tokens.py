"""
Token Embedding Modules for RmGPT

Implements the four types of tokens:
1. Signal Tokens: Encode raw sensor signals
2. Prompt Tokens: Task-specific adaptation tokens
3. Time-Frequency Task Tokens: Health status representation from time-frequency domain
4. Fault Tokens: Fault prototypes for different equipment types
"""
import torch
import torch.nn as nn
import numpy as np
from scipy import signal as scipy_signal
import pywt


class SignalTokenizer(nn.Module):
    """
    Signal Token Embedding Layer
    
    Converts raw multivariate time-series signals into token embeddings.
    Uses learnable patch embeddings similar to Vision Transformer.
    """
    def __init__(self, 
                 signal_dim: int,
                 patch_length: int,
                 embed_dim: int,
                 stride: int = None,
                 overlap: int = 0,
                 input_channels: int = None):
        """
        Args:
            signal_dim: Number of sensor channels (features) expected by patch_embed
            patch_length: Length of each signal patch (temporal window) - P in paper
            embed_dim: Embedding dimension
            stride: Stride between patches (S in paper). If None, uses patch_length - overlap
            overlap: Overlap between patches (deprecated, use stride instead)
            input_channels: Actual number of input channels. If different from signal_dim,
                          a projection layer will be added to map input_channels -> signal_dim
        """
        super().__init__()
        self.signal_dim = signal_dim
        self.patch_length = patch_length
        self.embed_dim = embed_dim
        self.overlap = overlap
        # Paper uses explicit stride (S) - if provided, use it; otherwise compute from overlap
        if stride is not None:
            self.stride = stride
        else:
            self.stride = patch_length - overlap
        
        # Note: Channel projection is now handled in RmGPT model
        # This tokenizer expects signals with signal_dim channels
        
        # Learnable patch embedding
        self.patch_embed = nn.Linear(signal_dim * patch_length, embed_dim)
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # Max 1000 patches
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Args:
            x: Input signals [batch_size, seq_len, signal_dim]
            Note: Signals should already be projected to signal_dim channels by RmGPT model
            
        Returns:
            token_embeddings: [batch_size, num_patches, embed_dim]
            num_patches: Number of patches created
        """
        batch_size, seq_len, signal_dim = x.shape
        
        # Verify signal_dim matches expected
        if signal_dim != self.signal_dim:
            raise ValueError(
                f"Input signal_dim ({signal_dim}) does not match expected signal_dim ({self.signal_dim}). "
                f"Signals should be projected to {self.signal_dim} channels before passing to this tokenizer."
            )
        
        # Create patches
        patches = []
        patch_idx = 0
        
        while patch_idx * self.stride + self.patch_length <= seq_len:
            start_idx = patch_idx * self.stride
            end_idx = start_idx + self.patch_length
            patch = x[:, start_idx:end_idx, :]  # [batch, patch_len, signal_dim]
            patch = patch.reshape(batch_size, -1)  # [batch, patch_len * signal_dim]
            patches.append(patch)
            patch_idx += 1
        
        if len(patches) == 0:
            # Handle case where signal is shorter than patch_length
            pad_size = self.patch_length - seq_len
            padded = torch.cat([x, x[:, -1:, :].repeat(1, pad_size, 1)], dim=1)
            patches = [padded.reshape(batch_size, -1)]
            patch_idx = 1
        
        # Stack patches: [batch, num_patches, patch_len * signal_dim]
        patches = torch.stack(patches, dim=1)
        num_patches = patches.shape[1]
        
        # Project to embedding dimension
        token_embeddings = self.patch_embed(patches)  # [batch, num_patches, embed_dim]
        
        # Add positional embedding
        if num_patches <= self.pos_embed.shape[1]:
            token_embeddings = token_embeddings + self.pos_embed[:, :num_patches, :]
        else:
            # Interpolate if needed
            pos_embed = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=num_patches,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            token_embeddings = token_embeddings + pos_embed
        
        token_embeddings = self.layer_norm(token_embeddings)
        
        return token_embeddings, num_patches


class PromptTokenizer(nn.Module):
    """
    Prompt Token Embedding Layer
    
    Learnable task-specific tokens for adaptive task adaptation.
    These tokens guide the model to perform specific tasks (diagnosis/prognosis).
    """
    def __init__(self, num_prompts: int, embed_dim: int):
        """
        Args:
            num_prompts: Number of prompt tokens
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        
        # Learnable prompt tokens
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, embed_dim))
        nn.init.normal_(self.prompts, std=0.02)
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size
            
        Returns:
            prompt_embeddings: [batch_size, num_prompts, embed_dim]
        """
        return self.prompts.expand(batch_size, -1, -1)


class TimeFreqTaskTokenizer(nn.Module):
    """
    Time-Frequency Task Token Embedding Layer
    
    Extracts time-frequency domain features (FFT, Wavelet) and converts
    them to tokens representing health status semantics.
    """
    def __init__(self, 
                 signal_dim: int,
                 embed_dim: int,
                 n_fft: int = 256,
                 wavelet: str = 'db4',
                 wavelet_levels: int = 4):
        """
        Args:
            signal_dim: Number of sensor channels
            embed_dim: Embedding dimension
            n_fft: FFT window size for frequency domain analysis
            wavelet: Wavelet type for time-frequency analysis
            wavelet_levels: Number of wavelet decomposition levels
        """
        super().__init__()
        self.signal_dim = signal_dim
        self.embed_dim = embed_dim
        self.n_fft = n_fft
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        
        # FFT feature dimension: concatenated across channels
        # Each channel: n_fft // 2 + 1 (positive frequencies)
        # Total: signal_dim * (n_fft // 2 + 1)
        fft_dim_per_channel = n_fft // 2 + 1
        fft_dim = signal_dim * fft_dim_per_channel
        
        # Wavelet feature dimension: signal_dim * wavelet_levels
        # (Each channel gets wavelet_levels features)
        wavelet_dim = signal_dim * wavelet_levels
        
        # Total time-freq feature dimension
        tf_dim = fft_dim + wavelet_dim
        
        # Projection to embedding dimension
        self.tf_proj = nn.Linear(tf_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def _extract_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract FFT features from signals"""
        # x: [batch, seq_len, signal_dim]
        batch_size, seq_len, signal_dim = x.shape
        
        # Apply FFT to each channel
        fft_features = []
        for ch in range(signal_dim):
            signal = x[:, :, ch]  # [batch, seq_len]
            
            # Compute FFT
            fft_vals = torch.fft.rfft(signal, n=self.n_fft, dim=1)
            
            # Take magnitude: [batch, n_fft//2 + 1]
            fft_mag = torch.abs(fft_vals)
            fft_features.append(fft_mag)
        
        # Concatenate across channels: [batch, signal_dim * (n_fft//2 + 1)]
        fft_features = torch.cat(fft_features, dim=1)
        
        return fft_features
    
    def _extract_wavelet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Wavelet features from signals"""
        # x: [batch, seq_len, signal_dim]
        batch_size, seq_len, signal_dim = x.shape
        
        # Convert to numpy for wavelet transform
        x_np = x.detach().cpu().numpy()
        wavelet_features_list = []
        
        for b in range(batch_size):
            batch_features = []
            for ch in range(signal_dim):
                signal = x_np[b, :, ch]
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.wavelet_levels)
                
                # Extract features from each level
                level_features = []
                for coeff in coeffs:
                    # Use mean and std as features
                    level_features.extend([np.mean(coeff), np.std(coeff)])
                
                batch_features.extend(level_features[:self.wavelet_levels])
            
            wavelet_features_list.append(batch_features)
        
        # Convert back to tensor
        wavelet_features = torch.tensor(wavelet_features_list, dtype=x.dtype, device=x.device)
        
        return wavelet_features  # [batch, signal_dim * wavelet_levels]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signals [batch_size, seq_len, signal_dim]
            
        Returns:
            tf_tokens: [batch_size, 1, embed_dim] (single time-freq token)
        """
        # Extract FFT features
        fft_features = self._extract_fft_features(x)  # [batch, fft_dim]
        
        # Extract Wavelet features
        try:
            wavelet_features = self._extract_wavelet_features(x)  # [batch, wavelet_dim]
        except:
            # Fallback if wavelet fails
            batch_size = x.shape[0]
            wavelet_features = torch.zeros(batch_size, self.signal_dim * self.wavelet_levels,
                                         dtype=x.dtype, device=x.device)
        
        # Concatenate features
        tf_features = torch.cat([fft_features, wavelet_features], dim=1)  # [batch, tf_dim]
        
        # Project to embedding dimension
        tf_tokens = self.tf_proj(tf_features)  # [batch, embed_dim]
        tf_tokens = tf_tokens.unsqueeze(1)  # [batch, 1, embed_dim]
        
        tf_tokens = self.layer_norm(tf_tokens)
        
        return tf_tokens


class FaultTokenizer(nn.Module):
    """
    Fault Token Embedding Layer
    
    Learnable fault prototypes representing different fault types across
    different equipment. Used for comparison-based diagnosis.
    """
    def __init__(self, num_faults: int, embed_dim: int):
        """
        Args:
            num_faults: Number of fault types (including normal state)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.num_faults = num_faults
        self.embed_dim = embed_dim
        
        # Learnable fault token embeddings
        self.fault_tokens = nn.Parameter(torch.randn(1, num_faults, embed_dim))
        nn.init.normal_(self.fault_tokens, std=0.02)
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size
            
        Returns:
            fault_embeddings: [batch_size, num_faults, embed_dim]
        """
        return self.fault_tokens.expand(batch_size, -1, -1)
    
    def get_fault_token(self, fault_id: int) -> torch.Tensor:
        """Get a specific fault token"""
        return self.fault_tokens[:, fault_id:fault_id+1, :]  # [1, 1, embed_dim]
