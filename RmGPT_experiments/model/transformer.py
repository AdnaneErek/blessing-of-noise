"""
Transformer Encoder Architecture for RmGPT

Implements the core transformer encoder blocks with multi-head attention,
feed-forward networks, and layer normalization.
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention Mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input [batch_size, seq_len, embed_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * embed_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [batch, num_heads, seq_len, seq_len]
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)  # [batch, seq_len, embed_dim]
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x), mask)
        
        # Feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder Stack"""
    
    def __init__(self,
                 num_layers: int,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
