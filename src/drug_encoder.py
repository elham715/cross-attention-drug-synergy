"""Transformer encoder for small-molecule chemical structures (SMILES)."""

import math
import torch
import torch.nn as nn
from typing import Optional

class DrugTransformerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block for chemical sequence representation."""
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        # key_padding_mask: True for positions to be IGNORED (i.e. ~mask)
        attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout1(attn_out)
        
        # Pre-LN Feedforward
        norm_x2 = self.norm2(x)
        ff_out = self.linear2(self.activation(self.linear1(norm_x2)))
        x = x + self.dropout2(ff_out)
        return x


class DrugTransformerEncoder(nn.Module):
    """Multi-layer Transformer Encoder processing tokenized SMILES strings."""
    
    def __init__(
        self,
        vocab_size: int = 75,
        max_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DrugTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            token_ids: LongTensor of shape [B, L]
            attention_mask: BoolTensor of shape [B, L] (True for valid tokens, False for pad)
            
        Returns:
            h_drug: Contextual atom representations of shape [B, L, d_model]
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # In PyTorch nn.MultiheadAttention, key_padding_mask requires True for PADDED tokens
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            
        return self.final_norm(x)
