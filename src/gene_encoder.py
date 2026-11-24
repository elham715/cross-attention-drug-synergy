"""Gene pathway encoder for tumor transcriptomic expression profiles."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .config import CANCER_PATHWAYS, PATHWAY_NAMES, NUM_PATHWAYS

class GenePathwayEncoder(nn.Module):
    """Encodes high-dimensional RNA-seq expression profiles into structured pathway tokens.
    
    Rather than treating gene expression as an unstructured flat vector, this encoder
    organizes the 128 landmark genes into 8 canonical oncogenic hallmark pathways.
    Pathway crosstalk is modeled via an inter-pathway self-attention layer.
    """
    
    def __init__(
        self,
        num_pathways: int = NUM_PATHWAYS,
        genes_per_pathway: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.genes_per_pathway = genes_per_pathway
        self.d_model = d_model
        
        # Pathway identity embeddings (learnable categorical ID for each of the 8 pathways)
        self.pathway_embedding = nn.Embedding(num_pathways, d_model)
        
        # Dedicated linear projections from 16-gene expression slice to d_model
        self.pathway_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(genes_per_pathway, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_pathways)
        ])
        
        # Inter-pathway Self-Attention to model biological signaling crosstalk (e.g. RTK -> RAS -> MAPK)
        self.crosstalk_norm = nn.LayerNorm(d_model)
        self.crosstalk_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.crosstalk_dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            gene_expression: FloatTensor of shape [B, 128] containing normalized RNA-seq z-scores.
            
        Returns:
            pathway_tokens: FloatTensor of shape [B, num_pathways, d_model]
        """
        batch_size = gene_expression.shape[0]
        device = gene_expression.device
        
        pathway_tokens = []
        for p_idx in range(self.num_pathways):
            start_col = p_idx * self.genes_per_pathway
            end_col = start_col + self.genes_per_pathway
            gene_slice = gene_expression[:, start_col:end_col]
            
            # Project gene slice to d_model
            p_vec = self.pathway_projections[p_idx](gene_slice)  # [B, d_model]
            
            # Add pathway identity prior
            p_id = torch.tensor([p_idx], device=device).expand(batch_size)
            p_id_emb = self.pathway_embedding(p_id)  # [B, d_model]
            
            pathway_tokens.append(p_vec + p_id_emb)
            
        # Stack into [B, num_pathways, d_model]
        h_pathway = torch.stack(pathway_tokens, dim=1)
        
        # Model pathway crosstalk via Self-Attention
        norm_h = self.crosstalk_norm(h_pathway)
        crosstalk_out, _ = self.crosstalk_attn(norm_h, norm_h, norm_h)
        h_pathway = h_pathway + self.crosstalk_dropout(crosstalk_out)
        
        return self.final_norm(h_pathway)
