"""Bi-modal Cross-Attention Transformer for Drug-Tumor Sensitivity Prediction."""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from .drug_encoder import DrugTransformerEncoder
from .gene_encoder import GenePathwayEncoder
from .config import MODEL_CONFIG

class CrossAttentionDrugSensitivityModel(nn.Module):
    """Predicts log(IC50) anticancer sensitivity using bi-modal cross-attention.
    
    The architecture enables bi-directional query-key information flow:
      1. Drug atoms attend to cancer pathways (Which oncogene does this pharmacophore target?)
      2. Cancer pathways attend to drug atoms (Which chemical group inhibits this pathway?)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        cfg = config or MODEL_CONFIG
        
        self.d_model = cfg.get("d_model", 128)
        self.n_heads = cfg.get("cross_attn_heads", 4)
        
        # 1. Chemical Drug Encoder
        self.drug_encoder = DrugTransformerEncoder(
            vocab_size=cfg.get("smiles_vocab_size", 75),
            max_len=cfg.get("max_smiles_len", 128),
            d_model=self.d_model,
            n_heads=cfg.get("n_heads", 4),
            num_layers=cfg.get("num_drug_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.10)
        )
        
        # 2. Genomic Pathway Encoder
        self.gene_encoder = GenePathwayEncoder(
            num_pathways=cfg.get("num_pathways", 8),
            genes_per_pathway=cfg.get("genes_per_pathway", 16),
            d_model=self.d_model,
            n_heads=cfg.get("n_heads", 4),
            dropout=cfg.get("dropout", 0.10)
        )
        
        # 3. Bi-modal Cross-Attention Blocks
        # A: Drug queries attend to Pathway keys/values
        self.drug_cross_norm = nn.LayerNorm(self.d_model)
        self.drug_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=cfg.get("dropout", 0.10),
            batch_first=True
        )
        
        # B: Pathway queries attend to Drug keys/values
        self.pathway_cross_norm = nn.LayerNorm(self.d_model)
        self.pathway_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=cfg.get("dropout", 0.10),
            batch_first=True
        )
        
        # 4. Multimodal Fusion & Non-Linear Regressor Head
        fusion_dim = 4 * self.d_model  # [drug, pathway, |drug - pathway|, drug * pathway]
        hidden_dims = cfg.get("regressor_hidden_dims", [128, 64])
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(cfg.get("dropout", 0.10)),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(cfg.get("dropout", 0.10)),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_expression: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            token_ids: LongTensor [B, L]
            attention_mask: BoolTensor [B, L] (True for valid tokens)
            gene_expression: FloatTensor [B, 128]
            
        Returns:
            predicted_ic50: FloatTensor [B, 1] representing natural log(IC50) in uM
        """
        # Step 1: Contextual encoding
        h_drug = self.drug_encoder(token_ids, attention_mask)      # [B, L, d]
        h_pathway = self.gene_encoder(gene_expression)             # [B, K, d]
        
        # Key padding mask for drug tokens (True = ignore)
        drug_key_padding = ~attention_mask
        
        # Step 2: Cross-Attention
        # Drug -> Pathway (Drug tokens query the cancer pathways)
        norm_drug = self.drug_cross_norm(h_drug)
        drug_context, _ = self.drug_cross_attn(
            query=norm_drug,
            key=h_pathway,
            value=h_pathway
        )
        h_drug_fused = h_drug + drug_context                       # [B, L, d]
        
        # Pathway -> Drug (Pathway tokens query the drug atoms)
        norm_pathway = self.pathway_cross_norm(h_pathway)
        pathway_context, _ = self.pathway_cross_attn(
            query=norm_pathway,
            key=h_drug,
            value=h_drug,
            key_padding_mask=drug_key_padding
        )
        h_pathway_fused = h_pathway + pathway_context              # [B, K, d]
        
        # Step 3: Masked Multimodal Pooling
        # Masked average pooling over drug tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()       # [B, L, 1]
        drug_pooled = (h_drug_fused * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0) # [B, d]
        
        # Average pooling over 8 pathway tokens
        pathway_pooled = h_pathway_fused.mean(dim=1)               # [B, d]
        
        # Step 4: Bilinear Multimodal Interaction Vector
        diff = torch.abs(drug_pooled - pathway_pooled)             # [B, d]
        prod = drug_pooled * pathway_pooled                        # [B, d]
        fusion_vector = torch.cat([drug_pooled, pathway_pooled, diff, prod], dim=-1) # [B, 4d]
        
        # Step 5: Regression to ln(IC50)
        return self.regressor(fusion_vector)

    def extract_attention_maps(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_expression: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-head cross-attention probability maps for interpretability.
        
        Returns:
            Dict containing:
              - 'drug_to_pathway_attn': [B, n_heads, L, K]
              - 'pathway_to_drug_attn': [B, n_heads, K, L]
              - 'predicted_ic50': [B, 1]
        """
        self.eval()
        with torch.no_grad():
            h_drug = self.drug_encoder(token_ids, attention_mask)
            h_pathway = self.gene_encoder(gene_expression)
            drug_key_padding = ~attention_mask
            
            # Drug -> Pathway
            norm_drug = self.drug_cross_norm(h_drug)
            _, drug_to_pathway_weights = self.drug_cross_attn(
                query=norm_drug,
                key=h_pathway,
                value=h_pathway,
                need_weights=True,
                average_attn_weights=False  # Keep all heads: [B, n_heads, L, K]
            )
            
            # Pathway -> Drug
            norm_pathway = self.pathway_cross_norm(h_pathway)
            _, pathway_to_drug_weights = self.pathway_cross_attn(
                query=norm_pathway,
                key=h_drug,
                value=h_drug,
                key_padding_mask=drug_key_padding,
                need_weights=True,
                average_attn_weights=False  # Keep all heads: [B, n_heads, K, L]
            )
            
            # Prediction
            pred = self.forward(token_ids, attention_mask, gene_expression)
            
            return {
                "drug_to_pathway_attn": drug_to_pathway_weights,
                "pathway_to_drug_attn": pathway_to_drug_weights,
                "predicted_ic50": pred
            }
