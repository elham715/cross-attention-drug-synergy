"""Unit tests for Bi-modal Cross-Attention model."""

import torch
import pytest
from src.cross_attention_transformer import CrossAttentionDrugSensitivityModel
from src.config import MODEL_CONFIG, NUM_GENES, NUM_PATHWAYS

def test_cross_attention_model_forward():
    batch_size = 4
    seq_len = 48
    d_model = 64
    
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = d_model
    cfg["max_smiles_len"] = 64
    cfg["n_heads"] = 2
    cfg["cross_attn_heads"] = 2
    cfg["num_drug_layers"] = 2
    cfg["dim_feedforward"] = 128
    
    model = CrossAttentionDrugSensitivityModel(cfg)
    
    token_ids = torch.randint(0, 70, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    mask[:, -8:] = False
    gene_expr = torch.randn(batch_size, NUM_GENES)
    
    pred_ic50 = model(token_ids, mask, gene_expr)
    
    assert pred_ic50.shape == (batch_size, 1)
    assert not torch.isnan(pred_ic50).any()

def test_extract_attention_maps():
    batch_size = 2
    seq_len = 32
    d_model = 64
    n_heads = 2
    
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = d_model
    cfg["max_smiles_len"] = 64
    cfg["n_heads"] = n_heads
    cfg["cross_attn_heads"] = n_heads
    cfg["num_drug_layers"] = 2
    
    model = CrossAttentionDrugSensitivityModel(cfg)
    
    token_ids = torch.randint(0, 70, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    gene_expr = torch.randn(batch_size, NUM_GENES)
    
    maps = model.extract_attention_maps(token_ids, mask, gene_expr)
    
    assert "drug_to_pathway_attn" in maps
    assert "pathway_to_drug_attn" in maps
    assert "predicted_ic50" in maps
    
    d2p = maps["drug_to_pathway_attn"]
    p2d = maps["pathway_to_drug_attn"]
    
    # Shape: [B, n_heads, seq_len, num_pathways]
    assert d2p.shape == (batch_size, n_heads, seq_len, NUM_PATHWAYS)
    # Shape: [B, n_heads, num_pathways, seq_len]
    assert p2d.shape == (batch_size, n_heads, NUM_PATHWAYS, seq_len)
    
    # Probabilities must sum to ~1 along key dimension
    assert torch.allclose(d2p.sum(dim=-1), torch.ones(batch_size, n_heads, seq_len), atol=1e-4)
