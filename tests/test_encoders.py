"""Unit tests for Drug and Gene encoders."""

import torch
import pytest
from src.drug_encoder import DrugTransformerEncoder
from src.gene_encoder import GenePathwayEncoder
from src.config import NUM_GENES, NUM_PATHWAYS

def test_drug_encoder_forward():
    batch_size = 4
    seq_len = 32
    d_model = 64
    
    encoder = DrugTransformerEncoder(
        vocab_size=75,
        max_len=64,
        d_model=d_model,
        n_heads=2,
        num_layers=2,
        dim_feedforward=128
    )
    
    token_ids = torch.randint(0, 70, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    # Mask out last 5 tokens as padding
    mask[:, -5:] = False
    
    out = encoder(token_ids, mask)
    assert out.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(out).any()

def test_gene_pathway_encoder_forward():
    batch_size = 4
    d_model = 64
    
    encoder = GenePathwayEncoder(
        num_pathways=NUM_PATHWAYS,
        genes_per_pathway=16,
        d_model=d_model,
        n_heads=2
    )
    
    gene_expr = torch.randn(batch_size, NUM_GENES)
    pathway_tokens = encoder(gene_expr)
    
    assert pathway_tokens.shape == (batch_size, NUM_PATHWAYS, d_model)
    assert not torch.isnan(pathway_tokens).any()
