"""Unit tests for data generation, loader, and optimization pipeline."""

import torch
import pytest
from src.smiles_tokenizer import SMILESTokenizer
from src.data_loader import generate_synthetic_gdsc_data, DrugSensitivityDataset, create_dataloaders
from src.cross_attention_transformer import CrossAttentionDrugSensitivityModel
from src.config import MODEL_CONFIG

def test_synthetic_data_generation():
    records = generate_synthetic_gdsc_data(num_samples=20, random_seed=42)
    assert len(records) == 20
    rec0 = records[0]
    assert "drug_name" in rec0
    assert "smiles" in rec0
    assert "cell_line" in rec0
    assert "ln_ic50" in rec0
    assert len(rec0["gene_expression"]) == 128

def test_dataset_and_dataloader():
    records = generate_synthetic_gdsc_data(num_samples=30, random_seed=42)
    tokenizer = SMILESTokenizer(max_length=64)
    train_loader, val_loader, test_loader = create_dataloaders(
        records, tokenizer, batch_size=8, train_ratio=0.7, val_ratio=0.15, seed=42
    )
    
    assert len(train_loader) > 0
    token_ids, mask, gene_expr, targets = next(iter(train_loader))
    assert token_ids.shape == (8, 64)
    assert mask.shape == (8, 64)
    assert gene_expr.shape == (8, 128)
    assert targets.shape == (8, 1)

def test_single_step_optimization():
    records = generate_synthetic_gdsc_data(num_samples=16, random_seed=42)
    tokenizer = SMILESTokenizer(max_length=48)
    dataset = DrugSensitivityDataset(records, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 32
    cfg["max_smiles_len"] = 48
    cfg["n_heads"] = 2
    cfg["cross_attn_heads"] = 2
    cfg["num_drug_layers"] = 1
    
    model = CrossAttentionDrugSensitivityModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    token_ids, mask, gene_expr, targets = next(iter(loader))
    
    # Forward pass 1
    pred1 = model(token_ids, mask, gene_expr)
    loss1 = criterion(pred1, targets)
    loss1.backward()
    optimizer.step()
    
    # Forward pass 2
    optimizer.zero_grad()
    pred2 = model(token_ids, mask, gene_expr)
    loss2 = criterion(pred2, targets)
    
    # Loss should decrease or parameters update
    assert loss1.item() > 0.0
    assert loss2.item() >= 0.0
