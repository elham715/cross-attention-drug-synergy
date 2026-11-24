"""Unit tests for SMILESTokenizer."""

import torch
import pytest
from src.smiles_tokenizer import SMILESTokenizer
from src.config import CURATED_DRUGS

def test_tokenizer_initialization():
    tokenizer = SMILESTokenizer(max_length=64)
    assert tokenizer.vocab_size > 30
    assert tokenizer.pad_id == 0
    assert tokenizer.cls_id == 1
    assert tokenizer.unk_id == 2

def test_smiles_tokenization_atoms():
    tokenizer = SMILESTokenizer(max_length=64)
    smiles = "CCO"  # Ethanol
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ["C", "C", "O"]

def test_smiles_complex_heteroatoms():
    tokenizer = SMILESTokenizer(max_length=64)
    # Osimertinib SMILES snippet
    smiles = "c1cc(Cl)c(F)cc1"
    tokens = tokenizer.tokenize(smiles)
    assert "Cl" in tokens
    assert "F" in tokens
    assert "c" in tokens

def test_encoding_shapes_and_mask():
    tokenizer = SMILESTokenizer(max_length=32)
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    token_ids, mask = tokenizer.encode(smiles)
    
    assert token_ids.shape == (32,)
    assert mask.shape == (32,)
    assert token_ids.dtype == torch.long
    assert mask.dtype == torch.bool
    
    # CLS token at position 0
    assert token_ids[0].item() == tokenizer.cls_id
    assert mask[0].item() is True
    
    # Valid tokens have mask=True, padding has mask=False
    num_valid = mask.sum().item()
    assert num_valid > 5
    assert (token_ids[num_valid:] == tokenizer.pad_id).all()

def test_batch_encode():
    tokenizer = SMILESTokenizer(max_length=48)
    smiles_list = [
        CURATED_DRUGS["Osimertinib"]["smiles"],
        CURATED_DRUGS["Gefitinib"]["smiles"]
    ]
    batch_ids, batch_masks = tokenizer.batch_encode(smiles_list)
    assert batch_ids.shape == (2, 48)
    assert batch_masks.shape == (2, 48)
