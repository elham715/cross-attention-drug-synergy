"""Data loading and biological benchmark generation for drug sensitivity prediction."""

import os
import csv
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .config import (
    CURATED_DRUGS,
    CANCER_PATHWAYS,
    ALL_GENES,
    PATHWAY_NAMES,
    NUM_GENES,
    NUM_PATHWAYS,
    REPRESENTATIVE_CELL_LINES,
)
from .smiles_tokenizer import SMILESTokenizer

def generate_synthetic_gdsc_data(
    num_samples: int = 600,
    random_seed: int = 42,
    csv_path: Optional[str] = None
) -> List[Dict]:
    """Generates realistic GDSC cancer drug screening benchmark records.
    
    Each record pairs a small-molecule targeted inhibitor with a cancer cell line
    having quantified RNA-seq expression across the 128 landmark genes, and an
    empirically grounded log(IC50) sensitivity value.
    """
    np.random.seed(random_seed)
    drug_names = list(CURATED_DRUGS.keys())
    cell_line_names = list(REPRESENTATIVE_CELL_LINES.keys())
    
    records = []
    
    for _ in range(num_samples):
        # Sample drug and cell line
        d_name = np.random.choice(drug_names)
        c_name = np.random.choice(cell_line_names)
        
        drug_info = CURATED_DRUGS[d_name]
        cell_info = REPRESENTATIVE_CELL_LINES[c_name]
        
        # 1. Synthesize 128-gene expression profile (z-scores around mean 0, std 1)
        expression = np.random.normal(loc=0.0, scale=0.8, size=NUM_GENES).astype(np.float32)
        
        # Modulate expression based on cell line biology
        for up_path in cell_info["upregulated"]:
            if up_path in CANCER_PATHWAYS:
                for gene in CANCER_PATHWAYS[up_path]:
                    idx = ALL_GENES.index(gene)
                    expression[idx] += np.random.uniform(1.2, 2.5)  # Upregulated oncogenes
                    
        for down_path in cell_info["downregulated"]:
            if down_path in CANCER_PATHWAYS:
                for gene in CANCER_PATHWAYS[down_path]:
                    idx = ALL_GENES.index(gene)
                    expression[idx] -= np.random.uniform(1.0, 2.0)  # Downregulated tumor suppressors
                    
        # 2. Compute Ground-Truth log(IC50) in uM based on biological synergy
        # Base IC50 for unselected drugs: moderate resistance (ln(IC50) around 2.0 to 4.0 uM)
        base_ln_ic50 = np.random.normal(loc=2.5, scale=0.6)
        
        target_pathway = drug_info["target_pathway"]
        
        # Case A: On-target oncogene addiction -> Exquisite sensitivity
        if target_pathway == cell_info["driver_pathway"]:
            # Specific high-potency pairs
            if (d_name == "Osimertinib" and c_name == "PC-9") or \
               (d_name == "Gefitinib" and c_name == "PC-9") or \
               (d_name == "Imatinib" and c_name == "K562") or \
               (d_name == "Vemurafenib" and c_name == "SK-MEL-28") or \
               (d_name == "Palbociclib" and c_name == "MCF-7") or \
               (d_name == "Olaparib" and c_name == "MDA-MB-436"):
                # Extreme sensitivity: sub-nanomolar to low nanomolar (ln(IC50) in -4.0 to -1.5 range)
                delta = -np.random.uniform(3.5, 5.0)
            else:
                # Moderate pathway inhibition
                delta = -np.random.uniform(2.0, 3.2)
        # Case B: Drug targets an upregulated secondary pathway -> Moderate sensitivity
        elif target_pathway in cell_info["upregulated"]:
            delta = -np.random.uniform(1.0, 2.0)
        # Case C: Primary resistance (e.g. KRAS mutant resistant to EGFR TKI)
        elif d_name in ["Osimertinib", "Gefitinib", "Lapatinib"] and c_name == "A549":
            delta = +np.random.uniform(1.5, 2.5)  # Resistance surge
        else:
            delta = 0.0
            
        final_ln_ic50 = float(base_ln_ic50 + delta + np.random.normal(0, 0.25))
        
        records.append({
            "drug_name": d_name,
            "smiles": drug_info["smiles"],
            "target_pathway": target_pathway,
            "cell_line": c_name,
            "cancer_type": cell_info["cancer_type"],
            "gene_expression": expression.tolist(),
            "ln_ic50": final_ln_ic50
        })
        
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            header = ["drug_name", "smiles", "target_pathway", "cell_line", "cancer_type", "ln_ic50"] + ALL_GENES
            writer.writerow(header)
            for rec in records:
                row = [
                    rec["drug_name"],
                    rec["smiles"],
                    rec["target_pathway"],
                    rec["cell_line"],
                    rec["cancer_type"],
                    f"{rec['ln_ic50']:.4f}"
                ] + [f"{x:.4f}" for x in rec["gene_expression"]]
                writer.writerow(row)
                
    return records


class DrugSensitivityDataset(Dataset):
    """PyTorch Dataset for multi-modal drug sensitivity."""
    
    def __init__(self, records: List[Dict], tokenizer: SMILESTokenizer):
        self.records = records
        self.tokenizer = tokenizer
        
        # Pre-tokenize all SMILES for maximum CPU throughput
        self.tokenized_drugs = {}
        unique_smiles = set(r["smiles"] for r in records)
        for s in unique_smiles:
            ids, mask = self.tokenizer.encode(s)
            self.tokenized_drugs[s] = (ids, mask)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        smiles = rec["smiles"]
        token_ids, attention_mask = self.tokenized_drugs[smiles]
        gene_expr = torch.tensor(rec["gene_expression"], dtype=torch.float32)
        target = torch.tensor([rec["ln_ic50"]], dtype=torch.float32)
        return token_ids, attention_mask, gene_expr, target


def create_dataloaders(
    records: List[Dict],
    tokenizer: SMILESTokenizer,
    batch_size: int = 32,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Splits dataset and creates PyTorch DataLoaders."""
    np.random.seed(seed)
    indices = np.random.permutation(len(records))
    
    n_total = len(records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]
    
    train_dataset = DrugSensitivityDataset(train_records, tokenizer)
    val_dataset = DrugSensitivityDataset(val_records, tokenizer)
    test_dataset = DrugSensitivityDataset(test_records, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
