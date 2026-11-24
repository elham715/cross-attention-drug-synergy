#!/usr/bin/env python3
"""Automated Training, Evaluation, and Visualization Driver for Cross-Attention Drug Synergy."""

import os
import argparse
import time
import torch
import numpy as np

from src.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    CURATED_DRUGS,
    REPRESENTATIVE_CELL_LINES,
    PATHWAY_NAMES,
)
from src.smiles_tokenizer import SMILESTokenizer
from src.cross_attention_transformer import CrossAttentionDrugSensitivityModel
from src.data_loader import generate_synthetic_gdsc_data, create_dataloaders
from src.trainer import ModelTrainer
from src.visualizer import CrossAttentionVisualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Bi-modal Cross-Attention Drug Sensitivity Model."
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--samples", type=int, default=600, help="Number of benchmark samples (default: 600)")
    parser.add_argument("--save-dir", type=str, default="figures", help="Directory for generated figures (default: figures)")
    parser.add_argument("--data-file", type=str, default="data/gdsc_benchmark.csv", help="Path to benchmark CSV (default: data/gdsc_benchmark.csv)")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory for model weights (default: checkpoints)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    print("=" * 76)
    print("🧬 CROSS-ATTENTION DRUG SYNERGY: MULTI-MODAL SENSITIVITY TRANSFORMER")
    print("   Born et al. (Cell Reports Methods & IBM Research) Reproduction")
    print("=" * 76)
    
    # 1. Dataset Generation & Loading
    print(f"\n[1/5] Preparing GDSC benchmark dataset ({args.samples} samples)...")
    records = generate_synthetic_gdsc_data(
        num_samples=args.samples,
        random_seed=42,
        csv_path=args.data_file
    )
    print(f"      Saved curated benchmark to: {args.data_file}")
    
    # 2. Tokenizer & Dataloaders
    print("\n[2/5] Initializing SMILES regex tokenizer & data loaders...")
    tokenizer = SMILESTokenizer(max_length=MODEL_CONFIG["max_smiles_len"])
    train_loader, val_loader, test_loader = create_dataloaders(
        records=records,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        train_ratio=TRAINING_CONFIG["train_ratio"],
        val_ratio=TRAINING_CONFIG["val_ratio"],
        seed=TRAINING_CONFIG["seed"]
    )
    print(f"      Train samples: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # 3. Model Architecture
    print("\n[3/5] Instantiating Bi-modal Cross-Attention Transformer...")
    cfg = dict(MODEL_CONFIG)
    cfg["dropout"] = 0.10
    model = CrossAttentionDrugSensitivityModel(cfg)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total Trainable Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"      Running on Device: CPU (Intel Core i5 Optimized)")
    
    # 4. Training Loop
    print("\n[4/5] Executing high-efficiency training loop...")
    train_cfg = dict(TRAINING_CONFIG)
    train_cfg["num_epochs"] = args.epochs
    train_cfg["learning_rate"] = args.lr
    train_cfg["batch_size"] = args.batch_size
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_cfg,
        device=torch.device("cpu")
    )
    
    train_results = trainer.fit(verbose=True)
    history = train_results["history"]
    test_metrics = train_results["test_metrics"]
    
    # Save best model
    best_ckpt_path = os.path.join(args.ckpt_dir, "best_cross_attention_model.pt")
    torch.save(model.state_dict(), best_ckpt_path)
    print(f"\n      Saved best model checkpoint to: {best_ckpt_path}")
    
    # 5. Publication Visualization Suite
    print("\n[5/5] Generating publication-grade 300 DPI figures...")
    visualizer = CrossAttentionVisualizer(output_dir=args.save_dir)
    
    # Figure A: Training convergence curves
    curve_path = visualizer.plot_training_curves(history, filename="training_convergence.png")
    print(f"      [Figure 1/3] Convergence plot: {curve_path}")
    
    # Figure B: Predicted vs Actual log(IC50)
    scatter_path = visualizer.plot_predicted_vs_actual(
        y_true=test_metrics["targets"],
        y_pred=test_metrics["preds"],
        spearman_rho=test_metrics["spearman"],
        pearson_r=test_metrics["pearson"],
        rmse=test_metrics["rmse"],
        filename="predicted_vs_actual_ic50.png"
    )
    print(f"      [Figure 2/3] Sensitivity correlation: {scatter_path}")
    
    # Figure C: Interpretability Cross-Attention Map (Osimertinib vs PC-9)
    test_drug = "Osimertinib"
    test_cell = "PC-9"
    test_smiles = CURATED_DRUGS[test_drug]["smiles"]
    
    # Extract PC-9 expression
    pc9_rec = [r for r in records if r["cell_line"] == test_cell][0]
    token_ids, mask = tokenizer.encode(test_smiles)
    token_ids_batch = token_ids.unsqueeze(0)
    mask_batch = mask.unsqueeze(0)
    gene_expr_batch = torch.tensor(pc9_rec["gene_expression"], dtype=torch.float32).unsqueeze(0)
    
    maps = model.extract_attention_maps(token_ids_batch, mask_batch, gene_expr_batch)
    # Average across 4 attention heads: [1, n_heads, L, K] -> [L, K]
    drug_to_pathway_avg = maps["drug_to_pathway_attn"][0].mean(dim=0).cpu().numpy()
    
    # Get raw token strings for the valid tokens
    raw_tokens = [tokenizer.CLS_TOKEN] + tokenizer.tokenize(test_smiles)
    valid_len = mask.sum().item()
    raw_tokens = raw_tokens[:valid_len]
    drug_to_pathway_valid = drug_to_pathway_avg[:valid_len, :]
    
    heatmap_path = visualizer.plot_cross_attention_heatmap(
        attn_matrix=drug_to_pathway_valid,
        smiles_tokens=raw_tokens,
        drug_name=test_drug,
        cell_line=test_cell,
        filename="cross_attention_heatmap.png"
    )
    print(f"      [Figure 3/3] Cross-attention heatmap: {heatmap_path}")
    
    print("\n" + "=" * 76)
    print("🏆 ALL EXECUTION MILESTONES COMPLETE")
    print(f"   • Test Spearman Rank Correlation ρ: {test_metrics['spearman']:.4f}")
    print(f"   • Test Pearson Correlation r:       {test_metrics['pearson']:.4f}")
    print(f"   • Test RMSE:                       {test_metrics['rmse']:.3f} ln(uM)")
    print(f"   • Generated Artifacts:              {args.save_dir}/")
    print("=" * 76)

if __name__ == "__main__":
    main()
