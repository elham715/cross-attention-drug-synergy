"""Publication-grade visualization suite for Cross-Attention Drug Synergy."""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .config import PATHWAY_NAMES

# 3Blue1Brown / Dark-mode color palette
BG_COLOR = "#0F111A"
CARD_BG = "#1A1D2B"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#8B949E"
CYAN_ACCENT = "#38BDF8"
GOLD_ACCENT = "#FACC15"
PURPLE_ACCENT = "#C084FC"
GREEN_ACCENT = "#4ADE80"
GRID_COLOR = "#2D3748"

class CrossAttentionVisualizer:
    """Renders 300 DPI publication-grade figures in 3B1B dark aesthetic."""
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._apply_style()

    def _apply_style(self):
        plt.rcParams.update({
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "axes.edgecolor": "#30363D",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "font.family": "serif",
            "font.size": 11,
        })

    def plot_cross_attention_heatmap(
        self,
        attn_matrix: np.ndarray,
        smiles_tokens: List[str],
        drug_name: str = "Osimertinib",
        cell_line: str = "PC-9",
        filename: str = "cross_attention_heatmap.png"
    ) -> str:
        """Plots drug atom tokens querying the 8 cancer hallmark pathways.
        
        Args:
            attn_matrix: 2D array of shape [L_valid, 8]
            smiles_tokens: List of string tokens of length L_valid
            drug_name: Name of the drug
            cell_line: Name of the cancer cell line
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # Clip to first 35 tokens for aesthetic display if too long
        display_len = min(len(smiles_tokens), 35)
        sub_matrix = attn_matrix[:display_len, :]
        sub_tokens = smiles_tokens[:display_len]
        
        cax = ax.imshow(sub_matrix, cmap="inferno", aspect="auto", interpolation="nearest")
        
        # Colorbar
        cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        cbar.set_label("Cross-Attention Weight $\\alpha_{ij}$", color=TEXT_COLOR, size=11, labelpad=10)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
        
        # Labels
        ax.set_xticks(np.arange(len(PATHWAY_NAMES)))
        ax.set_xticklabels(PATHWAY_NAMES, rotation=45, ha="right", fontsize=9, color=CYAN_ACCENT)
        ax.set_yticks(np.arange(display_len))
        ax.set_yticklabels([f"[{i:02d}] {tok}" for i, tok in enumerate(sub_tokens)], fontsize=8, color=TEXT_COLOR)
        
        ax.set_title(
            f"Bi-modal Cross-Attention: {drug_name} Atoms $\\longleftrightarrow$ Cancer Pathways ({cell_line})",
            fontsize=13,
            fontweight="bold",
            color=GOLD_ACCENT,
            pad=16
        )
        ax.set_xlabel("Genomic Hallmark Signaling Pathways", color=TEXT_COLOR, fontsize=11, labelpad=10)
        ax.set_ylabel("SMILES Chemical Tokens (Atoms & Bonds)", color=TEXT_COLOR, fontsize=11, labelpad=10)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path

    def plot_predicted_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        spearman_rho: float,
        pearson_r: float,
        rmse: float,
        filename: str = "predicted_vs_actual_ic50.png"
    ) -> str:
        """Plots experimental vs model-predicted log(IC50) values."""
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        
        # Scatter points with color mapped to error
        errors = np.abs(y_true - y_pred)
        scatter = ax.scatter(
            y_true,
            y_pred,
            c=errors,
            cmap="viridis",
            alpha=0.85,
            edgecolors="#FFFFFF",
            linewidths=0.4,
            s=55,
            zorder=3
        )
        
        # Identity line y = x
        min_val = min(y_true.min(), y_pred.min()) - 0.5
        max_val = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], color=GOLD_ACCENT, linestyle="--", linewidth=1.5, alpha=0.9, label="Identity Line ($y = x$)")
        
        # Best-fit regression line
        poly_fit = np.polyfit(y_true, y_pred, 1)
        fit_fn = np.poly1d(poly_fit)
        x_vals = np.linspace(min_val, max_val, 100)
        ax.plot(x_vals, fit_fn(x_vals), color=CYAN_ACCENT, linewidth=1.8, label=f"Fit: $y = {poly_fit[0]:.2f}x + {poly_fit[1]:.2f}$")
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Metrics Callout Card
        stats_text = (
            f"Spearman $\\rho$: {spearman_rho:.4f}\n"
            f"Pearson $r$: {pearson_r:.4f}\n"
            f"RMSE: {rmse:.3f} $\\ln(\\mu M)$"
        )
        ax.text(
            0.05, 0.93,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=CARD_BG, edgecolor=CYAN_ACCENT, alpha=0.9),
            color=TEXT_COLOR
        )
        
        ax.set_title("Cross-Attention Transformer Sensitivity Benchmark (GDSC)", fontsize=13, fontweight="bold", color=GOLD_ACCENT, pad=15)
        ax.set_xlabel("Experimental $\\ln(\\mathrm{IC}_{50})$ in $\\mu M$ (GDSC Ground Truth)", fontsize=11, color=TEXT_COLOR, labelpad=8)
        ax.set_ylabel("Predicted $\\ln(\\mathrm{IC}_{50})$ in $\\mu M$ (Cross-Attention Model)", fontsize=11, color=TEXT_COLOR, labelpad=8)
        ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor="#30363D", fontsize=9)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_convergence.png"
    ) -> str:
        """Plots training & validation loss and Spearman rank correlation progression."""
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        
        # Panel 1: MSE Loss
        ax1.plot(epochs, history["train_loss"], color=CYAN_ACCENT, marker="o", markersize=4, linewidth=2, label="Train MSE")
        ax1.plot(epochs, history["val_loss"], color=GOLD_ACCENT, marker="s", markersize=4, linewidth=2, label="Val MSE")
        ax1.set_title("Loss Convergence (MSE)", fontsize=12, fontweight="bold", color=GOLD_ACCENT)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Mean Squared Error", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(facecolor=CARD_BG, edgecolor="#30363D")
        
        # Panel 2: Spearman & Pearson Correlation
        ax2.plot(epochs, history["val_spearman"], color=GREEN_ACCENT, marker="^", markersize=4, linewidth=2, label="Spearman $\\rho$")
        ax2.plot(epochs, history["val_pearson"], color=PURPLE_ACCENT, marker="d", markersize=4, linewidth=2, label="Pearson $r$")
        ax2.set_title("Validation Correlation Metrics", fontsize=12, fontweight="bold", color=GOLD_ACCENT)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Correlation Coefficient", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(facecolor=CARD_BG, edgecolor="#30363D")
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path
