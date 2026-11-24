"""Configuration and clinical registry for Cross-Attention Drug Synergy."""

from typing import Dict, List, Any

# ============================================================================
# 1. 8 Hallmark Cancer Pathways & 128 Constituent Landmark Genes
# ============================================================================
CANCER_PATHWAYS: Dict[str, List[str]] = {
    "RTK_EGFR": [
        "EGFR", "ERBB2", "ERBB3", "ERBB4", "MET", "AXL", "KIT", "PDGFRA",
        "PDGFRB", "FGFR1", "FGFR2", "FGFR3", "IGF1R", "RET", "ROS1", "ALK"
    ],
    "PI3K_AKT_MTOR": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PTEN", "AKT1", "AKT2", "AKT3",
        "MTOR", "RPTOR", "RICTOR", "TSC1", "TSC2", "RHEB", "STK11", "PDK1"
    ],
    "RAS_RAF_MEK_MAPK": [
        "KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1",
        "MAPK3", "NF1", "SOS1", "GRB2", "SHC1", "PTPN11", "DUSP6", "ETV4"
    ],
    "CELL_CYCLE_CDK": [
        "CDK4", "CDK6", "CDK2", "CDK1", "CCND1", "CCND2", "CCNE1", "CCNA2",
        "CDKN2A", "CDKN1A", "CDKN1B", "RB1", "E2F1", "E2F3", "MYC", "WEE1"
    ],
    "DNA_DAMAGE_REPAIR": [
        "BRCA1", "BRCA2", "PARP1", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51",
        "TP53", "MDM2", "PALB2", "BAP1", "MLH1", "MSH2", "ERCC1", "XPA"
    ],
    "APOPTOSIS_SURVIVAL": [
        "BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BIM", "PUMA", "NOXA",
        "CASP3", "CASP8", "CASP9", "APAF1", "XIAP", "BIRC5", "FAS", "BID"
    ],
    "ANGIOGENESIS_METASTASIS": [
        "VEGFA", "VEGFB", "KDR", "FLT1", "FLT4", "HIF1A", "EPAS1", "VHL",
        "CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "MMP2", "MMP9"
    ],
    "EPIGENETIC_CHROMATIN": [
        "HDAC1", "HDAC2", "HDAC6", "EP300", "CREBBP", "EZH2", "ARID1A", "SMARCA4",
        "KMT2A", "KMT2D", "SETD2", "DNMT1", "DNMT3A", "TET2", "BRD4", "KAT6A"
    ]
}

# Flattened list of all 128 unique genes in deterministic order
ALL_GENES: List[str] = [gene for pathway in CANCER_PATHWAYS.values() for gene in pathway]
GENE_TO_INDEX: Dict[str, int] = {gene: idx for idx, gene in enumerate(ALL_GENES)}
PATHWAY_NAMES: List[str] = list(CANCER_PATHWAYS.keys())
NUM_PATHWAYS: int = len(PATHWAY_NAMES)
NUM_GENES: int = len(ALL_GENES)

# ============================================================================
# 2. Curated Oncology Small-Molecule Panel with Verified Canonical SMILES
# ============================================================================
CURATED_DRUGS: Dict[str, Dict[str, Any]] = {
    "Osimertinib": {
        "smiles": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
        "target_pathway": "RTK_EGFR",
        "primary_targets": ["EGFR T790M", "EGFR L858R", "HER2"],
        "drug_class": "3rd-Gen Covalent TKI",
        "molecular_weight": 499.6,
    },
    "Gefitinib": {
        "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "target_pathway": "RTK_EGFR",
        "primary_targets": ["EGFR L858R", "EGFR Exon 19 del"],
        "drug_class": "1st-Gen Reversible TKI",
        "molecular_weight": 446.9,
    },
    "Lapatinib": {
        "smiles": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
        "target_pathway": "RTK_EGFR",
        "primary_targets": ["EGFR", "ERBB2/HER2"],
        "drug_class": "Dual Reversible TKI",
        "molecular_weight": 581.1,
    },
    "Trametinib": {
        "smiles": "CC1=C(C(=O)N(C1=O)c1c(C)cccc1I)Nc1c(F)cc(C(=O)NC2CC2)cc1F",
        "target_pathway": "RAS_RAF_MEK_MAPK",
        "primary_targets": ["MEK1", "MEK2"],
        "drug_class": "Allosteric MEK Inhibitor",
        "molecular_weight": 615.4,
    },
    "Vemurafenib": {
        "smiles": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F",
        "target_pathway": "RAS_RAF_MEK_MAPK",
        "primary_targets": ["BRAF V600E"],
        "drug_class": "Type-I BRAF Kinase Inhibitor",
        "molecular_weight": 489.9,
    },
    "Palbociclib": {
        "smiles": "CC(=O)c1c(C)c2cnc(Nc3ccc(N4CCNCC4)cn3)nc2n(C2CCCC2)c1=O",
        "target_pathway": "CELL_CYCLE_CDK",
        "primary_targets": ["CDK4", "CDK6"],
        "drug_class": "Selective CDK4/6 Inhibitor",
        "molecular_weight": 447.5,
    },
    "Olaparib": {
        "smiles": "O=C(c1cc(Cc2n[nH]c(=O)c3ccccc23)ccc1F)N1CCN(C(=O)C2CC2)CC1",
        "target_pathway": "DNA_DAMAGE_REPAIR",
        "primary_targets": ["PARP1", "PARP2"],
        "drug_class": "PARP Catalytic Inhibitor",
        "molecular_weight": 434.5,
    },
    "Imatinib": {
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "target_pathway": "RTK_EGFR",
        "primary_targets": ["BCR-ABL", "KIT", "PDGFR"],
        "drug_class": "Type-II Kinase Inhibitor",
        "molecular_weight": 493.6,
    },
    "Sorafenib": {
        "smiles": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
        "target_pathway": "ANGIOGENESIS_METASTASIS",
        "primary_targets": ["VEGFR2", "PDGFRB", "C-RAF"],
        "drug_class": "Multi-Kinase Inhibitor",
        "molecular_weight": 464.8,
    },
    "Venetoclax": {
        "smiles": "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(Oc4cnc5[nH]ccc5c4)c3)CC2)=C(c2ccc(Cl)cc2)C1",
        "target_pathway": "APOPTOSIS_SURVIVAL",
        "primary_targets": ["BCL2"],
        "drug_class": "BH3 Mimetic BCL-2 Inhibitor",
        "molecular_weight": 868.4,
    },
    "Alpelisib": {
        "smiles": "CC(C)(C(=O)Nc1ncc(s1)-c1nc(NC(=O)C2CCCC2)ccn1)c1ccccc1",
        "target_pathway": "PI3K_AKT_MTOR",
        "primary_targets": ["PIK3CA (p110alpha)"],
        "drug_class": "Alpha-selective PI3K Inhibitor",
        "molecular_weight": 441.5,
    },
    "Vorinostat": {
        "smiles": "O=C(CCCCCCC(=O)NO)Nc1ccccc1",
        "target_pathway": "EPIGENETIC_CHROMATIN",
        "primary_targets": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"],
        "drug_class": "Pan-HDAC Inhibitor",
        "molecular_weight": 264.3,
    }
}

# ============================================================================
# 3. Representative Cancer Cell Lines with Genomic Drivers
# ============================================================================
REPRESENTATIVE_CELL_LINES: Dict[str, Dict[str, Any]] = {
    "PC-9": {
        "cancer_type": "NSCLC (Lung)",
        "driver_pathway": "RTK_EGFR",
        "driver_alteration": "EGFR Exon 19 del (Hyper-sensitive to EGFR TKIs)",
        "upregulated": ["RTK_EGFR", "PI3K_AKT_MTOR"],
        "downregulated": ["APOPTOSIS_SURVIVAL"]
    },
    "A549": {
        "cancer_type": "NSCLC (Lung)",
        "driver_pathway": "RAS_RAF_MEK_MAPK",
        "driver_alteration": "KRAS G12S + STK11 loss (Resistant to EGFR TKIs)",
        "upregulated": ["RAS_RAF_MEK_MAPK", "ANGIOGENESIS_METASTASIS"],
        "downregulated": ["DNA_DAMAGE_REPAIR"]
    },
    "MCF-7": {
        "cancer_type": "Breast (Luminal A)",
        "driver_pathway": "CELL_CYCLE_CDK",
        "driver_alteration": "PIK3CA E545K + ER+ (Sensitive to CDK4/6 & PI3K inhibitors)",
        "upregulated": ["CELL_CYCLE_CDK", "PI3K_AKT_MTOR"],
        "downregulated": ["ANGIOGENESIS_METASTASIS"]
    },
    "MDA-MB-231": {
        "cancer_type": "Breast (TNBC)",
        "driver_pathway": "ANGIOGENESIS_METASTASIS",
        "driver_alteration": "KRAS + BRAF + TP53 mut (Aggressive metastatic phenotype)",
        "upregulated": ["ANGIOGENESIS_METASTASIS", "EPIGENETIC_CHROMATIN"],
        "downregulated": ["DNA_DAMAGE_REPAIR"]
    },
    "K562": {
        "cancer_type": "CML (Leukemia)",
        "driver_pathway": "RTK_EGFR",
        "driver_alteration": "BCR-ABL1 fusion (Exquisitely sensitive to Imatinib)",
        "upregulated": ["RTK_EGFR", "RAS_RAF_MEK_MAPK"],
        "downregulated": ["APOPTOSIS_SURVIVAL"]
    },
    "SK-MEL-28": {
        "cancer_type": "Melanoma (Skin)",
        "driver_pathway": "RAS_RAF_MEK_MAPK",
        "driver_alteration": "BRAF V600E (Sensitive to Vemurafenib & Trametinib)",
        "upregulated": ["RAS_RAF_MEK_MAPK", "CELL_CYCLE_CDK"],
        "downregulated": ["APOPTOSIS_SURVIVAL"]
    },
    "MDA-MB-436": {
        "cancer_type": "Breast (BRCA1-null)",
        "driver_pathway": "DNA_DAMAGE_REPAIR",
        "driver_alteration": "BRCA1 5396+1G>A (Defective HR, synthetic lethality to Olaparib)",
        "upregulated": ["CELL_CYCLE_CDK", "APOPTOSIS_SURVIVAL"],
        "downregulated": ["DNA_DAMAGE_REPAIR"]
    },
    "HT-29": {
        "cancer_type": "Colorectal",
        "driver_pathway": "RAS_RAF_MEK_MAPK",
        "driver_alteration": "BRAF V600E + PIK3CA P449T",
        "upregulated": ["RAS_RAF_MEK_MAPK", "PI3K_AKT_MTOR"],
        "downregulated": ["DNA_DAMAGE_REPAIR"]
    },
    "U-87 MG": {
        "cancer_type": "Glioblastoma",
        "driver_pathway": "PI3K_AKT_MTOR",
        "driver_alteration": "PTEN null (Unchecked AKT/mTOR survival pathway)",
        "upregulated": ["PI3K_AKT_MTOR", "ANGIOGENESIS_METASTASIS"],
        "downregulated": ["APOPTOSIS_SURVIVAL"]
    },
    "OVCAR-3": {
        "cancer_type": "Ovarian High-Grade",
        "driver_pathway": "DNA_DAMAGE_REPAIR",
        "driver_alteration": "TP53 mut + CCNE1 amplification",
        "upregulated": ["CELL_CYCLE_CDK", "EPIGENETIC_CHROMATIN"],
        "downregulated": ["DNA_DAMAGE_REPAIR"]
    }
}

# ============================================================================
# 4. Model Architecture Configuration (Optimized for Laptop CPU Execution)
# ============================================================================
MODEL_CONFIG: Dict[str, Any] = {
    # SMILES Transformer Encoder
    "smiles_vocab_size": 75,
    "max_smiles_len": 128,
    "d_model": 128,
    "n_heads": 4,
    "num_drug_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.10,
    
    # Gene Pathway Representation
    "num_pathways": NUM_PATHWAYS,  # 8 pathways
    "num_genes": NUM_GENES,        # 128 landmark genes
    "genes_per_pathway": 16,
    
    # Bi-modal Cross-Attention
    "cross_attn_heads": 4,
    
    # Regressor Head
    "regressor_hidden_dims": [128, 64],
}

# ============================================================================
# 4. Training & Optimization Configuration
# ============================================================================
TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": 32,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "num_epochs": 15,
    "warmup_epochs": 2,
    "seed": 42,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
}
