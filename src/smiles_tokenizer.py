"""SMILES regex tokenizer for small-molecule chemical structures."""

import re
from typing import List, Dict, Tuple, Optional
import torch

class SMILESTokenizer:
    """Regex-based SMILES tokenizer with special tokens.
    
    Tokens include organic atoms, aromatic rings, charges, stereochemistry,
    and structural branches.
    """
    
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"
    UNK_TOKEN = "<unk>"
    
    # Comprehensive SMILES token vocabulary
    BASE_VOCAB = [
        PAD_TOKEN, CLS_TOKEN, UNK_TOKEN,
        # Organic atoms
        "C", "c", "N", "n", "O", "o", "S", "s", "P", "p",
        "F", "Cl", "Br", "I", "B", "Si", "Na", "K",
        # Bonds & Operators
        "=", "#", "-", ":", "/", "\\", ".",
        # Branches & Rings
        "(", ")", "[", "]",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        # Stereochemistry & Charges
        "@", "@@", "+", "-", "%", "H"
    ]
    
    # Regex pattern to match bracketed atoms, multi-char halogens, or single chars
    SMILES_REGEX_PATTERN = r"(\[[^\]]+]|Br|Cl|Si|Na|Se|@@|@[a-zA-Z]?|[A-Z][a-z]?|[cnosp]|[0-9]|\(|\)|\.|\=|\#|\-|\:|\/|\\|\+|\-|\%)"

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.vocab: Dict[str, int] = {token: idx for idx, token in enumerate(self.BASE_VOCAB)}
        self.inv_vocab: Dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.pad_id = self.vocab[self.PAD_TOKEN]
        self.cls_id = self.vocab[self.CLS_TOKEN]
        self.unk_id = self.vocab[self.UNK_TOKEN]
        self.regex = re.compile(self.SMILES_REGEX_PATTERN)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into chemical token substrings."""
        tokens = []
        for match in self.regex.finditer(smiles):
            token = match.group(0)
            tokens.append(token)
        return tokens

    def encode(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a SMILES string into token IDs and an attention mask.
        
        Returns:
            token_ids: Tensor of shape [max_length]
            attention_mask: Tensor of shape [max_length] (1 for valid tokens, 0 for pad)
        """
        raw_tokens = self.tokenize(smiles)
        # Prepend [CLS] token
        token_ids = [self.cls_id]
        for t in raw_tokens:
            token_ids.append(self.vocab.get(t, self.unk_id))
            if len(token_ids) >= self.max_length:
                break
                
        seq_len = len(token_ids)
        mask = [1] * seq_len
        
        # Pad up to max_length
        if seq_len < self.max_length:
            padding_len = self.max_length - seq_len
            token_ids.extend([self.pad_id] * padding_len)
            mask.extend([0] * padding_len)
            
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def batch_encode(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of SMILES strings."""
        batch_ids = []
        batch_masks = []
        for s in smiles_list:
            ids, mask = self.encode(s)
            batch_ids.append(ids)
            batch_masks.append(mask)
        return torch.stack(batch_ids, dim=0), torch.stack(batch_masks, dim=0)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back into a reconstructed SMILES string."""
        tokens = []
        for tid in token_ids.tolist():
            if tid == self.pad_id:
                continue
            if tid == self.cls_id:
                continue
            tokens.append(self.inv_vocab.get(tid, self.UNK_TOKEN))
        return "".join(tokens)
