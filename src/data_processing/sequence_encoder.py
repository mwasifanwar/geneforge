import torch
import numpy as np
from typing import List, Dict

class SequenceEncoder:
    def __init__(self):
        self.config = Config()
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
        self.vocab_size = len(self.amino_acids)
        self.max_length = self.config.get('data.max_sequence_length')
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        tokens = []
        for aa in sequence[:self.max_length]:
            if aa in self.amino_acids:
                tokens.append(self.amino_acids.index(aa))
            else:
                tokens.append(self.amino_acids.index('X'))
        
        while len(tokens) < self.max_length:
            tokens.append(self.vocab_size - 1)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode_sequence(self, tokens: torch.Tensor) -> str:
        sequence = []
        for token in tokens:
            if token < self.vocab_size:
                sequence.append(self.amino_acids[token])
            else:
                sequence.append('X')
        return ''.join(sequence).replace('X', '')
    
    def create_positional_encoding(self, sequence_length: int, d_model: int) -> torch.Tensor:
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(sequence_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def create_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != self.vocab_size - 1).float()
        return mask.unsqueeze(1).unsqueeze(2)
    
    def calculate_sequence_features(self, sequence: str) -> Dict:
        features = {}
        features['length'] = len(sequence)
        features['molecular_weight'] = self._calculate_molecular_weight(sequence)
        features['hydrophobicity'] = self._calculate_hydrophobicity(sequence)
        features['charge'] = self._calculate_net_charge(sequence)
        features['aromaticity'] = self._calculate_aromaticity(sequence)
        return features
    
    def _calculate_molecular_weight(self, sequence: str) -> float:
        aa_weights = {
            'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
            'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
            'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
            'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
        }
        return sum(aa_weights.get(aa, 110.0) for aa in sequence)
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        hydrophobic_aas = {'A', 'V', 'L', 'I', 'P', 'F', 'W', 'M'}
        return sum(1 for aa in sequence if aa in hydrophobic_aas) / len(sequence)
    
    def _calculate_net_charge(self, sequence: str) -> float:
        positive_aas = {'K', 'R', 'H'}
        negative_aas = {'D', 'E'}
        return sum(1 for aa in sequence if aa in positive_aas) - sum(1 for aa in sequence if aa in negative_aas)
    
    def _calculate_aromaticity(self, sequence: str) -> float:
        aromatic_aas = {'F', 'W', 'Y'}
        return sum(1 for aa in sequence if aa in aromatic_aas) / len(sequence)